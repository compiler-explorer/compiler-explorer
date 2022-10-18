// Copyright (c) 2018, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import * as fs from 'fs/promises';
import Path from 'path';

import Semver from 'semver';

import {BaseCompiler} from '../base-compiler';
import {SassAsmParser} from '../parsers/asm-parser-sass';
import {asSafeVer} from '../utils';

import {ClangParser} from './argument-parsers';

export class NvccCompiler extends BaseCompiler {
    static get key() {
        return 'nvcc';
    }

    constructor(info, env) {
        super(info, env);
        this.compiler.supportsOptOutput = true;
        this.compiler.supportsDeviceAsmView = true;
        this.deviceAsmParser = new SassAsmParser(this.compilerProps);
    }

    // TODO: (for all of CUDA)
    // * lots of whitespace from nvcc
    // * would be nice to try and filter unused `.func`s from e.g. clang output

    /**
     *
     * @param {import('../../types/features/filters.interfaces').ParseFilters} filters
     * @param {string} outputFilename
     * @param {string[]?} userOptions
     * @returns {string[]}
     */
    optionsForFilter(filters, outputFilename, userOptions) {
        const opts = ['-o', this.filename(outputFilename), '-g', '-lineinfo'];
        if (!filters.execute) {
            opts.push('-c', '-keep', '-keep-dir', Path.dirname(outputFilename));
            if (!filters.binary) {
                opts.push('-Xcompiler=-S');
            }
        }
        return opts;
    }

    getArgumentParser() {
        return ClangParser;
    }

    optOutputRequested(options) {
        return (
            super.optOutputRequested(options) ||
            options.includes('--optimization-info') ||
            options.includes('-opt-info')
        );
    }

    /**
     *
     * @param {string} outputFilename
     * @param {*} result
     * @param {number} maxOutput
     */
    async nvdisasm(outputFilename, result, maxOutput) {
        const {nvdisasm, semver} = this.compiler;

        const args = Semver.lt(asSafeVer(semver), '11.0.0', true)
            ? [outputFilename, '-c', '-g']
            : [outputFilename, '-c', '-g', '-hex'];

        const {code, execTime, stdout} = await this.exec(nvdisasm, args, {maxOutput, customCwd: result.dirPath});

        if (code !== 0) {
            result.asm = `<No output: ${Path.basename(nvdisasm)} returned ${code}>`;
        } else {
            result.objdumpTime = execTime;
            result.asm = this.postProcessObjdumpOutput(stdout);
        }
        return result;
    }

    /**
     *
     * @param {*} result
     * @param {string} outputFilename
     * @param {import('../../types/features/filters.interfaces').ParseFilters} filters
     */
    async postProcess(result, outputFilename, filters) {
        const maxSize = this.env.ceProps('max-asm-size', 64 * 1024 * 1024);
        const optPromise = result.hasOptOutput ? this.processOptOutput(result.optPath) : Promise.resolve('');
        const asmPromise = (
            !filters.binary
                ? fs.readFile(outputFilename, {encoding: 'utf8'})
                : this.objdump(outputFilename, {}, maxSize, filters.intel, filters.demangle, filters)
        ).then(asm => {
            result.asm = typeof asm === 'string' ? asm : asm.asm;
            return result;
        });
        return Promise.all([asmPromise, optPromise]);
    }

    /**
     *
     * @param {*} result
     * @param {*} filters
     * @param {import('../../types/compilation/compilation.interfaces').CompilationInfo} compilationInfo
     * @returns
     */
    async extractDeviceCode(result, filters, compilationInfo) {
        if (result.dirPath) {
            const outs = await fs.readdir(result.dirPath);
            const name = outs.find(f => f.endsWith(filters.binary ? '.cubin' : '.ptx'));
            if (name) {
                const path = Path.join(result.dirPath, name);
                const maxSize = this.env.ceProps('max-asm-size', 64 * 1024 * 1024);
                const asm = !filters.binary
                    ? await fs.readFile(path, {encoding: 'utf8'})
                    : await this.nvdisasm(path, {dirPath: result.dirPath}, maxSize).then(({asm}) => asm);
                result.devices = {
                    gpu: await this.postProcessAsm(
                        {
                            okToCache: filters.demangle,
                            ...this.deviceAsmParser.process(asm, filters),
                        },
                        filters,
                    ),
                };
            }
        }
        return result;
    }
}
