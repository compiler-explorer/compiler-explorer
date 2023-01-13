// Copyright (c) 2018, Elliot Saba
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

import path from 'path';

import {BaseCompiler} from '../base-compiler';

import {BaseParser} from './argument-parsers';

import * as utils from '../utils';

export class JuliaCompiler extends BaseCompiler {
    static get key() {
        return 'julia';
    }

    constructor(info, env) {
        super(info, env);
        this.compiler.demangler = null;
        this.demanglerClass = null;
        this.compilerWrapperPath =
            this.compilerProps('compilerWrapper') || utils.resolvePathFromAppRoot('etc', 'scripts', 'julia_wrapper.jl');
    }

    // No demangling for now
    postProcessAsm(result, filters) {
        return result;
    }

    processAsm(result, filters, options) {
        const lineRe = /^<(\d+) (\d+) ([^ ]+) ([^>]*)>$/;
        const bytecodeLines = result.asm.split('\n');
        const bytecodeResult = [];
        // Every method block starts with a introductory line
        //   <[source code line] [output line number] [function name] [method types]>
        // Check for the starting line, add the method block, skip other lines
        var i = 0;
        while (i < bytecodeLines.length) {
            var line = bytecodeLines[i];
            const match = line.match(lineRe);

            if (match) {
                var source = parseInt(match[1]);
                var linenum = parseInt(match[2]);
                linenum = Math.min(linenum, bytecodeLines.length);
                var funname = match[3];
                var types = match[4];
                var j = 0;
                bytecodeResult.push({text: '<' + funname + ' ' + types + '>', source: {line: source, file: null}});
                while (j < linenum) {
                    bytecodeResult.push({text: bytecodeLines[i + 1 + j], source: {line: source, file: null}});
                    j++;
                }
                bytecodeResult.push({text: '', source: {line: null, file: null}});
                i += linenum + 1;
                continue;
            }
            i++;
        }
        return {asm: bytecodeResult};
    }

    optionsForFilter(filters, outputFilename) {
        let opts = [outputFilename];
        if (filters.optOutput) {
            opts += ['--optimize'];
        }
        return opts;
    }

    getArgumentParser() {
        return BaseParser;
    }

    runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        // compiler wrapper, then input should be first argument, not last
        options.unshift(options.pop());
        options.unshift(this.compilerWrapperPath);

        return this.exec(compiler, options, execOptions).then(result => {
            result.inputFilename = inputFilename;
            const transformedInput = result.filenameTransform(inputFilename);
            result.stdout = utils.parseOutput(result.stdout, transformedInput);
            result.stderr = utils.parseOutput(result.stderr, transformedInput);
            return result;
        });
    }
}
