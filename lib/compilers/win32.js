// Copyright (c) 2018, Microsoft Corporation
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

import temp from 'temp';
import _ from 'underscore';

import { AsmParser } from '../asm-parser';
import { BaseCompiler } from '../base-compiler';
import { MapFileReaderVS } from '../map-file-vs';
import { PELabelReconstructor } from '../pe32-support';
import * as utils from '../utils';

export class Win32Compiler extends BaseCompiler {
    static get key() { return 'win32'; }

    constructor(compilerInfo, env) {
        super(compilerInfo, env);

        this.binaryAsmParser = new AsmParser(this.compilerProps);
    }

    newTempDir() {
        return new Promise((resolve, reject) => {
            temp.mkdir({prefix: 'compiler-explorer-compiler', dir: process.env.TMP}, (err, dirPath) => {
                if (err)
                    reject(`Unable to open temp file: ${err}`);
                else
                    resolve(dirPath);
            });
        });
    }

    getExecutableFilename(dirPath, outputFilebase) {
        return this.getOutputFilename(dirPath, outputFilebase) + '.exe';
    }

    getObjdumpOutputFilename(defaultOutputFilename) {
        return this.getExecutableFilename(path.dirname(defaultOutputFilename), 'output');
    }

    getSharedLibraryPathsAsArguments(libraries) {
        const libPathFlag = this.compiler.libpathFlag || '/LIBPATH:';

        return this.getSharedLibraryPaths(libraries).map(path => libPathFlag + path);
    }

    getSharedLibraryLinks(libraries) {
        return _.flatten(_.map(libraries, (selectedLib) => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (!foundVersion) return false;

            return _.map(foundVersion.liblink, (lib) => {
                if (lib) {
                    return '"' + lib + '.lib"';
                } else {
                    return false;
                }
            });
        }));
    }

    getStaticLibraryLinks(libraries) {
        return _.map(super.getSortedStaticLibraries(libraries), (lib) => {
            return '"' + lib + '.lib"';
        });
    }

    prepareArguments(userOptions, filters, backendOptions, inputFilename, outputFilename, libraries) {
        let options = this.optionsForFilter(filters, outputFilename, userOptions);
        backendOptions = backendOptions || {};

        if (this.compiler.options) {
            options = options.concat(utils.splitArguments(this.compiler.options));
        }

        if (this.compiler.supportsOptOutput && backendOptions.produceOptInfo) {
            options = options.concat(this.compiler.optArg);
        }

        const libIncludes = this.getIncludeArguments(libraries);
        const libOptions = this.getLibraryOptions(libraries);
        let libLinks = [];
        let libPaths = [];
        let preLink = [];
        let staticlibLinks = [];

        if (filters.binary) {
            preLink = ['/link'];
            libLinks = this.getSharedLibraryLinks(libraries);
            libPaths = this.getSharedLibraryPathsAsArguments(libraries);
            staticlibLinks = this.getStaticLibraryLinks(libraries);
        }

        userOptions = this.filterUserOptions(userOptions) || [];
        return options.concat(libIncludes, libOptions, userOptions, [this.filename(inputFilename)], preLink, libPaths,
            libLinks, staticlibLinks);
    }

    optionsForFilter(filters, outputFilename) {
        if (filters.binary) {
            const mapFilename = outputFilename + '.map';
            const mapFileReader = new MapFileReaderVS(mapFilename);

            filters.preProcessBinaryAsmLines = (asmLines) => {
                const reconstructor = new PELabelReconstructor(asmLines, false, mapFileReader);
                reconstructor.run('output.s.obj');

                return reconstructor.asmLines;
            };

            return [
                '/nologo',
                '/FA',
                '/Fa' + this.filename(outputFilename.replace(/\.exe$/, '')),
                '/Fo' + this.filename(outputFilename.replace(/\.exe$/, '') + '.obj'),
                '/Fm' + this.filename(mapFilename),
                '/Fe' + this.filename(this.getExecutableFilename(path.dirname(outputFilename), 'output')),
            ];
        } else {
            return [
                '/nologo',
                '/FA',
                '/c',
                '/Fa' + this.filename(outputFilename),
                '/Fo' + this.filename(outputFilename + '.obj'),
            ];
        }
    }

    processAsm(result, filters/*, options*/) {
        if (filters.binary) {
            filters.dontMaskFilenames = true;
            return this.binaryAsmParser.process(result.asm, filters);
        } else {
            return this.asm.process(result.asm, filters);
        }
    }

    exec(compiler, args, options_) {
        let options = Object.assign({}, options_);
        options.env = Object.assign({}, options.env);

        if (this.compiler.includePath) {
            options.env['INCLUDE'] = this.compiler.includePath;
        }
        if (this.compiler.libPath) {
            options.env['LIB'] = this.compiler.libPath.join(';');
        }
        for (const [env, to] of this.compiler.envVars) {
            options.env[env] = to;
        }

        return super.exec(compiler, args, options);
    }
}
