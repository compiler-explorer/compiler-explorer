// Copyright (c) 2019, Compiler Explorer Authors
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

import _ from 'underscore';

import { getToolchainPath } from '../toolchain-utils';
import * as utils from '../utils';

import { BaseTool } from './base-tool';

export class CompilerDropinTool extends BaseTool {
    static get key() { return 'compiler-dropin-tool'; }

    constructor(toolInfo, env) {
        super(toolInfo, env);

        this.addOptionsToToolArgs = false;
    }

    getToolchainPath(compilationInfo) {
        return getToolchainPath(compilationInfo.compiler.exe, compilationInfo.compiler.options);
    }

    getOrderedArguments(compilationInfo, includeflags, libOptions, args, sourcefile) {
        // order should be:
        //  1) options from the compiler config (compilationInfo.compiler.options)
        //  2) includes from the libraries (includeflags)
        //  3) options from the tool config (this.tool.options)
        //  4) options manually specified in the compiler tab (options)
        //  5) flags from the clang-tidy tab
        let compileFlags = [];
        let argsToFilterOut = new Set([sourcefile, '-stdlib=libc++']);

        const toolchainPath = this.getToolchainPath(compilationInfo);

        let compilerOptions = compilationInfo.compiler.options ?
            utils.splitArguments(compilationInfo.compiler.options) : [];

        if (toolchainPath) {
            // note: needs toolchain argument twice as the first time its sometimes ignored
            compileFlags = compileFlags.concat('--gcc-toolchain=' + toolchainPath);
            compileFlags = compileFlags.concat('--gcc-toolchain=' + toolchainPath);

            compilerOptions = _.filter(compilerOptions, (option) => {
                return !((option.indexOf('--gcc-toolchain=') === 0) ||
                    (option.indexOf('--gxx-name=') === 0));
            });
        } else {
            return false;
        }

        compilerOptions = compilerOptions.filter(option => !argsToFilterOut.has(option));

        compileFlags = compileFlags.concat(compilerOptions);
        compileFlags = compileFlags.concat(includeflags);

        const manualCompileFlags = compilationInfo.options.filter(option => !argsToFilterOut.has(option));
        compileFlags = compileFlags.concat(manualCompileFlags);
        const toolOptions = this.tool.options ? this.tool.options : [];
        compileFlags = compileFlags.concat(toolOptions);
        compileFlags = compileFlags.concat(libOptions);
        args ? args : [];
        compileFlags = compileFlags.concat(args);

        compileFlags = _.map(compileFlags, (option) => {
            if (option && option.length > 1) {
                if (option[0] === '/') {
                    return false;
                }
            }

            return option;
        });

        return _.filter(compileFlags);
    }

    async runTool(compilationInfo, inputFilepath, args, stdin, supportedLibraries) {
        const sourcefile = inputFilepath;

        const includeflags = super.getIncludeArguments(compilationInfo.libraries, supportedLibraries);
        const libOptions = super.getLibraryOptions(compilationInfo.libraries, supportedLibraries);

        const compileFlags = this.getOrderedArguments(compilationInfo, includeflags, libOptions, args, sourcefile);
        if (!compileFlags) {
            return new Promise((resolve) => {
                resolve(this.createErrorResponse('Unable to run tool with selected compiler'));
            });
        }

        return super.runTool(compilationInfo, sourcefile, compileFlags);
    }
}
