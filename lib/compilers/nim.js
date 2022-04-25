// Copyright (c) 2019, Bastien Penavayre
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

import fs from 'fs-extra';
import _ from 'underscore';

import {BaseCompiler} from '../base-compiler';

import {NimParser} from './argument-parsers';

const NimCommands = ['compile', 'compileToC', 'c', 'compileToCpp', 'cpp', 'cc', 'compileToOC', 'objc', 'js', 'check'];

export class NimCompiler extends BaseCompiler {
    static get key() {
        return 'nim';
    }

    constructor(info, env) {
        super(info, env);
        this.compiler.supportsIntel = true;
    }

    cacheDir(outputFilename) {
        return outputFilename + '.cache';
    }

    optionsForFilter(filters, outputFilename) {
        return [
            '--debugger:native', // debugging info
            '-o:' + outputFilename, //output file, only for js mode
            '--nolinking', //disable linking, only compile to nimcache
            '--nimcache:' + this.cacheDir(outputFilename), //output folder for the nimcache
        ];
    }

    filterUserOptions(userOptions) {
        //If none of the allowed commands is present in userOptions add 'compile' command
        if (_.intersection(userOptions, NimCommands).length === 0) {
            userOptions.unshift('compile');
        }

        return userOptions.filter(option => !['--run', '-r'].includes(option));
    }

    expectedExtensionFromCommand(command) {
        const isC = ['compile', 'compileToC', 'c'],
            isCpp = ['compileToCpp', 'cpp', 'cc'],
            isObjC = ['compileToOC', 'objc'];

        if (isC.includes(command)) return '.c.o';
        else if (isCpp.includes(command)) return '.cpp.o';
        else if (isObjC.includes(command)) return '.m.o';
        else return null;
    }

    getCacheFile(options, inputFilename, cacheDir) {
        const commandsInOptions = _.intersection(options, NimCommands);
        if (commandsInOptions.length === 0) return null;
        const command = commandsInOptions[0];
        const extension = this.expectedExtensionFromCommand(command);
        if (!extension) return null;
        const moduleName = path.basename(inputFilename);
        const resultName = '@m' + moduleName + extension;
        return path.join(cacheDir, resultName);
    }

    async postProcess(result, outputFilename, filters) {
        const options = result.compilationOptions;
        const cacheDir = this.cacheDir(outputFilename);
        try {
            if (_.intersection(options, ['js', 'check']).length > 0) filters.binary = false;
            else {
                filters.binary = true;
                const objFile = this.getCacheFile(options, result.inputFilename, cacheDir);
                await fs.move(objFile, outputFilename);
            }
            return super.postProcess(result, outputFilename, filters);
        } finally {
            await fs.remove(cacheDir);
        }
    }

    getSharedLibraryPathsAsArguments(/*libraries*/) {
        return [];
    }

    getArgumentParser() {
        return NimParser;
    }

    isCfgCompiler(/*compilerVersion*/) {
        return true;
    }

    orderArguments(options, inputFilename, libIncludes, libOptions, libPaths, libLinks, userOptions, staticLibLinks) {
        return options.concat(
            libIncludes,
            libOptions,
            libPaths,
            libLinks,
            userOptions,
            [this.filename(inputFilename)],
            staticLibLinks
        );
    }
}
