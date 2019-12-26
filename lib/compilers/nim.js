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

const BaseCompiler = require('../base-compiler'),
    _ = require('underscore'),
    path = require('path'),
    argumentParsers = require("./argument-parsers"),
    fs = require('fs-extra');

class NimCompiler extends BaseCompiler {
    constructor(info, env) {
        super(info, env);
        this.compiler.supportsIntel = true;
    }

    cacheDir(outputFilename) {
        return outputFilename + '.cache';
    }

    optionsForFilter(filters, outputFilename) {
        return [
            "-o:" + outputFilename, //output file, only for js mode
            "--nolinking", //disable linking, only compile to nimcache
            "--nimcache:" + this.cacheDir(outputFilename) //output folder for the nimcache
        ];
    }

    filterUserOptions(userOptions) {
        //Allowed commands
        const commands = [
            "compile", "compileToC", "c",
            "compileToCpp", "cpp", "cc",
            "compileToOC", "objc",
            "js",
            "check"
        ];

        //If none of the allowed commands is present in userOptions compile to C++
        if (_.intersection(userOptions, commands).length === 0) {
            userOptions.unshift("compileToCpp");
        }

        return userOptions.filter(option => !['--run', '-r'].includes(option));
    }

    postProcess(result, outputFilename, filters) {
        let options = result.compilationOptions;
        let setup = Promise.resolve("");
        const cacheDir = this.cacheDir(outputFilename);
        const cleanup = () => fs.remove(cacheDir);

        if (_.intersection(options, ["js", "check"]).length !== 0)
            filters.binary = false;
        else {
            filters.binary = true;

            const isC = ["compile", "compileToC", "c"],
                isCpp = ["compileToCpp", "cpp", "cc"],
                isObjC = ["compileToOC", "objc"];

            let extension;
            if (_.intersection(options, isC).length !== 0)
                extension = '.c.o';
            else if (_.intersection(options, isCpp).length !== 0)
                extension = '.cpp.o';
            else if (_.intersection(options, isObjC).length !== 0)
                extension = '.m.o';

            const moduleName = path.basename(result.inputFilename);
            const resultName = moduleName + extension;
            const objFile = path.join(cacheDir, resultName);
            setup = fs.move(objFile, outputFilename);
        }

        const postProcess = () => super.postProcess(result, outputFilename, filters);

        return setup.then(postProcess).finally(cleanup);
    }

    getSharedLibraryPathsAsArguments(/*libraries*/) {
        return [];
    }

    getArgumentParser() {
        return argumentParsers.Nim;
    }

    isCfgCompiler(/*compilerVersion*/) {
        return true;
    }
}

module.exports = NimCompiler;
