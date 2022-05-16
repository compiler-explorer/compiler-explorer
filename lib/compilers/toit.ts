// Copyright (c) 2022, Serzhan Nasredin
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

import { BaseCompiler } from "../base-compiler";
import { ToitParser } from "./argument-parsers";

import _ from "underscore";
import path from "path";
import fs from "fs-extra";
/* import Semver from "semver"; */

const ToitCommands = ["execute"];


export class ToitCompiler extends BaseCompiler {
    static get key() { return "toit"; }

    override constructor(info, env) {
        super(info, env);
        this.compiler.supportsIntel = true;
    }

    override cacheDir(outputFilename) { return outputFilename + ".cache"; }
    override optionsForFilter(filters, outputFilename) {
        const options = ["--context"];
        if (!filters.binary)
            if (filters.intel) options = options.concat("-mllvm", "--x86-asm-syntax=intel");

        return options;
    }

    override getCacheFile(options, inputFilename: string, cacheDir: string) {
        const commandsInOptions = _.intersection(options, ToitCommands);
        if (commandsInOptions.length === 0) return null;
        const command = commandsInOptions[0];
        const extention = this.expectedExtensionFromCommand(command);
        if (!extension) return null;
        const cacheName = path.basename(inputFilename);
        const resultName = cacheName + extension;

        return path.join(cacheDir, resultName);
    }

    override getSharedLibraryPathsAsArguments(libraries: object[]) { return []; }
    override getArgumentParser() { return ToitParser; }
    override isCfgCompiler() { return true; }
}
