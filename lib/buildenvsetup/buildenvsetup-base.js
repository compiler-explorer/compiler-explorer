// Copyright (c) 2020, Compiler Explorer Authors
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
"use strict";

const
    _ = require('underscore'),
    path = require('path'),
    fs = require('fs-extra'),
    exec = require('../exec');

class BuildEnvSetupBase {
    constructor(compilerInfo, env) {
        this.compiler = compilerInfo;
        this.env = env;

        this.compilerOptionsArr = this.compiler.options.split(" ");
        this.compilerArch = this.getCompilerArch();
        this.compilerTypeOrGCC = compilerInfo.compilerType ? compilerInfo.compilerType : "gcc";
        this.compilerSupportsX86 = true;

        if (this.compilerArch) {
            this.compilerSupportsX86 = false;
        } else {
            this.hasSupportForArch('x86').then(res => this.compilerSupportsX86 = res);
        }
    }

    async hasSupportForArch(arch) {
        let result = null;
        if (this.compilerTypeOrGCC === "gcc") {
            result = await exec.execute(this.compiler.exe, ['--target-help']);
        } else if (this.compilerTypeOrGCC === "clang") {
            const binpath = path.dirname(this.compiler.exe);
            const llc = path.join(binpath, 'llc');
            if (fs.existsSync(llc)) {
                result = await exec.execute(llc, ['--version']);
            }
        }

        if (result) {
            return result.stdout.includes(arch);
        }

        return false;
    }

    async setup(/*key, dirPath, selectedLibraries*/) {
        // override with specific implementation
        return Promise.resolve();
    }

    getCompilerArch() {
        let arch = _.find(this.compilerOptionsArr, (option) => {
            return  option.startsWith("-march=");
        });

        if (arch) return arch.substr(7);

        let target = _.find(this.compilerOptionsArr, (option) => {
            option.startsWith("-target=") ||
            option.startsWith("--target=");
        });

        if (target) return target.substr(target.indexOf('=') + 1);

        return false;
    }

    getLibcxx(key) {
        const match = this.compiler.options.match(/-stdlib=(\S*)/i);
        if (match) {
            return match[1];
        } else {
            const stdlibOption = _.find(key.options, (option) => {
                return option.startsWith("-stdlib=");
            });

            if (stdlibOption) {
                return stdlibOption.substr(8);
            }

            return "libstdc++";
        }
    }

    getTarget(key) {
        if (!this.compilerSupportsX86) return "";
        if (this.compilerArch) return this.compilerArch;

        if (key.options.includes('-m32')) {
            return "x86";
        } else {
            let target = _.find(key.options, (option) => {
                return option.startsWith("-target=") || option.startsWith("--target=");
            });

            if (target) {
                return target.substr(target.indexOf('=') + 1);
            }
        }

        return "x86_64";
    }
}

module.exports = BuildEnvSetupBase;
