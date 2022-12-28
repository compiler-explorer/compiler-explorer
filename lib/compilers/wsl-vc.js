// Copyright (c) 2017, Andrew Pardoe
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

// The main difference from wine-vc.js is that we translate
// compiler path from Unix mounted volume (/mnt/c/tmp) to Windows (c:/tmp)

import path from 'path';

import temp from 'temp';

import {VcAsmParser} from '../parsers/asm-parser-vc';

import {Win32VcCompiler} from './win32-vc';

export class WslVcCompiler extends Win32VcCompiler {
    static get key() {
        return 'wsl-vc';
    }

    constructor(info, env) {
        super(info, env);
        this.asm = new VcAsmParser();
    }

    filename(fn) {
        // AP: Need to translate compiler paths from what the Node.js process sees
        // on a Unix mounted volume (/mnt/c/tmp) to what CL sees on Windows (c:/tmp)
        // We know process.env.tmpDir is of format /mnt/X/dir where X is drive letter.
        const driveLetter = process.env.winTmp.substring(5, 6);
        const directoryPath = process.env.winTmp.substring(7);
        const windowsStyle = driveLetter.concat(':/', directoryPath);
        return fn.replace(process.env.winTmp, windowsStyle);
    }

    // AP: Create CE temp directory in winTmp directory instead of the tmpDir directory.
    // NPM temp package: https://www.npmjs.com/package/temp, see Affixes
    newTempDir() {
        return new Promise((resolve, reject) => {
            temp.mkdir({prefix: 'compiler-explorer-compiler', dir: process.env.winTmp}, (err, dirPath) => {
                if (err) reject(`Unable to open temp file: ${err}`);
                else resolve(dirPath);
            });
        });
    }

    exec(compiler, args, options_) {
        let options = Object.assign({}, options_);
        options.env = Object.assign({}, options.env);

        let old_env = options.env['WSLENV'];
        if (old_env) {
            old_env = ':' + old_env;
        } else {
            old_env = '';
        }
        options.env['WSLENV'] = 'INCLUDE:LIB' + old_env;

        return super.exec(compiler, args, options);
    }

    runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        // inputFilename is guaranteed to be Windows formatted (e.g. c:/) but
        // Node.js needs the Unix syntax
        const inputDirectory = path.dirname(inputFilename);
        const driveLetter = inputDirectory.substring(0, 1).toLowerCase();
        const directoryPath = inputDirectory.substring(2).trim();
        execOptions.customCwd = path.join('/mnt', driveLetter, directoryPath);

        return super.runCompiler(compiler, options, inputFilename, execOptions);
    }
}
