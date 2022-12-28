// Copyright (c) 2021, Compiler Explorer Authors
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

import fs from 'fs';
import path from 'path';

import _ from 'underscore';

import {BaseTool} from './base-tool';

export class BBCDiskifyTool extends BaseTool {
    static get key() {
        return 'bbcdiskify-tool';
    }

    async createBlankSSD(compilationInfo) {
        const execOptions = this.getDefaultExecOptions();
        execOptions.customCwd = path.dirname(compilationInfo.executableFilename);

        return this.exec(this.tool.exe, ['blank_ssd', 'disk.ssd'], execOptions);
    }

    async writeSourceFiles(compilationInfo) {
        const execOptions = this.getDefaultExecOptions();
        execOptions.customCwd = path.dirname(compilationInfo.executableFilename);

        const files = ['$.!Boot'];
        files.concat(_.pick(compilationInfo.files, 'filename'));
        files.concat(compilationInfo.inputFilename);

        return this.exec(this.tool.exe, ['putfile', 'disk.ssd'].concat(files), execOptions);
    }

    async assembleCode(compilationInfo) {
        const asmText = fs.readFileSync(compilationInfo.executableFilename);
        fs.writeFileSync(compilationInfo.executableFilename, '.org $3000\njsr _main\nrts\n\n' + asmText);

        const execOptions = this.getDefaultExecOptions();
        execOptions.customCwd = path.dirname(compilationInfo.executableFilename);

        await this.exec(
            '/opt/compiler-explorer/6502/cc65-trunk/bin/ca65',
            ['--include-dir', '/opt/compiler-explorer/6502/cc65-trunk/share/cc65/asminc', 'output.s', '-o', '$.!Boot'],
            execOptions,
        );
    }

    async runTool(compilationInfo, inputFilename) {
        if (compilationInfo.filters.binary) {
            await this.createBlankSSD(compilationInfo);
            await fs.copyFile(compilationInfo.executableFilename, '$.!Boot');
            await this.writeSourceFiles(compilationInfo);
        } else {
            await this.assembleCode(compilationInfo);
            await this.createBlankSSD(compilationInfo);
            await this.writeSourceFiles(compilationInfo);
        }

        const outputPath = path.dirname(compilationInfo.executableFilename);
        const file_buffer = fs.readFileSync(path.join(outputPath, 'disk.ssd'));
        const binary_base64 = file_buffer.toString('base64');

        const result = {
            stdout: binary_base64,
            stderr: '',
            filenameTransform: x => x,
        };

        const exeDir = path.dirname(this.tool.exe);
        return this.convertResult(result, inputFilename, exeDir);
    }
}
