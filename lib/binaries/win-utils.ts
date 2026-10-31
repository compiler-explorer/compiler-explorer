// Copyright (c) 2023, Compiler Explorer Authors
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

import fs from 'node:fs/promises';
import path from 'node:path';

import {ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {logger} from '../logger.js';
import * as utils from '../utils.js';

export class WinUtils {
    protected re_dll_name = /dll name: (.*\.dll)/i;
    protected objdumper: string;
    protected exec: any;
    protected alreadyDone: string[];
    protected execOptions: ExecutionOptionsWithEnv;
    protected skippable: string[];

    constructor(exec: typeof BaseCompiler.prototype.exec, objdumper: string, execOptions: ExecutionOptionsWithEnv) {
        this.exec = exec;
        this.objdumper = objdumper;
        this.execOptions = execOptions;
        this.alreadyDone = [];
        this.skippable = ['kernel32.dll', 'ucrtbase.dll'];
    }

    private async get_pe_machine(filename: string): Promise<number | undefined> {
        try {
            const file = await fs.open(filename, 'r');
            try {
                const header = Buffer.alloc(4);
                await file.read(header, 0, 4, 0x3c);
                const peHeader = Buffer.alloc(6);
                await file.read(peHeader, 0, 6, header.readUInt32LE(0));
                if (peHeader.toString('latin1', 0, 4) !== 'PE\0\0') return undefined;
                return peHeader.readUInt16LE(4);
            } finally {
                await file.close();
            }
        } catch {
            return undefined;
        }
    }

    private async find_first_file_in_dirs(
        paths: string[],
        filename: string,
        machine: number | undefined,
    ): Promise<string | false> {
        for (const p of paths) {
            const fullpath = path.join(p, filename);
            if (!(await utils.fileExists(fullpath))) continue;
            // A 64-bit process sees the 64-bit System32 on PATH, so 32-bit executables need their DLLs found elsewhere
            if (machine !== undefined && (await this.get_pe_machine(fullpath)) !== machine) continue;
            return fullpath;
        }

        return false;
    }

    public async get_dlls_used(executable: string): Promise<string[]> {
        const dlls_used: string[] = [];

        const dump_dlls_used = await this.exec(this.objdumper, ['--private-headers', executable], this.execOptions);

        const paths = this.execOptions.env.PATH.split(path.delimiter);
        const machine = await this.get_pe_machine(executable);

        const lines = utils.splitLines(dump_dlls_used.stdout);
        for (const line of lines) {
            const match = line.match(this.re_dll_name);
            if (match) {
                const dll_name = match[1];

                if (this.skippable.includes(dll_name.toLowerCase())) continue;
                if (this.alreadyDone.includes(dll_name)) continue;

                logger.debug('looking for ' + dll_name);

                const dll_path = await this.find_first_file_in_dirs(paths, dll_name, machine);
                if (dll_path) {
                    logger.debug('found at ' + dll_path);
                    dlls_used.push(dll_path);
                    this.alreadyDone.push(dll_name);

                    const recursed = await this.get_dlls_used(dll_path);
                    if (recursed.length > 0) {
                        dlls_used.push(...recursed);
                    }
                }
            }
        }

        return dlls_used;
    }
}

export async function copyNeededDlls(
    dirPath: string,
    executableFilename: string,
    execFunction: typeof BaseCompiler.prototype.exec,
    objdumper: string,
    execoptions: ExecutionOptionsWithEnv,
): Promise<void> {
    const winutils = new WinUtils(execFunction, objdumper, execoptions);
    try {
        const dlls = await winutils.get_dlls_used(executableFilename);
        for (const dll of dlls) {
            const infolder = path.join(dirPath, path.basename(dll));
            await fs.copyFile(dll, infolder);
        }
    } catch {
        logger.error(`Error while retreiving or copying dll dependencies of ${executableFilename}`);
    }
}
