// Copyright (c) 2026, Compiler Explorer Authors
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

import type {ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import * as delphiElf from '../delphi-elf-support.js';
import * as utils from '../utils.js';
import {PascalWinCompiler} from './pascal-win.js';

// Delphi cross-compiles to Linux from the Windows host via dcclinux64 (LLVM backend). Output is an
// ELF with DWARF, so the disassembly path is simpler than the Windows PE/PDB one - see
// delphi-elf-support.ts. The sysroot and LLVM linker are env-overridable; the Linux RTL is derived
// from the compiler's own path so this class serves any Delphi version's dcclinux64 unchanged.
const DCC_LINUX_SDK =
    process.env.CE_DELPHI_LINUX_SDK || 'C:\\Users\\jim\\Documents\\Embarcadero\\Studio\\SDKs\\fedora44.sdk';
const LD_LLD = process.env.CE_LD_LLD || 'C:\\Program Files\\LLVM\\bin\\ld.lld.exe';

export class PascalWinLinuxCompiler extends PascalWinCompiler {
    static get key() {
        return 'pascal-win-linux';
    }

    elfTarget = 'prog.dpr';
    elfLabel = 'prog';

    override getExecutableFilename(dirPath: string) {
        return path.join(dirPath, 'prog');
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, 'prog');
    }

    // <SDK>/usr/lib/gcc/x86_64-redhat-linux/<highest-version-dir> (forward slashes for ld.lld)
    async findGccLibDir(): Promise<string> {
        const base = DCC_LINUX_SDK.replace(/\//g, '\\') + '\\usr\\lib\\gcc\\x86_64-redhat-linux';
        let ver = '16';
        try {
            const entries = (await fs.readdir(base)).filter(e => /^\d+$/.test(e)).sort((a, b) => Number(b) - Number(a));
            if (entries.length > 0) ver = entries[0];
        } catch {
            // fall back to the pinned default
        }
        return DCC_LINUX_SDK.replace(/\\/g, '/') + '/usr/lib/gcc/x86_64-redhat-linux/' + ver;
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ) {
        if (!execOptions) execOptions = this.getDefaultExecOptions();

        const alreadyHasDPR = path.basename(inputFilename) === this.dprFilename;
        const tempPath = path.dirname(inputFilename);
        const projectFile = path.join(tempPath, this.dprFilename);
        inputFilename = inputFilename.replaceAll('/', '\\');

        // The user's own source file we annotate: the program's prog.dpr, or the unit's .pas whose
        // routines are force-referenced into the wrapper (reused from PascalWinCompiler).
        const unitFilepath = path.basename(inputFilename);
        this.elfTarget = alreadyHasDPR ? this.dprFilename : unitFilepath;
        this.elfLabel = this.elfTarget.replace(/\.(pas|dpr)$/i, '');

        if (!alreadyHasDPR) {
            const unitName = unitFilepath.replace(/.pas$/i, '');
            const unitSource = await fs.readFile(inputFilename, 'utf8');
            await this.saveDummyProjectFile(projectFile, unitName, unitFilepath, unitSource);
        }

        // dcclinux64 auto-reads <project>.cfg; RTL is <studio>/lib/Linux64/release next to bin/.
        const rtl = path.join(path.resolve(path.dirname(compiler), '..'), 'lib', 'Linux64', 'release');
        const gccLib = await this.findGccLibDir();
        const sdkFwd = DCC_LINUX_SDK.replace(/\\/g, '/');
        // biome-ignore format: keep the .cfg readable
        const cfg = [
            '-U"' + rtl + '"',
            '-O"' + rtl + '"',
            '-NU"' + tempPath + '"',
            '-NX"' + tempPath + '"',
            '-E"' + tempPath + '"',
            '--syslibroot:"' + DCC_LINUX_SDK + '"',
            '--linker-option:"-L' + gccLib + '"',
            '--linker-option:"-L' + sdkFwd + '/usr/lib64"',
            '--linker-option:"-L' + sdkFwd + '/lib64"',
            '--linker:"' + LD_LLD + '"',
            '--linker-option:"--allow-shlib-undefined"',
            '-V', '-$D+', '-GD', '-B',
        ].join('\r\n') + '\r\n';
        await fs.writeFile(path.join(tempPath, 'prog.cfg'), cfg);

        execOptions.customCwd = tempPath;
        return this.exec(compiler, ['prog.dpr'], execOptions).then(result => ({
            ...result,
            inputFilename,
            stdout: utils.parseOutput(result.stdout, inputFilename),
            stderr: utils.parseOutput(result.stderr, inputFilename),
        }));
    }

    override async objdump(outputFilename: string, result, maxSize: number, intelAsm: boolean) {
        const dirPath = path.dirname(outputFilename);
        const execBinary = this.getExecutableFilename(dirPath);
        const target = (await utils.fileExists(execBinary)) ? execBinary : outputFilename;

        // -l pulls DWARF line info so objdump interleaves <path>:<line> markers.
        let args = [...this.compiler.objdumperArgs, '-d', '-l', target];
        if (intelAsm) args = args.concat(['-M', 'intel']);
        return this.exec(this.compiler.objdumper, args, {maxOutput: 1024 * 1024 * 1024}).then(objResult => {
            result.asm =
                objResult.code === 0 ? objResult.stdout : '<No output: objdump returned ' + objResult.code + '>';
            return result;
        });
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions) {
        filters.binary = true;
        filters.dontMaskFilenames = true;
        filters.preProcessBinaryAsmLines = (asmLines: string[]) => {
            return delphiElf.annotateElfAsm(asmLines, this.elfTarget, this.elfLabel);
        };
        return [];
    }
}
