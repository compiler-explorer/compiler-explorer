// Copyright (c) 2017, Compiler Explorer Authors
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

import type {
    ExecutionOptions,
    ExecutionOptionsWithEnv,
    FiledataPair,
} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import * as delphiPdb from '../delphi-pdb-support.js';
import type {AsmRange} from '../delphi-pdb-support.js';
import * as utils from '../utils.js';
import * as pascalUtils from './pascal-utils.js';

// External tools for the map2pdb + llvm-pdbutil disassembly pipeline (see delphi-pdb-support.ts).
// Overridable via env so the service config can relocate them without a code change.
const MAP2PDB = process.env.CE_MAP2PDB || 'C:\\ci\\map2pdb\\map2pdb.exe';
const PDBUTIL = process.env.CE_PDBUTIL || 'C:\\Program Files\\LLVM\\bin\\llvm-pdbutil.exe';

export class PascalWinCompiler extends BaseCompiler {
    static get key() {
        return 'pascal-win';
    }

    mapFilename: string | null;
    dprFilename: string;

    // Result of the PDB line-mapping pipeline for the most recent compile, consumed by the
    // (synchronous) preProcessBinaryAsmLines hook. Set in runCompiler.
    pdbRanges: AsmRange[];
    pdbMarkerPath: string;
    pdbLabelName: string;

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        info.supportsFiltersInBinary = true;

        this.mapFilename = null;
        this.compileFilename = 'output.pas';
        this.dprFilename = 'prog.dpr';

        this.pdbRanges = [];
        this.pdbMarkerPath = 'C:/app/prog.dpr';
        this.pdbLabelName = 'prog';
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override exec(command: string, args: string[], options: ExecutionOptions) {
        if (process.platform === 'linux' || process.platform === 'darwin') {
            const wine = this.env.ceProps<string>('wine');

            args = args.slice(0);
            if (command.toLowerCase().endsWith('.exe')) {
                args.unshift(command);
                command = wine;
            }
        }

        return super.exec(command, args, options);
    }

    override getExecutableFilename(dirPath: string) {
        return path.join(dirPath, 'prog.exe');
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, 'prog.exe');
    }

    override filename(fn: string) {
        if (process.platform === 'linux' || process.platform === 'darwin') {
            return 'Z:' + fn;
        }
        return super.filename(fn);
    }

    override async objdump(outputFilename: string, result, maxSize: number, intelAsm: boolean) {
        const dirPath = path.dirname(outputFilename);
        const execBinary = this.getExecutableFilename(dirPath);
        if (await utils.fileExists(execBinary)) {
            outputFilename = execBinary;
        } else {
            outputFilename = this.getOutputFilename(path.dirname(outputFilename));
        }

        let args = [...this.compiler.objdumperArgs, '-d', outputFilename];
        if (intelAsm) args = args.concat(['-M', 'intel']);
        return this.exec(this.compiler.objdumper, args, {maxOutput: 1024 * 1024 * 1024}).then(objResult => {
            if (objResult.code === 0) {
                result.asm = objResult.stdout;
            } else {
                result.asm = '<No output: objdump returned ' + objResult.code + '>';
            }

            return result;
        });
    }

    async saveDummyProjectFile(filename: string, unitName: string, unitPath: string, unitSource: string) {
        // Delphi's linker drops any routine the wrapper never references, so a bare `uses X;`
        // wrapper compiles fine but silently omits the very code being tested. Call every
        // parseable top-level routine with synthesized dummy arguments to force the linker to
        // keep it - see pascal-utils.ts. Routines we can't confidently synthesize arguments for
        // (custom record/class/generic parameter types) are skipped rather than guessed at, so
        // this can under-cover but never emits invalid Pascal.
        const routines = pascalUtils.extractDeclaredRoutines(unitSource);
        const {varsBlock, statements} = pascalUtils.buildForceReferenceCallSite(routines);

        // biome-ignore format: keep as-is for readability
        await fs.writeFile(
            filename,
            'program prog;\n' +
            'uses ' + unitName + ' in \'' + unitPath + '\';\n' +
            varsBlock +
            'begin\n' +
            (statements ? statements + '\n' : '') +
            'end.\n',
        );
    }

    override async writeAllFiles(dirPath: string, source: string, files: FiledataPair[]) {
        let inputFilename: string;
        if (pascalUtils.isProgram(source)) {
            inputFilename = path.join(dirPath, this.dprFilename);
        } else {
            const unitName = pascalUtils.getUnitname(source);
            if (unitName) {
                inputFilename = path.join(dirPath, unitName + '.pas');
            } else {
                inputFilename = path.join(dirPath, this.compileFilename);
            }
        }

        await fs.writeFile(inputFilename, source);

        if (files && files.length > 0) {
            await this.writeMultipleFiles(files, dirPath);
        }

        return {
            inputFilename,
        };
    }

    // Run map2pdb + `llvm-pdbutil dump -l` on the freshly compiled artifacts and precompute the
    // per-line VA ranges of the user's own source file. Stored for the synchronous
    // preProcessBinaryAsmLines hook. Any failure degrades to empty ranges (a "no source-mapped
    // code" note), never a crash.
    async computePdbRanges(tempPath: string, targetBasename: string, labelName: string) {
        this.pdbRanges = [];
        this.pdbMarkerPath = 'C:/app/' + targetBasename;
        this.pdbLabelName = labelName;
        try {
            const execOpts: ExecutionOptions = {customCwd: tempPath, maxOutput: 1024 * 1024 * 256};
            const m2p = await this.exec(MAP2PDB, ['prog.map', '-pdb:prog.pdb'], execOpts);
            if (m2p.code !== 0) return;
            const dl = await this.exec(PDBUTIL, ['dump', '-l', 'prog.pdb'], execOpts);
            if (dl.code !== 0) return;

            const mapText = await fs.readFile(path.join(tempPath, 'prog.map'), 'utf8');
            const segBases = delphiPdb.parseSegmentBases(mapText);
            const contribs = delphiPdb.parseDumpLines(dl.stdout);
            this.pdbRanges = delphiPdb.buildRanges(contribs, segBases, targetBasename);
        } catch {
            this.pdbRanges = [];
        }
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        const alreadyHasDPR = path.basename(inputFilename) === this.dprFilename;

        const tempPath = path.dirname(inputFilename);
        const projectFile = path.join(tempPath, this.dprFilename);

        this.mapFilename = path.join(tempPath, 'prog.map');

        inputFilename = inputFilename.replaceAll('/', '\\');

        // The user's own source file (the code we annotate): the program's prog.dpr, or the unit's
        // .pas (its routines are force-referenced into the prog.dpr wrapper - see above).
        const unitFilepath = path.basename(inputFilename);
        const targetBasename = alreadyHasDPR ? this.dprFilename : unitFilepath;
        const labelName = targetBasename.replace(/\.(pas|dpr)$/i, '');

        if (!alreadyHasDPR) {
            const unitName = unitFilepath.replace(/.pas$/i, '');
            const unitSource = await fs.readFile(inputFilename, 'utf8');
            await this.saveDummyProjectFile(projectFile, unitName, unitFilepath, unitSource);
        }

        options.pop();

        options.unshift('-CC', '-W', '-H', '-GD', '-$D+', '-V', '-B');

        options.push(projectFile);
        execOptions.customCwd = tempPath;

        return this.exec(compiler, options, execOptions).then(async result => {
            if (result.code === 0) {
                await this.computePdbRanges(tempPath, targetBasename, labelName);
            } else {
                this.pdbRanges = [];
            }
            return {
                ...result,
                inputFilename,
                stdout: utils.parseOutput(result.stdout, inputFilename),
                stderr: utils.parseOutput(result.stderr, inputFilename),
            };
        });
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions) {
        filters.binary = true;
        filters.dontMaskFilenames = true;
        filters.preProcessBinaryAsmLines = (asmLines: string[]) => {
            return delphiPdb.annotateAsm(asmLines, this.pdbRanges, this.pdbMarkerPath, this.pdbLabelName);
        };

        return [];
    }
}
