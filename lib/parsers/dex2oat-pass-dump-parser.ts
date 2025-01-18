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

import {OptPipelineResults, Pass} from '../../types/compilation/opt-pipeline-output.interfaces.js';
import {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {assert} from '../assert.js';
import {logger} from '../logger.js';

type PassDump = {
    name: string;
    lines: ResultLine[];
};

// Checks that the pairing of (before) and (after) passes is correct.
function passesMatch(before: string, after: string) {
    assert(before.endsWith(' (before)'));
    assert(after.endsWith(' (after)'));
    before = before.slice(0, before.length - ' (before)'.length);
    after = after.slice(0, after.length - ' (after)'.length);
    return before === after;
}

export class Dex2OatPassDumpParser {
    nameRegex: RegExp;
    dexPcRegex: RegExp;
    offsetRegex: RegExp;

    constructor() {
        this.nameRegex = /^\s*name\s+"(.*)"\s*$/;
        this.dexPcRegex = /^.*dex_pc:([\w/]+)\s.*$/;
        this.offsetRegex = /^.*(0x\w+):.*$/;
    }

    // Parses the lines from the classes.cfg file and returns a mapping of
    // function names to per-function optimization pass dumps.
    parsePassDumpsForFunctions(ir: string[]) {
        const functionsToPassDumps: Record<string, PassDump[]> = {};

        let functionName;
        let passName;

        let inFunctionHeader = false;
        let inOptPass = false;
        let inBlock = false;

        let match;
        for (const l of ir) {
            if (l.match('begin_compilation')) {
                inFunctionHeader = true;
            } else if (l.match('end_compilation')) {
                inFunctionHeader = false;
            } else if (l.match('begin_cfg')) {
                inOptPass = true;
            } else if (l.match('end_cfg')) {
                inOptPass = false;
            } else {
                if (l.match('begin_block')) {
                    inBlock = true;
                } else if (l.match('end_block')) {
                    inBlock = false;
                }

                if (inFunctionHeader && this.nameRegex.test(l)) {
                    match = l.match(this.nameRegex);

                    functionName = match![1];
                    functionsToPassDumps[functionName] = [];
                } else if (inOptPass && !inBlock && this.nameRegex.test(l)) {
                    // We check !inBlock because blocks also contain a name
                    // field that will match nameRegex.
                    match = l.match(this.nameRegex);

                    passName = match![1];
                    functionsToPassDumps[functionName!].push({name: passName, lines: []});
                } else if (inOptPass) {
                    const passDump = functionsToPassDumps[functionName!].pop();

                    // pop() can return undefined, but we know that it won't
                    // because if we're in an opt pass, the previous case should
                    // have been met already.
                    if (passDump) {
                        passDump.lines.push({text: l});
                        functionsToPassDumps[functionName!].push(passDump);
                    } else {
                        logger.error(`passDump for function ${functionName} is undefined!`);
                    }
                }
            }
        }

        return functionsToPassDumps;
    }

    // Parses the final disassembly output step from classes.cfg and returns a
    // mapping of method names to instruction offsets + corresponding dex PCs.
    parsePassDumpsForDexPcs(ir: string[]) {
        const methodsAndOffsetsToDexPcs: Record<string, Record<number, number>> = {};
        const methodsToInstructions: Record<string, string[]> = {};

        let methodName;
        let passName;

        let inFunctionHeader = false;
        let inDisassembly = false;
        let inHir = false;

        let match;
        for (const l of ir) {
            // Collect function names.
            if (l.match('begin_compilation')) {
                inFunctionHeader = true;
            } else if (l.match('end_compilation')) {
                inFunctionHeader = false;
            } else if (inFunctionHeader && this.nameRegex.test(l)) {
                match = l.match(this.nameRegex);
                methodName = match![1];
                methodsAndOffsetsToDexPcs[methodName] = {};
                methodsToInstructions[methodName] = [];

                // Determine if we're in the disassembly pass.
            } else if (this.nameRegex.test(l)) {
                match = l.match(this.nameRegex);
                passName = match![1];
                if (passName === 'disassembly (after)') {
                    inDisassembly = true;
                }
            } else if (l.match('end_cfg')) {
                inDisassembly = false;

                // Determine if we're in an HIR section (only if we're in a
                // disassembly pass).
            } else if (inDisassembly && l.match('begin_HIR')) {
                inHir = true;
            } else if (l.match('end_HIR')) {
                inHir = false;

                // Collect all lines in HIR that do not end in '<|@'. This string
                // marks the end of an HIR instruction, so lines without it are
                // either HIR instructions that contain dex PCs, or the following
                // lines of disassembly.
            } else if (inHir && !l.endsWith('<|@')) {
                methodsToInstructions[methodName!].push(l);
            }
        }

        let dexPc = -1;
        for (const methodName in methodsToInstructions) {
            const remove: string[] = [];
            for (const instruction of methodsToInstructions[methodName]) {
                if (this.dexPcRegex.test(instruction)) {
                    match = instruction.match(this.dexPcRegex);
                    dexPc = Number.parseInt(match![1]);
                    remove.push(instruction);
                } else if (this.offsetRegex.test(instruction)) {
                    match = instruction.match(this.offsetRegex);
                    const offset = match![1];
                    methodsAndOffsetsToDexPcs[methodName][Number.parseInt(offset, 16)] = dexPc;
                } else {
                    dexPc = -1;
                    remove.push(instruction);
                }
            }
            dexPc = -1;

            // Remove lines that weren't actually instructions.
            methodsToInstructions[methodName] = methodsToInstructions[methodName].filter(e => remove.includes(e));

            // Remove methods that don't contain any instructions.
            // This only really applies to the first "method" with a name that
            // describes the ISA, ISA features, etc. For example,
            // 'isa:arm64 isa_features:a53,crc,-lse,-fp16,-dotprod,-sve...'
            if (Object.keys(methodsAndOffsetsToDexPcs[methodName]).length === 0) {
                delete methodsAndOffsetsToDexPcs[methodName];
            }
        }

        return methodsAndOffsetsToDexPcs;
    }

    // This method merges each function's (before) and (after) optimization
    // passes' text into a series of Pass objects.
    // This method was adapted from llvm-pass-dump-parser.ts.
    mergeBeforeAfterPassDumps(functionsToPassDumps: Record<string, PassDump[]>) {
        const finalOutput: OptPipelineResults = {};
        for (const [functionName, passDumps] of Object.entries(functionsToPassDumps)) {
            const passes: Pass[] = [];
            for (let i = 0; i < passDumps.length; ) {
                const pass: Pass = {
                    name: '',
                    machine: false,
                    before: [],
                    after: [],
                    irChanged: true,
                };
                const currentDump = passDumps[i];
                const nextDump = i < passDumps.length - 1 ? passDumps[i + 1] : null;
                if (currentDump.name.endsWith(' (after)')) {
                    pass.name = currentDump.name.slice(0, currentDump.name.length - ' (after)'.length);
                    pass.after = currentDump.lines;
                    i++;
                } else if (currentDump.name.endsWith(' (before)')) {
                    if (nextDump !== null && nextDump.name.endsWith(' (after)')) {
                        assert(passesMatch(currentDump.name, nextDump.name), '', currentDump.name, nextDump.name);
                        pass.name = currentDump.name.slice(0, currentDump.name.length - ' (before)'.length);
                        pass.before = currentDump.lines;
                        pass.after = nextDump.lines;
                        i += 2;
                    } else {
                        pass.name = currentDump.name.slice(0, currentDump.name.length - ' (before)'.length);
                        pass.before = currentDump.lines;
                        i++;
                    }
                } else {
                    assert(false, 'Unexpected pass name', currentDump.name);
                }
                pass.irChanged = pass.before.map(x => x.text).join('\n') !== pass.after.map(x => x.text).join('\n');
                passes.push(pass);
            }
            finalOutput[functionName] = passes;
        }
        return finalOutput;
    }

    process(rawText: string): OptPipelineResults {
        const lines = rawText.split(/\n/);

        // Remove leading [begin|end]_compilation section, as it doesn't correspond to any methods.
        const ir = lines.slice(lines.findIndex(l => l.match('end_compilation')) + 1);

        const functionsToPassDumps = this.parsePassDumpsForFunctions(ir);

        return this.mergeBeforeAfterPassDumps(functionsToPassDumps);
    }
}
