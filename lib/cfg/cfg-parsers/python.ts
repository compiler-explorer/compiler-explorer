// Copyright (c) 2025, Compiler Explorer Authors
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

import {CompilationResult} from '../../../types/compilation/compilation.interfaces.js';
import {ResultLine} from '../../../types/resultline/resultline.interfaces.js';
import {InstructionType} from '../instruction-sets/base.js';
import {AssemblyLine, BaseCFGParser, CanonicalBB} from './base.js';

export class PythonCFGParser extends BaseCFGParser {
    BBid = 0;

    static override get key() {
        return 'python3';
    }

    override isFunctionEnd(x: string) {
        return x.startsWith('Disassembly of');
    }

    override isBasicBlockEnd(inst: string, prevInst: string) {
        if (inst.includes('>>'))
            // jmp target
            return true;
        // Probably applicable to non-python CFGs too:
        return this.instructionSetInfo.getInstructionType(prevInst) !== InstructionType.notRetInst;
    }

    override filterData(bytecode: ResultLine[]): ResultLine[] {
        // Filter out module code before the first function
        let i = 0;
        while (
            i < bytecode.length &&
            !bytecode[i].text.startsWith('Disassembly of') &&
            !bytecode[i].text.startsWith('Function #')
        ) {
            i++;
        }
        return bytecode.slice(i).filter(x => x.text && x.text.trim() !== '');
    }

    override async processFuncNames(bytecode: AssemblyLine[], fullRes?: CompilationResult): Promise<AssemblyLine[]> {
        // replace 'Disassembly of' with function name
        const res: ResultLine[] = [];
        let src: string | null = null;

        let funcIdx = 0;

        for (let i = 0; i < bytecode.length; i++) {
            const line = bytecode[i];
            let funcName: string | undefined = undefined;
            if (line.text.startsWith('Disassembly of')) {
                const srcLineStr = line.text.match(/line (\d+)/)?.[1];
                const srcLineNum = srcLineStr ? Number.parseInt(srcLineStr) : null;
                if (srcLineNum && fullRes && fullRes.inputFilename) {
                    if (src === null) {
                        src = await fs.readFile(fullRes.inputFilename, 'utf8');
                        const srcLine = src.split('\n')[srcLineNum - 1];
                        funcName = srcLine.match(/def (\w+)\(/)?.[1];
                    }
                }
                if (funcName) line.text = funcName;
                else line.text = `Function #${funcIdx++}`;
            }
            res.push(line);
        }
        return res;
    }

    // In python <= 3.12:
    //      10 POP_JUMP_IF_FALSE        5 (to 22)   ===> captures line num "22"
    // In python >= 3.13:
    //         POP_JUMP_IF_FALSE        4 (to L1)   ===> captures label "L1"
    override extractJmpTargetName(inst: string) {
        const candidateName = inst.match(/\(to (\w+)\)$/);
        return candidateName ? candidateName[1] : '';
    }

    //'  6   >>   22 LOAD_FAST                0 (num):'   ==> '22'
    //'  4        12 LOAD_FAST                0 (num):'   ==> '12'
    //'      >>  140 FOR_ITER                98 (to 340)' ==> 140
    //'  5       L1: LOAD_FAST                0 (num)'    ==> 'L1'  (labels are present for python >= 3.13)
    //'  3           LOAD_FAST_LOAD_FAST      0 (num, num)'  ===> '3'
    //'              UNPACK_SEQUENCE          1'    ===>  Block #<id>
    override getBbId(firstInst: string): string {
        const label = firstInst.match(/(\w+):/)?.[1];
        if (label) return label;
        const bytecodeOffset = firstInst.match(/^\s*\d*?\s+>?>?\s+(\d+)/)?.[1];
        if (bytecodeOffset) return bytecodeOffset;

        return 'Block #' + this.BBid++;
    }

    override getBbFirstInstIdx(firstLine: number) {
        return firstLine;
    }

    override extractNodeIdFromInst(inst: string) {
        return this.getBbId(inst);
    }

    override extractAltJmpTargetName(asmArr: AssemblyLine[], bbIdx: number, arrBB: CanonicalBB[]): string {
        if (bbIdx >= arrBB.length) return '';
        return arrBB[bbIdx + 1].nameId;
    }
}
