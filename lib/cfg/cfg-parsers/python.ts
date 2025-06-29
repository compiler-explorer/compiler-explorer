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

import {ResultLine} from '../../../types/resultline/resultline.interfaces.js';
import {InstructionType} from '../instruction-sets/base.js';
import {AssemblyLine, BaseCFGParser, CanonicalBB} from './base.js';

export class PythonCFGParser extends BaseCFGParser {
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

    override filterData(bytecode: ResultLine[]) {
        // Filter out module prefix before first function,
        // replace 'Disassembly of' with 'Function #<idx>'
        const res: ResultLine[] = [];
        let i = 0;
        while (
            i < bytecode.length &&
            !bytecode[i].text.startsWith('Disassembly of') &&
            !bytecode[i].text.startsWith('Function #')
        ) {
            i++;
        }

        let funcIdx = 0;
        for (let j = i; j < bytecode.length; j++) {
            const line = bytecode[j];
            if (!line.text) continue;

            if (line.text.startsWith('Disassembly of')) {
                line.text = `Function #${funcIdx++}`;
            }
            res.push(line);
        }
        return res;
    }

    //      10 POP_JUMP_IF_FALSE        5 (to 22)   ===> captures "22"
    override extractJmpTargetName(inst: string) {
        const candidateName = inst.match(/\(to (\d+)\)$/);
        return candidateName ? candidateName[1] : '';
    }

    //'  6     >>   22 LOAD_FAST                0 (num):'   ==> '22'
    //'  4          12 LOAD_FAST                0 (num):'   ==> '12'
    //'        >>  140 FOR_ITER                98 (to 340)' ==> 140
    override getBbId(firstInst: string): string {
        return firstInst.match(/^\s*(\d+)?\s+>?>?\s+(\d+)/)?.[2] ?? '';
    }

    override getBbFirstInstIdx(firstLine: number) {
        return firstLine;
    }

    override extractNodeIdFromInst(inst: string) {
        return this.getBbId(inst);
    }

    override extractAltJmpTargetName(asmArr: AssemblyLine[], bbIdx: number, arrBB: CanonicalBB[]): string {
        const nextBbStart = arrBB[bbIdx + 1]?.start;
        if (!nextBbStart) return '';

        const inst = asmArr[nextBbStart];
        return this.getBbId(inst.text);
    }
}
