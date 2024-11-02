// Copyright (c) 2024, Compiler Explorer Authors
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

import _ from 'underscore';

import {EdgeColor} from '../../../types/compilation/cfg.interfaces.js';
import {logger} from '../../logger.js';
import {BaseInstructionSetInfo, InstructionType} from '../instruction-sets/base.js';

import {AssemblyLine, BaseCFGParser, BBRange, CanonicalBB, Edge, Range} from './base.js';

// This currently only covers the default arm64 output. To support dex2oat's
// other ISAs, we just need to make sure the correct isJmpInstruction() is being
// used, and parse out the destination addrs differently. For example:
// - x86/x86_64      0x00004071    jmp +34 (0x00004098)
// - arm    i        0x00004052    b 0x00004078
// - riscv64         0x0000411c    bltz t6, +40 ; 0x00004144
export class OatCFGParser extends BaseCFGParser {
    code: AssemblyLine[];

    jmpAddrRegex: RegExp;
    hexRegex: RegExp;

    constructor(instructionSetInfo: BaseInstructionSetInfo) {
        super(instructionSetInfo);
        this.code = [];
        this.jmpAddrRegex = /.*\(addr (0x.*)\)/;
        this.hexRegex = /0x0+(.*)/;
    }

    static override get key() {
        return 'oat';
    }

    // Generally the same as the base filterData(), but we keep empty lines
    // because there are no other indicators for the end of a function.
    override filterData(assembly: AssemblyLine[]) {
        const isCode = (x: AssemblyLine) => x && (x.source !== null || this.isFunctionName(x));
        return this.filterTextSection(assembly).map(_.clone).filter(isCode);
    }

    // For filtering through the 'Instruction set', 'Instruction set features',
    // and 'Compiler filter' lines at the top of the compiler output.
    isHeaderInfo(text: string) {
        return text.startsWith('Instruction ') || text.startsWith('Compiler ');
    }

    // Uses the same general flow as the overridden function, but accounts for
    // empty spaces between functions.
    override splitToFunctions(asmArr: AssemblyLine[]) {
        if (asmArr.length === 0) return [];
        const result: Range[] = [];
        let first = 0;
        while (!asmArr[first].text || this.isHeaderInfo(asmArr[first].text)) {
            ++first;
        }
        const last = asmArr.length;
        const fnRange: Range = {start: first, end: first};
        while (first !== last) {
            if (this.isFunctionEnd(asmArr[first].text)) {
                fnRange.end = first;
                if (fnRange.end > fnRange.start) {
                    result.push(_.clone(fnRange));
                }
                fnRange.start = first + 1;
            }
            ++first;
        }

        fnRange.end = last;
        if (fnRange.end > fnRange.start) {
            result.push(_.clone(fnRange));
        }

        return result;
    }

    // In these examples, '0x416c' and '0x8074' will be returned.
    //     0x00004144    b #+0x28 (addr 0x416c)
    //     0x00008050    b.hs #+0x24 (addr 0x00008074)
    getJmpAddr(inst: string): string {
        const match = inst.match(this.jmpAddrRegex);
        if (match) return this.shortenHex(match[1]);
        return '';
    }

    // In this example, '0x0000416c' will be returned.
    //     0x0000416c    ret
    getPc(inst: string) {
        return inst.trim().split(/\s+/)[0].toLowerCase();
    }

    // In this example, 'add' will be returned.
    //     0x00004168    add w0, w1, #0x3 (3)
    getOpcode(inst: string) {
        return inst.trim().split(/\s+/)[1].toLowerCase();
    }

    isJmpTarget(inst: string, jmpAddrs: string[]) {
        return jmpAddrs.includes(this.shortenHex(this.getPc(inst)));
    }

    // '0x00004168' -> '0x4168'
    shortenHex(pc: string) {
        const match = pc.match(this.hexRegex);
        if (match) return '0x' + match[1];
        return pc;
    }

    override splitToBasicBlocks(asmArr: AssemblyLine[], range: Range) {
        let first = range.start;
        const last = range.end;
        if (first === last) return [];

        // Collect branch targets so we know where to start new blocks.
        const jmpAddrs: string[] = [];
        while (first < last) {
            if (asmArr[first].text.includes('(addr ')) {
                const addr = this.getJmpAddr(asmArr[first].text);
                if (addr) jmpAddrs.push(addr);
            }
            ++first;
        }
        // range.start is the function name; we want blocks' ranges to begin
        // with the first instruction.
        first = range.start + 1;

        let rangeBb: BBRange = {
            nameId: this.shortenHex(this.getPc(asmArr[first].text)),
            start: first,
            end: 0,
            actionPos: [],
        };

        const newRangeWith = function (oldRange: BBRange, nameId: string, start: number) {
            return {
                nameId: nameId,
                start: start,
                actionPos: [],
                end: oldRange.end,
            };
        };

        const result: BBRange[] = [];
        while (first < last) {
            const inst = asmArr[first].text;
            const opcode = this.getOpcode(inst);
            if (this.isBasicBlockEnd(inst, asmArr[first - 1] ? asmArr[first - 1].text : '')) {
                rangeBb.end = first;
                result.push(_.clone(rangeBb));
                rangeBb = newRangeWith(rangeBb, this.extractNodeName(inst), first + 1);
            } else if (this.instructionSetInfo.isJmpInstruction(opcode)) {
                rangeBb.actionPos.push(first);
            } else if (this.isJmpTarget(inst, jmpAddrs)) {
                rangeBb.end = first;
                result.push(_.clone(rangeBb));
                rangeBb = newRangeWith(rangeBb, this.extractNodeName(inst), first);
            }
            ++first;
        }

        rangeBb.end = last;
        result.push(_.clone(rangeBb));
        return result;
    }

    // Empty lines indicate the end of a function.
    override isFunctionEnd(text: string) {
        return text.trim().length === 0;
    }

    // All nodes are named after the address of their first instruction.
    override extractNodeName(inst: string) {
        return this.shortenHex(this.getPc(inst));
    }

    // Identical to splitToCanonicalBasicBlock(), but with a different node
    // naming scheme.
    splitToCanonicalBasicBlockOat(code: AssemblyLine[], basicBlock: BBRange): CanonicalBB[] {
        const actionPos = basicBlock.actionPos;
        let actPosSz = actionPos.length;
        if (actionPos[actPosSz - 1] + 1 === basicBlock.end) {
            --actPosSz;
        }

        if (actPosSz === 0)
            return [
                {
                    nameId: basicBlock.nameId,
                    start: basicBlock.start,
                    end: basicBlock.end,
                },
            ];
        else if (actPosSz === 1)
            return [
                {nameId: basicBlock.nameId, start: basicBlock.start, end: actionPos[0] + 1},
                {
                    nameId: this.extractNodeName(this.code[actionPos[0] + 1].text),
                    start: actionPos[0] + 1,
                    end: basicBlock.end,
                },
            ];
        else {
            let first = 0;
            const last = actPosSz;
            const blockName = basicBlock.nameId;
            let tmp: CanonicalBB = {nameId: blockName, start: basicBlock.start, end: actionPos[first] + 1};
            const result: CanonicalBB[] = [];
            result.push(_.clone(tmp));
            while (first !== last - 1) {
                tmp.nameId = this.extractNodeName(this.code[actionPos[first] + 1].text);
                tmp.start = actionPos[first] + 1;
                ++first;
                tmp.end = actionPos[first] + 1;
                result.push(_.clone(tmp));
            }

            tmp = {
                nameId: this.extractNodeName(this.code[actionPos[first] + 1].text),
                start: actionPos[first] + 1,
                end: basicBlock.end,
            };
            result.push(_.clone(tmp));

            return result;
        }
    }

    override makeEdges(asmArr: AssemblyLine[], arrOfCanonicalBasicBlock: CanonicalBB[]) {
        const edges: Edge[] = [];

        const setEdge = (sourceNode: string, targetNode: string, color: EdgeColor) => ({
            from: sourceNode,
            to: targetNode,
            arrows: 'to',
            color: color,
        });

        for (const x of arrOfCanonicalBasicBlock) {
            let targetNode;
            const lastInst = asmArr[x.end - 1].text;
            const opcode = this.getOpcode(lastInst);
            switch (this.instructionSetInfo.getInstructionType(opcode)) {
                case InstructionType.jmp: {
                    targetNode = this.shortenHex(this.getJmpAddr(lastInst));
                    edges.push(setEdge(x.nameId, targetNode, 'blue'));
                    break;
                }
                case InstructionType.conditionalJmpInst: {
                    // Branch taken
                    targetNode = this.shortenHex(this.getJmpAddr(lastInst));
                    edges.push(setEdge(x.nameId, targetNode, 'green'));
                    // Branch not taken
                    targetNode = this.extractNodeName(asmArr[x.end].text);
                    edges.push(setEdge(x.nameId, targetNode, 'red'));
                    break;
                }
                case InstructionType.notRetInst: {
                    // No jmp, but the next instruction is in a different basic
                    // block because it is the target of another jmp.
                    if (asmArr[x.end]) {
                        targetNode = this.extractNodeName(asmArr[x.end].text);
                        edges.push(setEdge(x.nameId, targetNode, 'grey'));
                    }
                    break;
                }
                case InstructionType.retInst: {
                    break;
                }
            }
        }
        logger.debug(edges);
        return edges;
    }

    override generateFunctionCfg(code: AssemblyLine[], fn: Range) {
        this.code = _.clone(code);
        const basicBlocks = this.splitToBasicBlocks(code, fn);
        let arrOfCanonicalBasicBlock: CanonicalBB[] = [];
        for (const bb of basicBlocks) {
            // We don't want to use the base class's split method.
            const tmp = this.splitToCanonicalBasicBlockOat(code, bb);
            arrOfCanonicalBasicBlock = arrOfCanonicalBasicBlock.concat(tmp);
        }
        return {
            nodes: this.makeNodes(code, arrOfCanonicalBasicBlock),
            edges: this.makeEdges(code, arrOfCanonicalBasicBlock),
        };
    }
}
