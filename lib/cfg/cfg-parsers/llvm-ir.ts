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

import {assert, unwrap} from '../../assert.js';
import {SentryCapture} from '../../sentry.js';
import {BaseInstructionSetInfo} from '../instruction-sets/base.js';

import {AssemblyLine, BaseCFGParser, Edge, Node, Range} from './base.js';

export type BBRange = {
    namePrefix: string; // used to encode the function name in the first block
    nameId: string;
    start: number;
    end: number;
};

export class LlvmIrCfgParser extends BaseCFGParser {
    functionDefinition: RegExp;
    labelRe: RegExp;
    labelReference: RegExp;

    static override get key() {
        return 'llvm';
    }

    constructor(instructionSetInfo: BaseInstructionSetInfo) {
        super(instructionSetInfo);
        this.functionDefinition = /^define .+ @("?[^"]+"?)\(/;
        this.labelRe = /^("?[\w$.-]+"?):\s*(;.*)?$/;
        this.labelReference = /%("?[\w$.-]+"?)/g;
    }

    override filterData(asmArr: AssemblyLine[]) {
        return asmArr;
    }

    override splitToFunctions(asmArr: AssemblyLine[]) {
        if (asmArr.length === 0) return [];
        const result: Range[] = [];
        let i = 0;
        while (i < asmArr.length) {
            if (this.functionDefinition.test(asmArr[i].text)) {
                const start = i;
                do {
                    i++;
                } while (i < asmArr.length && asmArr[i].text !== '}');

                // start is the function define, end is the closing brace
                result.push({
                    start,
                    end: i,
                });
            }
            i++;
        }
        return result;
    }

    splitToLlvmBasicBlocks(code: AssemblyLine[], fn: Range): BBRange[] {
        const fMatch = code[fn.start].text.match(this.functionDefinition);
        const fnName = unwrap(fMatch)[1];
        const result: BBRange[] = [];
        let i = fn.start + 1;
        let bbStart = i;
        let currentName: string = '';
        let namePrefix: string = fnName + '\n\n';
        while (i < fn.end) {
            const match = code[i].text.match(this.labelRe);
            if (match) {
                const label = match[1];
                if (bbStart === i) {
                    assert(result.length === 0);
                    currentName = label;
                } else {
                    // start is the fn / label define, end is exclusive
                    result.push({
                        namePrefix: namePrefix,
                        nameId: currentName,
                        start: bbStart,
                        end: i,
                    });
                    currentName = label;
                    namePrefix = '';
                }
                bbStart = i + 1;
            }
            i++;
        }
        result.push({
            namePrefix: '',
            nameId: currentName,
            start: bbStart,
            end: i,
        });
        return result;
    }

    makeLlvmNodes(asms: AssemblyLine[], canonicalBasicBlocks: BBRange[]): Node[] {
        return canonicalBasicBlocks.map(e => {
            // Trim newlines at the end of a BB
            let end = e.end;
            while (end > e.start && asms[end - 1].text === '') {
                end--;
            }
            return {
                id: e.nameId,
                label: `${e.namePrefix}${e.nameId}${e.nameId.includes(':') ? '' : ':'}\n${this.concatInstructions(
                    asms,
                    e.start,
                    end,
                )}`,
            };
        });
    }

    makeLlvmEdges(asmArr: AssemblyLine[], canonicalBasicBlocks: BBRange[]) {
        const edges: Edge[] = [];
        for (const bb of canonicalBasicBlocks) {
            // Find the last instruction in the basic block. I think asmArr[bb.end] is always an empty line (except for
            // the last basic block) but this is just in case.
            let lastInst = bb.end - 1;
            while (lastInst >= bb.start && asmArr[lastInst].text === '') {
                lastInst--;
            }

            // Ad-hoc handling of a few known cases where LLVM splits a single instruction over multiple lines.
            const terminatingInstruction = (() => {
                if (asmArr[lastInst].text.trim().startsWith(']')) {
                    // Llvm likes to split switches over multiple lines:
                    //  switch i32 %0, label %5 [
                    //    i32 14, label %7
                    //    i32 60, label %2
                    //    i32 12, label %3
                    //    i32 4, label %4
                    //  ], !dbg !60
                    const end = lastInst--;
                    while (!asmArr[lastInst].text.includes('[')) {
                        lastInst--;
                    }
                    return this.concatInstructions(asmArr, lastInst, end + 1);
                } else if (
                    lastInst >= 1 &&
                    asmArr[lastInst].text.includes('unwind label') &&
                    asmArr[lastInst - 1].text.trim().includes('invoke ')
                ) {
                    // Handle multi-line `invoke` like:
                    // invoke void @__cxa_throw(ptr nonnull %exception, ptr nonnull @typeinfo for int, ptr null) #3
                    //          to label %unreachable unwind label %lpad
                    return this.concatInstructions(asmArr, lastInst - 1, lastInst + 1);
                } else if (
                    lastInst >= 1 &&
                    asmArr[lastInst - 1].text.includes('landingpad') &&
                    asmArr[lastInst].text.includes('catch')
                ) {
                    // Handle multi-line `landingpad` like:
                    // %0 = landingpad { ptr, i32 }
                    //         catch ptr null
                    return this.concatInstructions(asmArr, lastInst - 1, lastInst + 1);
                } else if (
                    lastInst >= 1 &&
                    asmArr[lastInst - 1].text.includes('callbr') &&
                    asmArr[lastInst].text.trim().startsWith('to label')
                ) {
                    // Handle multi-line `callbr` like:
                    // %2 = callbr i32 asm "mov ${1:l}, $0", "=r,!i,~{dirflag},~{fpsr},~{flags}"() #2
                    //      to label %asm.fallthrough1 [label %err.split2]
                    return this.concatInstructions(asmArr, lastInst - 1, lastInst + 1);
                } else {
                    return asmArr[lastInst].text;
                }
            })();
            let terminator;
            if (terminatingInstruction.includes('invoke ')) {
                terminator = 'invoke';
            } else if (terminatingInstruction.includes('callbr')) {
                terminator = 'callbr';
            } else {
                terminator = terminatingInstruction.trim().split(' ')[0].replaceAll(',', '');
            }

            const labels = [...terminatingInstruction.matchAll(this.labelReference)].map(m => m[1]);
            switch (terminator) {
                case 'ret':
                case 'unreachable': {
                    break;
                }
                case 'br': {
                    // br label %16, !dbg !41
                    // br i1 %13, label %59, label %14, !dbg !41
                    if (labels.length === 1) {
                        edges.push({
                            from: bb.nameId,
                            to: labels[0],
                            arrows: 'to',
                            color: 'blue',
                        });
                    } else if (labels.length === 3) {
                        edges.push(
                            {
                                from: bb.nameId,
                                to: labels[1],
                                arrows: 'to',
                                color: 'green',
                            },
                            {
                                from: bb.nameId,
                                to: labels[2],
                                arrows: 'to',
                                color: 'red',
                            },
                        );
                    } else if (labels.length === 2) {
                        //  br i1 true, label %bb1, label %bb4
                        edges.push(
                            {
                                from: bb.nameId,
                                to: labels[0],
                                arrows: 'to',
                                color: 'green',
                            },
                            {
                                from: bb.nameId,
                                to: labels[1],
                                arrows: 'to',
                                color: 'red',
                            },
                        );
                    } else {
                        SentryCapture(terminatingInstruction, 'makeLlvmEdges unexpected br');
                        assert(false);
                    }
                    break;
                }
                case 'switch': {
                    // switch i32 %val, label %default [ i32 0, label %onzero i32 1, label %onone i32 2, label %ontwo ]
                    for (const label of labels.slice(1)) {
                        edges.push({
                            from: bb.nameId,
                            to: label,
                            arrows: 'to',
                            color: 'blue',
                        });
                    }
                    break;
                }
                case 'indirectbr': {
                    // indirectbr ptr %Addr, [ label %bb1, label %bb2, label %bb3 ]
                    for (const label of labels.slice(1)) {
                        edges.push({
                            from: bb.nameId,
                            to: label,
                            arrows: 'to',
                            color: 'blue',
                        });
                    }
                    break;
                }
                case 'invoke': {
                    // %retval = invoke i32 @Test(i32 15) to label %Continue unwind label %TestCleanup
                    edges.push(
                        {
                            from: bb.nameId,
                            to: labels[labels.length - 2],
                            arrows: 'to',
                            color: 'green',
                        },
                        {
                            from: bb.nameId,
                            to: labels[labels.length - 1],
                            arrows: 'to',
                            color: 'grey',
                        },
                    );
                    break;
                }
                case 'callbr': {
                    // callbr void asm "", "r,!i"(i32 %x) to label %fallthrough [label %indirect]
                    {
                        const callbrLabelsPart = terminatingInstruction.slice(
                            terminatingInstruction.lastIndexOf('to label'),
                        );
                        const callbrLabels = [...callbrLabelsPart.matchAll(this.labelReference)].map(m => m[1]);
                        edges.push({
                            from: bb.nameId,
                            to: callbrLabels[0],
                            arrows: 'to',
                            color: 'grey',
                        });
                        for (const label of callbrLabels.slice(1)) {
                            edges.push({
                                from: bb.nameId,
                                to: label,
                                arrows: 'to',
                                color: 'blue',
                            });
                        }
                    }
                    break;
                }
                case 'resume': {
                    // TODO: Landing pads?
                    break;
                }
                case 'catchswitch': {
                    // %cs2 = catchswitch within %parenthandler [label %handler0] unwind label %cleanup
                    // TODO
                    break;
                }
                case 'catchret': {
                    // catchret from %catch to label %continue
                    // TODO
                    break;
                }
                case 'cleanupret': {
                    // cleanupret from %cleanup unwind label %continue
                    // TODO
                    break;
                }
                default: {
                    if (bb.start > lastInst) {
                        // this can happen when a basic block is empty, which can happen for the entry block
                    } else {
                        throw new Error(`Unexpected basic block terminator: ${terminatingInstruction}`);
                    }
                }
            }
        }
        return edges;
    }

    override generateFunctionCfg(code: AssemblyLine[], fn: Range) {
        const basicBlocks = this.splitToLlvmBasicBlocks(code, fn);
        return {
            nodes: this.makeLlvmNodes(code, basicBlocks),
            edges: this.makeLlvmEdges(code, basicBlocks),
        };
    }

    override getFnName(code: AssemblyLine[], fn: Range) {
        const match = code[fn.start].text.match(this.functionDefinition);
        return unwrap(match)[1];
    }
}
