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

import _ from 'underscore';

import type {CompilerInfo} from '../../types/compiler.interfaces.js';
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {makeDefaultedKeyedTypeGetter} from '../keyed-type.js';
import {logger} from '../logger.js';

import {BaseCFGParser} from './cfg-parsers/base.js';
import {ClangCFGParser} from './cfg-parsers/clang.js';
import {GccCFGParser} from './cfg-parsers/gcc.js';
import {BaseInstructionSetInfo, InstructionType} from './instruction-sets/base.js';

// TODO(jeremy-rifkin):
// I've done some work to split out the compiler / instruction set logic
// We'll want to do some work to fill in information for instruction sets and other compilers
// A good comparison https://godbolt.org/z/8EvqoWhYo
// MSVC especially is a little weird, LLVM is also a much different structure than normal asm

const parsers = makeDefaultedKeyedTypeGetter(
    'cfg parser provider',
    {
        ClangCFGParser,
        GccCFGParser,
    },
    BaseCFGParser,
);
const instructionSets = makeDefaultedKeyedTypeGetter('instruction set info provider', {}, BaseInstructionSetInfo);

type Range = {
    start: number;
    end: number;
};

function splitToFunctions(asmArr: ResultLine[], parser: BaseCFGParser) {
    if (asmArr.length === 0) return [];
    const result: Range[] = [];
    let first = 1;
    const last = asmArr.length;
    const fnRange: Range = {start: 0, end: 0};
    while (first !== last) {
        if (parser.isFunctionEnd(asmArr[first].text)) {
            fnRange.end = first;
            result.push(_.clone(fnRange));
            fnRange.start = first;
        }
        ++first;
    }

    fnRange.end = last;
    result.push(_.clone(fnRange));
    return result;
}

type BBRange = {
    nameId: string;
    start: number;
    end: number;
    actionPos: number[];
};

function splitToBasicBlocks(asmArr: ResultLine[], range: Range, parser: BaseCFGParser) {
    let first = range.start;
    const last = range.end;
    if (first === last) return [];
    const functionName = asmArr[first].text;
    ++first;

    let rangeBb: BBRange = {nameId: functionName, start: first, end: 0, actionPos: []};
    const result: BBRange[] = [];

    const newRangeWith = function (oldRange, nameId, start) {
        return {nameId: nameId, start: start, actionPos: [], end: oldRange.end};
    };

    while (first < last) {
        const inst = asmArr[first].text;
        if (parser.isBasicBlockEnd(inst, asmArr[first - 1] ? asmArr[first - 1].text : '')) {
            rangeBb.end = first;
            result.push(_.clone(rangeBb));
            //inst is expected to be .L*: where * in 1,2,...
            rangeBb = newRangeWith(rangeBb, inst, first + 1);
        } else if (parser.instructionSetInfo.isJmpInstruction(inst)) {
            rangeBb.actionPos.push(first);
        }
        ++first;
    }

    rangeBb.end = last;
    result.push(_.clone(rangeBb));
    return result;
}

type CanonicalBB = {
    nameId: string;
    start: number;
    end: number;
};

function splitToCanonicalBasicBlock(basicBlock: BBRange): CanonicalBB[] {
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
            {nameId: basicBlock.nameId + '@' + (actionPos[0] + 1), start: actionPos[0] + 1, end: basicBlock.end},
        ];
    else {
        let first = 0;
        const last = actPosSz;
        const blockName = basicBlock.nameId;
        let tmp: CanonicalBB = {nameId: blockName, start: basicBlock.start, end: actionPos[first] + 1};
        const result: CanonicalBB[] = [];
        result.push(_.clone(tmp));
        while (first !== last - 1) {
            tmp.nameId = blockName + '@' + (actionPos[first] + 1);
            tmp.start = actionPos[first] + 1;
            ++first;
            tmp.end = actionPos[first] + 1;
            result.push(_.clone(tmp));
        }

        tmp = {nameId: blockName + '@' + (actionPos[first] + 1), start: actionPos[first] + 1, end: basicBlock.end};
        result.push(_.clone(tmp));

        return result;
    }
}

function concatInstructions(asmArr: ResultLine[], first: number, last: number) {
    return asmArr
        .slice(first, last)
        .map(x => x.text)
        .join('\n');
}

type Node = {
    id: string;
    label: string;
};

function makeNodes(asms: ResultLine[], arrOfCanonicalBasicBlock: CanonicalBB[]): Node[] {
    return arrOfCanonicalBasicBlock.map(e => ({
        id: e.nameId,
        label: `${e.nameId}${e.nameId.includes(':') ? '' : ':'}\n${concatInstructions(asms, e.start, e.end)}`,
    }));
}

type Edge = {
    from: string;
    to: string;
    arrows: string;
    color: string;
};

function makeEdges(asmArr: ResultLine[], arrOfCanonicalBasicBlock: CanonicalBB[], parser: BaseCFGParser) {
    const edges: Edge[] = [];

    const setEdge = (sourceNode: string, targetNode: string, color: string) => ({
        from: sourceNode,
        to: targetNode,
        arrows: 'to',
        color: color,
    });
    const isBasicBlockEnd = parser.isBasicBlockEnd;

    const hasName = (asmArr: ResultLine[], cbb: CanonicalBB) => {
        const asm = asmArr[cbb.end];
        return asm ? isBasicBlockEnd(asm.text, '') : false;
    };

    const generateName = function (name: string, suffix: number) {
        const pos = name.indexOf('@');
        if (pos === -1) return `${name}@${suffix}`;

        return name.substring(0, pos + 1) + suffix;
    };
    /* note: x.end-1 possible values:
        jmp .L*, {jne,je,jg,...} .L*, ret/rep ret, call and any other instruction that doesn't change control flow
    */

    for (const x of arrOfCanonicalBasicBlock) {
        let targetNode;
        const lastInst = asmArr[x.end - 1].text;
        switch (parser.instructionSetInfo.getInstructionType(lastInst)) {
            case InstructionType.jmp: {
                //we have to deal only with jmp destination, jmp instruction are always taken.
                //edge from jump inst
                targetNode = parser.extractNodeName(lastInst);
                edges.push(setEdge(x.nameId, targetNode, 'blue'));
                break;
            }
            case InstructionType.conditionalJmpInst: {
                //deal with : branch taken, branch not taken
                targetNode = parser.extractNodeName(lastInst);
                edges.push(setEdge(x.nameId, targetNode, 'green'));
                targetNode = hasName(asmArr, x) ? asmArr[x.end].text : generateName(x.nameId, x.end);
                edges.push(setEdge(x.nameId, targetNode, 'red'));
                break;
            }
            case InstructionType.notRetInst: {
                //precondition: lastInst is not last instruction in asmArr (but it is in canonical basic block)
                //note : asmArr[x.end] expected to be .L*:(name of a basic block)
                //       this .L*: has to be exactly after the last instruction in the current canonical basic block
                if (asmArr[x.end]) {
                    targetNode = asmArr[x.end].text;
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

function isLLVMBased({compilerType, version}: CompilerInfo) {
    return (
        version.includes('clang') ||
        version.includes('LLVM') ||
        version.includes('rustc') ||
        compilerType === 'swift' ||
        compilerType === 'zig' ||
        compilerType === 'ispc'
    );
}

type CFG = {
    nodes: Node[];
    edges: Edge[];
};

export function generateStructure(compilerInfo: CompilerInfo, asmArr: ResultLine[]) {
    const compilerGroup = isLLVMBased(compilerInfo) ? 'clang' : compilerInfo.group;
    const instructionSet = new (instructionSets(compilerInfo.instructionSet))();
    const parser = new (parsers(compilerGroup))(instructionSet);
    const code = parser.filterData(asmArr);
    const functions = splitToFunctions(code, parser);
    if (functions.length === 0) {
        return functions;
    }
    const result: Record<string, CFG> = {};
    for (const fn of functions) {
        const basicBlocks = splitToBasicBlocks(code, fn, parser);
        let arrOfCanonicalBasicBlock: CanonicalBB[] = [];
        for (const bb of basicBlocks) {
            const tmp = splitToCanonicalBasicBlock(bb);
            arrOfCanonicalBasicBlock = arrOfCanonicalBasicBlock.concat(tmp);
        }

        result[code[fn.start].text] = {
            nodes: makeNodes(code, arrOfCanonicalBasicBlock),
            edges: makeEdges(code, arrOfCanonicalBasicBlock, parser),
        };
    }

    return result;
}
