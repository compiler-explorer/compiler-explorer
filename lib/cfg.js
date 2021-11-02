// Copyright (c) 2017, Najjar Chedy
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

import { logger } from './logger';
import * as utils from './utils';

const InstructionType_jmp = 0;
const InstructionType_conditionalJmpInst = 1;
const InstructionType_notRetInst = 2;
const InstructionType_retInst = 3;

// deal with typeinfo name for, typeinfo for, vtable for
const isFunctionName = x => x.text.trim().indexOf('.') !== 0;

function getAsmDirective(txt) {
    const pattern = /^\s*(\.[^ L]\S*)/;
    const match = txt.match(pattern);
    if (match !== null) {
        return match[1];
    }
    return null;
}

function filterTextSection(data) {
    let useCurrentSection = true;
    const result = [];
    for (let i in data) {
        const x = data[i];
        const directive = getAsmDirective(x.text);
        if (directive != null) {
            if (directive === '.text' || directive === '.data') {
                useCurrentSection = directive === '.text';
            } else if (directive === '.section') {
                // Only patttern match for now.
                // Extracting section name would require adjusting demangling code
                // as demangled name could contain various symbols including ','.
                useCurrentSection = /\.section\s*"?\.text/.test(x.text);
            } else if (useCurrentSection) {
                result.push(x);
            }
        } else if (useCurrentSection) {
            result.push(x);
        }
    }
    return result;
}

const gcc = {
    filterData: asmArr => {
        const jmpLabelRegex = /\.L\d+:/;
        const isCode = x => x && x.text && (x.source !== null || jmpLabelRegex.test(x.text) || isFunctionName(x));
        return _.chain(filterTextSection(asmArr))
            .map(_.clone)
            .filter(isCode)
            .value();
    },
    isFunctionEnd: x => x[0] !== ' ' && x[0] !== '.' && x.includes(':'),

    isBasicBlockEnd: (inst, prevInst) => inst[0] === '.' || prevInst.includes(' ret'),

    isJmpInstruction: x => x.trim()[0] === 'j' || x.match(/\bb\.*(eq|ne|cs|hs|cc|lo|hi|ls|ge|lt|gt|le|rge|rlt)?\b/),

    getInstructionType: inst => {
        if (inst.includes('jmp') || inst.includes(' b ')) return InstructionType_jmp;
        else if (gcc.isJmpInstruction(inst)) return InstructionType_conditionalJmpInst;
        else if (!inst.includes(' ret')) return InstructionType_notRetInst;
        else return InstructionType_retInst;
    },

    extractNodeName: inst => inst.match(/\.L\d+/) + ':',
};

const clang = {
    filterData: asmArr => {
        const jmpLabelRegex = /\.LBB\d+_\d+:/;
        const isCode = x => x && x.text && (x.source !== null || jmpLabelRegex.test(x.text) || isFunctionName(x));

        const removeComments = x => {
            const pos_x86 = x.text.indexOf('# ');
            const pos_arm = x.text.indexOf('// ');
            if (pos_x86 !== -1)
                x.text = utils.trimRight(x.text.substring(0, pos_x86));
            if (pos_arm !== -1)
                x.text = utils.trimRight(x.text.substring(0, pos_arm));
            return x;
        };

        return _.chain(filterTextSection(asmArr))
            .map(_.clone)
            .filter(isCode)
            .map(removeComments)
            .value();
    },
    isFunctionEnd: x => x[0] !== ' ' && x[0] !== '.' && x.includes(':'),

    isBasicBlockEnd: (inst, prevInst) => inst[0] === '.' || prevInst.includes(' ret'),

    isJmpInstruction: x => x.trim()[0] === 'j' || x.match(/\bb\.*(eq|ne|cs|hs|cc|lo|hi|ls|ge|lt|gt|le|rge|rlt)?\b/),

    getInstructionType: function (inst) {
        if (inst.includes('jmp') || inst.includes(' b ')) return InstructionType_jmp;
        else if (clang.isJmpInstruction(inst)) return InstructionType_conditionalJmpInst;
        else if (!inst.includes(' ret')) return InstructionType_notRetInst;
        else return InstructionType_retInst;
    },

    extractNodeName: inst => inst.match(/\.LBB\d+_\d+/) + ':',
};

function splitToFunctions(asmArr, isEnd) {
    if (asmArr.length === 0) return [];
    const result = [];
    let first = 1;
    const last = asmArr.length;
    const fnRange = {start: 0, end: null};
    while (first !== last) {
        if (isEnd(asmArr[first].text)) {
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

function splitToBasicBlocks(asmArr, range, isEnd, isJmp) {
    let first = range.start;
    const last = range.end;
    if (first === last) return [];
    const functionName = asmArr[first].text;
    ++first;

    let rangeBb = {nameId: functionName.substr(0, 50), start: first, end: null, actionPos: []};
    const result = [];

    const newRangeWith = function (oldRange, nameId, start) {
        return {nameId: nameId, start: start, actionPos: [], end: oldRange.end};
    };

    while (first < last) {
        const inst = asmArr[first].text;
        if (isEnd(inst, asmArr[first - 1] ? asmArr[first - 1].text : '')) {
            rangeBb.end = first;
            result.push(_.clone(rangeBb));
            //inst is expected to be .L*: where * in 1,2,...
            rangeBb = newRangeWith(rangeBb, inst, first + 1);
        } else if (isJmp(inst)) {
            rangeBb.actionPos.push(first);
        }
        ++first;
    }

    rangeBb.end = last;
    result.push(_.clone(rangeBb));
    return result;
}

function splitToCanonicalBasicBlock(basicBlock) {
    const actionPos = basicBlock.actionPos;
    let actPosSz = actionPos.length;
    if (actionPos[actPosSz - 1] + 1 === basicBlock.end) {
        --actPosSz;
    }

    if (actPosSz === 0)
        return [{
            nameId: basicBlock.nameId,
            start: basicBlock.start,
            end: basicBlock.end,
        }];
    else if (actPosSz === 1)
        return [
            {nameId: basicBlock.nameId, start: basicBlock.start, end: actionPos[0] + 1},
            {nameId: basicBlock.nameId + '@' + (actionPos[0] + 1), start: actionPos[0] + 1, end: basicBlock.end},
        ];
    else {
        let first = 0;
        const last = actPosSz;
        const blockName = basicBlock.nameId;
        let tmp = {nameId: blockName, start: basicBlock.start, end: actionPos[first] + 1};
        const result = [];
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

function concatInstructions(asmArr, first, last) {
    return _.chain(asmArr.slice(first, last))
        .map(x => x.text.substr(0, 50))
        .value()
        .join('\n');
}

function makeNodes(asms, arrOfCanonicalBasicBlock) {
    return _.map(arrOfCanonicalBasicBlock, e => {
        return {
            id: e.nameId,
            label: `${e.nameId}${e.nameId.includes(':') ? '' : ':'}\n${concatInstructions(asms, e.start, e.end)}`,
            color: '#99ccff',
            shape: 'box',
        };
    });
}

function makeEdges(asmArr, arrOfCanonicalBasicBlock, rules) {
    const edge = {};
    const edges = [];

    const setEdge = function (edge, sourceNode, targetNode, color) {
        edge.from = sourceNode;
        edge.to = targetNode;
        edge.arrows = 'to';
        edge.color = color;
    };
    const isBasicBlockEnd = rules.isBasicBlockEnd;

    const hasName = function (asmArr, cbb) {
        const asm = asmArr[cbb.end];
        return asm ? isBasicBlockEnd(asm.text, '') : false;
    };

    const generateName = function (name, suffix) {
        const pos = name.indexOf('@');
        if (pos === -1)
            return name + '@' + suffix;

        return name.substring(0, pos + 1) + suffix;
    };
    /* note: x.end-1 possible values:
        jmp .L*, {jne,je,jg,...} .L*, ret/rep ret, call and any other instruction that doesn't change control flow
    */

    _.each(arrOfCanonicalBasicBlock, function (x) {
        let targetNode;
        const lastInst = asmArr[x.end - 1].text;
        switch (rules.getInstructionType(lastInst)) {
            case InstructionType_jmp: {
                //we have to deal only with jmp destination, jmp instruction are always taken.
                //edge from jump inst
                targetNode = rules.extractNodeName(lastInst);
                setEdge(edge, x.nameId, targetNode, 'blue');
                edges.push(_.clone(edge));
                break;
            }
            case InstructionType_conditionalJmpInst: {
                //deal with : branch taken, branch not taken
                targetNode = rules.extractNodeName(lastInst);
                setEdge(edge, x.nameId, targetNode, 'green');
                edges.push(_.clone(edge));
                targetNode = hasName(asmArr, x) ? asmArr[x.end].text : generateName(x.nameId, x.end);
                setEdge(edge, x.nameId, targetNode, 'red');
                edges.push(_.clone(edge));
                logger.debug(edge);
                break;
            }
            case InstructionType_notRetInst: {
                //precondition: lastInst is not last instruction in asmArr (but it is in canonical basic block)
                //note : asmArr[x.end] expected to be .L*:(name of a basic block)
                //       this .L*: has to be exactly after the last instruction in the current canonical basic block
                if (asmArr[x.end]) {
                    targetNode = asmArr[x.end].text;
                    setEdge(edge, x.nameId, targetNode, 'grey');
                    edges.push(_.clone(edge));
                }
                break;
            }
            case InstructionType_retInst:
                break;
        }
    });
    logger.debug(edges);
    return edges;
}

function isLLVMBased(compilerType, version) {
    return version.includes('clang') ||
        version.includes('LLVM') ||
        version.includes('rustc') ||
        compilerType === 'swift' ||
        compilerType === 'zig' ||
        compilerType === 'ispc';
}

export function generateStructure(compilerType, version, asmArr) {
    const rules = isLLVMBased(compilerType, version) ? clang : gcc;
    const code = rules.filterData(asmArr);
    const funcs = splitToFunctions(code, rules.isFunctionEnd);
    if (funcs.length === 0) {
        return funcs;
    }
    const result = {};
    _.each(funcs, _.bind(function (rng) {
        const basicBlocks = splitToBasicBlocks(code, rng, rules.isBasicBlockEnd, rules.isJmpInstruction);
        let arrOfCanonicalBasicBlock = [];
        _.each(basicBlocks, _.bind(function (elm) {
            const tmp = splitToCanonicalBasicBlock(elm);
            arrOfCanonicalBasicBlock = arrOfCanonicalBasicBlock.concat(tmp);
        }, this));

        result[code[rng.start].text] = {
            nodes: makeNodes(code, arrOfCanonicalBasicBlock),
            edges: makeEdges(code, arrOfCanonicalBasicBlock, rules),
        };
    }, this));

    return result;
}
