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

import {describe, expect, it} from 'vitest';
import {LabelContext, LabelProcessor} from '../lib/parsers/label-processor.js';
import {ParsedAsmResultLine} from '../types/asmresult/asmresult.interfaces.js';

describe('LabelProcessor tests', () => {
    const processor = new LabelProcessor();
    const mockContext: LabelContext = {
        hasOpcode: (line: string) => /^\s*[A-Za-z]/.test(line),
        checkVLIWpacket: () => false,
        labelDef: /^([.A-Z_a-z][\w$.]*):$/,
        dataDefn: /^\s*\.(ascii|byte|word|quad)/,
        commentRe: /[#;]/,
        instructionRe: /^\s*[A-Za-z]+/,
        identifierFindRe: /([$.@A-Z_a-z]\w*)(?:@\w+)*/g,
        definesGlobal: /^\s*\.global\s*([.A-Z_a-z][\w$.]*)/,
        definesWeak: /^\s*\.weak\s*([.A-Z_a-z][\w$.]*)/,
        definesAlias: /^\s*\.set\s*([.A-Z_a-z][\w$.]*\s*),\s*\.\s*(\+\s*0)?$/,
        definesFunction: /^\s*\.type.*,\s*[#%@]function$/,
        cudaBeginDef: /\.(entry|func)\s+(?:\([^)]*\)\s*)?([$.A-Z_a-z][\w$.]*)$/,
        startAppBlock: /^#APP.*$/,
        endAppBlock: /^#NO_APP.*$/,
        startAsmNesting: /^# Begin ASM.*$/,
        endAsmNesting: /^# End ASM.*$/,
        mipsLabelDefinition: /^\$[\w$.]+:/,
        labelFindNonMips: /[.A-Z_a-z][\w$.]*/g,
        labelFindMips: /[$.A-Z_a-z][\w$.]*/g,
        fixLabelIndentation: (line: string) => line.replace(/^\s+/, ''),
    };

    describe('getLabelFind', () => {
        it('should return MIPS regex for MIPS assembly', () => {
            const asmLines = ['$label1:', 'mov $t0, $t1'];
            const result = processor.getLabelFind(asmLines, mockContext);
            expect(result).toBe(mockContext.labelFindMips);
        });

        it('should return non-MIPS regex for non-MIPS assembly', () => {
            const asmLines = ['label1:', 'mov rax, rbx'];
            const result = processor.getLabelFind(asmLines, mockContext);
            expect(result).toBe(mockContext.labelFindNonMips);
        });
    });

    describe('getUsedLabelsInLine', () => {
        it('should extract labels from instruction line', () => {
            const line = '    mov rax, label1';
            const result = processor.getUsedLabelsInLine(line, mockContext);
            expect(result).toHaveLength(2);
            expect(result[0].name).toBe('rax');
            expect(result[1].name).toBe('label1');
        });

        it('should handle lines with comments', () => {
            const line = '    mov rax, label1 ; comment';
            const result = processor.getUsedLabelsInLine(line, mockContext);
            expect(result).toHaveLength(2);
            expect(result[0].name).toBe('rax');
            expect(result[1].name).toBe('label1');
        });

        it('should return empty array for label definition', () => {
            const line = 'label1:';
            const result = processor.getUsedLabelsInLine(line, mockContext);
            expect(result).toHaveLength(0);
        });
    });

    describe('removeLabelsWithoutDefinition', () => {
        it('should remove labels without definitions', () => {
            const asm: ParsedAsmResultLine[] = [
                {
                    text: 'mov rax, label1',
                    source: null,
                    labels: [
                        {name: 'label1', range: {startCol: 10, endCol: 16}},
                        {name: 'undefined_label', range: {startCol: 18, endCol: 32}},
                    ],
                },
            ];
            const labelDefinitions = {label1: 1};

            processor.removeLabelsWithoutDefinition(asm, labelDefinitions);

            expect(asm[0].labels).toHaveLength(1);
            expect(asm[0].labels![0].name).toBe('label1');
        });
    });

    describe('shouldFilterLabel', () => {
        it('should filter unused labels when filters.labels is true', () => {
            const match = ['label1:', 'label1', undefined] as any;
            const line = 'label1:';
            const labelsUsed = new Set<string>();

            const result = processor.shouldFilterLabel(match, line, labelsUsed, true);
            expect(result).toBe(true);
        });

        it('should not filter used labels', () => {
            const match = ['label1:', 'label1', undefined] as any;
            const line = 'label1:';
            const labelsUsed = new Set(['label1']);

            const result = processor.shouldFilterLabel(match, line, labelsUsed, true);
            expect(result).toBe(false);
        });

        it('should not filter when filters.labels is false', () => {
            const match = ['label1:', 'label1', undefined] as any;
            const line = 'label1:';
            const labelsUsed = new Set<string>();

            const result = processor.shouldFilterLabel(match, line, labelsUsed, false);
            expect(result).toBe(false);
        });
    });
});
