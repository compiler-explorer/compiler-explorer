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

import {beforeEach, describe, expect, it, vi} from 'vitest';

import {SPIRVAsmParser} from '../lib/parsers/asm-parser-spirv.js';

describe('SPIRVAsmParser', () => {
    let parser: SPIRVAsmParser;

    beforeEach(() => {
        parser = new SPIRVAsmParser();
    });

    describe('getUsedLabelsInLine override', () => {
        it('should detect labels in OpFunctionCall', () => {
            const line = 'OpFunctionCall %void %func_main';
            const labels = parser.getUsedLabelsInLine(line);

            expect(labels).toHaveLength(1);
            expect(labels[0].name).toBe('%func_main');
        });

        it('should detect labels in OpBranch', () => {
            const line = 'OpBranch %label_endif';
            const labels = parser.getUsedLabelsInLine(line);

            expect(labels).toHaveLength(1);
            expect(labels[0].name).toBe('%label_endif');
        });

        it('should detect multiple labels in OpBranchConditional', () => {
            const line = 'OpBranchConditional %cond %true_label %false_label';
            const labels = parser.getUsedLabelsInLine(line);

            expect(labels).toHaveLength(2);
            expect(labels[0].name).toBe('%true_label');
            expect(labels[1].name).toBe('%false_label');
        });

        it('should detect labels in OpSelectionMerge', () => {
            const line = 'OpSelectionMerge %merge_label None';
            const labels = parser.getUsedLabelsInLine(line);

            expect(labels).toHaveLength(1);
            expect(labels[0].name).toBe('%merge_label');
        });

        it('should detect labels in OpLoopMerge', () => {
            const line = 'OpLoopMerge %merge_label %continue_label None';
            const labels = parser.getUsedLabelsInLine(line);

            expect(labels).toHaveLength(2);
            expect(labels[0].name).toBe('%merge_label');
            expect(labels[1].name).toBe('%continue_label');
        });

        it('should detect labels in OpSwitch', () => {
            const line = 'OpSwitch %selector %default 1 %case1 2 %case2';
            const labels = parser.getUsedLabelsInLine(line);

            expect(labels).toHaveLength(3);
            expect(labels[0].name).toBe('%default');
            expect(labels[1].name).toBe('%case1');
            expect(labels[2].name).toBe('%case2');
        });

        it('should be called during assembly processing', () => {
            const spy = vi.spyOn(parser, 'getUsedLabelsInLine');

            const spirvCode = `
                OpBranch %exit_label
                %exit_label = OpLabel
            `;

            parser.processAsm(spirvCode, {
                directives: false,
                labels: false,
                commentOnly: false,
            });

            expect(spy).toHaveBeenCalled();
        });
    });

    describe('SPIR-V assembly processing', () => {
        it('should detect SPIR-V percent labels with custom getUsedLabelsInLine', () => {
            const spirvCode = [
                '%main = OpFunction %void None %1',
                '%entry = OpLabel',
                'OpBranch %exit',
                '%exit = OpLabel',
                'OpReturn',
            ].join('\n');

            const result = parser.processAsm(spirvCode, {
                directives: false,
                labels: false,
                commentOnly: false,
            });

            // Should detect SPIR-V %label definitions
            expect(result.labelDefinitions).toHaveProperty('%main');
            expect(result.labelDefinitions).toHaveProperty('%entry');
            expect(result.labelDefinitions).toHaveProperty('%exit');

            // Should detect SPIR-V %label usage in OpBranch
            const branchLine = result.asm.find(line => line.text?.includes('OpBranch'));
            expect(branchLine?.labels?.some(label => label.name === '%exit')).toBe(true);
        });

        it('should handle SPIR-V control flow instructions with multiple label references', () => {
            const spirvCode = [
                'OpSelectionMerge %merge None',
                'OpBranchConditional %condition %true_block %false_block',
                '%true_block = OpLabel',
                '%false_block = OpLabel',
                '%merge = OpLabel',
            ].join('\n');

            const result = parser.processAsm(spirvCode, {
                directives: false,
                labels: false,
                commentOnly: false,
            });

            // Should identify all label definitions
            expect(result.labelDefinitions).toHaveProperty('%true_block');
            expect(result.labelDefinitions).toHaveProperty('%false_block');
            expect(result.labelDefinitions).toHaveProperty('%merge');

            // Should detect multiple labels in OpBranchConditional
            const branchCondLine = result.asm.find(line => line.text?.includes('OpBranchConditional'));
            const labelNames = branchCondLine?.labels?.map(label => label.name) || [];
            expect(labelNames).toContain('%true_block');
            expect(labelNames).toContain('%false_block');

            // Should detect label in OpSelectionMerge
            const mergeLine = result.asm.find(line => line.text?.includes('OpSelectionMerge'));
            expect(mergeLine?.labels?.some(label => label.name === '%merge')).toBe(true);
        });
    });
});
