// Copyright (c) 2026, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//     Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
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

import * as monaco from 'monaco-editor';
import TomSelect from 'tom-select';
import {describe, expect, it} from 'vitest';

import type {CompilationResult} from '../../../types/compilation/compilation.interfaces.js';
import type {CompilerInfo} from '../../../types/compiler.interfaces.js';
import {DiffType} from '../../panes/diff.interfaces.js';
import {Diff, DiffStateObject} from '../../panes/diff.js';

const GCC = {id: 'gcc', name: 'x86-64 gcc'} as CompilerInfo;
const GNAT = {id: 'gnat', name: 'x86-64 gnat', supportsGnatDebugViews: true} as CompilerInfo;
const CLANG = {id: 'clang', name: 'x86-64 clang', supportsAstView: true, supportsIrView: true} as CompilerInfo;

const ASM = DiffType.ASM.toString();
const GNAT_TYPES = [DiffType.GNAT_ExpandedCode.toString(), DiffType.GNAT_Tree.toString()];
const CLANG_TYPES = [DiffType.AstOutput.toString(), DiffType.IrOutput.toString()];

function makePicker(): TomSelect {
    const select = document.createElement('select');
    document.body.append(select);
    return new TomSelect(select, {valueField: 'id', labelField: 'name', options: [], items: []});
}

function makeSide(id?: number, difftype = DiffType.ASM) {
    return new DiffStateObject(id, monaco.editor.createModel('', 'asm'), difftype);
}

function makeDiff(lhs = makeSide(), rhs = makeSide()) {
    const diff = Object.create(Diff.prototype) as Diff;
    Object.assign(diff, {
        compilers: {},
        sources: {},
        lhs,
        rhs,
        selectize: {lhs: makePicker(), rhs: makePicker(), lhsdifftype: makePicker(), rhsdifftype: makePicker()},
        eventHub: {emit: () => {}},
        updateTitle: () => {},
        updateState: () => {},
    });
    return diff;
}

function offeredTypes(picker: TomSelect): string[] {
    return Object.keys(picker.options);
}

// Opens compiler panes #1..#n; the first one is auto-selected on both sides.
function openCompilers(diff: Diff, ...compilers: CompilerInfo[]) {
    for (const [index, compiler] of compilers.entries()) {
        diff.onCompiler(index + 1, compiler, '', 1, 0);
    }
}

function selectRhs(diff: Diff, id: number) {
    diff.setSelectedEntry(diff.rhs, diff.selectize.rhsdifftype, diff.compilers[id]);
}

describe('Diff type picker', () => {
    it('offers GNAT views only for compilers that produce them', () => {
        const diff = makeDiff();
        openCompilers(diff, GCC, GNAT);
        selectRhs(diff, 2);

        for (const type of GNAT_TYPES) {
            expect(offeredTypes(diff.selectize.lhsdifftype)).not.toContain(type);
        }
        expect(offeredTypes(diff.selectize.rhsdifftype)).toEqual(expect.arrayContaining(GNAT_TYPES));
    });

    it("keeps each side to its own compiler's views when a result arrives", () => {
        const diff = makeDiff();
        openCompilers(diff, GCC, CLANG);
        selectRhs(diff, 2);

        diff.onCompileResult(2, CLANG, {} as CompilationResult);

        for (const type of CLANG_TYPES) {
            expect(offeredTypes(diff.selectize.lhsdifftype)).not.toContain(type);
        }
        expect(offeredTypes(diff.selectize.rhsdifftype)).toEqual(expect.arrayContaining(CLANG_TYPES));
    });

    it('falls back to assembly when the newly selected compiler cannot produce the selected view', () => {
        const diff = makeDiff();
        openCompilers(diff, GCC, GNAT);
        selectRhs(diff, 2);
        diff.rhs.difftype = DiffType.GNAT_Tree;

        selectRhs(diff, 1);

        expect(diff.rhs.difftype).toBe(DiffType.ASM);
        expect(diff.selectize.rhsdifftype.getValue()).toBe(ASM);
        for (const type of GNAT_TYPES) {
            expect(offeredTypes(diff.selectize.rhsdifftype)).not.toContain(type);
        }
    });

    it('restores a saved GNAT view once its compiler announces itself', () => {
        const diff = makeDiff(makeSide(1, DiffType.GNAT_Tree), makeSide(1, DiffType.GNAT_Tree));

        openCompilers(diff, GNAT);

        expect(diff.lhs.difftype).toBe(DiffType.GNAT_Tree);
        expect(diff.selectize.lhsdifftype.getValue()).toBe(DiffType.GNAT_Tree.toString());
    });

    it('drops a saved GNAT view whose compiler cannot produce it', () => {
        const diff = makeDiff(makeSide(1, DiffType.GNAT_Tree), makeSide(1, DiffType.GNAT_Tree));

        openCompilers(diff, GCC);

        expect(diff.lhs.difftype).toBe(DiffType.ASM);
        expect(diff.selectize.lhsdifftype.getValue()).toBe(ASM);
    });

    it('tells the user how to get GNAT output rather than showing an empty pane', () => {
        const diff = makeDiff(makeSide(1, DiffType.GNAT_Tree));
        openCompilers(diff, GNAT);

        diff.onCompileResult(1, GNAT, {} as CompilationResult);

        expect(diff.lhs.model.getValue()).toBe("<select 'Add new...' → 'GNAT Debug Tree' in this compiler's pane>");
    });

    describe('tool output', () => {
        const MCA = {id: 'llvm-mcatrunk', name: 'llvm-mca', code: 0, stdout: [{text: 'Iterations: 100'}], stderr: []};
        const TIDY = {id: 'clangtidytrunk', name: 'clang-tidy', code: 1, stdout: [], stderr: [{text: 'warning: x'}]};
        const mcaOption = encodeToolValue(MCA.id);

        function encodeToolValue(id: string): string {
            return `${DiffType.ToolOutput}:${id}`;
        }

        function selectTool(diff: Diff, id: string) {
            diff.lhs.difftype = DiffType.ToolOutput;
            diff.lhs.extraoption = id;
        }

        it('offers each tool that has a result, by name', () => {
            const diff = makeDiff();
            openCompilers(diff, GCC);

            diff.onCompileResult(1, GCC, {tools: [MCA, TIDY]} as unknown as CompilationResult);

            const options = diff.selectize.lhsdifftype.options;
            expect(options[mcaOption].name).toBe('llvm-mca');
            expect(options[encodeToolValue(TIDY.id)].name).toBe('clang-tidy');
        });

        it('offers no tools when the result has none', () => {
            const diff = makeDiff();
            openCompilers(diff, GCC);

            diff.onCompileResult(1, GCC, {} as CompilationResult);

            expect(
                offeredTypes(diff.selectize.lhsdifftype).filter(id => id.startsWith(`${DiffType.ToolOutput}:`)),
            ).toEqual([]);
        });

        it('reads tools from a nested result as well', () => {
            const diff = makeDiff();
            openCompilers(diff, GCC);

            diff.onCompileResult(1, GCC, {result: {tools: [MCA]}} as unknown as CompilationResult);

            expect(offeredTypes(diff.selectize.lhsdifftype)).toContain(mcaOption);
        });

        it("shows the selected tool's stdout followed by its stderr", () => {
            const diff = makeDiff(makeSide(1));
            openCompilers(diff, GCC);
            selectTool(diff, TIDY.id);

            diff.onCompileResult(1, GCC, {
                tools: [MCA, {...TIDY, stdout: [{text: 'out'}]}],
            } as unknown as CompilationResult);

            expect(diff.lhs.model.getValue()).toBe('out\nwarning: x');
        });

        it('tells the user how to get the output while the tool has no result', () => {
            const diff = makeDiff(makeSide(1));
            openCompilers(diff, GCC);
            selectTool(diff, MCA.id);

            diff.onCompileResult(1, GCC, {} as CompilationResult);

            expect(diff.lhs.model.getValue()).toBe("<select 'Add tool...' → this tool in this compiler's pane>");
        });

        it('restores a saved tool selection once its compiler announces itself', () => {
            const lhs = makeSide(1, DiffType.ToolOutput);
            lhs.extraoption = MCA.id;
            const diff = makeDiff(lhs);

            openCompilers(diff, GCC);

            expect(diff.lhs.difftype).toBe(DiffType.ToolOutput);
            expect(diff.selectize.lhsdifftype.getValue()).toBe(mcaOption);
        });

        it('keeps the selected tool when switching compiler, so the same tool can be compared across panes', () => {
            const diff = makeDiff(makeSide(1), makeSide(1));
            openCompilers(diff, GCC, CLANG);
            diff.onCompileResult(1, GCC, {tools: [MCA]} as unknown as CompilationResult);
            diff.rhs.difftype = DiffType.ToolOutput;
            diff.rhs.extraoption = MCA.id;

            selectRhs(diff, 2);
            expect(diff.rhs.difftype).toBe(DiffType.ToolOutput);
            expect(diff.selectize.rhsdifftype.getValue()).toBe(mcaOption);

            diff.onCompileResult(2, CLANG, {
                tools: [{...MCA, stdout: [{text: 'Iterations: 200'}]}],
            } as unknown as CompilationResult);
            expect(diff.rhs.model.getValue()).toBe('Iterations: 200');
        });
    });
});
