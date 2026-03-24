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

import {beforeAll, describe, expect, it} from 'vitest';

import {LlvmPassDumpParser} from '../lib/parsers/llvm-pass-dump-parser.js';
import {ResultLine} from '../types/resultline/resultline.interfaces.js';

function deepCopy(obj: ResultLine[]): ResultLine[] {
    return JSON.parse(JSON.stringify(obj));
}

describe('llvm-pass-dump-parser filter', () => {
    let llvmPassDumpParser: LlvmPassDumpParser;

    beforeAll(() => {
        llvmPassDumpParser = new LlvmPassDumpParser();
    });
    // biome-ignore format: keep as-is for readability
    const rawFuncIR: ResultLine[] = [
        {text: '  # Machine code for function f(S1&, S2 const&): NoPHIs, TracksLiveness, TiedOpsRewritten'},
        {text: 'define dso_local void @f(S1&, S2 const&)(%struct.S1* noundef nonnull align 8 dereferenceable(16) %s1, %struct.S2* noundef nonnull align 8 dereferenceable(16) %s2) #0 !dbg !7 {',},
        {text: 'entry:'},
        {text: '  %s1.addr = alloca %struct.S1*, align 8'},
        {text: '  store %struct.S1* %s1, %struct.S1** %s1.addr, align 8, !tbaa !32'},
        {text: '  call void @llvm.dbg.declare(metadata %struct.S1** %s1.addr, metadata !30, metadata !DIExpression()), !dbg !36',},
        {text: '  call void @llvm.dbg.value(metadata %struct.S1* %s1, metadata !30, metadata !DIExpression()), !dbg !32',},
        {text: '  tail call void @llvm.dbg.declare(metadata i16* %p.addr, metadata !24, metadata !DIExpression()), !dbg !12',},
        {text: '  tail call void @llvm.dbg.value(metadata i32 0, metadata !20, metadata !DIExpression()), !dbg !21'},
        {text: '    #dbg_declare(i16* %x.addr, !44, !DIExpression(), !42)'},
        {text: '    #dbg_value(i32 10, !40, !DIExpression(), !41)'},
        {text: '  DBG_VALUE $rdi, $noreg, !"s1", !DIExpression(), debug-location !32; example.cpp:0 line no:7'},
        {text: '  store %struct.S2* %s2, %struct.S2** %s2.addr, align 8, !tbaa !32'},
        {text: '  %0 = load %struct.S2*, %struct.S2** %s2.addr, align 8, !dbg !38, !tbaa !32'},
        {text: '  %a = getelementptr inbounds %struct.S2, %struct.S2* %0, i32 0, i32 0, !dbg !39'},
        {text: '  %1 = load i64, i64* %t, align 8, !dbg !40, !tbaa !41'},
        {text: '  %2 = load %struct.S1*, %struct.S1** %s1.addr, align 8, !dbg !46, !tbaa !32'},
        {text: '  store i64 %1, i64* %t2, align 8, !dbg !49, !tbaa !50'},
        {text: '  %t3 = getelementptr inbounds %struct.Wrapper2, %struct.Wrapper2* %b, i32 0, i32 0, !dbg !54'},
        {text: '  ret void, !dbg !61'},
    ];

    it('should not filter out dbg metadata', () => {
        const options = {filterDebugInfo: false};
        expect(llvmPassDumpParser.applyIrFilters(deepCopy(rawFuncIR), options)).toEqual(rawFuncIR);
    });

    it('should filter out dbg metadata too', () => {
        const options = {filterDebugInfo: true};
        // biome-ignore format: keep as-is for readability
        expect(llvmPassDumpParser.applyIrFilters(deepCopy(rawFuncIR), options)).toEqual([
            {text: '  # Machine code for function f(S1&, S2 const&): NoPHIs, TracksLiveness, TiedOpsRewritten'},
            {text: 'define dso_local void @f(S1&, S2 const&)(%struct.S1* noundef nonnull align 8 dereferenceable(16) %s1, %struct.S2* noundef nonnull align 8 dereferenceable(16) %s2) {',},
            {text: 'entry:'},
            {text: '  %s1.addr = alloca %struct.S1*, align 8'},
            {text: '  store %struct.S1* %s1, %struct.S1** %s1.addr, align 8, !tbaa !32'},
            {text: '  store %struct.S2* %s2, %struct.S2** %s2.addr, align 8, !tbaa !32'},
            {text: '  %0 = load %struct.S2*, %struct.S2** %s2.addr, align 8, !tbaa !32'},
            {text: '  %a = getelementptr inbounds %struct.S2, %struct.S2* %0, i32 0, i32 0'},
            {text: '  %1 = load i64, i64* %t, align 8, !tbaa !41'},
            {text: '  %2 = load %struct.S1*, %struct.S1** %s1.addr, align 8, !tbaa !32'},
            {text: '  store i64 %1, i64* %t2, align 8, !tbaa !50'},
            {text: '  %t3 = getelementptr inbounds %struct.Wrapper2, %struct.Wrapper2* %b, i32 0, i32 0'},
            {text: '  ret void'},
        ]);
    });

    it('should filter out instruction metadata and object attribute group, leave debug instructions in place', () => {
        // 'hide IR metadata' aims to decrease more visual noise than `hide debug info`
        const options = {filterDebugInfo: false, filterIRMetadata: true};
        // biome-ignore format: keep as-is for readability
        expect(llvmPassDumpParser.applyIrFilters(deepCopy(rawFuncIR), options)).toEqual([
            {text: '  # Machine code for function f(S1&, S2 const&): NoPHIs, TracksLiveness, TiedOpsRewritten'},
            {text: 'define dso_local void @f(S1&, S2 const&)(%struct.S1* noundef nonnull align 8 dereferenceable(16) %s1, %struct.S2* noundef nonnull align 8 dereferenceable(16) %s2) {',},
            {text: 'entry:'},
            {text: '  %s1.addr = alloca %struct.S1*, align 8'},
            {text: '  store %struct.S1* %s1, %struct.S1** %s1.addr, align 8'},
            {text: '  call void @llvm.dbg.declare(metadata %struct.S1** %s1.addr, metadata !30, metadata !DIExpression())',},
            {text: '  call void @llvm.dbg.value(metadata %struct.S1* %s1, metadata !30, metadata !DIExpression())'},
            {text: '  tail call void @llvm.dbg.declare(metadata i16* %p.addr, metadata !24, metadata !DIExpression())'},
            {text: '  tail call void @llvm.dbg.value(metadata i32 0, metadata !20, metadata !DIExpression())'},
            {text: '    #dbg_declare(i16* %x.addr, !44, !DIExpression(), !42)'},
            {text: '    #dbg_value(i32 10, !40, !DIExpression(), !41)'},
            {text: '  DBG_VALUE $rdi, $noreg, !"s1", !DIExpression(), debug-location !32; example.cpp:0 line no:7'},
            {text: '  store %struct.S2* %s2, %struct.S2** %s2.addr, align 8'},
            {text: '  %0 = load %struct.S2*, %struct.S2** %s2.addr, align 8'},
            {text: '  %a = getelementptr inbounds %struct.S2, %struct.S2* %0, i32 0, i32 0'},
            {text: '  %1 = load i64, i64* %t, align 8'},
            {text: '  %2 = load %struct.S1*, %struct.S1** %s1.addr, align 8'},
            {text: '  store i64 %1, i64* %t2, align 8'},
            {text: '  %t3 = getelementptr inbounds %struct.Wrapper2, %struct.Wrapper2* %b, i32 0, i32 0'},
            {text: '  ret void'},
        ]);
    });
});

describe('llvm-pass-dump-parser Old style IR Dump header', () => {
    let llvmPassDumpParser: LlvmPassDumpParser;

    beforeAll(() => {
        llvmPassDumpParser = new LlvmPassDumpParser();
    });

    const rawFuncIR = [
        {text: '*** IR Dump After NoOpModulePass on [module] ***'},
        {text: 'define void @foo() {'},
        {text: '  ret void'},
        {text: '}'},
        {text: 'define void @bar() {'},
        {text: 'entry:'},
        {text: '  br label %my-loop'},
        {text: 'my-loop:                                          ; preds = %my-loop, %entry'},
        {text: '  br label %my-loop'},
        {text: '}'},
    ];

    it('should recognize dump', () => {
        const brokenDown = llvmPassDumpParser.breakdownOutputIntoPassDumps(deepCopy(rawFuncIR));

        expect(brokenDown).toEqual([
            {
                affectedFunction: undefined,
                header: 'IR Dump After NoOpModulePass on [module]',
                lines: [
                    {text: 'define void @foo() {'},
                    {text: '  ret void'},
                    {text: '}'},
                    {text: 'define void @bar() {'},
                    {text: 'entry:'},
                    {text: '  br label %my-loop'},
                    {text: 'my-loop:                                          ; preds = %my-loop, %entry'},
                    {text: '  br label %my-loop'},
                    {text: '}'},
                ],
                machine: false,
            },
        ]);
    });
});

describe('llvm-pass-dump-parser New style IR Dump header', () => {
    let llvmPassDumpParser: LlvmPassDumpParser;

    beforeAll(() => {
        llvmPassDumpParser = new LlvmPassDumpParser();
    });

    const rawFuncIR = [
        {text: '; *** IR Dump After NoOpModulePass on [module] ***'},
        {text: 'define void @foo() {'},
        {text: '  ret void'},
        {text: '}'},
        {text: 'define void @bar() {'},
        {text: 'entry:'},
        {text: '  br label %my-loop'},
        {text: 'my-loop:                                          ; preds = %my-loop, %entry'},
        {text: '  br label %my-loop'},
        {text: '}'},
    ];

    it('should recognize dump', () => {
        const brokenDown = llvmPassDumpParser.breakdownOutputIntoPassDumps(deepCopy(rawFuncIR));

        expect(brokenDown).toEqual([
            {
                affectedFunction: undefined,
                header: 'IR Dump After NoOpModulePass on [module]',
                lines: [
                    {text: 'define void @foo() {'},
                    {text: '  ret void'},
                    {text: '}'},
                    {text: 'define void @bar() {'},
                    {text: 'entry:'},
                    {text: '  br label %my-loop'},
                    {text: 'my-loop:                                          ; preds = %my-loop, %entry'},
                    {text: '  br label %my-loop'},
                    {text: '}'},
                ],
                machine: false,
            },
        ]);
    });
});

describe('breakdownPassDumpsIntoFunctions', () => {
    let parser: LlvmPassDumpParser;

    beforeAll(() => {
        parser = new LlvmPassDumpParser();
    });

    it('should handle a single IR function', () => {
        const dump = {
            header: 'IR Dump After InstCombinePass on foo',
            affectedFunction: undefined,
            machine: false,
            lines: [
                {text: 'define i32 @foo(i32 %x) {'},
                {text: 'entry:'},
                {text: '  ret i32 %x'},
                {text: '}'},
            ] as ResultLine[],
        };
        const result = parser.breakdownPassDumpsIntoFunctions(dump);
        expect(result).toEqual({
            header: 'IR Dump After InstCombinePass on foo',
            machine: false,
            functions: {
                foo: [{text: 'define i32 @foo(i32 %x) {'}, {text: 'entry:'}, {text: '  ret i32 %x'}, {text: '}'}],
            },
        });
    });

    it('should handle multiple IR functions in a single dump', () => {
        const dump = {
            header: 'IR Dump After NoOpModulePass on [module]',
            affectedFunction: undefined,
            machine: false,
            lines: [
                {text: 'define void @foo() {'},
                {text: '  ret void'},
                {text: '}'},
                {text: ''},
                {text: 'define i32 @bar(i32 %x) {'},
                {text: '  ret i32 %x'},
                {text: '}'},
            ] as ResultLine[],
        };
        const result = parser.breakdownPassDumpsIntoFunctions(dump);
        expect(result).toEqual({
            header: 'IR Dump After NoOpModulePass on [module]',
            machine: false,
            functions: {
                foo: [{text: 'define void @foo() {'}, {text: '  ret void'}, {text: '}'}],
                bar: [{text: 'define i32 @bar(i32 %x) {'}, {text: '  ret i32 %x'}, {text: '}'}],
            },
        });
    });

    it('should handle loop dumps (starting with "; Preheader:")', () => {
        const dump = {
            header: 'IR Dump Before LoopDeletionPass on loop',
            affectedFunction: undefined,
            machine: false,
            lines: [
                {text: '; Preheader:'},
                {text: '  br label %loop'},
                {text: '; Loop:'},
                {text: '  br i1 %cond, label %loop, label %exit'},
            ] as ResultLine[],
        };
        const result = parser.breakdownPassDumpsIntoFunctions(dump);
        expect(result).toEqual({
            header: 'IR Dump Before LoopDeletionPass on loop',
            machine: false,
            functions: {
                '<loop>': [
                    {text: '; Preheader:'},
                    {text: '  br label %loop'},
                    {text: '; Loop:'},
                    {text: '  br i1 %cond, label %loop, label %exit'},
                ],
            },
        });
    });

    it('should handle machine function dumps', () => {
        const dump = {
            header: 'IR Dump After PHIElimination',
            affectedFunction: undefined,
            machine: true,
            lines: [
                {text: '# Machine code for function _Z3fooi:'},
                {text: 'bb.0:'},
                {text: '  RET64'},
                {text: '# End machine code for function _Z3fooi.'},
            ] as ResultLine[],
        };
        const result = parser.breakdownPassDumpsIntoFunctions(dump);
        expect(result).toEqual({
            header: 'IR Dump After PHIElimination',
            machine: true,
            functions: {
                _Z3fooi: [
                    {text: '# Machine code for function _Z3fooi:'},
                    {text: 'bb.0:'},
                    {text: '  RET64'},
                    {text: '# End machine code for function _Z3fooi.'},
                ],
            },
        });
    });

    it('should skip blank lines outside functions', () => {
        const dump = {
            header: 'IR Dump After SomePass',
            affectedFunction: undefined,
            machine: false,
            lines: [
                {text: ''},
                {text: ''},
                {text: 'define void @foo() {'},
                {text: '  ret void'},
                {text: '}'},
                {text: ''},
            ] as ResultLine[],
        };
        const result = parser.breakdownPassDumpsIntoFunctions(dump);
        expect(Object.keys(result.functions)).toEqual(['foo']);
    });

    it('should handle empty dump with no functions', () => {
        const dump = {
            header: 'IR Dump After SomePass',
            affectedFunction: undefined,
            machine: false,
            lines: [] as ResultLine[],
        };
        const result = parser.breakdownPassDumpsIntoFunctions(dump);
        expect(result.functions).toEqual({});
    });
});

describe('breakdownIntoPassDumpsByFunction', () => {
    let parser: LlvmPassDumpParser;

    beforeAll(() => {
        parser = new LlvmPassDumpParser();
    });

    it('should transpose from pass->functions to function->passes', () => {
        const passDumps = [
            {
                header: 'IR Dump Before InstCombinePass',
                machine: false,
                functions: {
                    foo: [{text: 'define void @foo() {'}, {text: '  ret void'}, {text: '}'}] as ResultLine[],
                    bar: [{text: 'define void @bar() {'}, {text: '  ret void'}, {text: '}'}] as ResultLine[],
                },
            },
        ];
        const result = parser.breakdownIntoPassDumpsByFunction(passDumps);
        expect(Object.keys(result).sort()).toEqual(['bar', 'foo']);
        expect(result['foo']).toHaveLength(1);
        expect(result['foo'][0].header).toBe('IR Dump Before InstCombinePass');
        expect(result['foo'][0].affectedFunction).toBeUndefined();
        expect(result['foo'][0].machine).toBe(false);
        expect(result['bar']).toHaveLength(1);
    });

    it('should attribute loop dumps to the previous function', () => {
        const passDumps: {header: string; machine: boolean; functions: Record<string, ResultLine[]>}[] = [
            {
                header: 'IR Dump Before InstCombinePass',
                machine: false,
                functions: {
                    myFunc: [{text: 'define void @myFunc() {'}, {text: '}'}],
                },
            },
            {
                header: 'IR Dump Before LoopDeletionPass',
                machine: false,
                functions: {
                    '<loop>': [{text: '; Preheader:'}, {text: '  br label %loop'}],
                },
            },
        ];
        const result = parser.breakdownIntoPassDumpsByFunction(passDumps);
        expect(Object.keys(result)).toEqual(['myFunc']);
        expect(result['myFunc']).toHaveLength(2);
        expect(result['myFunc'][1].header).toBe('IR Dump Before LoopDeletionPass');
    });

    it('should preserve previousFunction when pass header ends with (invalidated)', () => {
        // Issue #4195: SimpleLoopUnswitchPass can dump multiple functions when invalidated,
        // but the next pass can still be loop-only in the same function
        const passDumps: {header: string; machine: boolean; functions: Record<string, ResultLine[]>}[] = [
            {
                header: 'IR Dump Before SimpleLoopUnswitchPass',
                machine: false,
                functions: {
                    myFunc: [{text: '; Preheader:'}],
                },
            },
            {
                header: 'IR Dump After SimpleLoopUnswitchPass (invalidated)',
                machine: false,
                functions: {
                    myFunc: [{text: 'define void @myFunc() {'}, {text: '}'}],
                    otherFunc: [{text: 'define void @otherFunc() {'}, {text: '}'}],
                },
            },
            {
                header: 'IR Dump Before LoopDeletionPass',
                machine: false,
                functions: {
                    '<loop>': [{text: '; Preheader:'}, {text: '  br label %loop'}],
                },
            },
        ];
        const result = parser.breakdownIntoPassDumpsByFunction(passDumps);
        // The loop dump should still be attributed to myFunc (previousFunction preserved)
        expect(result['myFunc']).toHaveLength(3);
        expect(result['myFunc'][2].header).toBe('IR Dump Before LoopDeletionPass');
    });

    it('should clear previousFunction for multi-function dumps without (invalidated)', () => {
        const passDumps: {header: string; machine: boolean; functions: Record<string, ResultLine[]>}[] = [
            {
                header: 'IR Dump Before SomePass',
                machine: false,
                functions: {
                    myFunc: [{text: 'define void @myFunc() {'}, {text: '}'}],
                },
            },
            {
                header: 'IR Dump After SomeModulePass',
                machine: false,
                functions: {
                    foo: [{text: 'define void @foo() {'}, {text: '}'}],
                    bar: [{text: 'define void @bar() {'}, {text: '}'}],
                },
            },
        ];
        const result = parser.breakdownIntoPassDumpsByFunction(passDumps);
        // After a multi-function dump without (invalidated), previousFunction is null
        expect(Object.keys(result).sort()).toEqual(['bar', 'foo', 'myFunc']);
    });

    it('should handle empty function list in a pass (Printing <null> Function)', () => {
        const passDumps = [
            {
                header: 'IR Dump After PrintingNullFunction',
                machine: false,
                functions: {},
            },
        ];
        const result = parser.breakdownIntoPassDumpsByFunction(passDumps);
        expect(result).toEqual({});
    });

    it('should accumulate multiple passes for the same function', () => {
        const passDumps = [
            {
                header: 'IR Dump Before Pass1',
                machine: false,
                functions: {
                    foo: [{text: 'v1'}] as ResultLine[],
                },
            },
            {
                header: 'IR Dump After Pass1',
                machine: false,
                functions: {
                    foo: [{text: 'v2'}] as ResultLine[],
                },
            },
        ];
        const result = parser.breakdownIntoPassDumpsByFunction(passDumps);
        expect(result['foo']).toHaveLength(2);
        expect(result['foo'][0].header).toBe('IR Dump Before Pass1');
        expect(result['foo'][1].header).toBe('IR Dump After Pass1');
    });
});

describe('matchPassDumps', () => {
    let parser: LlvmPassDumpParser;

    beforeAll(() => {
        parser = new LlvmPassDumpParser();
    });

    it('should pair Before and After dumps', () => {
        const input: Record<string, any[]> = {
            foo: [
                {
                    header: 'IR Dump Before InstCombinePass on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: '  %a = add i32 %x, 0'}],
                },
                {
                    header: 'IR Dump After InstCombinePass on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: '  ret i32 %x'}],
                },
            ],
        };
        const result = parser.matchPassDumps(input);
        expect(result['foo']).toHaveLength(1);
        expect(result['foo'][0].name).toBe('InstCombinePass on foo');
        expect(result['foo'][0].before).toEqual([{text: '  %a = add i32 %x, 0'}]);
        expect(result['foo'][0].after).toEqual([{text: '  ret i32 %x'}]);
        expect(result['foo'][0].irChanged).toBe(true);
        expect(result['foo'][0].machine).toBe(false);
    });

    it('should handle Before with no After (e.g., Delete dead loops)', () => {
        const input: Record<string, any[]> = {
            foo: [
                {
                    header: 'IR Dump Before LoopDeletionPass on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: '; Preheader:'}],
                },
            ],
        };
        const result = parser.matchPassDumps(input);
        expect(result['foo']).toHaveLength(1);
        expect(result['foo'][0].name).toBe('LoopDeletionPass on foo');
        expect(result['foo'][0].before).toEqual([{text: '; Preheader:'}]);
        expect(result['foo'][0].after).toEqual([]);
        // before is non-empty, after is empty -> irChanged is true
        expect(result['foo'][0].irChanged).toBe(true);
    });

    it('should handle After with no Before', () => {
        const input: Record<string, any[]> = {
            foo: [
                {
                    header: 'IR Dump After SomePass on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: '  ret void'}],
                },
            ],
        };
        const result = parser.matchPassDumps(input);
        expect(result['foo']).toHaveLength(1);
        expect(result['foo'][0].name).toBe('SomePass on foo');
        expect(result['foo'][0].before).toEqual([]);
        expect(result['foo'][0].after).toEqual([{text: '  ret void'}]);
        expect(result['foo'][0].irChanged).toBe(true);
    });

    it('should detect irChanged=false when before and after are identical', () => {
        const lines = [{text: 'define void @foo() {'}, {text: '  ret void'}, {text: '}'}] as ResultLine[];
        const input: Record<string, any[]> = {
            foo: [
                {
                    header: 'IR Dump Before InstCombinePass on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: deepCopy(lines),
                },
                {
                    header: 'IR Dump After InstCombinePass on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: deepCopy(lines),
                },
            ],
        };
        const result = parser.matchPassDumps(input);
        expect(result['foo'][0].irChanged).toBe(false);
    });

    it('should handle (invalidated) suffix in after header matching', () => {
        const input: Record<string, any[]> = {
            foo: [
                {
                    header: 'IR Dump Before SimpleLoopUnswitchPass on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: '  br label %loop'}],
                },
                {
                    header: 'IR Dump After SimpleLoopUnswitchPass on foo (invalidated)',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: '  ret void'}],
                },
            ],
        };
        const result = parser.matchPassDumps(input);
        expect(result['foo']).toHaveLength(1);
        expect(result['foo'][0].name).toBe('SimpleLoopUnswitchPass on foo');
        expect(result['foo'][0].irChanged).toBe(true);
    });

    it('should grab previous IR pass after as before for first machine pass (IR->MIR boundary)', () => {
        const irAfter = [{text: 'define void @foo() {'}, {text: '  ret void'}, {text: '}'}] as ResultLine[];
        const mirBefore = [{text: '# Machine code for function foo:'}, {text: '  RET'}] as ResultLine[];
        const mirAfter = [{text: '# Machine code for function foo:'}, {text: '  RET'}] as ResultLine[];

        const input: Record<string, any[]> = {
            foo: [
                {
                    header: 'IR Dump Before InstCombinePass on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: 'define void @foo() {'}, {text: '  %a = add i32 1, 0'}, {text: '}'}],
                },
                {
                    header: 'IR Dump After InstCombinePass on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: irAfter,
                },
                {
                    header: 'IR Dump Before PHIElimination on foo',
                    affectedFunction: undefined,
                    machine: true,
                    lines: mirBefore,
                },
                {
                    header: 'IR Dump After PHIElimination on foo',
                    affectedFunction: undefined,
                    machine: true,
                    lines: mirAfter,
                },
            ],
        };
        const result = parser.matchPassDumps(input);
        expect(result['foo']).toHaveLength(2);
        // The first machine pass should have the IR pass's after as its before
        expect(result['foo'][1].before).toEqual(irAfter);
        expect(result['foo'][1].after).toEqual(mirAfter);
        expect(result['foo'][1].machine).toBe(true);
    });

    it('should handle multiple consecutive before/after pairs', () => {
        const input: Record<string, any[]> = {
            foo: [
                {
                    header: 'IR Dump Before Pass1 on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: 'v1'}],
                },
                {
                    header: 'IR Dump After Pass1 on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: 'v2'}],
                },
                {
                    header: 'IR Dump Before Pass2 on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: 'v2'}],
                },
                {
                    header: 'IR Dump After Pass2 on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: 'v3'}],
                },
            ],
        };
        const result = parser.matchPassDumps(input);
        expect(result['foo']).toHaveLength(2);
        expect(result['foo'][0].name).toBe('Pass1 on foo');
        expect(result['foo'][1].name).toBe('Pass2 on foo');
        expect(result['foo'][0].irChanged).toBe(true);
        expect(result['foo'][1].irChanged).toBe(true);
    });

    it('should handle multiple functions independently', () => {
        const input: Record<string, any[]> = {
            foo: [
                {
                    header: 'IR Dump Before Pass1 on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: 'foo-before'}],
                },
                {
                    header: 'IR Dump After Pass1 on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: 'foo-after'}],
                },
            ],
            bar: [
                {
                    header: 'IR Dump Before Pass1 on bar',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: 'bar-before'}],
                },
                {
                    header: 'IR Dump After Pass1 on bar',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: 'bar-after'}],
                },
            ],
        };
        const result = parser.matchPassDumps(input);
        expect(Object.keys(result).sort()).toEqual(['bar', 'foo']);
        expect(result['foo'][0].name).toBe('Pass1 on foo');
        expect(result['bar'][0].name).toBe('Pass1 on bar');
    });

    it('should handle Before followed by another Before (skipped After)', () => {
        const input: Record<string, any[]> = {
            foo: [
                {
                    header: 'IR Dump Before LoopDeletionPass on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: 'loop-ir'}],
                },
                {
                    header: 'IR Dump Before InstCombinePass on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: 'before-combine'}],
                },
                {
                    header: 'IR Dump After InstCombinePass on foo',
                    affectedFunction: undefined,
                    machine: false,
                    lines: [{text: 'after-combine'}],
                },
            ],
        };
        const result = parser.matchPassDumps(input);
        expect(result['foo']).toHaveLength(2);
        // First pass: Before with no After
        expect(result['foo'][0].name).toBe('LoopDeletionPass on foo');
        expect(result['foo'][0].before).toEqual([{text: 'loop-ir'}]);
        expect(result['foo'][0].after).toEqual([]);
        // Second pass: normal Before/After pair
        expect(result['foo'][1].name).toBe('InstCombinePass on foo');
    });
});

describe('associateFullDumpsWithFunctions', () => {
    let parser: LlvmPassDumpParser;

    beforeAll(() => {
        parser = new LlvmPassDumpParser();
    });

    it('should handle function-annotated passes and create <Full Module> entry', () => {
        const passDumps = [
            {
                header: 'IR Dump Before InstCombinePass',
                affectedFunction: 'foo',
                machine: false,
                lines: [{text: 'define void @foo() {'}, {text: '}'}] as ResultLine[],
            },
            {
                header: 'IR Dump After InstCombinePass',
                affectedFunction: 'foo',
                machine: false,
                lines: [{text: 'define void @foo() {'}, {text: '  ret void'}, {text: '}'}] as ResultLine[],
            },
        ];
        const result = parser.associateFullDumpsWithFunctions(passDumps);
        expect(Object.keys(result).sort()).toEqual(['<Full Module>', 'foo']);
        expect(result['foo']).toHaveLength(2);
        // Headers get annotated with function name in parentheses
        expect(result['foo'][0].header).toBe('IR Dump Before InstCombinePass (foo)');
        expect(result['foo'][0].affectedFunction).toBe('foo');
        expect(result['foo'][1].header).toBe('IR Dump After InstCombinePass (foo)');
        // Full Module gets all function-specific passes too
        expect(result['<Full Module>']).toHaveLength(2);
        expect(result['<Full Module>'][0].header).toBe('IR Dump Before InstCombinePass (foo)');
    });

    it('should handle loop annotations (%-prefixed) attributed to previous function', () => {
        const passDumps = [
            {
                header: 'IR Dump Before LICM',
                affectedFunction: 'myFunc',
                machine: false,
                lines: [{text: 'define void @myFunc() {'}, {text: '}'}] as ResultLine[],
            },
            {
                header: 'IR Dump Before LoopRotate',
                affectedFunction: '%loop',
                machine: false,
                lines: [{text: '; Preheader:'}, {text: '  br label %loop'}] as ResultLine[],
            },
        ];
        const result = parser.associateFullDumpsWithFunctions(passDumps);
        // %loop should be attributed to myFunc (previous function)
        expect(Object.keys(result).sort()).toEqual(['<Full Module>', 'myFunc']);
        expect(result['myFunc']).toHaveLength(2);
        // Header includes the resolved function name, not %loop
        expect(result['myFunc'][1].header).toBe('IR Dump Before LoopRotate (myFunc)');
        expect(result['myFunc'][1].affectedFunction).toBe('myFunc');
    });

    it('should broadcast module-level passes (no affectedFunction) to all functions', () => {
        const passDumps = [
            {
                header: 'IR Dump Before InstCombinePass',
                affectedFunction: 'foo',
                machine: false,
                lines: [{text: 'foo ir'}] as ResultLine[],
            },
            {
                header: 'IR Dump Before InstCombinePass',
                affectedFunction: 'bar',
                machine: false,
                lines: [{text: 'bar ir'}] as ResultLine[],
            },
            {
                header: 'IR Dump After GlobalOptPass',
                affectedFunction: undefined,
                machine: false,
                lines: [{text: 'module level ir'}] as ResultLine[],
            },
        ];
        const result = parser.associateFullDumpsWithFunctions(passDumps);
        expect(Object.keys(result).sort()).toEqual(['<Full Module>', 'bar', 'foo']);
        // Module-level pass should appear in all entries
        expect(result['foo'].some(p => p.header === 'IR Dump After GlobalOptPass')).toBe(true);
        expect(result['bar'].some(p => p.header === 'IR Dump After GlobalOptPass')).toBe(true);
        expect(result['<Full Module>'].some(p => p.header === 'IR Dump After GlobalOptPass')).toBe(true);
        // Module-level pass has undefined affectedFunction
        const modulePass = result['foo'].find(p => p.header === 'IR Dump After GlobalOptPass');
        expect(modulePass?.affectedFunction).toBeUndefined();
    });

    it('should clear previousFunction after a module-level pass', () => {
        const passDumps = [
            {
                header: 'IR Dump Before LICM',
                affectedFunction: 'myFunc',
                machine: false,
                lines: [{text: 'ir'}] as ResultLine[],
            },
            {
                header: 'IR Dump After GlobalOptPass',
                affectedFunction: undefined,
                machine: false,
                lines: [{text: 'module ir'}] as ResultLine[],
            },
        ];
        const result = parser.associateFullDumpsWithFunctions(passDumps);
        // previousFunction should be null now; verify structure is correct
        expect(Object.keys(result).sort()).toEqual(['<Full Module>', 'myFunc']);
    });

    it('should create <Full Module> entry even with no function-annotated passes', () => {
        const passDumps = [
            {
                header: 'IR Dump After GlobalOptPass',
                affectedFunction: undefined,
                machine: false,
                lines: [{text: 'module ir'}] as ResultLine[],
            },
        ];
        const result = parser.associateFullDumpsWithFunctions(passDumps);
        expect(Object.keys(result)).toEqual(['<Full Module>']);
        expect(result['<Full Module>']).toHaveLength(1);
    });

    it('should handle multiple functions with interleaved passes', () => {
        const passDumps = [
            {
                header: 'IR Dump Before Pass1',
                affectedFunction: 'foo',
                machine: false,
                lines: [{text: 'foo-before'}] as ResultLine[],
            },
            {
                header: 'IR Dump After Pass1',
                affectedFunction: 'foo',
                machine: false,
                lines: [{text: 'foo-after'}] as ResultLine[],
            },
            {
                header: 'IR Dump Before Pass1',
                affectedFunction: 'bar',
                machine: false,
                lines: [{text: 'bar-before'}] as ResultLine[],
            },
            {
                header: 'IR Dump After Pass1',
                affectedFunction: 'bar',
                machine: false,
                lines: [{text: 'bar-after'}] as ResultLine[],
            },
        ];
        const result = parser.associateFullDumpsWithFunctions(passDumps);
        expect(Object.keys(result).sort()).toEqual(['<Full Module>', 'bar', 'foo']);
        expect(result['foo']).toHaveLength(2);
        expect(result['bar']).toHaveLength(2);
        // <Full Module> gets everything
        expect(result['<Full Module>']).toHaveLength(4);
    });
});

describe('breakdownOutputIntoPassDumps additional headers', () => {
    let parser: LlvmPassDumpParser;

    beforeAll(() => {
        parser = new LlvmPassDumpParser();
    });

    it('should parse machine code dump headers (# *** ... ***:)', () => {
        const input: ResultLine[] = [
            {text: '# *** IR Dump After PHIElimination ***:'},
            {text: '# Machine code for function foo:'},
            {text: '  RET'},
        ];
        const result = parser.breakdownOutputIntoPassDumps(input);
        expect(result).toHaveLength(1);
        expect(result[0].header).toBe('IR Dump After PHIElimination');
        expect(result[0].machine).toBe(true);
        expect(result[0].affectedFunction).toBeUndefined();
        expect(result[0].lines).toEqual([{text: '# Machine code for function foo:'}, {text: '  RET'}]);
    });

    it('should parse CIR dump headers (// -----// ... //----- //)', () => {
        const input: ResultLine[] = [
            {text: '// -----// IR Dump Before CIRSimplify (cir-simplify) //----- //'},
            {text: 'cir.func @main() {'},
            {text: '  cir.return'},
            {text: '}'},
        ];
        const result = parser.breakdownOutputIntoPassDumps(input);
        expect(result).toHaveLength(1);
        expect(result[0].header).toBe('IR Dump Before CIRSimplify (cir-simplify)');
        expect(result[0].machine).toBe(false);
        expect(result[0].affectedFunction).toBeUndefined();
    });

    it('should parse headers with function annotations (function: name)', () => {
        const input: ResultLine[] = [
            {text: '; *** IR Dump Before InstCombinePass on foo ***  (function: foo)'},
            {text: 'define void @foo() {'},
            {text: '  ret void'},
            {text: '}'},
        ];
        const result = parser.breakdownOutputIntoPassDumps(input);
        expect(result).toHaveLength(1);
        expect(result[0].affectedFunction).toBe('foo');
    });

    it('should parse headers with loop annotations (loop: %name)', () => {
        const input: ResultLine[] = [
            {text: '; *** IR Dump Before LICM on foo ***  (loop: %loop)'},
            {text: '; Preheader:'},
            {text: '  br label %loop'},
        ];
        const result = parser.breakdownOutputIntoPassDumps(input);
        expect(result).toHaveLength(1);
        expect(result[0].affectedFunction).toBe('%loop');
    });

    it('should handle multiple consecutive pass dumps splitting correctly', () => {
        const input: ResultLine[] = [
            {text: '*** IR Dump After Pass1 ***'},
            {text: 'line1'},
            {text: '*** IR Dump After Pass2 ***'},
            {text: 'line2'},
        ];
        const result = parser.breakdownOutputIntoPassDumps(input);
        expect(result).toHaveLength(2);
        expect(result[0].header).toBe('IR Dump After Pass1');
        expect(result[0].lines).toEqual([{text: 'line1'}]);
        expect(result[1].header).toBe('IR Dump After Pass2');
        expect(result[1].lines).toEqual([{text: 'line2'}]);
    });

    it('should collapse duplicate blank lines within a pass body', () => {
        const input: ResultLine[] = [
            {text: '*** IR Dump After Pass1 ***'},
            {text: 'line1'},
            {text: ''},
            {text: ''},
            {text: ''},
            {text: 'line2'},
        ];
        const result = parser.breakdownOutputIntoPassDumps(input);
        expect(result[0].lines).toEqual([{text: 'line1'}, {text: ''}, {text: 'line2'}]);
    });

    it('should skip leading blank lines after header', () => {
        const input: ResultLine[] = [{text: '*** IR Dump After Pass1 ***'}, {text: ''}, {text: ''}, {text: 'line1'}];
        const result = parser.breakdownOutputIntoPassDumps(input);
        expect(result[0].lines).toEqual([{text: 'line1'}]);
    });

    it('should handle header with trailing comment (;...)', () => {
        const input: ResultLine[] = [
            {text: '*** IR Dump After InstCombinePass on foo ***;some trailing comment'},
            {text: 'define void @foo() {'},
            {text: '}'},
        ];
        const result = parser.breakdownOutputIntoPassDumps(input);
        expect(result).toHaveLength(1);
        expect(result[0].header).toBe('IR Dump After InstCombinePass on foo');
    });
});

describe('process() end-to-end', () => {
    let parser: LlvmPassDumpParser;

    beforeAll(() => {
        parser = new LlvmPassDumpParser();
    });

    it('should crop junk before the first header', () => {
        const output: ResultLine[] = [
            {text: 'warning: some compiler warning'},
            {text: 'another junk line'},
            {text: '; *** IR Dump Before InstCombinePass on foo ***'},
            {text: 'define void @foo() {'},
            {text: '  ret void'},
            {text: '}'},
            {text: '; *** IR Dump After InstCombinePass on foo ***'},
            {text: 'define void @foo() {'},
            {text: '  ret void'},
            {text: '}'},
        ];
        const result = parser.process(output, {} as any, {});
        expect(result['foo']).toHaveLength(1);
        expect(result['foo'][0].name).toBe('InstCombinePass on foo');
        // Verify the junk lines don't appear
        expect(result['foo'][0].before.every(l => !l.text.includes('warning'))).toBe(true);
    });

    it('should handle a realistic multi-pass multi-function non-fullModule dump', () => {
        // biome-ignore format: keep as-is for readability
        const output: ResultLine[] = [
            {text: '; *** IR Dump Before InstCombinePass on foo ***'},
            {text: 'define i32 @foo(i32 %x) {'},
            {text: '  %a = add i32 %x, 0'},
            {text: '  ret i32 %a'},
            {text: '}'},
            {text: '; *** IR Dump After InstCombinePass on foo ***'},
            {text: 'define i32 @foo(i32 %x) {'},
            {text: '  ret i32 %x'},
            {text: '}'},
            {text: '; *** IR Dump Before InstCombinePass on bar ***'},
            {text: 'define void @bar() {'},
            {text: '  ret void'},
            {text: '}'},
            {text: '; *** IR Dump After InstCombinePass on bar ***'},
            {text: 'define void @bar() {'},
            {text: '  ret void'},
            {text: '}'},
        ];
        const result = parser.process(output, {} as any, {});
        expect(Object.keys(result).sort()).toEqual(['bar', 'foo']);
        expect(result['foo']).toHaveLength(1);
        expect(result['foo'][0].irChanged).toBe(true);
        expect(result['bar']).toHaveLength(1);
        expect(result['bar'][0].irChanged).toBe(false);
    });

    it('should handle fullModule=true mode with function annotations', () => {
        // biome-ignore format: keep as-is for readability
        const output: ResultLine[] = [
            {text: '; *** IR Dump Before InstCombinePass on foo ***  (function: foo)'},
            {text: 'define void @foo() {'},
            {text: '  ret void'},
            {text: '}'},
            {text: '; *** IR Dump After InstCombinePass on foo ***  (function: foo)'},
            {text: 'define void @foo() {'},
            {text: '  ret void'},
            {text: '}'},
        ];
        const result = parser.process(output, {} as any, {fullModule: true});
        expect(Object.keys(result).sort()).toEqual(['<Full Module>', 'foo']);
        expect(result['foo']).toHaveLength(1);
        expect(result['foo'][0].name).toBe('InstCombinePass on foo (foo)');
        expect(result['<Full Module>']).toHaveLength(1);
    });

    it('should handle machine code headers in a multi-pass flow', () => {
        const output: ResultLine[] = [
            {text: '; *** IR Dump Before InstCombinePass on foo ***'},
            {text: 'define void @foo() {'},
            {text: '  ret void'},
            {text: '}'},
            {text: '; *** IR Dump After InstCombinePass on foo ***'},
            {text: 'define void @foo() {'},
            {text: '  ret void'},
            {text: '}'},
            {text: '# *** IR Dump After PHIElimination ***:'},
            {text: '# Machine code for function foo:'},
            {text: 'bb.0:'},
            {text: '  RET64'},
            {text: '# End machine code for function foo.'},
        ];
        const result = parser.process(output, {} as any, {});
        expect(result['foo']).toHaveLength(2);
        expect(result['foo'][0].machine).toBe(false);
        expect(result['foo'][1].machine).toBe(true);
        // First machine pass should get the IR pass's after as its before
        expect(result['foo'][1].before).toEqual([{text: 'define void @foo() {'}, {text: '  ret void'}, {text: '}'}]);
    });

    it('should assert when CIR body lines appear outside a recognized function define', () => {
        const output: ResultLine[] = [
            {text: '// -----// IR Dump Before CIRSimplify (cir-simplify) //----- //'},
            {text: 'cir.func @main() {'},
            {text: '  cir.return'},
            {text: '}'},
            {text: '// -----// IR Dump After CIRSimplify (cir-simplify) //----- //'},
            {text: 'cir.func @main() {'},
            {text: '  cir.return'},
            {text: '}'},
        ];
        // CIR doesn't use standard `define` so the parser hits an assertion
        // when it encounters body lines outside a recognized function
        expect(() => parser.process(output, {} as any, {})).toThrow(/Assertion failed/);
    });

    it('should filter debug info when filterDebugInfo is set', () => {
        // biome-ignore format: keep as-is for readability
        const output: ResultLine[] = [
            {text: '; *** IR Dump Before InstCombinePass on foo ***'},
            {text: 'define void @foo() !dbg !7 {'},
            {text: '  call void @llvm.dbg.value(metadata i32 0, metadata !20, metadata !DIExpression()), !dbg !21'},
            {text: '  ret void, !dbg !22'},
            {text: '}'},
            {text: '; *** IR Dump After InstCombinePass on foo ***'},
            {text: 'define void @foo() !dbg !7 {'},
            {text: '  ret void, !dbg !22'},
            {text: '}'},
        ];
        const result = parser.process(output, {} as any, {filterDebugInfo: true});
        expect(result['foo']).toHaveLength(1);
        // The dbg.value line should be filtered out
        const beforeLines = result['foo'][0].before.map(l => l.text);
        expect(beforeLines.some(l => l.includes('dbg.value'))).toBe(false);
        // The function define should have !dbg stripped
        expect(beforeLines[0]).toBe('define void @foo() {');
    });

    it('should assert when output has no recognized headers (junk passed to breakdownOutput)', () => {
        const output: ResultLine[] = [{text: 'just some random output'}, {text: 'no headers here'}];
        // findIndex returns -1, slice(-1) gets last line, which isn't a header
        // so breakdownOutputIntoPassDumps hits an assertion on the non-header line
        expect(() => parser.process(output, {} as any, {})).toThrow(/Assertion failed/);
    });

    it('should handle Before with no After in end-to-end flow', () => {
        const output: ResultLine[] = [
            {text: '; *** IR Dump Before LoopDeletionPass on myloop ***'},
            {text: 'define void @myFunc() {'},
            {text: 'entry:'},
            {text: '  br label %loop'},
            {text: 'loop:'},
            {text: '  br label %loop'},
            {text: '}'},
        ];
        const result = parser.process(output, {} as any, {});
        expect(result['myFunc']).toHaveLength(1);
        expect(result['myFunc'][0].name).toBe('LoopDeletionPass on myloop');
        expect(result['myFunc'][0].after).toEqual([]);
        expect(result['myFunc'][0].irChanged).toBe(true);
    });

    it('should filter IR metadata when filterIRMetadata is set', () => {
        const output: ResultLine[] = [
            {text: '; *** IR Dump Before InstCombinePass on foo ***'},
            {text: 'define void @foo() {'},
            {text: '  store i32 1, i32* %x, align 4, !tbaa !5'},
            {text: '  ret void'},
            {text: '}'},
            {text: '; *** IR Dump After InstCombinePass on foo ***'},
            {text: 'define void @foo() {'},
            {text: '  store i32 1, i32* %x, align 4, !tbaa !5'},
            {text: '  ret void'},
            {text: '}'},
        ];
        const result = parser.process(output, {} as any, {filterIRMetadata: true});
        expect(result['foo']).toHaveLength(1);
        // !tbaa metadata should be stripped
        const beforeLines = result['foo'][0].before.map(l => l.text);
        expect(beforeLines[1]).toBe('  store i32 1, i32* %x, align 4');
    });

    it('should handle fullModule with module-level passes broadcast to all functions', () => {
        const output: ResultLine[] = [
            {text: '; *** IR Dump Before InstCombinePass on foo ***  (function: foo)'},
            {text: 'define void @foo() {'},
            {text: '  ret void'},
            {text: '}'},
            {text: '; *** IR Dump After InstCombinePass on foo ***  (function: foo)'},
            {text: 'define void @foo() {'},
            {text: '  ret void'},
            {text: '}'},
            {text: '; *** IR Dump Before InstCombinePass on bar ***  (function: bar)'},
            {text: 'define void @bar() {'},
            {text: '  ret void'},
            {text: '}'},
            {text: '; *** IR Dump After InstCombinePass on bar ***  (function: bar)'},
            {text: 'define void @bar() {'},
            {text: '  ret void'},
            {text: '}'},
            {text: '; *** IR Dump After GlobalOptPass on [module] ***'},
            {text: 'define void @foo() {'},
            {text: '  ret void'},
            {text: '}'},
            {text: 'define void @bar() {'},
            {text: '  ret void'},
            {text: '}'},
        ];
        const result = parser.process(output, {} as any, {fullModule: true});
        expect(Object.keys(result).sort()).toEqual(['<Full Module>', 'bar', 'foo']);
        // Each function should have the module-level pass
        expect(result['foo'].some(p => p.name.includes('GlobalOptPass'))).toBe(true);
        expect(result['bar'].some(p => p.name.includes('GlobalOptPass'))).toBe(true);
    });
});
