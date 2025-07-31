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

import {beforeAll, describe, expect, it} from 'vitest';

import {MlirPassDumpParser} from '../lib/parsers/mlir-pass-dump-parser.js';
import * as properties from '../lib/properties.js';

function deepCopy<T>(obj: T): T {
    return JSON.parse(JSON.stringify(obj));
}

describe('mlir-pass-dump-parser', () => {
    let mlirPassDumpParser: MlirPassDumpParser;

    beforeAll(() => {
        const fakeProps = new properties.CompilerProps({mlir: {id: 'mlir'}}, properties.fakeProps({}));
        const compilerProps = (fakeProps.get as any).bind(fakeProps, 'mlir');
        mlirPassDumpParser = new MlirPassDumpParser(compilerProps);
    });

    const rawMlirDump = [
        {
            text: "// -----// IR Dump Before Inliner (inline) ('builtin.module' operation) //----- //",
        },
        {text: 'module {'},
        {
            text: '  tt.func public @add_kernel() attributes {noinline = false} {',
        },
        {text: '    %0 = tt.get_program_id x : i32 loc(#loc1)'},
        {text: '    tt.return loc(#loc2)'},
        {text: '  } loc(#loc)'},
        {text: '} loc(#loc)'},
        {text: '#loc = loc("/app/example.py":7:0)'},
        {text: '#loc1 = loc("/app/example.py":8:24)'},
        {text: '#loc2 = loc("/app/example.py":8:4)'},
        {text: ''},
        {text: ''},
        {
            text: "// -----// IR Dump Before Canonicalizer (canonicalize) ('tt.func' operation: @add_kernel) //----- //",
        },
        {text: 'module {'},
        {
            text: '  tt.func public @add_kernel() attributes {noinline = false} {',
        },
        {text: '    %0 = tt.get_program_id x : i32 loc(#loc1)'},
        {text: '    tt.return loc(#loc2)'},
        {text: '  } loc(#loc)'},
        {text: '} loc(#loc)'},
        {text: '#loc = loc("/app/example.py":7:0)'},
        {text: '#loc1 = loc("/app/example.py":8:24)'},
        {text: '#loc2 = loc("/app/example.py":8:4)'},
        {text: ''},
        {text: ''},
        {
            text: "// -----// IR Dump Before TritonRewriteTensorPointer (triton-rewrite-tensor-pointer) ('builtin.module' operation) //----- //",
        },
        {text: 'module {'},
        {
            text: '  tt.func public @add_kernel() attributes {noinline = false} {',
        },
        {text: '    tt.return loc(#loc1)'},
        {text: '  } loc(#loc)'},
        {text: '} loc(#loc)'},
        {text: '#loc = loc("/app/example.py":7:0)'},
        {text: '#loc1 = loc("/app/example.py":8:4)'},
        {text: ''},
        {text: ''},
    ];

    it('should break down output into pass dumps', () => {
        const passDumps = mlirPassDumpParser.breakdownOutputIntoPassDumps(deepCopy(rawMlirDump));

        expect(passDumps.length).toBe(3);
        expect(passDumps[0].header).toBe("IR Dump Before Inliner (inline) ('builtin.module' operation)");
        expect(passDumps[1].header).toBe(
            "IR Dump Before Canonicalizer (canonicalize) ('tt.func' operation: @add_kernel)",
        );
        expect(passDumps[2].header).toBe(
            "IR Dump Before TritonRewriteTensorPointer (triton-rewrite-tensor-pointer) ('builtin.module' operation)",
        );

        // Check that the first pass dump has the correct lines
        expect(passDumps[0].lines.length).toBe(10);
        expect(passDumps[0].lines[0].text).toBe('module {');
        expect(passDumps[0].lines[1].text).toBe('  tt.func public @add_kernel() attributes {noinline = false} {');
    });

    it('should break down pass dumps into functions', () => {
        const passDumps = mlirPassDumpParser.breakdownOutputIntoPassDumps(deepCopy(rawMlirDump));
        const splitPassDump = mlirPassDumpParser.breakdownPassDumpsIntoFunctions(passDumps[0]);

        expect(splitPassDump.header).toBe("IR Dump Before Inliner (inline) ('builtin.module' operation)");
        expect(Object.keys(splitPassDump.functions)).toContain('add_kernel');
        expect(splitPassDump.functions['add_kernel'].length).toBe(10);
    });

    it('should apply IR filters to remove location information', () => {
        const filtered = mlirPassDumpParser.applyIrFilters(deepCopy(rawMlirDump.slice(0, 7)));

        // Should filter out location references but keep the lines
        expect(filtered.length).toBe(7);
        expect(filtered[3].text).toBe('    %0 = tt.get_program_id x : i32');
        expect(filtered[4].text).toBe('    tt.return');
        expect(filtered[5].text).toBe('  }');
        expect(filtered[6].text).toBe('}');
    });

    it('should break down output into pass dumps by function', () => {
        const passDumps = mlirPassDumpParser.breakdownOutputIntoPassDumps(deepCopy(rawMlirDump));
        const splitPassDumps = passDumps.map(dump => mlirPassDumpParser.breakdownPassDumpsIntoFunctions(dump));
        const passDumpsByFunction = mlirPassDumpParser.breakdownIntoPassDumpsByFunction(splitPassDumps);

        expect(Object.keys(passDumpsByFunction)).toContain('add_kernel');
        expect(passDumpsByFunction['add_kernel'].length).toBe(3);

        // Check that the function has all three passes
        const headers = passDumpsByFunction['add_kernel'].map(dump => dump.header);
        expect(headers).toContain("IR Dump Before Inliner (inline) ('builtin.module' operation)");
        expect(headers).toContain("IR Dump Before Canonicalizer (canonicalize) ('tt.func' operation: @add_kernel)");
        expect(headers).toContain(
            "IR Dump Before TritonRewriteTensorPointer (triton-rewrite-tensor-pointer) ('builtin.module' operation)",
        );
    });

    it('should detect IR changes between passes', () => {
        // Create two different IR dumps to test change detection
        const before = [
            {text: 'module {'},
            {text: '  tt.func public @add_kernel() {'},
            {text: '    %0 = tt.get_program_id x : i32'},
            {text: '    tt.return'},
            {text: '  }'},
            {text: '}'},
        ];

        const afterNoChange = deepCopy(before);
        expect(mlirPassDumpParser.isIrChanged(before, afterNoChange)).toBe(false);

        const afterWithChange = [
            {text: 'module {'},
            {text: '  tt.func public @add_kernel() {'},
            {text: '    tt.return'}, // Line removed
            {text: '  }'},
            {text: '}'},
        ];
        expect(mlirPassDumpParser.isIrChanged(before, afterWithChange)).toBe(true);
    });

    it('should match pass dumps and detect changes', () => {
        const passDumps = mlirPassDumpParser.breakdownOutputIntoPassDumps(deepCopy(rawMlirDump));
        const splitPassDumps = passDumps.map(dump => mlirPassDumpParser.breakdownPassDumpsIntoFunctions(dump));
        const passDumpsByFunction = mlirPassDumpParser.breakdownIntoPassDumpsByFunction(splitPassDumps);
        const matchedPasses = mlirPassDumpParser.matchPassDumps(passDumpsByFunction);

        expect(Object.keys(matchedPasses)).toContain('add_kernel');

        const addKernelPasses = matchedPasses['add_kernel'];
        expect(addKernelPasses.length).toBe(2); // We should have 2 passes (comparing the 3 dumps)

        // Check the first pass (Inliner to Canonicalizer)
        expect(addKernelPasses[0].name).toBe("Inliner (inline) ('builtin.module' operation)");
        expect(addKernelPasses[0].irChanged).toBe(false); // No changes between these two dumps

        // Check the second pass (Canonicalizer to TritonRewriteTensorPointer)
        expect(addKernelPasses[1].name).toBe("Canonicalizer (canonicalize) ('tt.func' operation: @add_kernel)");
        expect(addKernelPasses[1].irChanged).toBe(true); // There are changes between these two dumps
    });

    it('should process the complete output correctly', () => {
        const optPipelineOptions = {
            fullModule: false,
            filterDebugInfo: false,
            filterIRMetadata: false,
            noDiscardValueNames: true,
            demangle: true,
            libraryFunctions: false,
        };
        const result = mlirPassDumpParser.process(deepCopy(rawMlirDump), {}, optPipelineOptions);

        expect(Object.keys(result)).toContain('add_kernel');
        expect(result['add_kernel'].length).toBe(2);

        // Verify the passes are correctly identified
        expect(result['add_kernel'][0].name).toBe("Inliner (inline) ('builtin.module' operation)");
        expect(result['add_kernel'][1].name).toBe("Canonicalizer (canonicalize) ('tt.func' operation: @add_kernel)");

        // Verify the IR changes are correctly detected
        expect(result['add_kernel'][0].irChanged).toBe(false);
        expect(result['add_kernel'][1].irChanged).toBe(true);
    });
});
