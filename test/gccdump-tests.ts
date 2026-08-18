// Copyright (c) 2026, Compiler Explorer Authors
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

import fs from 'node:fs/promises';
import path from 'node:path';

import {afterEach, beforeAll, describe, expect, it, vi} from 'vitest';

import {BaseCompiler} from '../lib/base-compiler.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import * as utils from '../lib/utils.js';
import {GccDumpOptions} from '../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {
    'c++': {id: 'c++'},
} as const;

describe('GCC dump output processing', () => {
    let ce: CompilationEnvironment;
    let compiler: BaseCompiler;

    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
        const info = makeFakeCompilerInfo({
            exe: '/usr/bin/gcc',
            lang: 'c++',
            ldPath: [],
            supportsGccDump: true,
            removeEmptyGccDump: true,
        });
        compiler = new BaseCompiler(info as CompilerInfo, ce);
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    describe('passObjectFromDumpFilename', () => {
        it('parses tree/rtl/ipa dump file names into descriptors', () => {
            expect(compiler.passObjectFromDumpFilename('example.cpp.255r.expand')).toEqual({
                nnn: 255,
                filename_suffix: 'r.expand',
                name: 'expand (rtl)',
                command_prefix: '-fdump-rtl-expand',
            });
            expect(compiler.passObjectFromDumpFilename('example.cpp.081i.cp')).toEqual({
                nnn: 81,
                filename_suffix: 'i.cp',
                name: 'cp (ipa)',
                command_prefix: '-fdump-ipa-cp',
            });
            expect(compiler.passObjectFromDumpFilename('example.cpp.005t.original')).toEqual({
                nnn: 5,
                filename_suffix: 't.original',
                name: 'original (tree)',
                command_prefix: '-fdump-tree-original',
            });
        });

        it('keeps hyphenated/numbered pass names intact', () => {
            expect(compiler.passObjectFromDumpFilename('example.cpp.050t.local-pure-const1')).toEqual({
                nnn: 50,
                filename_suffix: 't.local-pure-const1',
                name: 'local-pure-const1 (tree)',
                command_prefix: '-fdump-tree-local-pure-const1',
            });
            expect(compiler.passObjectFromDumpFilename('example.cpp.274r.loop2_unroll')?.name).toEqual(
                'loop2_unroll (rtl)',
            );
        });

        it('returns null for non-dump files', () => {
            expect(compiler.passObjectFromDumpFilename('example.cpp.s')).toBeNull();
            expect(compiler.passObjectFromDumpFilename('example.cpp')).toBeNull();
            expect(compiler.passObjectFromDumpFilename('example.cpp.o')).toBeNull();
            expect(compiler.passObjectFromDumpFilename('example.dump')).toBeNull();
        });
    });

    describe('trimGccDumpHeaderFunctions', () => {
        const userBlock = [
            ';; Function main (main, funcdef_no=1, decl_uid=1, cgraph_uid=1, symbol_order=1)',
            '[example.cpp:3:5] x = 1;',
            '',
        ].join('\n');
        const headerBlock = [
            ';; Function std::foo (_ZSt3foo, funcdef_no=2, decl_uid=2, cgraph_uid=2, symbol_order=2)',
            '[/usr/include/c++/13/foo.h:10:2] y = 2;',
            '',
        ].join('\n');

        // 4th arg = isRtlDump (RTL dumps are left untouched by the lineno strip).
        it('keeps source-defined functions and drops header-defined ones', () => {
            const trimmed = compiler.trimGccDumpHeaderFunctions(userBlock + headerBlock, 'example.cpp', true, false);
            expect(trimmed).toContain(';; Function main');
            expect(trimmed).not.toContain(';; Function std::foo');
        });

        it('keeps a preamble before the first function', () => {
            const withPreamble = ';; preamble line\n' + headerBlock;
            const trimmed = compiler.trimGccDumpHeaderFunctions(withPreamble, 'example.cpp', true, false);
            expect(trimmed).toContain(';; preamble line');
            expect(trimmed).not.toContain(';; Function std::foo');
        });

        it('returns dumps without function markers unchanged (e.g. IPA summaries)', () => {
            const cgraph = 'Callgraph:\nmain/1 (main)\n  called by:\n';
            expect(compiler.trimGccDumpHeaderFunctions(cgraph, 'example.cpp', true, false)).toEqual(cgraph);
        });

        it('keeps a function block whose origin cannot be determined (no header path)', () => {
            const noLoc = ';; Function mystery (mystery)\n  some body\n';
            expect(compiler.trimGccDumpHeaderFunctions(noLoc, 'example.cpp', true, false)).toEqual(noLoc);
        });

        it('keeps RTL blocks that reference the source via quotes/nested brackets (#regression)', () => {
            // RTL dumps don't use a clean leading [file:line]; they have "file":line and noise
            // like [orig:N] / [N [file:line]]. The block must still be kept because it references
            // the source file. A syntax-based "first location" match used to wrongly empty these.
            const rtlBlock = [
                ';; Function main (main, funcdef_no=0)',
                '(note 1 0 3 [orig:117] NOTE_INSN_DELETED)',
                '(insn 16 14 17 (set (reg:SI r0) (mem:SI)) "example.cpp":5:100 [3 [example.cpp:5:100]])',
                '',
            ].join('\n');
            const trimmed = compiler.trimGccDumpHeaderFunctions(rtlBlock, 'example.cpp', true, true);
            expect(trimmed).toContain(';; Function main');
            expect(trimmed).toContain('[orig:117]'); // RTL brackets left intact
        });

        it('keeps non-lineno RTL brackets ([orig:N], hex, %) but drops the insn filename when lineno is disabled', () => {
            const rtlBlock =
                ';; Function main (main)\n' +
                '(note 1 0 3 [orig:117] NOTE_INSN_DELETED)\n' +
                '(insn 8 7 9 (set (reg:DI 139) (const_int -32 [0xffffffffffffffe0])) "/app/example.cpp":12:15 discrim 1 -1)\n' +
                '  goto <bb 3>; [94.50%]\n';
            const trimmed = compiler.trimGccDumpHeaderFunctions(rtlBlock, 'example.cpp', false, true);
            expect(trimmed).toContain('[orig:117]'); // not a path -> kept
            expect(trimmed).toContain('[0xffffffffffffffe0]'); // hex operand -> kept
            expect(trimmed).toContain('[94.50%]'); // branch probability -> kept
            expect(trimmed).toContain(':12:15 discrim 1'); // line:col of the insn location -> kept
            expect(trimmed).not.toContain('"/app/example.cpp"'); // redundant source filename -> dropped
        });

        it('strips forced -lineno [path:line] prefixes from RTL dumps when lineno is disabled (#8826)', () => {
            // -fdump-rtl-expand-details prints the gimple statements it expands; -lineno prefixes
            // each with [path:line:col], which pinskia reported made the dump hard to read.
            const rtlBlock =
                ';; Function main (main)\n' +
                '  [/app/example.cpp:12:15 discrim 1] __builtin_memcpy (&v1, data_22(D), 32);\n' +
                '  _32 = BIT_FIELD_REF <[/app/example.cpp:16:41] [/app/example.cpp:16:37] v.val[3], 16, 0>;\n' +
                ';; [/app/example.cpp:12:15 discrim 1] __builtin_memcpy (&v1, data_22(D), 32);\n';
            const trimmed = compiler.trimGccDumpHeaderFunctions(rtlBlock, 'example.cpp', false, true);
            expect(trimmed).not.toContain('[/app/example.cpp:12:15 discrim 1]');
            expect(trimmed).not.toContain('[/app/example.cpp:16:41]');
            expect(trimmed).toContain('  __builtin_memcpy (&v1, data_22(D), 32);');
            expect(trimmed).toContain('  _32 = BIT_FIELD_REF <v.val[3], 16, 0>;');
            expect(trimmed).toContain(';; __builtin_memcpy (&v1, data_22(D), 32);');
        });

        it('keeps forced -lineno prefixes in RTL dumps when lineno is enabled', () => {
            const rtlBlock =
                ';; Function main (main)\n  [/app/example.cpp:12:15 discrim 1] __builtin_memcpy (&v1, x);\n';
            const trimmed = compiler.trimGccDumpHeaderFunctions(rtlBlock, 'example.cpp', true, true);
            expect(trimmed).toContain('[/app/example.cpp:12:15 discrim 1]');
        });

        it('strips location annotations from tree dumps when lineno is disabled', () => {
            const trimmed = compiler.trimGccDumpHeaderFunctions(userBlock, 'example.cpp', false, false);
            expect(trimmed).toContain('x = 1;');
            expect(trimmed).not.toContain('[example.cpp:3:5]');
        });

        it('strips location annotations from IPA GIMPLE-body dumps (icf, inline, ...) when lineno is off', () => {
            // IPA passes that print GIMPLE bodies carry the same [file:line] prefixes as tree
            // dumps; they must be stripped too (isRtlDump=false), which the old tree-only gate missed.
            const ipaBody =
                ';; Function main (main)\n[example.cpp:10:8] # DEBUG BEGIN_STMT\n[example.cpp:11:23] x = 1;\n';
            const trimmed = compiler.trimGccDumpHeaderFunctions(ipaBody, 'example.cpp', false, false);
            expect(trimmed).toContain('x = 1;');
            expect(trimmed).not.toContain('[example.cpp:10:8]');
        });
    });

    describe('processGccDumpOutput (enumeration path)', () => {
        const rootDir = '/tmp/ce-test';
        const inputFilename = path.join(rootDir, 'example.cpp');

        const dumpFiles: Record<string, string> = {
            'example.cpp.005t.original': ';; Function main (main, funcdef_no=1)\n[example.cpp:1:1] original body\n',
            'example.cpp.081i.cp': ';; Function main (main, funcdef_no=1)\n[example.cpp:1:1] cp body\n',
            'example.cpp.099i.ipa-clones': '', // empty dump (pass ran but emitted nothing) -> excluded
            'example.cpp.255r.expand':
                ';; Function main (main, funcdef_no=1)\n[example.cpp:1:1] expand body\n' +
                ';; Function std::lib (_ZSt3lib)\n[/usr/include/lib.h:2:2] header body\n',
        };

        function mockFs() {
            const names = [
                ...Object.keys(dumpFiles),
                'example.cpp.s', // compiler output, must be excluded
                'example.cpp', // source, must be excluded
                'example.cpp.o',
            ];
            vi.spyOn(fs, 'readdir').mockResolvedValue(names as unknown as any);
            vi.spyOn(utils, 'tryReadTextFile').mockImplementation(
                async (filename: string) => dumpFiles[path.basename(filename)],
            );
        }

        const baseOpts = (): GccDumpOptions => ({
            opened: true,
            treeDump: true,
            rtlDump: true,
            ipaDump: true,
            dumpFlags: {
                gimpleFe: false,
                address: false,
                alias: false,
                slim: false,
                raw: false,
                details: false,
                stats: false,
                blocks: false,
                vops: false,
                lineno: true,
                uid: false,
                all: false,
            },
        });

        it('enumerates produced dumps, orders by pass number, excludes non-dump files', async () => {
            mockFs();
            const result: any = {inputFilename, stderr: []};
            const output = await compiler.processGccDumpOutput(baseOpts(), result, true, 'example.s');

            expect(output.all.map((p: any) => p.filename_suffix)).toEqual(['t.original', 'i.cp', 'r.expand']);
            expect(Object.keys(output.passDumps!).sort()).toEqual(['i.cp', 'r.expand', 't.original']);
            // No .s / source / .o leaked into the dropdown.
            expect(output.all.map((p: any) => p.name)).not.toContain('s (rtl)');
            // The empty ipa-clones dump is excluded from both the drop-down and the cache.
            expect(output.all.map((p: any) => p.filename_suffix)).not.toContain('i.ipa-clones');
            expect(output.passDumps).not.toHaveProperty('i.ipa-clones');
        });

        it('trims header functions out of the cached dump contents', async () => {
            mockFs();
            const result: any = {inputFilename, stderr: []};
            const output = await compiler.processGccDumpOutput(baseOpts(), result, true, 'example.s');

            expect(output.passDumps!['r.expand']).toContain(';; Function main');
            expect(output.passDumps!['r.expand']).not.toContain(';; Function std::lib');
        });

        it('strips lineno annotations from tree dumps when Line Numbers is off', async () => {
            mockFs();
            const opts = baseOpts();
            opts.dumpFlags!.lineno = false; // user did NOT enable Line Numbers
            const result: any = {inputFilename, stderr: []};
            const output = await compiler.processGccDumpOutput(opts, result, true, 'example.s');

            expect(output.passDumps!['t.original']).toContain('original body');
            expect(output.passDumps!['t.original']).not.toContain('[example.cpp:1:1]');
        });

        it('keeps lineno annotations when Line Numbers is on', async () => {
            mockFs();
            const opts = baseOpts(); // baseOpts has lineno: true
            const result: any = {inputFilename, stderr: []};
            const output = await compiler.processGccDumpOutput(opts, result, true, 'example.s');

            expect(output.passDumps!['t.original']).toContain('[example.cpp:1:1]');
        });

        it('serves the selected pass content and enables syntax highlight', async () => {
            mockFs();
            const opts = baseOpts();
            opts.pass = {
                filename_suffix: 'r.expand',
                name: 'expand (rtl)',
                command_prefix: '-fdump-rtl-expand',
                selectedPass: null,
            };
            const result: any = {inputFilename, stderr: []};
            const output = await compiler.processGccDumpOutput(opts, result, true, 'example.s');

            expect(output.selectedPass).toEqual(opts.pass);
            expect(output.currentPassOutput).toContain('expand body');
            expect(output.syntaxHighlight).toBe(true);
        });

        it('clears the selection when the requested pass produced no dump', async () => {
            mockFs();
            const opts = baseOpts();
            opts.pass = {
                filename_suffix: 'r.doesnotexist',
                name: 'doesnotexist (rtl)',
                command_prefix: '-fdump-rtl-doesnotexist',
                selectedPass: null,
            };
            const result: any = {inputFilename, stderr: []};
            const output = await compiler.processGccDumpOutput(opts, result, true, 'example.s');

            expect(output.selectedPass).toBeNull();
        });

        it('respects the category filters (rtl only)', async () => {
            mockFs();
            const opts = baseOpts();
            opts.treeDump = false;
            opts.ipaDump = false;
            const result: any = {inputFilename, stderr: []};
            const output = await compiler.processGccDumpOutput(opts, result, true, 'example.s');

            expect(output.all.map((p: any) => p.filename_suffix)).toEqual(['r.expand']);
        });
    });
});
