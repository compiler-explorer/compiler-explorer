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
import os from 'node:os';
import path from 'node:path';

import {afterEach, beforeAll, describe, expect, it, vi} from 'vitest';

import {BaseCompiler} from '../lib/base-compiler.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import {C3Compiler} from '../lib/compilers/index.js';
import {CompilationResult} from '../types/compilation/compilation.interfaces.js';
import {LLVMIrBackendOptions} from '../types/compilation/ir.interfaces.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';
import {makeCompilationEnvironment} from './utils.js';

const languages = {
    c3: {id: 'c3'},
};

function makeInfo(semver: string): CompilerInfo {
    return {
        exe: '/opt/compiler-explorer/c3-test/c3c',
        remote: true,
        lang: languages.c3.id,
        semver,
    } as unknown as CompilerInfo;
}

describe('C3 compiler', () => {
    let env: CompilationEnvironment;
    const dirs: string[] = [];

    beforeAll(() => {
        env = makeCompilationEnvironment({languages});
    });

    afterEach(async () => {
        vi.restoreAllMocks();
        while (dirs.length > 0) await fs.rm(dirs.pop()!, {recursive: true, force: true});
    });

    async function tempDir(): Promise<string> {
        const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-c3c-test-'));
        dirs.push(dir);
        return dir;
    }

    it('should not crash on instantiation', () => {
        new C3Compiler(makeInfo('0.8.2'), env);
    });

    describe('module name to output file mapping', () => {
        async function baseNamesFor(source: string, filename = 'example.c3'): Promise<string[]> {
            const dir = await tempDir();
            const inputFilename = path.join(dir, filename);
            await fs.writeFile(inputFilename, source);
            return await new C3Compiler(makeInfo('0.8.2'), env).getModuleBaseNames(inputFilename);
        }

        it('uses the module name', async () => {
            expect(await baseNamesFor('module test;\nfn int sq(int x) { return x * x; }\n')).toEqual(['test']);
        });

        it('replaces the path separator in a nested module', async () => {
            expect(await baseNamesFor('module foo::bar;\n')).toEqual(['foo.bar']);
        });

        it('tolerates spaces around the path separator', async () => {
            expect(await baseNamesFor('module foo :: bar;\n')).toEqual(['foo.bar']);
        });

        it('returns every module in declaration order', async () => {
            expect(await baseNamesFor('module alpha;\nfn void a() {}\nmodule beta;\nfn void b() {}\n')).toEqual([
                'alpha',
                'beta',
            ]);
        });

        it('joins a repeated module in only once', async () => {
            // A module may be declared in several sections; c3c appends them all to one file.
            expect(await baseNamesFor('module alpha;\nfn void a() {}\nmodule alpha;\nfn void b() {}\n')).toEqual([
                'alpha',
            ]);
        });

        it('falls back to the file name when no module is declared', async () => {
            expect(await baseNamesFor('fn int sq(int x) { return x * x; }\n')).toEqual(['example']);
        });

        it('keeps a user module that shadows a standard library name', async () => {
            // Legal in C3. The name is still the user's; whether c3c writes a file for it is up to c3c.
            expect(await baseNamesFor('module std::io;\nfn void mine() {}\n')).toEqual(['std.io']);
        });

        it('ignores attributes after the module name', async () => {
            expect(await baseNamesFor('module test @export;\nfn int sq(int x) { return x * x; }\n')).toEqual(['test']);
            expect(await baseNamesFor('module test @if(env::LINUX);\n')).toEqual(['test']);
        });

        it('handles a generic module', async () => {
            expect(await baseNamesFor('module gen<Type>;\nfn void f() {}\n')).toEqual(['gen']);
        });

        it('accepts an indented declaration but not one embedded mid-line', async () => {
            expect(await baseNamesFor('    module indented;\n')).toEqual(['indented']);
            expect(await baseNamesFor('module real;\nfn void f() { int modules = 1; }\n')).toEqual(['real']);
            expect(await baseNamesFor('module real;\nimport std::io;\n')).toEqual(['real']);
        });

        it('sanitises the derived name the way c3c does', async () => {
            // c3c turns my-file.c3 into module my_file, so the file it writes is my_file.s.
            expect(await baseNamesFor('fn void f() {}\n', 'my-file.c3')).toEqual(['my_file']);
        });

        it('falls back to the derived name when the source cannot be read', async () => {
            const dir = await tempDir();
            const compiler = new C3Compiler(makeInfo('0.8.2'), env);
            expect(await compiler.getModuleBaseNames(path.join(dir, 'missing.c3'))).toEqual(['missing']);
        });

        describe('ignores declarations that are not code', () => {
            it('in a line comment', async () => {
                expect(await baseNamesFor('module mine;\n// module std::io;\n')).toEqual(['mine']);
            });

            it('in a block comment', async () => {
                expect(await baseNamesFor('/*\nmodule std::io;\n*/\nmodule mine;\n')).toEqual(['mine']);
            });

            it('in a doc comment', async () => {
                expect(await baseNamesFor('<*\nmodule std::io;\n*>\nmodule mine;\n')).toEqual(['mine']);
            });

            it('in a raw string', async () => {
                expect(await baseNamesFor('module mine;\nString s = `\nmodule std::io;\n`;\n')).toEqual(['mine']);
            });

            it('but still finds a declaration on the line after a comment', async () => {
                // Blanking must preserve line structure, or the anchors shift.
                expect(await baseNamesFor('/* c */\nmodule mine;\n')).toEqual(['mine']);
            });

            it('in a nested block comment', async () => {
                // C3 block comments nest: the first */ closes only the inner one.
                expect(await baseNamesFor('/* outer /* inner */\nmodule std::io;\n*/\nmodule mine;\n')).toEqual([
                    'mine',
                ]);
                expect(await baseNamesFor('/* a /* b /* c */ */\nmodule std::io;\n*/\nmodule mine;\n')).toEqual([
                    'mine',
                ]);
            });

            it('without mistaking a comment opener in a string or character literal', async () => {
                expect(await baseNamesFor('module mine;\nString s = "/*";\nmodule second;\n')).toEqual([
                    'mine',
                    'second',
                ]);
                expect(await baseNamesFor("module mine;\nchar c = '\"';\nmodule second;\n")).toEqual([
                    'mine',
                    'second',
                ]);
            });

            it('honouring escapes in literals', async () => {
                expect(await baseNamesFor('module mine;\nString s = "a\\"/*";\nmodule second;\n')).toEqual([
                    'mine',
                    'second',
                ]);
                expect(await baseNamesFor('module mine;\nString s = `a``/*`;\nmodule second;\n')).toEqual([
                    'mine',
                    'second',
                ]);
            });

            it('treating comment openers inside a line comment as plain text', async () => {
                expect(await baseNamesFor('// a /* b\nmodule mine;\n')).toEqual(['mine']);
            });

            it('to the end of the file for an unterminated comment', async () => {
                expect(await baseNamesFor('module mine;\n/* /* */\nmodule std::io;\n')).toEqual(['mine']);
            });
        });

        describe('blankNonCode', () => {
            it('blanks every non-code span, keeping code, newlines and length', () => {
                const source = 'a /* x /* y */ z */ b\n// c\nd "e" `f` \'g\' <* h *> i\n';
                const blanked = C3Compiler.blankNonCode(source);
                expect(blanked).toHaveLength(source.length);
                expect(blanked.split('\n')).toEqual([
                    'a' + ' '.repeat(19) + 'b',
                    ' '.repeat(4),
                    'd' + ' '.repeat(21) + 'i',
                    '',
                ]);
            });

            it('does not nest doc comments, matching c3c', () => {
                // c3c ends a <* *> doc comment at the first *>, so the trailing code is code.
                expect(C3Compiler.blankNonCode('<* a <* b *> c *>').trim()).toEqual('c *>');
            });
        });
    });

    describe('joining per-module output', () => {
        it('passes a single module through untouched', async () => {
            const dir = await tempDir();
            await fs.writeFile(path.join(dir, 'alpha.s'), 'alpha asm');
            const compiler = new C3Compiler(makeInfo('0.8.2'), env);
            const destination = path.join(dir, 'output.s');

            await compiler.joinModuleOutputs(dir, ['alpha'], '.s', destination);

            expect(await fs.readFile(destination, 'utf8')).toEqual('alpha asm');
        });

        it('labels the chunks when several modules were emitted', async () => {
            const dir = await tempDir();
            await fs.writeFile(path.join(dir, 'alpha.s'), 'alpha asm');
            await fs.writeFile(path.join(dir, 'beta.s'), 'beta asm');
            const compiler = new C3Compiler(makeInfo('0.8.2'), env);
            const destination = path.join(dir, 'output.s');

            await compiler.joinModuleOutputs(dir, ['alpha', 'beta'], '.s', destination);

            expect(await fs.readFile(destination, 'utf8')).toEqual(
                '# module alpha\nalpha asm\n# module beta\nbeta asm',
            );
        });

        it('comments the labels correctly for IR', async () => {
            const dir = await tempDir();
            await fs.writeFile(path.join(dir, 'alpha.ll'), 'alpha ir');
            await fs.writeFile(path.join(dir, 'beta.ll'), 'beta ir');
            const compiler = new C3Compiler(makeInfo('0.8.2'), env);
            const destination = path.join(dir, 'output.ll');

            await compiler.joinModuleOutputs(dir, ['alpha', 'beta'], '.ll', destination);

            expect(await fs.readFile(destination, 'utf8')).toEqual('; module alpha\nalpha ir\n; module beta\nbeta ir');
        });

        it('skips modules that produced no file', async () => {
            const dir = await tempDir();
            await fs.writeFile(path.join(dir, 'alpha.s'), 'alpha asm');
            const compiler = new C3Compiler(makeInfo('0.8.2'), env);
            const destination = path.join(dir, 'output.s');

            await compiler.joinModuleOutputs(dir, ['alpha', 'missing'], '.s', destination);

            expect(await fs.readFile(destination, 'utf8')).toEqual('alpha asm');
        });

        it('writes nothing at all when no module produced a file', async () => {
            const dir = await tempDir();
            const compiler = new C3Compiler(makeInfo('0.8.2'), env);
            const destination = path.join(dir, 'output.s');

            await compiler.joinModuleOutputs(dir, ['nothing'], '.s', destination);

            // Absent rather than empty, so the usual "no output file" reporting still applies.
            await expect(fs.access(destination)).rejects.toThrow();
        });

        it("only joins the user's modules, never c3c's compiler_rt or standard library files", async () => {
            const dir = await tempDir();
            const inputFilename = path.join(dir, 'example.c3');
            await fs.writeFile(inputFilename, 'module test;\nfn void f() {}\n');
            await fs.writeFile(path.join(dir, 'test.s'), 'user asm');
            await fs.writeFile(path.join(dir, 'compiler_rt.s'), 'compiler_rt asm');
            await fs.writeFile(path.join(dir, 'std.io.s'), 'stdlib asm');
            const compiler = new C3Compiler(makeInfo('0.8.2'), env);
            const destination = path.join(dir, 'output.s');

            await compiler.joinModulesFor(inputFilename, '.s', destination);

            expect(await fs.readFile(destination, 'utf8')).toEqual('user asm');
        });

        it('does not copy the destination onto itself for a module named output', async () => {
            const dir = await tempDir();
            const destination = path.join(dir, 'output.s');
            await fs.writeFile(destination, 'stale');
            await fs.writeFile(path.join(dir, 'other.s'), 'real asm');
            const compiler = new C3Compiler(makeInfo('0.8.2'), env);

            await compiler.joinModuleOutputs(dir, ['output', 'other'], '.s', destination);

            expect(await fs.readFile(destination, 'utf8')).toEqual('real asm');
        });
    });

    describe('postProcess wiring', () => {
        async function runPostProcess(setup: (dir: string) => Promise<void>, code = 0) {
            const dir = await tempDir();
            const inputFilename = path.join(dir, 'example.c3');
            await fs.writeFile(inputFilename, 'module test;\nfn void f() {}\n');
            await setup(dir);
            const outputFilename = path.join(dir, 'output.s');
            const result = {code, inputFilename, asm: '', stdout: [], stderr: []} as unknown as CompilationResult;
            await new C3Compiler(makeInfo('0.8.2'), env).postProcess(
                result,
                outputFilename,
                {} as ParseFiltersAndOutputOptions,
            );
            return {result, outputFilename};
        }

        it('joins the module output and refreshes the recorded size', async () => {
            // The size is stat'd by the caller before postProcess runs, i.e. before the file exists.
            // Unrefreshed, BaseCompiler reports "<No output file>" even though we just wrote one.
            const {result, outputFilename} = await runPostProcess(async dir =>
                fs.writeFile(path.join(dir, 'test.s'), 'the asm'),
            );
            expect(await fs.readFile(outputFilename, 'utf8')).toEqual('the asm');
            expect(result.asmSize).toEqual('the asm'.length);
        });

        it('leaves the size unset when the compiler produced nothing', async () => {
            const {result} = await runPostProcess(async () => {});
            expect(result.asmSize).toBeUndefined();
        });

        it('does nothing when the compilation failed', async () => {
            const {result, outputFilename} = await runPostProcess(
                async dir => fs.writeFile(path.join(dir, 'test.s'), 'stale asm'),
                1,
            );
            await expect(fs.access(outputFilename)).rejects.toThrow();
            expect(result.asmSize).toBeUndefined();
        });
    });

    describe('options', () => {
        const filters = {} as ParseFiltersAndOutputOptions;
        const optionsFor = (semver: string) => new C3Compiler(makeInfo(semver), env).optionsForFilter();

        it('keeps unused user code but skips writing out the standard library', () => {
            const options = optionsFor('0.8.2');
            expect(options).toContain('--strip-unused=no');
            expect(options).toContain('--emit-stdlib=no');
        });

        it('uses the option spelling from 0.5.0 on', () => {
            expect(optionsFor('0.5.0')).toEqual(expect.arrayContaining(['--strip-unused=no', '--emit-stdlib=no']));
        });

        it('uses the --no-* spelling on 0.4, which predates the =yes|no form', () => {
            const options = optionsFor('0.4.0');
            expect(options).toEqual(expect.arrayContaining(['--no-strip-unused', '--no-emit-stdlib']));
            expect(options).not.toContain('--strip-unused=no');
            expect(options).not.toContain('--emit-stdlib=no');
        });

        it('only redirects output from 0.6.8, which added --asm-out and --llvm-out', () => {
            expect(optionsFor('0.6.8')).toEqual(expect.arrayContaining(['--asm-out', '--llvm-out']));
            expect(optionsFor('0.6.7')).not.toContain('--asm-out');
            expect(optionsFor('0.6.7')).not.toContain('--llvm-out');
        });

        it('puts user options after ours, so the user can turn stripping back on', async () => {
            // c3c honours the last occurrence of an option.
            const dir = await tempDir();
            const args = new C3Compiler(makeInfo('0.8.2'), env).prepareArguments(
                ['--strip-unused=yes', '--emit-stdlib=yes'],
                filters,
                {},
                path.join(dir, 'example.c3'),
                path.join(dir, 'output.s'),
                [],
                [],
            );
            expect(args.indexOf('--strip-unused=no')).toBeGreaterThanOrEqual(0);
            expect(args.indexOf('--strip-unused=no')).toBeLessThan(args.indexOf('--strip-unused=yes'));
            expect(args.indexOf('--emit-stdlib=no')).toBeLessThan(args.indexOf('--emit-stdlib=yes'));
        });
    });

    describe('IR', () => {
        const filters = {} as ParseFiltersAndOutputOptions;
        const irOptions = {} as LLVMIrBackendOptions;

        it('does not also emit the assembly in the IR run', async () => {
            const spy = vi
                .spyOn(BaseCompiler.prototype, 'generateIR')
                .mockResolvedValue({code: 0, asm: [], compilationOptions: []});
            const compiler = new C3Compiler(makeInfo('0.8.2'), env);
            const options = [...compiler.optionsForFilter(), '-O1', 'example.c3'];

            await compiler.generateIR('example.c3', options, irOptions, false, filters);

            const irRunOptions = spy.mock.calls[0][1];
            expect(irRunOptions).not.toContain('--emit-asm');
            expect(irRunOptions).not.toContain('--asm-out');
            expect(irRunOptions).toEqual([
                'compile-only',
                '-g',
                '-l',
                'pthread',
                '--no-obj',
                '--llvm-out',
                '.',
                '--strip-unused=no',
                '--emit-stdlib=no',
                '-O1',
                'example.c3',
            ]);
        });

        it('joins the per-module IR into the file the IR pane reads', async () => {
            const dir = await tempDir();
            const inputFilename = path.join(dir, 'example.c3');
            await fs.writeFile(inputFilename, 'module test;\nfn int sq(int x) { return x * x; }\n');
            await fs.writeFile(path.join(dir, 'test.ll'), 'define i32 @test.sq(i32 %0) {\n  ret i32 %0\n}\n');
            await fs.writeFile(path.join(dir, 'compiler_rt.ll'), 'define void @compiler_rt_thing() {\n}\n');
            const compiler = new C3Compiler(makeInfo('0.8.2'), env);

            await compiler.processIrOutput(
                {code: 0, inputFilename} as unknown as CompilationResult,
                irOptions,
                filters,
            );

            const joined = await fs.readFile(compiler.getIrOutputFilename(inputFilename), 'utf8');
            expect(joined).toContain('@test.sq');
            expect(joined).not.toContain('compiler_rt');
        });
    });
});
