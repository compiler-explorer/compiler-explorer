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

import {afterEach, beforeAll, describe, expect, it} from 'vitest';

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {C3Compiler} from '../lib/compilers/index.js';
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
            // Legal in C3: the section is appended to the standard library module's own output file,
            // so it must not be filtered out as "not the user's code".
            expect(await baseNamesFor('module std::io;\nfn void mine() {}\n')).toEqual(['std.io']);
        });

        it('ignores attributes after the module name', async () => {
            expect(await baseNamesFor('module test @export;\nfn int sq(int x) { return x * x; }\n')).toEqual(['test']);
            expect(await baseNamesFor('module test @if(env::LINUX);\n')).toEqual(['test']);
        });

        it('does not treat an indented or embedded "module" as a declaration', async () => {
            expect(await baseNamesFor('module real;\nfn void f() { int modules = 1; }\n')).toEqual(['real']);
        });
    });

    describe('joining per-module output', () => {
        it('concatenates the modules that were emitted', async () => {
            const dir = await tempDir();
            await fs.writeFile(path.join(dir, 'alpha.s'), 'alpha asm');
            await fs.writeFile(path.join(dir, 'beta.s'), 'beta asm');
            const compiler = new C3Compiler(makeInfo('0.8.2'), env);
            const destination = path.join(dir, 'output.s');

            await compiler.joinModuleOutputs(dir, ['alpha', 'beta'], '.s', destination);

            expect(await fs.readFile(destination, 'utf8')).toEqual('alpha asm\nbeta asm');
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

    describe('strip-unused default', () => {
        const filters = {} as ParseFiltersAndOutputOptions;

        it('asks for unreferenced code on versions that support it', () => {
            const options = new C3Compiler(makeInfo('0.8.2'), env).optionsForFilter(filters, 'output.s');
            expect(options).toContain('--strip-unused=no');
        });

        it('is not passed to versions predating the option', () => {
            // --strip-unused arrived in 0.5.0.
            const options = new C3Compiler(makeInfo('0.4.0'), env).optionsForFilter(filters, 'output.s');
            expect(options).not.toContain('--strip-unused=no');
        });

        it('comes before user options so the user can override it', () => {
            // c3c honours the last occurrence, and BaseCompiler appends user options after these.
            const options = new C3Compiler(makeInfo('0.8.2'), env).optionsForFilter(filters, 'output.s');
            expect(options.indexOf('--strip-unused=no')).toBeGreaterThanOrEqual(0);
            expect(options.at(-1)).toEqual('--strip-unused=no');
        });
    });
});
