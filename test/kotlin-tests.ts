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

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {KotlinCompiler} from '../lib/compilers/index.js';
import {CompilationResult} from '../types/compilation/compilation.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo, newTempDir} from './utils.js';

const languages = {
    kotlin: {id: 'kotlin'},
};

function makeCompiler(env: CompilationEnvironment, semver?: string) {
    return new KotlinCompiler(makeFakeCompilerInfo({exe: 'kotlinc', remote: undefined, lang: 'kotlin', semver}), env);
}

function optionValue(options: string[], prefix: string) {
    return options.find(option => option.startsWith(prefix))?.substring(prefix.length);
}

describe('Kotlin IR in the AST view', () => {
    let env: CompilationEnvironment;

    beforeAll(() => {
        env = makeCompilationEnvironment({languages});
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('is offered', () => {
        expect(makeCompiler(env, '2.2.20').couldSupportASTDump()).toBe(true);
    });

    describe('Generating the IR', () => {
        it('dumps the IR from a compile of its own inside the working directory', async () => {
            const compiler = makeCompiler(env, '2.2.20');
            const sourceDir = newTempDir();
            const inputFilename = path.join(sourceDir, 'example.kt');
            const runCompiler = vi
                .spyOn(compiler, 'runCompiler')
                .mockImplementation(async (exe: string, options: string[]) => {
                    const moduleDir = path.join(optionValue(options, '-Xdump-directory=')!, 'main');
                    await fs.mkdir(moduleDir, {recursive: true});
                    await fs.writeFile(
                        path.join(moduleDir, '00_BEFORE.ExternalPackageParentPatcherLowering.ir'),
                        'MODULE_FRAGMENT name:<main>\n',
                    );
                    return {code: 0} as CompilationResult;
                });

            const result = await compiler.generateAST(inputFilename, ['-Xopt-in=kotlin.RequiresOptIn', inputFilename]);

            expect(result).toEqual([{text: 'MODULE_FRAGMENT name:<main>'}]);
            const [exe, options] = runCompiler.mock.calls[0];
            expect(exe).toEqual('kotlinc');
            expect(options).toContain('-Xphases-to-dump-before=ExternalPackageParentPatcherLowering');
            expect(options.slice(-2)).toEqual(['-Xopt-in=kotlin.RequiresOptIn', inputFilename]);
            expect(optionValue(options, '-Xdump-directory=')).toEqual(path.join(sourceDir, 'kotlin-ir'));
            expect(options[options.indexOf('-d') + 1]).toEqual(path.join(sourceDir, 'kotlin-ir.jar'));
        });

        async function generatedOptions(userOptions: string[]) {
            const compiler = makeCompiler(env, '2.2.20');
            const runCompiler = vi.spyOn(compiler, 'runCompiler').mockResolvedValue({code: 0} as CompilationResult);
            const inputFilename = path.join(newTempDir(), 'example.kt');
            await compiler.generateAST(inputFilename, [...userOptions, inputFilename]);
            return runCompiler.mock.calls[0][1];
        }

        it.each(['-Xphases-to-dump=Inline', '-Xphases-to-dump-before=Inline', '-Xphases-to-dump-after=Inline'])(
            'dumps the phases the user asked for with %s',
            async userPhases => {
                const options = await generatedOptions([userPhases]);
                expect(options.filter(option => option.startsWith('-Xphases-to-dump'))).toEqual([userPhases]);
            },
        );

        it("keeps its own output locations over the user's", async () => {
            const options = await generatedOptions(['-Xdump-directory=/elsewhere', '-d', '.', '-Xdump-fqname=main']);
            expect(options.filter(option => option.startsWith('-Xdump-directory'))).toHaveLength(1);
            expect(options.some(option => option.includes('/elsewhere'))).toBe(false);
            expect(options.filter(option => option === '-d')).toHaveLength(1);
            expect(options).not.toContain('.');
            expect(options).toContain('-Xdump-fqname=main');
        });

        it('reports a failed compilation', async () => {
            const compiler = makeCompiler(env, '2.2.20');
            vi.spyOn(compiler, 'runCompiler').mockResolvedValue({code: 1} as CompilationResult);
            const inputFilename = path.join(newTempDir(), 'example.kt');
            expect(await compiler.generateAST(inputFilename, [inputFilename])).toEqual([
                {text: 'Failed to run compiler to get Kotlin IR'},
            ]);
        });
    });

    describe('Reading the dumps', () => {
        it('concatenates every dump file', async () => {
            const compiler = makeCompiler(env, '2.2.20');
            const dumpDir = path.join(newTempDir(), 'kotlin-ir');
            const moduleDir = path.join(dumpDir, 'main');
            await fs.mkdir(moduleDir, {recursive: true});
            await fs.writeFile(path.join(moduleDir, '10_BEFORE.Second.ir'), 'second\n');
            await fs.writeFile(path.join(moduleDir, '02_BEFORE.First.ir'), 'first\nline\n');

            const result = await compiler.readIrDumps(dumpDir);
            expect(result.map(line => line.text)).toEqual(['first', 'line', '', 'second']);
        });

        it('orders the dumps by phase, before ahead of after', async () => {
            const compiler = makeCompiler(env, '2.2.20');
            const dumpDir = path.join(newTempDir(), 'kotlin-ir');
            // As kotlinc lays them out: module-level phases under the module, per-file phases at the top level
            const dumps = {
                'main/00_BEFORE.ExternalPackageParentPatcherLowering.ir': 'module 0 before',
                'main/00_AFTER.ExternalPackageParentPatcherLowering.ir': 'module 0 after',
                '17_AFTER.TypeAliasAnnotationMethodsLowering.ir': 'file 17 after',
                '17_BEFORE.TypeAliasAnnotationMethodsLowering.ir': 'file 17 before',
                '100_AFTER.TypeSwitchTransformation.ir': 'file 100 after',
                '100_BEFORE.TypeSwitchTransformation.ir': 'file 100 before',
                'main/103_BEFORE.JvmValidateIrAfterLowering.ir': 'module 103 before',
            };
            for (const [name, content] of Object.entries(dumps)) {
                await fs.mkdir(path.dirname(path.join(dumpDir, name)), {recursive: true});
                await fs.writeFile(path.join(dumpDir, name), content);
            }

            const result = await compiler.readIrDumps(dumpDir);
            expect(result.map(line => line.text).filter(text => text)).toEqual([
                'module 0 before',
                'module 0 after',
                'file 17 before',
                'file 17 after',
                'file 100 before',
                'file 100 after',
                'module 103 before',
            ]);
        });

        it('reports when nothing was dumped', async () => {
            const compiler = makeCompiler(env, '2.2.20');
            const result = await compiler.readIrDumps(path.join(newTempDir(), 'kotlin-ir'));
            expect(result).toEqual([{text: '<No Kotlin IR generated>'}]);
        });
    });
});
