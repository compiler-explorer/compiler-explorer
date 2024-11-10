// Copyright (c) 2017, Compiler Explorer Authors
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

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {JavaCompiler} from '../lib/compilers/index.js';
import * as utils from '../lib/utils.js';
import {ParsedAsmResultLine} from '../types/asmresult/asmresult.interfaces.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';

import {fs, makeCompilationEnvironment} from './utils.js';

const languages = {
    java: {id: 'java'},
};

const info = {
    exe: null,
    remote: true,
    lang: languages.java.id,
} as unknown as CompilerInfo;

describe('Basic compiler setup', () => {
    let env: CompilationEnvironment;

    beforeAll(() => {
        env = makeCompilationEnvironment({languages});
    });

    it('Should not crash on instantiation', () => {
        new JavaCompiler(info, env);
    });

    it('should ignore second param for getOutputFilename', () => {
        // Because javac produces a class files based on user provided class names,
        // it's not possible to determine the main class file before compilation/parsing
        const compiler = new JavaCompiler(info, env);
        if (process.platform === 'win32') {
            expect(compiler.getOutputFilename('/tmp/')).toEqual('\\tmp\\example.class');
        } else {
            expect(compiler.getOutputFilename('/tmp/')).toEqual('/tmp/example.class');
        }
    });

    describe('Forbidden compiler arguments', () => {
        it('JavaCompiler should not allow -d parameter', () => {
            const compiler = new JavaCompiler(info, env);
            expect(compiler.filterUserOptions(['hello', '-d', '--something', '--something-else'])).toEqual([
                'hello',
                '--something-else',
            ]);
            expect(compiler.filterUserOptions(['hello', '-d'])).toEqual(['hello']);
            expect(compiler.filterUserOptions(['-d', 'something', 'something-else'])).toEqual(['something-else']);
        });

        it('JavaCompiler should not allow -s parameter', () => {
            const compiler = new JavaCompiler(info, env);
            expect(compiler.filterUserOptions(['hello', '-s', '--something', '--something-else'])).toEqual([
                'hello',
                '--something-else',
            ]);
            expect(compiler.filterUserOptions(['hello', '-s'])).toEqual(['hello']);
            expect(compiler.filterUserOptions(['-s', 'something', 'something-else'])).toEqual(['something-else']);
        });

        it('JavaCompiler should not allow --source-path parameter', () => {
            const compiler = new JavaCompiler(info, env);
            expect(compiler.filterUserOptions(['hello', '--source-path', '--something', '--something-else'])).toEqual([
                'hello',
                '--something-else',
            ]);
            expect(compiler.filterUserOptions(['hello', '--source-path'])).toEqual(['hello']);
            expect(compiler.filterUserOptions(['--source-path', 'something', 'something-else'])).toEqual([
                'something-else',
            ]);
        });

        it('JavaCompiler should not allow -sourcepath parameter', () => {
            const compiler = new JavaCompiler(info, env);
            expect(compiler.filterUserOptions(['hello', '-sourcepath', '--something', '--something-else'])).toEqual([
                'hello',
                '--something-else',
            ]);
            expect(compiler.filterUserOptions(['hello', '-sourcepath'])).toEqual(['hello']);
            expect(compiler.filterUserOptions(['-sourcepath', 'something', 'something-else'])).toEqual([
                'something-else',
            ]);
        });
    });
});

describe('javap parsing', () => {
    let compiler: JavaCompiler;
    let env: CompilationEnvironment;
    beforeAll(() => {
        env = makeCompilationEnvironment({languages});
        compiler = new JavaCompiler(info, env);
    });

    async function testJava(baseFolder: string, ...classNames: string[]) {
        const compiler = new JavaCompiler(info, env);

        const asm = classNames.map(className => {
            return {text: fs.readFileSync(`${baseFolder}/${className}.asm`).toString()};
        });

        const output = utils.splitLines(fs.readFileSync(`${baseFolder}/output.asm`).toString());
        const expectedSegments = output.map(line => {
            const match = line.match(/^line (\d+):(.*)$/);
            if (match) {
                return {
                    text: match[2],
                    source: {
                        line: parseInt(match[1]),
                        file: null,
                    },
                };
            }
            return {
                text: line,
                source: null,
            };
        });

        const result = {
            asm,
        };

        const processed = await compiler.processAsm(result);
        expect(processed).toHaveProperty('asm');
        const asmSegments = (processed as {asm: ParsedAsmResultLine[]}).asm;
        expect(asmSegments).toEqual(expectedSegments);
    }

    it('should handle errors', async () => {
        const result = {
            asm: '<Compilation failed>',
        };

        await expect(compiler.processAsm(result)).resolves.toEqual({
            asm: [{text: '<Compilation failed>', source: null}],
        });
    });

    it('Parses simple class with one method', () => {
        return Promise.all([testJava('test/java/square', 'javap-square')]);
    });

    it('Preserves ordering of multiple classes', () => {
        return Promise.all([
            testJava('test/java/two-classes', 'ZFirstClass', 'ASecondClass'),
            testJava('test/java/two-classes', 'ASecondClass', 'ZFirstClass'),
        ]);
    });

    it('Properly parses lookupswitch blocks', () => {
        return testJava('test/java/lookupswitch-bug-2995', 'Main');
    });
});
