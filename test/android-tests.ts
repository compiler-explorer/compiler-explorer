// Copyright (c) 2024, Compiler Explorer Authors
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
import {Dex2OatCompiler} from '../lib/compilers/index.js';
import * as utils from '../lib/utils.js';
import {ParsedAsmResultLine} from '../types/asmresult/asmresult.interfaces.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';

import {fs, makeCompilationEnvironment} from './utils.js';

const languages = {
    androidJava: {id: 'android-java'},
    androidKotlin: {id: 'android-kotlin'},
};

const androidJavaInfo = {
    exe: null,
    remote: true,
    lang: languages.androidJava.id,
} as unknown as CompilerInfo;

const androidKotlinInfo = {
    exe: null,
    remote: true,
    lang: languages.androidKotlin.id,
} as unknown as CompilerInfo;

describe('dex2oat', () => {
    let env: CompilationEnvironment;

    beforeAll(() => {
        env = makeCompilationEnvironment({languages});
    });

    describe('android-java', () => {
        it('Should not crash on instantiation', () => {
            new Dex2OatCompiler(androidJavaInfo, env);
        });

        it('Output is shown as-is if full output mode is enabled', () => {
            return testParse(androidJavaInfo, 'test/android/java', true);
        });

        if (process.platform !== 'win32') {
            it('Output is parsed and formatted if full output mode is disabled', () => {
                return testParse(androidJavaInfo, 'test/android/java', false);
            });
        }
    });

    describe('android-kotlin', () => {
        it('Should not crash on instantiation', () => {
            new Dex2OatCompiler(androidKotlinInfo, env);
        });

        it('Output is shown as-is if full output mode is enabled', () => {
            return testParse(androidKotlinInfo, 'test/android/kotlin', true);
        });

        if (process.platform !== 'win32') {
            it('Output is parsed and formatted if full output mode is disabled', () => {
                return testParse(androidKotlinInfo, 'test/android/kotlin', false);
            });
        }
    });

    async function testParse(info: CompilerInfo, baseFolder: string, fullOutput: boolean) {
        const compiler = new Dex2OatCompiler(info, env);
        compiler.fullOutput = fullOutput;

        // The "result" of running oatdump.
        const asm = [{text: fs.readFileSync(`${baseFolder}/oatdump.asm`).toString()}];
        const objdumpResult = {
            asm,
        };
        const processed = await compiler.processAsm(objdumpResult);
        expect(processed).toHaveProperty('asm');
        const actualSegments = (processed as {asm: ParsedAsmResultLine[]}).asm;

        // fullOutput results in no processing, with the entire oatdump text
        // being returned as one long string.
        const output = fullOutput
            ? [fs.readFileSync(`${baseFolder}/oatdump.asm`).toString()]
            : utils.splitLines(fs.readFileSync(`${baseFolder}/output.asm`).toString());
        const expectedSegments = output.map(line => {
            return {
                text: line,
                source: null,
            };
        });

        expect(actualSegments).toEqual(expectedSegments);
    }
});
