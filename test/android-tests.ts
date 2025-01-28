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
            return testParseOatdump(androidJavaInfo, 'test/android/java', true);
        });

        if (process.platform !== 'win32') {
            it('Output is parsed and formatted if full output mode is disabled', () => {
                return testParseOatdump(androidJavaInfo, 'test/android/java', false);
            });
        }

        it('Line numbers are correctly extracted from .smali files', () => {
            return testParseSmaliForLineNumbers(androidJavaInfo, 'test/android/parse-data');
        });

        it("Classes with 'L' in the name compile correctly", () => {
            return testParseSmaliClassWithL(androidJavaInfo, 'test/android/parse-data');
        });

        it('Inner classes are correctly read in .smali files', () => {
            return testParseSmaliInnerClasses(androidJavaInfo, 'test/android/parse-data');
        });

        it('Dex PCs are correctly extracted from classes.cfg', () => {
            return testParsePassDumpsForDexPcs(androidJavaInfo, 'test/android/parse-data');
        });

        it('Internal primitive descriptors are correctly translated to full names', () => {
            return testPrettyDescriptorsPrimitive(androidJavaInfo);
        });

        it('Internal reference descriptors are correctly prettified', () => {
            return testPrettyDescriptorsReference(androidJavaInfo);
        });

        it('Internal descriptors with dimensions are correctly prettified', () => {
            return testPrettyDescriptorsDimensions(androidJavaInfo);
        });

        it('Method parameters are correctly split into component parts', () => {
            return testSplitMethodParameters(androidJavaInfo);
        });

        it('Method signature is prettified correctly', () => {
            return testPrettyMethodSignature(androidJavaInfo);
        });
    });

    describe('android-kotlin', () => {
        it('Should not crash on instantiation', () => {
            new Dex2OatCompiler(androidKotlinInfo, env);
        });

        it('Output is shown as-is if full output mode is enabled', () => {
            return testParseOatdump(androidKotlinInfo, 'test/android/kotlin', true);
        });

        if (process.platform !== 'win32') {
            it('Output is parsed and formatted if full output mode is disabled', () => {
                return testParseOatdump(androidKotlinInfo, 'test/android/kotlin', false);
            });
        }

        it('Line numbers are correctly extracted from .smali files', () => {
            return testParseSmaliForLineNumbers(androidKotlinInfo, 'test/android/parse-data');
        });

        it("Classes with 'L' in the name compile correctly", () => {
            return testParseSmaliClassWithL(androidKotlinInfo, 'test/android/parse-data');
        });

        it('Inner classes are correctly read in .smali files', () => {
            return testParseSmaliInnerClasses(androidKotlinInfo, 'test/android/parse-data');
        });

        it('Dex PCs are correctly extracted from classes.cfg', () => {
            return testParsePassDumpsForDexPcs(androidKotlinInfo, 'test/android/parse-data');
        });

        it('Internal primitive descriptors are correctly translated to full names', () => {
            return testPrettyDescriptorsPrimitive(androidKotlinInfo);
        });

        it('Internal reference descriptors are correctly prettified', () => {
            return testPrettyDescriptorsReference(androidKotlinInfo);
        });

        it('Internal descriptors with dimensions are correctly prettified', () => {
            return testPrettyDescriptorsDimensions(androidKotlinInfo);
        });

        it('Method parameters are correctly split into component parts', () => {
            return testSplitMethodParameters(androidKotlinInfo);
        });

        it('Method signature is prettified correctly', () => {
            return testPrettyMethodSignature(androidKotlinInfo);
        });
    });

    async function testParseOatdump(info: CompilerInfo, baseFolder: string, fullOutput: boolean) {
        const compiler = new Dex2OatCompiler(info, env);
        compiler.fullOutput = fullOutput;

        // The "result" of running oatdump.
        const asm = [{text: fs.readFileSync(`${baseFolder}/oatdump.asm`).toString()}];
        const objdumpResult = {
            asm,
        };
        const processed = await compiler.processAsm(objdumpResult, compiler.getDefaultFilters());
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

    async function testParseSmaliForLineNumbers(info: CompilerInfo, baseFolder: string) {
        const compiler = new Dex2OatCompiler(info, env);
        const rawSmaliText = fs.readFileSync(`${baseFolder}/Square.smali`, {encoding: 'utf8'});

        const dexPcsToLines: Record<string, Record<number, number>> = {};
        compiler.parseSmaliForLineNumbers(dexPcsToLines, rawSmaliText.split(/\n/));

        expect(Object.keys(dexPcsToLines)).toHaveLength(2);
        expect(dexPcsToLines).toHaveProperty('void Square.<init>()', {0: 12, 3: 12});
        expect(dexPcsToLines).toHaveProperty('int Square.square(int)', {0: 14, 1: 14});
    }

    async function testParseSmaliClassWithL(info: CompilerInfo, baseFolder: string) {
        const compiler = new Dex2OatCompiler(info, env);
        const rawSmaliText = fs.readFileSync(`${baseFolder}/ClassWithL.smali`, {encoding: 'utf8'});

        const dexPcsToLines: Record<string, Record<number, number>> = {};
        compiler.parseSmaliForLineNumbers(dexPcsToLines, rawSmaliText.split(/\n/));

        expect(Object.keys(dexPcsToLines)).toHaveLength(2);
        expect(dexPcsToLines).toHaveProperty('void LSLqLuLaLrLeL.<init>()', {0: 12, 3: 12});
        expect(dexPcsToLines).toHaveProperty('int LSLqLuLaLrLeL.square(int)', {0: 14, 1: 14});
    }

    async function testParseSmaliInnerClasses(info: CompilerInfo, baseFolder: string) {
        const compiler = new Dex2OatCompiler(info, env);
        const rawSmaliText = fs.readFileSync(`${baseFolder}/InnerClassCases.smali`, {encoding: 'utf8'});

        const dexPcsToLines: Record<string, Record<number, number>> = {};
        compiler.parseSmaliForLineNumbers(dexPcsToLines, rawSmaliText.split(/\n/));

        expect(Object.keys(dexPcsToLines)).toHaveLength(6);

        // Self
        expect(dexPcsToLines).toHaveProperty('void InnerClassCases.<init>()', {0: 1, 3: 1});

        // Non-static
        expect(dexPcsToLines).toHaveProperty('void InnerClassCases$InnerClass.<init>(InnerClassCases)', {
            0: 2,
            2: 2,
            5: 2,
        });
        expect(dexPcsToLines).toHaveProperty('void InnerClassCases$FinalInnerClass.<init>(InnerClassCases)', {
            0: 3,
            2: 3,
            5: 3,
        });

        // Static
        expect(dexPcsToLines).toHaveProperty('void InnerClassCases$StaticInnerClass.<init>()', {0: 4, 3: 4});
        expect(dexPcsToLines).toHaveProperty('void InnerClassCases$StaticFinalInnerClass.<init>()', {0: 5, 3: 5});
        expect(dexPcsToLines).toHaveProperty('void InnerClassCases$LStartsWithL.<init>()', {0: 6, 3: 6});
    }

    async function testParsePassDumpsForDexPcs(info: CompilerInfo, baseFolder: string) {
        const compiler = new Dex2OatCompiler(info, env);
        const rawCfgText = fs.readFileSync(`${baseFolder}/classes.cfg`, {encoding: 'utf8'});

        const methodsAndOffsetsToDexPcs = compiler.passDumpParser.parsePassDumpsForDexPcs(rawCfgText.split(/\n/));

        expect(Object.keys(methodsAndOffsetsToDexPcs)).toHaveLength(2);
        expect(methodsAndOffsetsToDexPcs).toHaveProperty('void Square.<init>()', {0: 3});
        expect(methodsAndOffsetsToDexPcs).toHaveProperty('int Square.square(int)', {0: 0, 4: 1});
    }

    async function testPrettyDescriptorsPrimitive(info: CompilerInfo) {
        const compiler = new Dex2OatCompiler(info, env);
        const primitiveMap = {
            V: 'void',
            Z: 'boolean',
            B: 'byte',
            S: 'short',
            C: 'char',
            I: 'int',
            J: 'long',
            F: 'float',
            D: 'double',
        };
        for (const input in primitiveMap) {
            expect(compiler.prettyDescriptor(input)).toEqual(primitiveMap[input]);
        }
    }

    async function testPrettyDescriptorsReference(info: CompilerInfo) {
        const compiler = new Dex2OatCompiler(info, env);
        const referenceMap = {
            'Ljava/lang/String;': 'java.lang.String',
            'Landroid/util/Log;': 'android.util.Log',
            'Landroidx/annotation/DoNotInline;': 'androidx.annotation.DoNotInline',
        };
        for (const input in referenceMap) {
            expect(compiler.prettyDescriptor(input)).toEqual(referenceMap[input]);
        }
    }

    async function testPrettyDescriptorsDimensions(info: CompilerInfo) {
        const compiler = new Dex2OatCompiler(info, env);
        const dimensionsMap = {
            '[I': 'int[]',
            '[[Z': 'boolean[][]',
            '[[[Ljava/lang/Integer;': 'java.lang.Integer[][][]',
            '[[[[Landroid/graphics/Bitmap;': 'android.graphics.Bitmap[][][][]',
        };
        for (const input in dimensionsMap) {
            expect(compiler.prettyDescriptor(input)).toEqual(dimensionsMap[input]);
        }
    }

    async function testSplitMethodParameters(info: CompilerInfo) {
        const compiler = new Dex2OatCompiler(info, env);
        const parametersMap = {
            I: ['I'],
            II: ['I', 'I'],
            IIZ: ['I', 'I', 'Z'],
            'Ljava/lang/String;': ['Ljava/lang/String;'],
            'IILjava/lang/String;': ['I', 'I', 'Ljava/lang/String;'],
            '[JJ[Ljava/lang/String;Z': ['[J', 'J', '[Ljava/lang/String;', 'Z'],
        };
        for (const input in parametersMap) {
            expect(compiler.splitMethodParameters(input)).toEqual(parametersMap[input]);
        }
    }

    async function testPrettyMethodSignature(info: CompilerInfo) {
        const compiler = new Dex2OatCompiler(info, env);
        const methodMap = {
            'square(I)I': 'int square(int)',
            'stringAppend(Ljava/lang/String;)Ljava/lang/String;': 'java.lang.String stringAppend(java.lang.String)',
            'push([II)[I': 'int[] push(int[], int)',
        };
        for (const input in methodMap) {
            expect(compiler.prettyMethodSignature(input)).toEqual(methodMap[input]);
        }
    }
});
