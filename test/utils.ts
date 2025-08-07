// Copyright (c) 2020, Compiler Explorer Authors
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

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

import {afterEach, expect, onTestFinished} from 'vitest';
import * as temp from '../lib/temp.js';

// Check if expensive tests should be skipped (e.g., during pre-commit hooks)
export const skipExpensiveTests = process.env.SKIP_EXPENSIVE_TESTS === 'true';

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {CompilationQueue} from '../lib/compilation-queue.js';
import {AsmParser} from '../lib/parsers/asm-parser.js';
import {CC65AsmParser} from '../lib/parsers/asm-parser-cc65.js';
import {AsmEWAVRParser} from '../lib/parsers/asm-parser-ewavr.js';
import {PTXAsmParser} from '../lib/parsers/asm-parser-ptx.js';
import {SassAsmParser} from '../lib/parsers/asm-parser-sass.js';
import {VcAsmParser} from '../lib/parsers/asm-parser-vc.js';

// Test helper class that extends AsmParser to allow setting protected properties for testing
class AsmParserForTest extends AsmParser {
    setBinaryHideFuncReForTest(regex: RegExp | null) {
        this.binaryHideFuncRe = regex;
    }
}

import {CompilerProps, fakeProps} from '../lib/properties.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';
import {Language} from '../types/languages.interfaces.js';

function ensureTempCleanup() {
    // Sometimes we're called from inside a test, sometimes from outside. Handle both.
    afterEach(async () => await temp.cleanup());
    try {
        onTestFinished(async () => await temp.cleanup());
    } catch (_) {
        // ignore; we weren't in a test body.
    }
}

// TODO: Find proper type for options
export function makeCompilationEnvironment(options: Record<string, any>): CompilationEnvironment {
    ensureTempCleanup();
    const compilerProps = new CompilerProps(options.languages, fakeProps(options.props || {}));
    const compilationQueue = options.queue || new CompilationQueue(options.concurrency || 1, options.timeout, 100_000);
    return new CompilationEnvironment(compilerProps, fakeProps({}), compilationQueue, options.doCache);
}

export function makeFakeCompilerInfo(props: Partial<CompilerInfo>): CompilerInfo {
    return props as CompilerInfo;
}

export function makeFakeLanguage(props: Partial<Language>): Language {
    return props as Language;
}

export function makeFakeParseFiltersAndOutputOptions(
    options: Partial<ParseFiltersAndOutputOptions>,
): ParseFiltersAndOutputOptions {
    return options as ParseFiltersAndOutputOptions;
}

// This combines a should assert and a type guard
// Example:
//
//  let a: null|number = 1;
//  if(shouldExist(a)) {}
//    a.should.equal(1); /* No longer need ! because of type guard
//  }
//
//  a = null;
//  shouldExist(a); /* throws should.exist assertion
export function shouldExist<T>(value: T, message?: string): value is Exclude<T, null | undefined> {
    // TODO: if the message is set we should have a proper message here; since the move to vitest we lost it.
    expect(value).toEqual(expect.anything());
    return true;
}

/***
 * Absolute path to the root of the tests
 */
export const TEST_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)));

export function resolvePathFromTestRoot(...args: string[]): string {
    return path.resolve(TEST_ROOT, ...args);
}

// Tracked temporary directories.
export function newTempDir() {
    ensureTempCleanup();
    return temp.mkdirSync('compiler-explorer-tests');
}

export function processAsm(filename: string, filters: ParseFiltersAndOutputOptions) {
    const file = fs.readFileSync(filename, 'utf8');
    let parser: AsmParser;
    if (file.includes('Microsoft')) parser = new VcAsmParser();
    else if (filename.includes('sass-')) parser = new SassAsmParser();
    else if (filename.includes('ptx-')) parser = new PTXAsmParser();
    else if (filename.includes('cc65-')) parser = new CC65AsmParser(fakeProps({}));
    else if (filename.includes('ewarm-')) parser = new AsmEWAVRParser(fakeProps({}));
    else {
        const testParser = new AsmParserForTest();
        testParser.setBinaryHideFuncReForTest(
            /^(__.*|_(init|start|fini)|(de)?register_tm_clones|call_gmon_start|frame_dummy|\.plt.*|_dl_relocate_static_pie)$/,
        );
        parser = testParser;
    }
    return parser.process(file, filters);
}
