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

import os from 'os';
import path from 'path';
import {fileURLToPath} from 'url';

import fs from 'fs-extra';
import temp from 'temp';
import {expect} from 'vitest';

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {CompilationQueue} from '../lib/compilation-queue.js';
import {CompilerProps, fakeProps} from '../lib/properties.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';
import {Language} from '../types/languages.interfaces.js';

// TODO: Find proper type for options
export function makeCompilationEnvironment(options: Record<string, any>): CompilationEnvironment {
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
    temp.track(true);
    return temp.mkdirSync({prefix: 'compiler-explorer-tests', dir: os.tmpdir()});
}

// eslint-disable-next-line -- do not rewrite exports
export {path, fs};
