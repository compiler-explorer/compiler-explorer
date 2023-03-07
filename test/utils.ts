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

import path from 'path';
import {fileURLToPath} from 'url';

import chai from 'chai';
import fs from 'fs-extra';

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {CompilationQueue} from '../lib/compilation-queue.js';
import {CompilerProps, fakeProps} from '../lib/properties.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';

// TODO: Find proper type for options
export function makeCompilationEnvironment(options: Record<string, any>): CompilationEnvironment {
    const compilerProps = new CompilerProps(options.languages, fakeProps(options.props || {}));
    const compilationQueue = options.queue || new CompilationQueue(options.concurrency || 1, options.timeout);
    return new CompilationEnvironment(compilerProps, compilationQueue, options.doCache);
}

export function makeFakeCompilerInfo(props: Partial<CompilerInfo>): CompilerInfo {
    return props as CompilerInfo;
}

export function makeFakeParseFiltersAndOutputOptions(
    options: Partial<ParseFiltersAndOutputOptions>,
): ParseFiltersAndOutputOptions {
    return options as ParseFiltersAndOutputOptions;
}

export const should = chai.should();

// This compbines a shoudl assert and a type guard
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
    should.exist(value, message);
    return true;
}

/***
 * Absolute path to the root of the tests
 */
export const TEST_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)));

export function resolvePathFromTestRoot(...args: string[]): string {
    return path.resolve(TEST_ROOT, ...args);
}

// eslint-disable-next-line -- do not rewrite exports
export {chai, path, fs};
