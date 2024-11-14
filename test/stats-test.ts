// Copyright (c) 2023, Compiler Explorer Authors
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

import {describe, expect, it} from 'vitest';

import {filterCompilerOptions, KnownBuildMethod, makeSafe} from '../lib/stats.js';
import {getHash} from '../lib/utils.js';
import {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';

describe('Stats', () => {
    const someDate = new Date(Date.UTC(2023, 6, 12, 2, 4, 6));
    it('should correctly parse and remove sensitive info from ParseRequests', () => {
        const source = 'This should never be seen';
        const executionParameters = {args: ['should', 'not', 'be', 'seen'], stdin: 'This should also not be seen'};
        expect(
            makeSafe(
                someDate,
                'g130',
                {
                    source: source,
                    options: ['-DDEBUG', '-O2', '-fsanitize=undefined'],
                    backendOptions: {},
                    filters: {
                        binary: false,
                        binaryObject: false,
                        execute: false,
                        demangle: true,
                        intel: true,
                        labels: true,
                        libraryCode: true,
                        directives: true,
                        commentOnly: true,
                        trim: true,
                        debugCalls: false,
                        dontMaskFilenames: true,
                        optOutput: true,
                        preProcessLines: lines => lines,
                        preProcessBinaryAsmLines: lines => lines,
                    },
                    bypassCache: 0,
                    tools: [],
                    executeParameters: executionParameters,
                    libraries: [],
                },
                [],
                KnownBuildMethod.Compile,
            ),
        ).toEqual({
            compilerId: 'g130',
            bypassCache: false,
            executionParamsHash: getHash(executionParameters),
            filters: {
                binary: false,
                binaryobject: false,
                commentonly: true,
                debugcalls: false,
                demangle: true,
                directives: true,
                dontmaskfilenames: true,
                execute: false,
                intel: true,
                labels: true,
                librarycode: true,
                optoutput: true,
                trim: true,
            },
            libraries: [],
            options: ['-O2', '-fsanitize=undefined'],
            sourceHash: getHash(source + '[]'),
            time: '2023-07-12T02:04:06.000Z',
            tools: [],
            overrides: [],
            runtimeTools: [],
            backendOptions: [],
            buildMethod: KnownBuildMethod.Compile,
        });
    });

    it('should filter compiler arguments', () => {
        expect(filterCompilerOptions(['-moo', 'foo', '/moo'])).toEqual(['-moo', '/moo']);
        expect(filterCompilerOptions(['-Dsecret=1234', '/Dsecret'])).toEqual([]);
        expect(filterCompilerOptions(['-ithings', '/Ithings'])).toEqual([]);
    });

    it('should sanitize some duplications', () => {
        const executionParameters = {};
        expect(
            makeSafe(
                someDate,
                'g130',
                {
                    source: '',
                    options: ['-O2', '-fsanitize=undefined'],
                    backendOptions: {
                        overrides: [{name: 'test', value: '123'}],
                        skipAsm: false,
                        SKIPASM: 'hello123',
                    },
                    filters: {
                        binary: false,
                        binaryObject: false,
                        execute: false,
                        demangle: true,
                        intel: true,
                        labels: true,
                        libraryCode: true,
                        directives: true,
                        commentOnly: true,
                        trim: true,
                        debugCalls: false,
                        dontMaskFilenames: true,
                        skipAsm: true,
                        SKIPASM: true,
                        skipasm: true,
                        optOutput: true,
                    } as ParseFiltersAndOutputOptions,
                    bypassCache: 0,
                    tools: [],
                    executeParameters: executionParameters,
                    libraries: [],
                },
                [],
                KnownBuildMethod.Compile,
            ),
        ).toEqual({
            compilerId: 'g130',
            bypassCache: false,
            executionParamsHash: getHash(executionParameters),
            filters: {
                binary: false,
                binaryobject: false,
                commentonly: true,
                debugcalls: false,
                demangle: true,
                directives: true,
                dontmaskfilenames: true,
                execute: false,
                intel: true,
                labels: true,
                librarycode: true,
                optoutput: true,
                skipasm: true,
                trim: true,
            },
            libraries: [],
            options: ['-O2', '-fsanitize=undefined'],
            sourceHash: getHash('[]'),
            time: '2023-07-12T02:04:06.000Z',
            tools: [],
            overrides: ['test=123'],
            backendOptions: ['skipasm=1'],
            runtimeTools: [],
            buildMethod: KnownBuildMethod.Compile,
        });
    });
});
