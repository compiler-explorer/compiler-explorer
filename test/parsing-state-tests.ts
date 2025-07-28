// Copyright (c) 2025, Compiler Explorer Authors
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

import {beforeEach, describe, expect, it} from 'vitest';

import {ParsingState} from '../lib/parsers/parsing-state.js';

describe('ParsingState tests', () => {
    const files = {1: '/path/to/file.cpp'};
    let state: ParsingState;

    beforeEach(() => {
        state = new ParsingState(files, null, '', false, false, []);
    });

    describe('construction and initialization', () => {
        it('should initialize with correct default values', () => {
            expect(state.files).toBe(files);
            expect(state.source).toBeNull();
            expect(state.prevLabel).toBe('');
            expect(state.prevLabelIsUserFunction).toBe(false);
            expect(state.dontMaskFilenames).toBe(false);
            expect(state.mayRemovePreviousLabel).toBe(true);
            expect(state.keepInlineCode).toBe(false);
            expect(state.inNvccDef).toBe(false);
            expect(state.inNvccCode).toBe(false);
            expect(state.inCustomAssembly).toBe(0);
            expect(state.inVLIWpacket).toBe(false);
            expect(state.getCurrentLineIndex()).toBe(0);
        });
    });

    describe('source management', () => {
        it('should update source and track own source', () => {
            const source = {file: null, line: 42, mainsource: true};
            state.updateSource(source);

            expect(state.source).toBe(source);
            expect(state.lastOwnSource).toBe(source);
        });

        it('should not update lastOwnSource for library sources', () => {
            const librarySource = {file: '/usr/include/stdio.h', line: 100};
            state.updateSource(librarySource);

            expect(state.source).toBe(librarySource);
            expect(state.lastOwnSource).toBeNull();
        });

        it('should reset state on block end', () => {
            state.updateSource({file: null, line: 42});
            state.updatePrevLabel('test_label');

            state.resetToBlockEnd();

            expect(state.source).toBeNull();
            expect(state.prevLabel).toBe('');
            expect(state.lastOwnSource).toBeNull();
        });
    });

    describe('custom assembly handling', () => {
        it('should track custom assembly nesting', () => {
            expect(state.isInCustomAssembly()).toBe(false);

            state.enterCustomAssembly();
            expect(state.isInCustomAssembly()).toBe(true);
            expect(state.inCustomAssembly).toBe(1);

            state.enterCustomAssembly();
            expect(state.inCustomAssembly).toBe(2);

            state.exitCustomAssembly();
            expect(state.inCustomAssembly).toBe(1);
            expect(state.isInCustomAssembly()).toBe(true);

            state.exitCustomAssembly();
            expect(state.isInCustomAssembly()).toBe(false);
        });
    });

    describe('NVCC state management', () => {
        it('should handle NVCC definition state', () => {
            expect(state.inNvccDef).toBe(false);
            expect(state.inNvccCode).toBe(false);

            state.enterNvccDef();
            expect(state.inNvccDef).toBe(true);
            expect(state.inNvccCode).toBe(true);

            state.exitNvccDef();
            expect(state.inNvccDef).toBe(false);
            expect(state.inNvccCode).toBe(true);
        });
    });

    describe('library code filtering', () => {
        it('should filter library code when conditions are met', () => {
            state.updatePrevLabel('lib_func', false); // not user function
            state.updateSource({file: '/usr/lib/library.so', line: 100}); // library source

            const result = state.shouldFilterLibraryCode({libraryCode: true});
            expect(result).toBe(true);
        });

        it('should not filter when user function', () => {
            state.updatePrevLabel('user_func', true); // user function
            state.updateSource({file: '/usr/lib/library.so', line: 100});

            const result = state.shouldFilterLibraryCode({libraryCode: true});
            expect(result).toBe(false);
        });

        it('should not filter when has own source', () => {
            state.updatePrevLabel('lib_func', false);
            state.updateSource({file: null, line: 42, mainsource: true}); // own source

            const result = state.shouldFilterLibraryCode({libraryCode: true});
            expect(result).toBe(false);
        });

        it('should not filter when source is undefined', () => {
            state.updatePrevLabel('lib_func', false); // not user function
            state.updateSource(undefined); // no source

            const result = state.shouldFilterLibraryCode({libraryCode: true});
            expect(result).toBe(false);
        });

        it('should not filter when source is null', () => {
            state.updatePrevLabel('lib_func', false); // not user function
            state.updateSource(null); // no source

            const result = state.shouldFilterLibraryCode({libraryCode: true});
            expect(result).toBe(false);
        });
    });

    describe('label management', () => {
        it('should update and clear prev label', () => {
            state.updatePrevLabel('test_label', true);
            expect(state.prevLabel).toBe('test_label');
            expect(state.prevLabelIsUserFunction).toBe(true);

            state.clearPrevLabel();
            expect(state.prevLabel).toBe('');
            expect(state.prevLabelIsUserFunction).toBe(false);
        });
    });

    describe('line iteration', () => {
        it('should iterate through lines', () => {
            const testLines = ['line1', 'line2', 'line3'];
            const iterableState = new ParsingState({}, null, '', false, false, testLines);

            const lines: string[] = [];
            for (const line of iterableState) {
                lines.push(line);
            }

            expect(lines).toEqual(testLines);
        });

        it('should track current index during iteration', () => {
            const testLines = ['line1', 'line2'];
            const iterableState = new ParsingState({}, null, '', false, false, testLines);

            expect(iterableState.getCurrentLineIndex()).toBe(0);

            const iterator = iterableState[Symbol.iterator]();
            iterator.next();
            expect(iterableState.getCurrentLineIndex()).toBe(1);

            iterator.next();
            expect(iterableState.getCurrentLineIndex()).toBe(2);
        });
    });

    describe('flags management', () => {
        it('should manage removal and inline code flags', () => {
            expect(state.shouldRemovePreviousLabel()).toBe(true);
            expect(state.shouldKeepInlineCode()).toBe(false);

            state.setMayRemovePreviousLabel(false);
            state.setKeepInlineCode(true);

            expect(state.shouldRemovePreviousLabel()).toBe(false);
            expect(state.shouldKeepInlineCode()).toBe(true);
        });
    });
});
