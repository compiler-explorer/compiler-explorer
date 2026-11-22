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

import $ from 'jquery';
import {afterEach, describe, expect, it, vi} from 'vitest';

import type {CompilerInfo} from '../../../types/compiler.interfaces.js';
import type {Library} from '../../options.interfaces.js';
import {compilersService} from '../../services/compilers.service.js';
import {type LanguageMap, languagesService} from '../../services/languages.service.js';
import {
    type CompilerLibs,
    DEFAULT_COMPILER_KEY,
    LibsWidget,
    stateLibsToLibs,
    toCompilerKey,
} from '../../widgets/libs-widget.js';

const fakeCompiler = {id: 'g142', name: 'x86-64 gcc 14.2'} as unknown as CompilerInfo;

describe('LibsWidget compiler key', () => {
    it('reads the id off a compiler object, as the compiler and executor panes pass', () => {
        expect(toCompilerKey(fakeCompiler)).toBe('g142');
    });

    it('takes an id string as-is, as the conformance view passes', () => {
        expect(toCompilerKey('g142|clang1500')).toBe('g142|clang1500');
    });

    it('agrees with itself across the two call shapes for the same compiler', () => {
        expect(toCompilerKey('g142')).toBe(toCompilerKey(fakeCompiler));
    });

    it('falls back to the default key for every empty shape', () => {
        expect(toCompilerKey(undefined)).toBe(DEFAULT_COMPILER_KEY);
        expect(toCompilerKey(null)).toBe(DEFAULT_COMPILER_KEY);
        expect(toCompilerKey('')).toBe(DEFAULT_COMPILER_KEY);
    });

    it('never yields undefined, which is what keyed the conformance view into a dead bucket', () => {
        for (const shape of [fakeCompiler, 'g142|clang1500', '', null, undefined]) {
            expect(typeof toCompilerKey(shape)).toBe('string');
        }
    });
});

describe('LibsWidget saved state libraries', () => {
    it('reads the {name, ver} shape links are written with today', () => {
        expect(stateLibsToLibs([{name: 'abseil', ver: '202601071'}])).toEqual([{name: 'abseil', ver: '202601071'}]);
    });

    it('reads the {id, version} shape older links carry', () => {
        expect(stateLibsToLibs([{id: 'abseil', version: '202601071'}])).toEqual([{name: 'abseil', ver: '202601071'}]);
    });

    it('has nothing to say about missing or empty state', () => {
        expect(stateLibsToLibs(undefined)).toEqual([]);
        expect(stateLibsToLibs([])).toEqual([]);
    });

    it('drops half-specified entries rather than asking for version undefined', () => {
        expect(stateLibsToLibs([{name: 'abseil'}, {ver: '202601071'}, {}])).toEqual([]);
    });
});

function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>(r => {
        resolve = r;
    });
    return {promise, resolve};
}

const flush = () => new Promise(resolve => setTimeout(resolve, 0));

const fakeLib = (name: string, version = '1'): Library => ({
    name,
    dependencies: [],
    versions: {[version]: {alias: [], hidden: false, libId: name, used: false, version}},
});

async function makeWidget(possibleLibs: CompilerLibs = {}) {
    Object.assign(window, {compilerExplorerOptions: {}});
    if ($('#lib-search-result-tpl').length === 0) {
        $('body').append('<div id="lib-search-result-tpl"><div><button class="lib-info-button"></button></div></div>');
    }
    vi.spyOn(languagesService, 'getLanguages').mockResolvedValue({});
    vi.spyOn(compilersService, 'getCompilersForLang').mockResolvedValue({});
    const onChange = vi.fn();
    const widget = new LibsWidget('c++', 'g162', $('<button>'), {}, onChange, possibleLibs);
    await widget.stateLoaded;
    return {widget, onChange};
}

describe('LibsWidget compiler and language changes', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('carries the selected libraries over to the next compiler', async () => {
        const {widget} = await makeWidget({fmt: fakeLib('fmt')});
        widget.markLibrary('fmt', '1', true);

        await widget.setNewLangId('c++', 'clang2010', {fmt: fakeLib('fmt')});

        expect(widget.get()).toEqual([{name: 'fmt', ver: '1'}]);
    });

    it("selects the new language's default libraries", async () => {
        const {widget} = await makeWidget();
        vi.spyOn(languagesService, 'getLanguages').mockResolvedValue({
            ___c: {defaultLibs: 'fmt.1'},
        } as unknown as LanguageMap);

        await widget.setNewLangId('___c', 'cg162', {fmt: fakeLib('fmt'), boost: fakeLib('boost')});

        expect(widget.get()).toEqual([{name: 'fmt', ver: '1'}]);
    });

    it('survives a compiler change that arrives while a language change is still loading', async () => {
        const {widget} = await makeWidget();

        const languagesLoaded = deferred<LanguageMap>();
        vi.spyOn(languagesService, 'getLanguages').mockReturnValueOnce(languagesLoaded.promise);
        const toC = widget.setNewLangId('___c', 'cg162', {fmt: fakeLib('fmt')});
        await flush();

        const cCompilersLoaded = deferred<Record<string, CompilerInfo>>();
        vi.spyOn(compilersService, 'getCompilersForLang').mockReturnValueOnce(cCompilersLoaded.promise);
        const toClang = widget.setNewLangId('___c', 'cclang2010', {boost: fakeLib('boost')});

        languagesLoaded.resolve({});
        await flush();
        cCompilersLoaded.resolve({});

        await expect(toC).resolves.toBeUndefined();
        await expect(toClang).resolves.toBeUndefined();
        expect(widget.getLibInfoById('boost')).toBeDefined();
    });

    it("keeps each compiler's own library list when two compiler changes overlap", async () => {
        const {widget, onChange} = await makeWidget();

        const gccNameLoaded = deferred<Record<string, CompilerInfo>>();
        const clangNameLoaded = deferred<Record<string, CompilerInfo>>();
        vi.spyOn(compilersService, 'getCompilersForLang')
            .mockReturnValueOnce(gccNameLoaded.promise)
            .mockReturnValueOnce(clangNameLoaded.promise);

        const toGcc = widget.setNewLangId('c++', 'g152', {fmt: fakeLib('fmt')});
        const toClang = widget.setNewLangId('c++', 'clang2010', {boost: fakeLib('boost')});

        gccNameLoaded.resolve({});
        await flush();
        clangNameLoaded.resolve({});
        await Promise.all([toGcc, toClang]);

        expect(widget.getLibInfoById('boost')).toBeDefined();
        expect(widget.getLibInfoById('fmt')).toBeUndefined();
        expect(onChange).toHaveBeenCalledTimes(1);
    });
});
