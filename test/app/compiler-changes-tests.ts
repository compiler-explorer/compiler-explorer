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

import {afterEach, beforeEach, describe, expect, it, vi} from 'vitest';

import {setupCompilerChangeHandling} from '../../lib/app/compiler-changes.js';
import {AppArguments} from '../../lib/app.interfaces.js';
import {CompilerFinder} from '../../lib/compiler-finder.js';
import {RouteAPI} from '../../lib/handlers/route-api.js';
import {logger} from '../../lib/logger.js';
import {ClientOptionsHandler} from '../../lib/options-handler.js';
import * as properties from '../../lib/properties.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import {Language, LanguageKey} from '../../types/languages.interfaces.js';

describe('compiler-changes module', () => {
    const languages = {fake: {id: 'fake'}} as unknown as Record<LanguageKey, Language>;
    const initialCompilers = [{id: 'gcc', name: 'GCC', lang: 'c++'}] as unknown as CompilerInfo[];

    let clientOptionsHandler: ClientOptionsHandler;
    let apiHandler: {
        setCompilers: ReturnType<typeof vi.fn>;
        setLanguages: ReturnType<typeof vi.fn>;
        setOptions: ReturnType<typeof vi.fn>;
    };
    let routeApi: RouteAPI;
    let compilerFinder: CompilerFinder;

    beforeEach(() => {
        vi.spyOn(logger, 'info').mockImplementation(() => logger);
        vi.spyOn(logger, 'debug').mockImplementation(() => logger);
        clientOptionsHandler = {setCompilers: vi.fn().mockResolvedValue(undefined)} as unknown as ClientOptionsHandler;
        apiHandler = {setCompilers: vi.fn(), setLanguages: vi.fn(), setOptions: vi.fn()};
        routeApi = {apiHandler} as unknown as RouteAPI;
        compilerFinder = {find: vi.fn()} as unknown as CompilerFinder;
    });

    afterEach(() => {
        vi.restoreAllMocks();
        vi.useRealTimers();
    });

    function setup(props: Record<string, any>, appArgs?: Partial<AppArguments>) {
        return setupCompilerChangeHandling(
            initialCompilers,
            clientOptionsHandler,
            routeApi,
            languages,
            properties.fakeProps(props),
            compilerFinder,
            appArgs as AppArguments,
        );
    }

    it('should apply the initial compiler set', async () => {
        await setup({});

        expect(clientOptionsHandler.setCompilers).toHaveBeenCalledWith(initialCompilers);
        expect(apiHandler.setCompilers).toHaveBeenCalledWith(initialCompilers);
        expect(apiHandler.setLanguages).toHaveBeenCalledWith(languages);
        expect(apiHandler.setOptions).toHaveBeenCalledWith(clientOptionsHandler);
    });

    it('should not rescan when compilers were prediscovered', async () => {
        vi.useFakeTimers();
        await setup({rescanCompilerSecs: 1}, {prediscovered: 'compilers.json'});

        await vi.advanceTimersByTimeAsync(5000);
        expect(compilerFinder.find).not.toHaveBeenCalled();
    });

    it('should not reapply an unchanged compiler set on rescan', async () => {
        vi.useFakeTimers();
        vi.mocked(compilerFinder.find).mockResolvedValue({compilers: [...initialCompilers], foundClash: false});
        await setup({rescanCompilerSecs: 1}, {});

        await vi.advanceTimersByTimeAsync(1000);
        expect(compilerFinder.find).toHaveBeenCalledTimes(1);
        expect(clientOptionsHandler.setCompilers).toHaveBeenCalledTimes(1);
        expect(apiHandler.setCompilers).toHaveBeenCalledTimes(1);
    });

    it('should apply a changed compiler set on rescan', async () => {
        vi.useFakeTimers();
        const newCompilers = [{id: 'clang', name: 'Clang', lang: 'c++'}] as unknown as CompilerInfo[];
        vi.mocked(compilerFinder.find).mockResolvedValue({compilers: newCompilers, foundClash: false});
        await setup({rescanCompilerSecs: 1}, {});

        await vi.advanceTimersByTimeAsync(1000);
        expect(clientOptionsHandler.setCompilers).toHaveBeenCalledTimes(2);
        expect(clientOptionsHandler.setCompilers).toHaveBeenLastCalledWith(newCompilers);
        expect(apiHandler.setCompilers).toHaveBeenLastCalledWith(newCompilers);
    });
});
