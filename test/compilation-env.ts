// Copyright (c) 2016, Compiler Explorer Authors
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

import './utils.js';
import {beforeAll, describe, expect, it} from 'vitest';

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {FormattingService} from '../lib/formatting-service.js';
import {CompilerProps, fakeProps} from '../lib/properties.js';

const props = {
    optionsAllowedRe: '.*',
    optionsForbiddenRe: '^(-W[alp],)?((-wrapper|-fplugin.*|-specs|-load|-plugin|(@.*)|-I|-i)(=.*)?|--)$',
    cacheConfig: 'InMemory(10)',
};

describe('Compilation environment', () => {
    let compilerProps;

    beforeAll(() => {
        compilerProps = new CompilerProps({}, fakeProps(props));
    });

    it('Should cache by default', async () => {
        // TODO: Work will need to be done here when CompilationEnvironment's constructor is typed better
        const ce = new CompilationEnvironment(
            compilerProps,
            fakeProps({}),
            undefined,
            new FormattingService(),
            undefined,
        );
        await expect(ce.cacheGet('foo')).resolves.toBeNull();
        await ce.cachePut('foo', {res: 'bar'}, undefined);
        await expect(ce.cacheGet('foo')).resolves.toEqual({res: 'bar'});
        await expect(ce.cacheGet('baz')).resolves.toBeNull();
    });
    it('Should cache when asked', async () => {
        const ce = new CompilationEnvironment(compilerProps, fakeProps({}), undefined, new FormattingService(), true);
        await expect(ce.cacheGet('foo')).resolves.toBeNull();
        await ce.cachePut('foo', {res: 'bar'}, undefined);
        await expect(ce.cacheGet('foo')).resolves.toEqual({res: 'bar'});
    });
    it("Shouldn't cache when asked", async () => {
        // TODO: Work will need to be done here when CompilationEnvironment's constructor is typed better
        const ce = new CompilationEnvironment(compilerProps, fakeProps({}), undefined, new FormattingService(), false);
        await expect(ce.cacheGet('foo')).resolves.toBeNull();
        await ce.cachePut('foo', {res: 'bar'}, undefined);
        await expect(ce.cacheGet('foo')).resolves.toBeNull();
    });
    it('Should filter bad options', () => {
        // TODO: Work will need to be done here when CompilationEnvironment's constructor is typed better
        const ce = new CompilationEnvironment(
            compilerProps,
            fakeProps({}),
            undefined,
            new FormattingService(),
            undefined,
        );
        expect(ce.findBadOptions(['-O3', '-flto'])).toEqual([]);
        expect(ce.findBadOptions(['-O3', '-plugin'])).toEqual(['-plugin']);
    });
});
