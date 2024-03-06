// Copyright (c) 2017, Compiler Explorer Authors
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

import './utils.js';
import {SymbolStore} from '../lib/symbol-store.js';

describe('SymbolStore', () => {
    it('should be empty initially', () => {
        const store = new SymbolStore();
        expect(store.listSymbols().length).toEqual(0);
        expect(store.listTranslations().length).toEqual(0);
    });

    it('should be able to add an item', () => {
        const store = new SymbolStore();
        store.add('test');
        expect(store.listSymbols().length).toEqual(1);
        expect(store.listTranslations().length).toEqual(1);

        expect(store.listSymbols()[0]).toEqual('test');

        const translations = store.listTranslations();
        expect(translations[0][0]).toEqual('test');
        expect(translations[0][1]).toEqual('test');
    });

    it('should not contain duplicate items', () => {
        const store = new SymbolStore();
        store.add('test');
        store.add('test');
        expect(store.listSymbols().length).toEqual(1);
        expect(store.listTranslations().length).toEqual(1);

        expect(store.listSymbols()[0]).toEqual('test');

        const translations = store.listTranslations();
        expect(translations[0][0]).toEqual('test');
        expect(translations[0][1]).toEqual('test');
    });

    it('should return a sorted list', () => {
        const store = new SymbolStore();
        store.add('test123');
        store.add('test123456');
        expect(store.listSymbols().length).toEqual(2);
        expect(store.listTranslations().length).toEqual(2);

        const translations = store.listTranslations();
        expect(translations[0][0]).toEqual('test123456');
        expect(translations[1][0]).toEqual('test123');
    });

    it('should be able to add an array of items', () => {
        const store = new SymbolStore();
        store.addMany(['test123', 'test123456', 'test123']);
        expect(store.listSymbols().length).toEqual(2);
        expect(store.listTranslations().length).toEqual(2);

        const translations = store.listTranslations();
        expect(translations[0][0]).toEqual('test123456');
        expect(translations[1][0]).toEqual('test123');
    });

    it('should be possible to exclude items in another store', () => {
        const store1 = new SymbolStore();
        store1.addMany(['test123', 'test123456', 'test123']);

        const store2 = new SymbolStore();
        store2.addMany(['test123']);

        store1.exclude(store2);
        const translations = store1.listTranslations();
        expect(translations.length).toEqual(1);
        expect(translations[0][0]).toEqual('test123456');
    });

    it('should be possible to exclude items that partially match', () => {
        const store1 = new SymbolStore();
        store1.addMany(['test123', 'test123456', 'test123']);

        const store2 = new SymbolStore();
        store2.addMany(['est123']);

        store1.softExclude(store2);
        const translations = store1.listTranslations();
        expect(translations.length).toEqual(1);
        expect(translations[0][0]).toEqual('test123456');
    });

    it('should be able to check contents', () => {
        const store = new SymbolStore();
        store.addMany(['test123', 'test123456', 'test123']);

        expect(store.contains('test123')).toEqual(true);
        expect(store.contains('test123456')).toEqual(true);
        expect(store.contains('test456')).toEqual(false);

        expect(store.listSymbols().length).toEqual(2);
    });
});
