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

import './utils';
import {SymbolStore} from '../lib/symbol-store';

describe('SymbolStore', function () {
    it('should be empty initially', function () {
        const store = new SymbolStore();
        store.listSymbols().length.should.equal(0);
        store.listTranslations().length.should.equal(0);
    });

    it('should be able to add an item', function () {
        const store = new SymbolStore();
        store.add('test');
        store.listSymbols().length.should.equal(1);
        store.listTranslations().length.should.equal(1);

        store.listSymbols()[0].should.equal('test');

        const translations = store.listTranslations();
        translations[0][0].should.equal('test');
        translations[0][1].should.equal('test');
    });

    it('should not contain duplicate items', function () {
        const store = new SymbolStore();
        store.add('test');
        store.add('test');
        store.listSymbols().length.should.equal(1);
        store.listTranslations().length.should.equal(1);

        store.listSymbols()[0].should.equal('test');

        const translations = store.listTranslations();
        translations[0][0].should.equal('test');
        translations[0][1].should.equal('test');
    });

    it('should return a sorted list', function () {
        const store = new SymbolStore();
        store.add('test123');
        store.add('test123456');
        store.listSymbols().length.should.equal(2);
        store.listTranslations().length.should.equal(2);

        const translations = store.listTranslations();
        translations[0][0].should.equal('test123456');
        translations[1][0].should.equal('test123');
    });

    it('should be able to add an array of items', function () {
        const store = new SymbolStore();
        store.addMany(['test123', 'test123456', 'test123']);
        store.listSymbols().length.should.equal(2);
        store.listTranslations().length.should.equal(2);

        const translations = store.listTranslations();
        translations[0][0].should.equal('test123456');
        translations[1][0].should.equal('test123');
    });

    it('should be possible to exclude items in another store', function () {
        const store1 = new SymbolStore();
        store1.addMany(['test123', 'test123456', 'test123']);

        const store2 = new SymbolStore();
        store2.addMany(['test123']);

        store1.exclude(store2);
        var translations = store1.listTranslations();
        translations.length.should.equal(1);
        translations[0][0].should.equal('test123456');
    });

    it('should be possible to exclude items that partially match', function () {
        const store1 = new SymbolStore();
        store1.addMany(['test123', 'test123456', 'test123']);

        const store2 = new SymbolStore();
        store2.addMany(['est123']);

        store1.softExclude(store2);
        var translations = store1.listTranslations();
        translations.length.should.equal(1);
        translations[0][0].should.equal('test123456');
    });

    it('should be able to check contents', function () {
        const store = new SymbolStore();
        store.addMany(['test123', 'test123456', 'test123']);

        store.contains('test123').should.equal(true);
        store.contains('test123456').should.equal(true);
        store.contains('test456').should.equal(false);

        store.listSymbols().length.should.equal(2);
    });
});
