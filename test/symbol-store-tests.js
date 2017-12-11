// Copyright (c) 2017, Patrick Quist
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

var chai = require('chai');
var should = chai.should();
var assert = chai.assert;
var SymbolStore = require('../lib/symbol-store').SymbolStore;
var logger = require('../lib/logger').logger;

describe('Basic examples', function () {
    it('Empty', function () {
        var store = new SymbolStore();
        store.ListSymbols().length.should.equal(0);
        store.ListTranslations().length.should.equal(0);
    });

    it('One item', function () {
        var store = new SymbolStore();
        store.Add("test");
        store.ListSymbols().length.should.equal(1);
        store.ListTranslations().length.should.equal(1);

        store.ListSymbols()[0].should.equal("test");

        var translations = store.ListTranslations();
        translations[0][0].should.equal("test");
        translations[0][1].should.equal("test");
    });

    it('Duplicate items', function () {
        var store = new SymbolStore();
        store.Add("test");
        store.Add("test");
        store.ListSymbols().length.should.equal(1);
        store.ListTranslations().length.should.equal(1);

        store.ListSymbols()[0].should.equal("test");

        var translations = store.ListTranslations();
        translations[0][0].should.equal("test");
        translations[0][1].should.equal("test");
    });

    it('Sorted items', function () {
        var store = new SymbolStore();
        store.Add("test123");
        store.Add("test123456");
        store.ListSymbols().length.should.equal(2);
        store.ListTranslations().length.should.equal(2);

        var translations = store.ListTranslations();
        translations[0][0].should.equal("test123456");
        translations[1][0].should.equal("test123");
    });

    it('Add-many', function () {
        var store = new SymbolStore();
        store.AddMany(["test123", "test123456", "test123"]);
        store.ListSymbols().length.should.equal(2);
        store.ListTranslations().length.should.equal(2);

        var translations = store.ListTranslations();
        translations[0][0].should.equal("test123456");
        translations[1][0].should.equal("test123");
    });
});
