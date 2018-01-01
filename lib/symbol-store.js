// Copyright (c) 2012-2018, Patrick Quist
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
"use strict";

var logger = require('../lib/logger').logger;

class SymbolStore {
    constructor() {
        this.uniqueSymbols = {};
        this.sortedSymbols = [];
        this.isSorted = true;
    }

    Sort() {
        this.sortedSymbols = [];
        for (var symbol in this.uniqueSymbols) {
            this.sortedSymbols.push([symbol, this.uniqueSymbols[symbol]]);
        }

        this.sortedSymbols.sort(function(a, b) {
            return b[0].length - a[0].length;
        });

        this.isSorted = true;
    }

    Add(symbol, demangled) {
        if (demangled !== undefined) {
            this.uniqueSymbols[symbol] = demangled;
        } else {
            this.uniqueSymbols[symbol] = symbol;
        }

        this.isSorted = false;
    }

    AddMany(symbols) {
        symbols.forEach(symbol => {
            this.uniqueSymbols[symbol] = symbol;
        });

        this.isSorted = false;
    }

    ListSymbols() {
        if (!this.isSorted) this.Sort();

        return this.sortedSymbols.map(function(elem) {
            return elem[0];
        });
    }

    ListTranslations() {
        if (!this.isSorted) this.Sort();

        return this.sortedSymbols;
    }
}

exports.SymbolStore = SymbolStore;
