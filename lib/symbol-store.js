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

export class SymbolStore {
    constructor() {
        this.uniqueSymbols = {};
        this.sortedSymbols = [];
        this.isSorted = true;
    }

    sort() {
        this.sortedSymbols = [];
        for (let symbol in this.uniqueSymbols) {
            this.sortedSymbols.push([symbol, this.uniqueSymbols[symbol]]);
        }

        this.sortedSymbols.sort(function (a, b) {
            return b[0].length - a[0].length;
        });

        this.isSorted = true;
    }

    add(symbol, demangled) {
        if (demangled !== undefined) {
            this.uniqueSymbols[symbol] = demangled;
        } else {
            this.uniqueSymbols[symbol] = symbol;
        }

        this.isSorted = false;
    }

    contains(symbol) {
        return this.uniqueSymbols[symbol] !== undefined;
    }

    exclude(otherStore) {
        for (const symbol in otherStore.uniqueSymbols) {
            delete this.uniqueSymbols[symbol];
        }

        this.isSorted = false;
    }

    softExclude(otherStore) {
        for (const symbol in otherStore.uniqueSymbols) {
            let shouldExclude = false;
            let checksymbol;
            for (checksymbol in this.uniqueSymbols) {
                if (checksymbol.endsWith(symbol)) {
                    shouldExclude = true;
                    break;
                }
            }

            if (shouldExclude) delete this.uniqueSymbols[checksymbol];
        }

        this.isSorted = false;
    }

    addMany(symbols) {
        symbols.forEach(symbol => {
            this.uniqueSymbols[symbol] = symbol;
        });

        this.isSorted = false;
    }

    listSymbols() {
        if (!this.isSorted) this.sort();

        return this.sortedSymbols.map(function (elem) {
            return elem[0];
        });
    }

    listTranslations() {
        if (!this.isSorted) this.sort();

        return this.sortedSymbols;
    }
}
