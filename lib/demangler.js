// Copyright (c) 2018, Compiler Explorer Team
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

const
    SymbolStore = require('./symbol-store').SymbolStore,
    utils = require('../lib/utils'),
    AsmRegex = require('../lib/asmregex').AsmRegex;

class Demangler extends AsmRegex {
    /**
     * 
     * @param {string} demanglerExe 
     * @param {BaseCompiler} compiler 
     */
    constructor(demanglerExe, compiler) {
        super();

        this.demanglerExe = demanglerExe;
        this.demanglerArguments = [];
        this.symbolstore = null;
        this.othersymbols = new SymbolStore();
        this.result = {};
        this.input = [];
        this.includeMetadata = true;
        this.compiler = compiler;

        this.jumpDef = /(j\w+|b|bl|blx)\s+([a-z_$][a-z0-9$_@]*)/i;
        this.callDef = /call[q]?\s+([.a-z_$][a-z0-9$_@.]*)/i;

        // symbols in a mov or lea command starting with an underscore
        this.movUnderscoreDef = /mov.*\s(_[a-z0-9$_@.]*)/i;
        this.leaUnderscoreDef = /lea.*\s(_[a-z0-9$_@.]*)/i;
        this.quadUnderscoreDef = /\.quad\s*(_[a-z0-9$_@.]*)/i;
    }

    addMatchToOtherSymbols(matches) {
        if (!matches) return false;

        const midx = matches.length - 1;
        this.othersymbols.add(matches[midx], matches[midx]);

        return true;
    }

    collectLabels() {
        for (var j = 0; j < this.result.asm.length; ++j) {
            const line = this.result.asm[j].text;

            let matches = line.match(this.labelDef);
            if (matches) {
                const midx = matches.length - 1;
                this.symbolstore.add(matches[midx], matches[midx]);
            }

            if (this.addMatchToOtherSymbols(line.match(this.jumpDef))) continue;
            if (this.addMatchToOtherSymbols(line.match(this.callDef))) continue;
            if (this.addMatchToOtherSymbols(line.match(this.movUnderscoreDef))) continue;
            if (this.addMatchToOtherSymbols(line.match(this.leaUnderscoreDef))) continue;
            if (this.addMatchToOtherSymbols(line.match(this.quadUnderscoreDef))) continue;
        }

        this.othersymbols.exclude(this.symbolstore);
    }

    getInput() {
        this.input = [];
        this.input = this.input.concat(this.symbolstore.listSymbols());
        this.input = this.input.concat(this.othersymbols.listSymbols());

        return this.input.join("\n");
    }

    /**
     * 
     * @param {string} symbol 
     */
    getMetadata() {
        let metadata = [];
        
        return metadata;
    }

    /**
     * 
     * @param {string} symbol 
     * @param {string} translation 
     */
    addTranslation(symbol, translation) {
        let metadataStr = "";

        if (this.includeMetadata) {
            const metadata = this.getMetadata(symbol);
            metadataStr = metadata.map((meta) => " [" + meta.description + "]").join();
        }

        if (this.symbolstore.contains(symbol)) {
            this.symbolstore.add(symbol, translation + metadataStr);
        } else {
            this.othersymbols.add(symbol, translation + metadataStr);
        }
    }

    processOutput(output) {
        const lines = utils.splitLines(output.stdout);
        for (let i = 0; i < lines.length; ++i)
            this.addTranslation(this.input[i], lines[i]);

        let translations = [];
        translations = translations.concat(this.symbolstore.listTranslations());
        translations = translations.concat(this.othersymbols.listTranslations());

        for (let i = 0; i < this.result.asm.length; ++i) {
            let line = this.result.asm[i].text;
            for (let j = 0; j < translations.length; ++j) {
                line = line.replace(translations[j][0], translations[j][1]);
                line = line.replace(translations[j][0], translations[j][1]);
            }
            this.result.asm[i].text = line;
        }

        return this.result;
    }

    execDemangler(options) {
        options.maxOutput = -1;

        return this.compiler.exec(
            this.demanglerExe,
            this.demanglerArguments,
            options
        );
    }

    process(result, execOptions) {
        let options = execOptions || {};
        this.result = result;

        if (!this.symbolstore) {
            this.symbolstore = new SymbolStore();
            this.collectLabels();
        }

        options.input = this.getInput();

        if (options.input === "") {
            return new Promise((resolve) => resolve(this.result));
        } else {
            return this.execDemangler(options).then((output) => this.processOutput(output));
        }
    }
}

exports.Demangler = Demangler;
