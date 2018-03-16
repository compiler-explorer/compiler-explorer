// Copyright (c) 2018, Patrick Quist
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
    exec = require('../lib/exec');

class Demangler {
    /**
     * 
     * @param {string} demanglerExe 
     * @param {SymbolStore} symbolstore 
     */
    constructor(demanglerExe, symbolstore) {
        this.demanglerExe = demanglerExe;
        this.symbolstore = symbolstore;
        this.othersymbols = new SymbolStore();
        this.result = {};
        this.input = [];
        this.includeMetadata = true;

        this.labelDef = /^([.a-z_$][a-z0-9$_@.]*):/i;
        this.jumpDef = /j.+\s+([.a-z_$][a-z0-9$_@.]*)/i;
        this.callDef = /call[q]?\s+([.a-z_$][a-z0-9$_@.]*)/i;

        // symbols in a mov or lea command starting with an underscore
        this.movUnderscoreDef = /mov.*\s(_[a-z0-9$_@.]*)/i;
        this.leaUnderscoreDef = /lea.*\s(_[a-z0-9$_@.]*)/i;
        this.quadUnderscoreDef = /\.quad\s*(_[a-z0-9$_@.]*)/i;
    }

    AddMatchToOtherSymbols(matches) {
        if (!matches) return false;

        const midx = matches.length - 1;
        this.othersymbols.Add(matches[midx], matches[midx]);

        return true;
    }

    CollectLabels() {
        for (var j = 0; j < this.result.asm.length; ++j) {
            const line = this.result.asm[j].text;

            let matches = line.match(this.labelDef);
            if (matches) {
                const midx = matches.length - 1;
                this.symbolstore.Add(matches[midx], matches[midx]);
            }

            if (this.AddMatchToOtherSymbols(line.match(this.jumpDef))) continue;
            if (this.AddMatchToOtherSymbols(line.match(this.callDef))) continue;
            if (this.AddMatchToOtherSymbols(line.match(this.movUnderscoreDef))) continue;
            if (this.AddMatchToOtherSymbols(line.match(this.leaUnderscoreDef))) continue;
            if (this.AddMatchToOtherSymbols(line.match(this.quadUnderscoreDef))) continue;
        }

        this.othersymbols.Exclude(this.symbolstore);
    }

    GetInput() {
        this.input = [];
        this.input = this.input.concat(this.symbolstore.ListSymbols());
        this.input = this.input.concat(this.othersymbols.ListSymbols());

        return this.input.join("\n");
    }

    /**
     * 
     * @param {string} symbol 
     */
    GetMetadata() {
        let metadata = [];
        
        return metadata;
    }

    /**
     * 
     * @param {string} symbol 
     * @param {string} translation 
     */
    AddTranslation(symbol, translation) {
        let metadataStr = "";

        if (this.includeMetadata) {
            const metadata = this.GetMetadata(symbol);
            metadataStr = metadata.map((meta) => " [" + meta.description + "]").join();
        }

        if (this.symbolstore.Contains(symbol)) {
            this.symbolstore.Add(symbol, translation + metadataStr);
        } else {
            this.othersymbols.Add(symbol, translation + metadataStr);
        }
    }

    ProcessOutput(output) {
        const lines = utils.splitLines(output.stdout);
        for (let i = 0; i < lines.length; ++i)
            this.AddTranslation(this.input[i], lines[i]);

        const translations = this.symbolstore.ListTranslations();
        const extratranslations = this.othersymbols.ListTranslations();

        for (let i = 0; i < this.result.asm.length; ++i) {
            let line = this.result.asm[i].text;
            for (let j = 0; j < translations.length; ++j) {
                line = line.replace(translations[j][0], translations[j][1]);
                line = line.replace(translations[j][0], translations[j][1]);
            }
            for (let j = 0; j < extratranslations.length; ++j) {
                line = line.replace(extratranslations[j][0], extratranslations[j][1]);
                line = line.replace(extratranslations[j][0], extratranslations[j][1]);
            }
            this.result.asm[i].text = line;
        }

        return this.result;
    }

    ExecDemangler(options) {
        return exec.execute(
            this.demanglerExe,
            [],
            options
        );
    }

    Process(result, execOptions) {
        let options = execOptions || {};
        this.result = result;

        if (!this.symbolstore) {
            this.symbolstore = new SymbolStore();
            this.CollectLabels();
        }

        options.input = this.GetInput();

        if (options.input === "") {
            return new Promise((resolve) => resolve(this.result));
        } else {
            return this.ExecDemangler(options).then((output) => this.ProcessOutput(output));
        }
    }
}

exports.Demangler = Demangler;
