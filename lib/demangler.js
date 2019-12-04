// Copyright (c) 2018, Compiler Explorer Authors
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
    logger = require('./logger').logger,
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
        this.callPtrDef1 = /call[q]?.*ptr\s\[[a-z]*\s\+\s([.a-z_$][a-z0-9$_@.]*)\]/i;
        this.callPtrDef2 = /call[q]?\s+([*.a-z_$][a-z0-9$_@.]*)/i;
        this.callPtrDef3 = /call[q]?.*\[qword ptr\s([.a-z_$][a-z0-9$_@.]*).*\]/i;
        this.callPtrDef4 = /call[q]?.*qword\sptr\s\[[a-z]*\s\+\s([.a-z_$][a-z0-9$_@.]*)\+?[0-9]?\]/i;

        // symbols in a mov or lea command starting with an underscore
        this.movUnderscoreDef = /mov.*\s(_[a-z0-9$_@.]*)/i;
        this.leaUnderscoreDef = /lea.*\s(_[a-z0-9$_@.]*)/i;
        this.quadUnderscoreDef = /\.quad\s*(_[a-z0-9$_@.]*)/i;
    }

    // Iterates over the labels, demangle the label names and updates the start and
    // end position of the label.
    demangleLabels(labels, value, newValue) {
        if (!labels || !labels.length) return;

        labels.forEach((label, index) => {
            if (label.name === value) {
                label.name = newValue;
                label.range.endCol = label.range.startCol + newValue.length;

                // Update the startCol value for each further labels.
                for (let j = index + 1; j < labels.length; j++) {
                    labels[j].range.startCol += newValue.length - value.length;
                }
            }
        });
    }

    demangleLabelDefinitions(labelDefinitions, translations) {
        if (!labelDefinitions) return;

        for (const [oldValue, newValue] of translations) {
            const value = labelDefinitions[oldValue];
            if (value && oldValue !== newValue) {
                labelDefinitions[newValue] = value;
                delete labelDefinitions[oldValue];
            }
        }
    }

    addMatchToOtherSymbols(matches) {
        if (!matches) return false;

        const midx = matches.length - 1;
        this.othersymbols.add(matches[midx], matches[midx]);

        return true;
    }

    collectLabels() {
        for (let j = 0; j < this.result.asm.length; ++j) {
            const line = this.result.asm[j].text;

            let matches = line.match(this.labelDef);
            if (matches) {
                const midx = matches.length - 1;
                this.symbolstore.add(matches[midx], matches[midx]);
            }

            if (this.addMatchToOtherSymbols(line.match(this.jumpDef))) continue;
            if (this.addMatchToOtherSymbols(line.match(this.callPtrDef4))) continue;
            if (this.addMatchToOtherSymbols(line.match(this.callPtrDef3))) continue;
            if (this.addMatchToOtherSymbols(line.match(this.callPtrDef2))) continue;
            if (this.addMatchToOtherSymbols(line.match(this.callPtrDef1))) continue;
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
        if (lines.length > this.input.length) {
            logger.error(`Demangler output issue ${lines.length} > ${this.input.length}`,
                this.input, output);
            throw new Error("Internal issue in demangler");
        }
        for (let i = 0; i < lines.length; ++i)
            this.addTranslation(this.input[i], lines[i]);

        let translations = [];
        translations = translations.concat(this.symbolstore.listTranslations());
        translations = translations.concat(this.othersymbols.listTranslations());

        for (let i = 0; i < this.result.asm.length; ++i) {
            let line = this.result.asm[i].text;
            for (let j = 0; j < translations.length; ++j) {
                const oldValue = translations[j][0];
                const newValue = translations[j][1];

                line = line.replace(oldValue, newValue);
                line = line.replace(oldValue, newValue);

                this.demangleLabels(this.result.asm[i].labels,
                    oldValue, newValue);
            }
            this.result.asm[i].text = line;
        }

        this.demangleLabelDefinitions(
            this.result.labelDefinitions, translations);

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
