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

const
    SymbolStore = require('./symbol-store').SymbolStore,
    _ = require('underscore-node'),
    utils = require('../lib/utils'),
    exec = require('../lib/exec'),
    logger = require('../lib/logger').logger;

class Demangler {
    /**
     * 
     * @param {string} demanglerExe 
     * @param {SymbolStore} symbolstore 
     */
    constructor(demanglerExe, symbolstore) {
        this.demanglerExe = demanglerExe;
        this.symbolstore = symbolstore;
        this.result = {};
    }

    CollectLabels() {
        const labelDef = /^([.a-zA-Z_$][a-zA-Z0-9$_.]*):/;

        for (var j = 0; j < this.result.asm.length; ++j) {
            const line = this.result.asm[j].text;
            const matches = line.match(labelDef);
            if (matches) {
                this.symbolstore.Add(matches[1], matches[1]);
            }
        }
    }

    ProcessNew(result) {

        // for (var j = 0; j < result.asm.length; ++j)
        //     result.asm[j].text = demangleIfNeeded(result.asm[j].text);

    }

    GetInput() {
        return _.pluck(this.result.asm, 'text').join("\n");

        //return this.symbolstore.ListSymbols().join("\n");
    }

    ProcessOutput(output) {
        const lines = utils.splitLines(output.stdout);
        for (let i = 0; i < this.result.asm.length; ++i)
            this.result.asm[i].text = lines[i];
        return this.result;
    }

    Process(result) {
        this.result = result;

        if (!this.symbolstore) {
            this.symbolstore = new SymbolStore();
            this.CollectLabels();
        }

        const options = {input: this.GetInput()};
        if (options.input.length == 0) return this.result;

        return exec.execute(
            this.demanglerExe,
            [],
            options
        ).then((output) => this.ProcessOutput(output));
    }
}

exports.Demangler = Demangler;
