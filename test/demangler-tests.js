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

const
    chai = require('chai'),
    chaiAsPromised = require('chai-as-promised'),
    SymbolStore = require('../lib/symbol-store').SymbolStore,
    Demangler = require('../lib/demangler').Demangler,
    logger = require('../lib/logger').logger;

chai.use(chaiAsPromised);
const should = chai.should();
const expect = chai.expect;
const assert = chai.assert;

describe('Basic demangling', function () {
    it('One line of asm', function () {
        const result = {};
        result.asm = [{"text": "Hello, World!"}];

        const demangler = new Demangler("c++filt");

        return Promise.all([
            demangler.Process(result).then((output) => {
                output.asm[0].text.should.equal("Hello, World!");
            })
        ]);
    });

    it('One label and some asm', function () {
        const result = {};
        result.asm = [
            {"text": "_Z6squarei:"},
            {"text": "  ret"}
        ];

        const demangler = new Demangler("c++filt");

        return Promise.all([
            demangler.Process(result).then((output) => {
                output.asm[0].text.should.equal("square(int):");
                output.asm[1].text.should.equal("  ret");
            })
        ]);
    });
});
