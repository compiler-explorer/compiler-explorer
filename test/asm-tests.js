// Copyright (c) 2018, Matt Godbolt
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

const AsmParser = require('../lib/asm-parser-vc');
const AsmRegex = require('../lib/asmregex').AsmRegex;
require('chai').should();

describe('ASM CL parser', () => {
    it('should work for error documents', () => {
        const parser = new AsmParser();
        const result = parser.process("<Compilation failed>", {
            directives: true
        });

        result.asm.should.deep.equal([{
            "source": null,
            "text": "<Compilation failed>"
        }]);
    });
});


describe('ASM regex base class', () => {
    it('should leave unfiltered lines alone', () => {
        const line = "     this    is    a line";
        AsmRegex.filterAsmLine(line, {}).should.equal(line);
    });
    it('should use up internal whitespace when asked', () => {
        AsmRegex.filterAsmLine("     this    is    a line", {trim: true}).should.equal("  this is a line");
        AsmRegex.filterAsmLine("this    is    a line", {trim: true}).should.equal("this is a line");
    });
    it('should keep whitespace in strings', () => {
        AsmRegex.filterAsmLine('equs     "this    string"', {trim: true}).should.equal('equs "this    string"');
        AsmRegex.filterAsmLine('     equs     "this    string"', {trim: true}).should.equal('  equs "this    string"');
        AsmRegex.filterAsmLine('equs     "this    \\"  string  \\""', {trim: true}).should.equal('equs "this    \\"  string  \\""');
    });
    it('should not get upset by mismatched strings', () => {
        AsmRegex.filterAsmLine("a   \"string    'yeah", {trim: true}).should.equal("a \"string 'yeah");
    });
});