// Copyright (c) 2023, Compiler Explorer Authors
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

import {addDigitSeparator, escapeHTML} from '../shared/common-utils.js';

describe('HTML Escape Test Cases', () => {
    it('should prevent basic injection', () => {
        escapeHTML("<script>alert('hi');</script>").should.equal(`&lt;script&gt;alert(&#x27;hi&#x27;);&lt;/script&gt;`);
    });
    it('should prevent tag injection', () => {
        escapeHTML('\'"`>').should.equal(`&#x27;&quot;&#x60;&gt;`);
    });
});

describe('digit separator', () => {
    it('handles short numbers', () => {
        addDigitSeparator('42', '_', 3).should.equal('42');
    });
    it('handles long numbers', () => {
        addDigitSeparator('1234', '_', 3).should.equal('1_234');
        addDigitSeparator('123456789', "'", 3).should.equal("123'456'789");
        addDigitSeparator('1234567890', "'", 3).should.equal("1'234'567'890");
    });
    it('handles hex numbers', () => {
        addDigitSeparator('AABBCCDD12345678', '_', 4).should.equal('AABB_CCDD_1234_5678');
        addDigitSeparator('01AABBCCDD12345678', '_', 4).should.equal('01_AABB_CCDD_1234_5678');
    });
    it('handles negative numbers', () => {
        addDigitSeparator('-42', '_', 3).should.equal('-42');
        addDigitSeparator('-420', '_', 3).should.equal('-420');
        addDigitSeparator('-4200', '_', 3).should.equal('-4_200');
        addDigitSeparator('-123456789', '_', 3).should.equal('-123_456_789');
    });
});
