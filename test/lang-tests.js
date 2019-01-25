// Copyright (c) 2017, Matt Godbolt & Rubén Rincón
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

const should = require('chai').should(),
    languages = require('../lib/languages').list,
    fs = require('fs-extra'),
    path = require('path');


describe('Language definitions tests', () => {
    it('Has id equal to object key', () => {
        Object.keys(languages).forEach(languageKey => should.equal(languages[languageKey].id, languageKey));
    });
    it('Has extensions with leading dots', () => {
        Object.keys(languages).forEach(languageKey => should.equal(languages[languageKey].extensions[0][0], '.'));
    });
    it('Has examples & are initialized', () => {
        Object.keys(languages).forEach(languageKey => {
            const lang = languages[languageKey];
            const example = fs.readFileSync(path.join('examples', lang.id, 'default' + lang.extensions[0]), 'utf-8');
            should.equal(example, lang.example);
        });
    });
});
