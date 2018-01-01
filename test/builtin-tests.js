// Copyright (c) 2012-2018, Matt Godbolt & Rubén Rincón
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
    builtin = require('../lib/sources/builtin'),
    languages = require('../lib/languages').list,
    _ = require('underscore-node');

describe('Builtin sources', () => {
    it('Does not include default code', () => {
        builtin.list((_, list) => {
            list.forEach(example => {
                should.not.equal(example.name, "default");
            });
        });
    });
    it('Has a valid language listed', () => {
        builtin.list((_, list) => {
            list.forEach(example => {
                should.not.equal(languages[example.lang], undefined, `Builtin ${example.name} has unrecognised language ${example.lang}`);
            });
        });
    });
    it('Has at least one example for each language', () => {
        builtin.list((placeholder, list) => {
            _.each(languages, lang => {
                should.not.equal(_.find(list, elem => elem.lang === lang.id), undefined, `Language ${lang.name} does not have any builtins`);
            });
        });
    });
    it('Reports a string error if no example found', () => {
        builtin.load('BADLANG', 'BADFILE', (error, result) => {
            should.equal(result, undefined, 'A result should not be returned for bad requests');
            error.should.be.a("string");
        });
    });
    it('Reports something for every defined example', () => {
        builtin.list((placeholder, examples) => {
            examples.forEach(example => {
                builtin.load(example.lang, example.file, (error, result) => {
                    should.not.exist(error, `Can't read ${example.name} for ${example.lang} in ${example.file}`);
                    result.file.should.be.a('string');
                });
            });
        });
    });
});