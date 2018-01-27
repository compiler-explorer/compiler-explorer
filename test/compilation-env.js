// Copyright (c) 2012-2018, Matt Godbolt
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
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ,
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

const chai = require('chai');
const CompilationEnvironment = require('../lib/compilation-env');

const should = chai.should();

const props = (key, deflt) => {
    switch (key) {
        case 'optionsWhitelistRe':
            return '.*';
        case 'optionsBlacklistRe':
            return '^(-W[alp],)?((-wrapper|-fplugin.*|-specs|-load|-plugin|(@.*)|-I|-i)(=.*)?|--)$';
        case 'cacheMb':
            return 10;
    }
    return deflt;
};

describe('Compilation environment', () => {
    const ce = new CompilationEnvironment(props);
    it('Should cache', () => {
        should.not.exist(ce.cacheGet('foo'));
        ce.cachePut('foo', 'bar');
        ce.cacheGet('foo').should.equal('bar');
        should.not.exist(ce.cacheGet('baz'));
    });
    it('Should filter bad options', () => {
        ce.findBadOptions(['-O3', '-flto']).should.be.empty;
        ce.findBadOptions(['-O3', '-plugin']).should.eql(['-plugin']);
    });
});
