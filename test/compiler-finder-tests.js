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

const chai = require('chai');
const chaiAsPromised = require("chai-as-promised");
const CompilerFinder = require('../lib/compiler-finder');
const _ = require('underscore');

chai.use(chaiAsPromised);
const should = chai.should();

const props = (key, deflt) => deflt;
const propsL = (lang, key, def) => props(key, def);
const propsAT = (langs, f, key, def) => {
    let response = {};
    _.each(langs, lang => {
        response[langs.id] = f(propsL(lang.id, key, def));
    });
    return response;
};

describe('Compiler-finder', function () {
    it('should not hang for undefined groups (Bug #860)', () => {
        // Contrived setup. I know
        const tweakedAT = (langs, f, key, def) => {
            let response = {};
            _.each(langs, lang => {
                let val = null;
                switch (key) {
                    case "compilers":
                        val = "goodCompiler:&badCompiler";
                        break;
                    default:
                        val = propsL(lang.id, key, def);
                        break;
                }
                response[langs.id] = f(val);
            });
            return response;
        };
        const finder = new CompilerFinder({}, propsL, tweakedAT, props, props, {'a-lang': {id:'a-lang'}}, {});
        return Promise.all(finder.getCompilers()).should.eventually.have.lengthOf(2);
        //return finder.recurseGetCompilers("lang", "", propsL).then();
    })
});
