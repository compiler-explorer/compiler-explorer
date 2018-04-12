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
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ,
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

const chai = require('chai');
const CompilerFinder = require('../lib/compiler-finder');
const properties = require('../lib/properties');
const fakeLangs = require('./fake-langs');
const _ = require('underscore-node');
class CompileHandler {
    constructor() {
        this.compilers = [];
    }

    setCompilers(compilers) {
        this.compilers = compilers;
    }
}

chai.should();

properties.initialize(`test/example-config/`, ['from-path', 'empty-groups']);

function makeCompilerFinder() {
    let compilerPropsFuncsL = {};
    _.each(fakeLangs, lang => compilerPropsFuncsL[lang.id] = properties.propsFor(lang.id));

    const fakeProps = properties.propsFor('fake-lang');
    const compilerPropsL = (lang, prop, def) => {
        const forLanguage = compilerPropsFuncsL[lang];
        if (forLanguage) {
            const forCompiler = forLanguage(prop);
            if (forCompiler !== undefined) return forCompiler;
        }
        return def;
    };
    const compilerPropsAT = (langs, fn, prop, def) => {
        let forLanguages = {};
        _.each(langs, lang => {
            forLanguages[lang.id] = fn(compilerPropsL(lang.id, prop, def), lang);
        });
        return forLanguages;
    };
    let compileHandler = new CompileHandler;
    return new CompilerFinder(compileHandler, compilerPropsL, compilerPropsAT, fakeProps, fakeProps, fakeLangs, {});
}

describe('Compiler finder', () => {
    it('should be able to find compilers from a path', () => {
        let finder = makeCompilerFinder();
        finder.find().then(() => {
            finder.compileHandler.compilers.should.have.lengthOf(2);
        });
    });
    it('should not hang for empty groups', () => {
        let finder = makeCompilerFinder();
        finder.find().then(() => {
            finder.compileHandler.compilers.should.be.empty;
        });
    });
});
