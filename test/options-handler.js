// Copyright (c) 2018, Compiler Explorer Team
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
const OptionsHandler = require('../lib/options-handler');
const _ = require('underscore-node');

chai.use(chaiAsPromised);
const should = chai.should();

const props = (key, def) => def || "";

const optionsHandler = new OptionsHandler([], ['test'], props, {compilerPropsA: props, compilerPropsL: props}, {});

const makeFakeCompiler = (id, lang, group, semver, isSemver) => {
    return {
        id: id,
        exe: '/dev/null',
        name: id,
        lang: lang,
        group: group,
        isSemVer: isSemver,
        semver: semver
    };
};

describe('Options handler', () => {
    it('should order libraries as expected', () => {
        const compilers = [
            makeFakeCompiler('a1', 'fake', 'a', '0.0.1', true),
            makeFakeCompiler('a2', 'fake', 'a', '0.2.0', true),
            makeFakeCompiler('a3', 'fake', 'a', '0.2.1', true),

            makeFakeCompiler('b1', 'fake', 'b', 'trunk', true),
            makeFakeCompiler('b2', 'fake', 'b', '1.0.0', true),
            makeFakeCompiler('b3', 'fake', 'b', '0.5.0', true),

            makeFakeCompiler('c1', 'fake', 'c', '3.0.0', true),
            makeFakeCompiler('c2', 'fake', 'c', '3.0.0', true),
            makeFakeCompiler('c3', 'fake', 'c', '3.0.0', true),

            makeFakeCompiler('d1', 'fake', 'd', 1.0, true),
            makeFakeCompiler('d2', 'fake', 'd', '2.0.0', true),
            makeFakeCompiler('d3', 'fake', 'd', '0.0.5', true),

            makeFakeCompiler('e1', 'fake', 'e', '0..0', false),
            makeFakeCompiler('e2', 'fake', 'e', undefined, false)
        ];
        const expectedOrder = {
            a: {
                a1: -0,
                a2: -1,
                a3: -2
            },
            b: {
                b1: -2,
                b2: -1,
                b3: -0
            },
            c: {
                c1: -0,
                c2: -1,
                c3: -2
            },
            d: {
                d1: -2,
                d2: -1,
                d3: -0
            },
            e: {
                e1: undefined,
                e2: undefined
            }
        };
        optionsHandler.setCompilers(compilers);
        _.each(optionsHandler.get().compilers, compiler => {
            should.equal(compiler['$order'], expectedOrder[compiler.group][compiler.id]);
        });
    });
});
