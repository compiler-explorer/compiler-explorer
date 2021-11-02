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

import './utils';
import { CompilerFinder } from '../lib/compiler-finder';
import * as properties from '../lib/properties';

const languages = {
    'a-lang': {
        id: 'a-lang',
    },
};

const libs = {
    'a-lang': {
        fmt: {
            versions: {
                trunk: {
                    version: '(trunk)',
                    libPath: '/fmt/trunk/lib',
                },
            },
        },
        catch2: {
            versions: {
                2101: {
                    version: '2.1.0.1',
                    libPath: '/catch2/2.1.0.1/lib/x86_64',
                },
                2102: {
                    version: '2.1.0.2',
                    libPath: '/catch2/2.1.0.2/lib/x86_64',
                },
            },
        },
    },
};

const props = {
    compilers: 'goodCompiler:&badCompiler',
};

const noOptionsAtAll = {
    compilers: 'goodCompiler',
};

const noBaseOptions = {
    compilers: 'goodCompiler',
    options: 'bar',
};

const onlyBaseOptions = {
    compilers: 'goodCompiler',
    baseOptions: 'foo',
};

const bothOptions = {
    compilers: 'goodCompiler',
    baseOptions: 'foo',
    options: 'bar',
};

const supportsLibrariesOptions = {
    compilers: 'goodCompiler',
    supportsLibraries: 'fmt:catch2.2101',
};

describe('Compiler-finder', function () {
    let compilerProps;

    let noOptionsAtAllProps;
    let noBaseOptionsProps;
    let onlyBaseOptionsProps;
    let bothOptionsProps;
    let libraryCompilerProps;

    let optionsHandler;

    before(() => {
        compilerProps = new properties.CompilerProps(languages, properties.fakeProps(props));

        noOptionsAtAllProps = new properties.CompilerProps(languages, properties.fakeProps(noOptionsAtAll));
        noBaseOptionsProps = new properties.CompilerProps(languages, properties.fakeProps(noBaseOptions));
        onlyBaseOptionsProps = new properties.CompilerProps(languages, properties.fakeProps(onlyBaseOptions));
        bothOptionsProps = new properties.CompilerProps(languages, properties.fakeProps(bothOptions));

        libraryCompilerProps = new properties.CompilerProps(languages, properties.fakeProps(supportsLibrariesOptions));

        optionsHandler = {
            get: () => {
                return {
                    libs: libs,
                    tools: {},
                };
            },
        };
    });

    it('should not hang for undefined groups (Bug #860)', () => {

        const finder = new CompilerFinder({}, compilerProps, properties.fakeProps({}), {}, optionsHandler);
        return finder.getCompilers().should.eventually.have.lengthOf(2);
    });

    it('should behave properly if no options are provided at all', async () => {
        const finder = new CompilerFinder({}, noOptionsAtAllProps, properties.fakeProps({}), {}, optionsHandler);
        const compilers = await finder.getCompilers();
        compilers[0].options.should.equal('');
    });

    it('should behave properly if no base options are provided', async () => {
        const finder = new CompilerFinder({}, noBaseOptionsProps, properties.fakeProps({}), {}, optionsHandler);
        const compilers = await finder.getCompilers();
        compilers[0].options.should.equal('bar');
    });

    it('should behave properly if only base options are provided', async () => {
        const finder = new CompilerFinder({}, onlyBaseOptionsProps, properties.fakeProps({}), {}, optionsHandler);
        const compilers = await finder.getCompilers();
        compilers[0].options.should.equal('foo');
    });

    it('should behave properly if both options are provided', async () => {
        const finder = new CompilerFinder({}, bothOptionsProps, properties.fakeProps({}), {}, optionsHandler);
        const compilers = await finder.getCompilers();
        compilers[0].options.should.equal('foo bar');
    });

    it('should be able to filter libraries', async () => {
        const finder = new CompilerFinder({}, libraryCompilerProps, properties.fakeProps({}), {}, optionsHandler);
        const compilers = await finder.getCompilers();
        const libsArr = compilers[0].libsArr;
        libsArr.should.deep.equal(['fmt', 'catch2.2101']);
    });
});
