// Copyright (c) 2021, Compiler Explorer Authors
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

import {RouteAPI} from '../lib/handlers/route-api';

import {fs} from './utils';

describe('Basic unfurls', () => {
    const router = null;
    let config;
    let routeApi;

    before(() => {
        config = {
            ceProps: () => {},
            clientOptionsHandler: {
                options: {
                    urlShortenService: 'memory',
                },
            },
            storageHandler: {
                expandId: async id => {
                    const json = await fs.readFile('test/state/' + id + '.json');
                    return {
                        config: json,
                    };
                },
                incrementViewCount: async () => {},
            },
        };
    });

    it('Too many editors to meta', async () => {
        const prom = new Promise((resolve, reject) => {
            config.renderGoldenLayout = (config, metadata) => {
                resolve({metadata});
            };

            routeApi = new RouteAPI(router, config);
            routeApi.apiHandler = {
                languages: {
                    'c++': {
                        name: 'C++',
                    },
                },
                compilers: [],
            };
            routeApi.storedStateHandler({params: {id: '../example-states/default-state'}}, null, () => {
                reject('Error in test');
            });
        });

        const res = await prom;
        res.metadata.should.deep.equal({
            ogAuthor: null,
            ogDescription: '',
            ogTitle: 'Compiler Explorer',
        });
    });

    it('Just one editor', async () => {
        const prom = new Promise((resolve, reject) => {
            config.renderGoldenLayout = (config, metadata) => {
                resolve({metadata});
            };

            routeApi = new RouteAPI(router, config);
            routeApi.apiHandler = {
                languages: {
                    'c++': {
                        name: 'C++',
                    },
                },
                compilers: [],
            };
            routeApi.storedStateHandler({params: {id: 'andthekitchensink'}}, null, () => {
                reject('Error in test');
            });
        });

        const res = await prom;
        res.metadata.should.deep.equal({
            ogAuthor: null,
            ogDescription:
                '\ntemplate&lt;typename T&gt;\nconcept TheSameAndAddable = requires(T a, T b) {\n    {a+b} -&gt; T;\n};\n\ntemplate&lt;TheSameAndAddable T&gt;\nT sum(T x, T y) {\n    return x + y;\n}\n\n#include &lt;string&gt;\n\nint main() {\n    int z = 0;\n    int w;\n\n    return sum(z, w);\n}\n',
            ogTitle: 'Compiler Explorer - C++',
        });
    });

    it('Tree things', async () => {
        const prom = new Promise((resolve, reject) => {
            config.renderGoldenLayout = (config, metadata) => {
                resolve({metadata});
            };

            routeApi = new RouteAPI(router, config);
            routeApi.apiHandler = {
                languages: {
                    'c++': {
                        name: 'C++',
                    },
                },
                compilers: [],
            };
            routeApi.storedStateHandler({params: {id: 'tree-gl'}}, null, () => {
                reject('Error in test');
            });
        });

        const res = await prom;
        res.metadata.should.deep.equal({
            ogAuthor: null,
            ogDescription: 'project(hello)\n\nadd_executable(output.s\n    example.cpp\n    square.cpp)\n',
            ogTitle: 'Compiler Explorer - C++',
        });
    });
});
