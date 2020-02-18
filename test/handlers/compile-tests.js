// Copyright (c) 2017, Matt Godbolt
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

require('../../lib/handlers/compile').SetTestMode();

const chai = require('chai'),
    CompilationEnvironment = require('../../lib/compilation-env'),
    CompileHandler = require('../../lib/handlers/compile').Handler,
    express = require('express'),
    bodyParser = require('body-parser'),
    properties = require('../../lib/properties');
chai.use(require("chai-http"));
chai.should();

const languages = {
    a: {id: 'a'},
    b: {id: 'b'},
    d: {id: 'd'}
};


describe('Compiler tests', () => {
    let app, compileHandler;

    before(() => {
        const compilerProps = new properties.CompilerProps(languages, properties.fakeProps({}));
        const compilationEnvironment = new CompilationEnvironment(compilerProps);
        compileHandler = new CompileHandler(compilationEnvironment);

        app = express();
        app.use(bodyParser.json()).use(bodyParser.text());
        app.post('/:compiler/compile', compileHandler.handle.bind(compileHandler));
    });

    it('throws for unknown compilers', () => {
        return chai.request(app)
            .post('/NOT_A_COMPILER/compile')
            .then((res) => {
                res.should.have.status(404);
            });
    });

    describe('JSON API', () => {
        it('handles text output', () => {
            return compileHandler.setCompilers([{
                compilerType: "fake-for-test",
                exe: "fake",
                fakeResult: {
                    code: 0,
                    stdout: [{text: "Something from stdout"}],
                    stderr: [{text: "Something from stderr"}],
                    asm: [{text: "ASMASMASM"}]
                }
            }]).then(() => {
                return chai.request(app)
                    .post('/fake-for-test/compile')
                    .send({
                        options: '',
                        source: 'I am a program'
                    })
                    .then(res => {
                        res.should.have.status(200);
                        res.should.be.text;
                        res.text.should.contain("Something from stdout");
                        res.text.should.contain("Something from stderr");
                        res.text.should.contain("ASMASMASM");
                    })
                    .catch(err => {
                        throw err;
                    });
            });
        });

        function makeFakeJson(source, options, fakeResult) {
            return compileHandler.setCompilers([{
                compilerType: "fake-for-test",
                exe: "fake",
                fakeResult: fakeResult || {}
            }])
                .then(() => chai.request(app)
                    .post('/fake-for-test/compile')
                    .set('Accept', 'application/json')
                    .send({
                        options: options || {},
                        source: source || ''
                    }));
        }

        it('handles JSON output', () => {
            return makeFakeJson('I am a program', {}, {
                code: 0,
                stdout: [{text: "Something from stdout"}],
                stderr: [{text: "Something from stderr"}],
                asm: [{text: "ASMASMASM"}]
            }).then(res => {
                res.should.have.status(200);
                res.should.be.json;
                res.body.should.deep.equals({
                    asm: [{text: "ASMASMASM"}],
                    code: 0,
                    input: {
                        filters: [],
                        options: [],
                        source: "I am a program"
                    },
                    stderr: [{text: "Something from stderr"}],
                    stdout: [{text: "Something from stdout"}]
                });
            })
                .catch(err => {
                    throw err;
                });
        });

        it('parses options and filters', () => {
            return makeFakeJson('I am a program', {
                userArguments: '-O1 -monkey "badger badger"',
                filters: {a: true, b: true, c: true}
            })
                .then(res => {
                    res.should.have.status(200);
                    res.should.be.json;
                    res.body.input.options.should.deep.equals(['-O1', '-monkey', 'badger badger']);
                    res.body.input.filters.should.deep.equals({a: true, b: true, c: true});
                });
        });
    });

    describe('Query API', () => {
        function makeFakeQuery(source, query, fakeResult) {
            return compileHandler.setCompilers([{
                compilerType: "fake-for-test",
                exe: "fake",
                fakeResult: fakeResult || {}
            }])
                .then(() => chai.request(app)
                    .post('/fake-for-test/compile')
                    .query(query || {})
                    .set('Accept', 'application/json')
                    .send(source || ""));
        }

        it('handles filters set directly', () => {
            return makeFakeQuery("source", {filters: 'a,b,c'})
                .then(res => {
                    res.should.have.status(200);
                    res.should.be.json;
                    res.body.input.options.should.deep.equals([]);
                    res.body.input.filters.should.deep.equals({a: true, b: true, c: true});
                })
                .catch(err => {
                    throw err;
                });
        });

        it('handles filters added', () => {
            return makeFakeQuery("source", {filters: 'a', addFilters: 'e,f'})
                .then(res => {
                    res.should.have.status(200);
                    res.should.be.json;
                    res.body.input.options.should.deep.equals([]);
                    res.body.input.filters.should.deep.equals({a: true, e: true, f: true});
                })
                .catch(err => {
                    throw err;
                });
        });
        it('handles filters removed', () => {
            return makeFakeQuery("source", {filters: 'a,b,c', removeFilters: 'b,c,d'})
                .then(res => {
                    res.should.have.status(200);
                    res.should.be.json;
                    res.body.input.options.should.deep.equals([]);
                    res.body.input.filters.should.deep.equals({a: true});
                })
                .catch(err => {
                    throw err;
                });
        });
        it('handles filters added and removed', () => {
            return makeFakeQuery("source", {filters: 'a,b,c', addFilters: 'c,g,h', removeFilters: 'b,c,d,h'})
                .then(res => {
                    res.should.have.status(200);
                    res.should.be.json;
                    res.body.input.options.should.deep.equals([]);
                    res.body.input.filters.should.deep.equals({a: true, g: true});
                })
                .catch(err => {
                    throw err;
                });
        });
    });

    describe('Multi language', () => {
        function makeFakeJson(compiler, lang) {
            return compileHandler.setCompilers([
                {
                    compilerType: "fake-for-test",
                    id: 'a',
                    lang: 'a',
                    exe: "fake",
                    fakeResult: {code: 0, stdout: [], stderr: [], asm: [{text: "LANG A"}]}
                },
                {
                    compilerType: "fake-for-test",
                    id: 'b',
                    lang: 'b',
                    exe: "fake",
                    fakeResult: {code: 0, stdout: [], stderr: [], asm: [{text: "LANG B"}]}
                },
                {
                    compilerType: "fake-for-test",
                    id: 'a',
                    lang: 'b',
                    exe: "fake",
                    fakeResult: {code: 0, stdout: [], stderr: [], asm: [{text: "LANG B but A"}]}
                }
            ])
                .then(() => chai.request(app)
                    .post(`/${compiler}/compile`)
                    .set('Accept', 'application/json')
                    .send({
                        lang: lang,
                        options: {},
                        source: ''
                    }));
        }

        it('finds without language', () => {
            return makeFakeJson("b", {})
                .then(res => {
                    res.should.have.status(200);
                    res.should.be.json;
                    res.body.asm.should.deep.equals([{text: "LANG B"}]);
                })
                .catch(err => {
                    throw err;
                });
        });

        it('disambiguates by language, choosing A', () => {
            return makeFakeJson("a", "a")
                .then(res => {
                    res.should.have.status(200);
                    res.should.be.json;
                    res.body.asm.should.deep.equals([{text: "LANG A"}]);
                })
                .catch(err => {
                    throw err;
                });
        });

        it('disambiguates by language, choosing B', () => {
            return makeFakeJson("a", "b")
                .then(res => {
                    res.should.have.status(200);
                    res.should.be.json;
                    res.body.asm.should.deep.equals([{text: "LANG B but A"}]);
                })
                .catch(err => {
                    throw err;
                });
        });
    });
});
