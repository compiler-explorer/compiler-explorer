// Copyright (c) 2017, Compiler Explorer Authors
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

import express, {Express} from 'express';
import request from 'supertest';
import {beforeAll, describe, expect, it} from 'vitest';

import {CompileHandler, SetTestMode} from '../../lib/handlers/compile.js';
import {fakeProps} from '../../lib/properties.js';
import {ActiveTool, BypassCache, LegacyCompatibleActiveTool} from '../../types/compilation/compilation.interfaces.js';
import {makeCompilationEnvironment} from '../utils.js';

SetTestMode();

const languages = {
    a: {id: 'a', name: 'A lang'},
    b: {id: 'b', name: 'B lang'},
    d: {id: 'd', name: 'D lang'},
};

describe('Compiler tests', () => {
    let app: Express, compileHandler;

    beforeAll(() => {
        const compilationEnvironment = makeCompilationEnvironment({languages});
        compileHandler = new CompileHandler(compilationEnvironment, fakeProps({}));

        const textParser = express.text({type: () => true});
        const formParser = express.urlencoded({extended: false});

        app = express();
        app.use(express.json());

        app.post('/noscript/compile', formParser, compileHandler.handle.bind(compileHandler));
        app.post('/:compiler/compile', textParser, compileHandler.handle.bind(compileHandler));
        app.post('/:compiler/cmake', compileHandler.handleCmake.bind(compileHandler));
    });

    it('throws for unknown compilers', async () => {
        await request(app).post('/NOT_A_COMPILER/compile').expect(404);
    });

    describe('Noscript API', () => {
        it('supports compile', async () => {
            await compileHandler.setCompilers([
                {
                    compilerType: 'fake-for-test',
                    exe: 'fake',
                    fakeResult: {
                        code: 0,
                        stdout: [{text: 'Something from stdout'}],
                        stderr: [{text: 'Something from stderr'}],
                        asm: [{text: 'ASMASMASM'}],
                    },
                },
            ]);
            const res = await request(app)
                .post('/noscript/compile')
                .set('Content-Type', 'application/x-www-form-urlencoded')
                .send('compiler=fake-for-test&source=I am a program')
                .expect(200)
                .expect('Content-Type', /text/);
            expect(res.text).toContain('Something from stdout');
            expect(res.text).toContain('Something from stderr');
            expect(res.text).toContain('ASMASMASM');
        });
    });

    describe('Curl API', () => {
        it('supports compile', async () => {
            await compileHandler.setCompilers([
                {
                    compilerType: 'fake-for-test',
                    exe: 'fake',
                    fakeResult: {
                        code: 0,
                        stdout: [{text: 'Something from stdout'}],
                        stderr: [{text: 'Something from stderr'}],
                        asm: [{text: 'ASMASMASM'}],
                    },
                },
            ]);
            const res = await request(app)
                .post('/fake-for-test/compile')
                .set('Content-Type', 'application/x-www-form-urlencoded')
                .send('I am a program /* &compiler=NOT_A_COMPILER&source=etc */')
                .expect(200)
                .expect('Content-Type', /text/);
            expect(res.text).toContain('Something from stdout');
            expect(res.text).toContain('Something from stderr');
            expect(res.text).toContain('ASMASMASM');
        });

        it('supports alias compile', async () => {
            await compileHandler.setCompilers([
                {
                    id: 'newcompilerid',
                    alias: ['oldid1', 'oldid2'],
                    compilerType: 'fake-for-test',
                    exe: 'fake',
                    fakeResult: {
                        code: 0,
                        stdout: [{text: 'Something from stdout'}],
                        stderr: [{text: 'Something from stderr'}],
                        asm: [{text: 'ASMASMASM'}],
                    },
                },
            ]);
            const res = await request(app)
                .post('/oldid1/compile')
                .set('Content-Type', 'application/x-www-form-urlencoded')
                .send('I am a program /* &compiler=NOT_A_COMPILER&source=etc */')
                .expect(200)
                .expect('Content-Type', /text/);
            expect(res.text).toContain('Something from stdout');
            expect(res.text).toContain('Something from stderr');
            expect(res.text).toContain('ASMASMASM');
        });
    });

    async function setFakeResult(fakeResult?: any) {
        await compileHandler.setCompilers([
            {
                compilerType: 'fake-for-test',
                exe: 'fake',
                fakeResult: fakeResult || {},
            },
        ]);
    }

    describe('JSON API', () => {
        it('handles text output', async () => {
            await compileHandler.setCompilers([
                {
                    compilerType: 'fake-for-test',
                    exe: 'fake',
                    fakeResult: {
                        code: 0,
                        stdout: [{text: 'Something from stdout'}],
                        stderr: [{text: 'Something from stderr'}],
                        asm: [{text: 'ASMASMASM'}],
                    },
                },
            ]);
            const res = await request(app)
                .post('/fake-for-test/compile')
                .send({
                    options: {},
                    source: 'I am a program',
                })
                .expect(200)
                .expect('Content-Type', /text/);
            expect(res.text).toContain('Something from stdout');
            expect(res.text).toContain('Something from stderr');
            expect(res.text).toContain('ASMASMASM');
        });

        function makeFakeJson(source: string, options?: any) {
            return request(app)
                .post('/fake-for-test/compile')
                .set('Accept', 'application/json')
                .send({
                    options: options || {},
                    source: source || '',
                });
        }

        function makeFakeWithExtraFilesJson(source: string, options?: any, files?: any) {
            return request(app)
                .post('/fake-for-test/compile')
                .set('Accept', 'application/json')
                .send({
                    options: options || {},
                    source: source || '',
                    files: files || [],
                });
        }

        function makeFakeCmakeJson(source: string, options?: any, files?: any) {
            return request(app)
                .post('/fake-for-test/cmake')
                .set('Accept', 'application/json')
                .send({
                    options: options || {},
                    source: source || '',
                    files: files || [],
                });
        }

        it('handles JSON output', async () => {
            await setFakeResult({
                code: 0,
                stdout: [{text: 'Something from stdout'}],
                stderr: [{text: 'Something from stderr'}],
                asm: [{text: 'ASMASMASM'}],
            });
            await makeFakeJson('I am a program')
                .expect('Content-Type', /json/)
                .expect(200, {
                    asm: [{text: 'ASMASMASM'}],
                    code: 0,
                    input: {
                        backendOptions: {},
                        files: [],
                        filters: {},
                        options: [],
                        source: 'I am a program',
                        tools: [],
                    },
                    stderr: [{text: 'Something from stderr'}],
                    stdout: [{text: 'Something from stdout'}],
                });
        });

        it('parses options and filters', async () => {
            await setFakeResult();
            const res = await makeFakeJson('I am a program', {
                userArguments: '-O1 -monkey "badger badger"',
                filters: {a: true, b: true, c: true},
            })
                .expect('Content-Type', /json/)
                .expect(200);
            expect(res.body.input.options).toEqual(['-O1', '-monkey', 'badger badger']);
            expect(res.body.input.filters).toEqual({a: true, b: true, c: true});
        });

        it('parses tools with array args', async () => {
            await setFakeResult();
            const res = await makeFakeJson('I am a program', {
                tools: [{id: 'tool', args: ['one', 'two', 'and three'], stdin: ''} as ActiveTool],
            })
                .expect('Content-Type', /json/)
                .expect(200);
            expect(res.body.input.tools).toEqual([{id: 'tool', args: ['one', 'two', 'and three'], stdin: ''}]);
        });
        it('parses tools with string args', async () => {
            await setFakeResult();
            const res = await makeFakeJson('I am a program', {
                tools: [{id: 'tool', args: 'one two "and three" and string', stdin: ''} as LegacyCompatibleActiveTool],
            })
                .expect('Content-Type', /json/)
                .expect(200);
            expect(res.body.input.tools).toEqual([
                {
                    id: 'tool',
                    args: ['one', 'two', 'and three', 'and', 'string'],
                    stdin: '',
                },
            ]);
        });

        it('parses extra files', async () => {
            await setFakeResult();
            const res = await makeFakeWithExtraFilesJson(
                'I am a program',
                {
                    userArguments: '-O1 -monkey "badger badger"',
                    filters: {a: true, b: true, c: true},
                },
                [
                    {
                        filename: 'myresource.txt',
                        contents: 'Hello, World!\nHow are you?\n',
                    },
                ],
            )
                .expect('Content-Type', /json/)
                .expect(200);
            expect(res.body.input.options).toEqual(['-O1', '-monkey', 'badger badger']);
            expect(res.body.input.filters).toEqual({a: true, b: true, c: true});
            expect(res.body.input.files).toEqual([
                {
                    filename: 'myresource.txt',
                    contents: 'Hello, World!\nHow are you?\n',
                },
            ]);
        });

        it('cmakes', async () => {
            await setFakeResult();
            const res = await makeFakeCmakeJson(
                'I am a program',
                {
                    userArguments: '-O1 -monkey "badger badger"',
                    filters: {a: true, b: true, c: true},
                },
                [
                    {
                        filename: 'myresource.txt',
                        contents: 'Hello, World!\nHow are you?\n',
                    },
                ],
            )
                .expect('Content-Type', /json/)
                .expect(200);
            expect(res.body.input.options).toEqual({
                backendOptions: {},
                bypassCache: BypassCache.None,
                executeParameters: {
                    args: [],
                    runtimeTools: [],
                    stdin: '',
                },
                filters: {
                    a: true,
                    b: true,
                    c: true,
                },
                libraries: [],
                options: ['-O1', '-monkey', 'badger badger'],
                source: 'I am a program',
                tools: [],
            });
            expect(res.body.input.files).toEqual([
                {
                    filename: 'myresource.txt',
                    contents: 'Hello, World!\nHow are you?\n',
                },
            ]);
        });
    });

    describe('Query API', () => {
        function makeFakeQuery(source?: any, query?: any) {
            return request(app)
                .post('/fake-for-test/compile')
                .query(query || {})
                .set('Accept', 'application/json')
                .send(source || '');
        }

        it('error on empty request body', async () => {
            await setFakeResult();
            await request(app).post('/fake-for-test/compile').set('Accept', 'application/json').expect(500);
        });

        it('handles filters set directly', async () => {
            await setFakeResult();
            const res = await makeFakeQuery('source', {filters: 'a,b,c'}).expect('Content-Type', /json/).expect(200);
            expect(res.body.input.options).toEqual([]);
            expect(res.body.input.filters).toEqual({a: true, b: true, c: true});
        });

        it('handles filters added', async () => {
            await setFakeResult();
            const res = await makeFakeQuery('source', {filters: 'a', addFilters: 'e,f'})
                .expect('Content-Type', /json/)
                .expect(200);
            expect(res.body.input.options).toEqual([]);
            expect(res.body.input.filters).toEqual({a: true, e: true, f: true});
        });

        it('handles filters removed', async () => {
            await setFakeResult();
            const res = await makeFakeQuery('source', {filters: 'a,b,c', removeFilters: 'b,c,d'})
                .expect('Content-Type', /json/)
                .expect(200);
            expect(res.body.input.options).toEqual([]);
            expect(res.body.input.filters).toEqual({a: true});
        });

        it('handles filters added and removed', async () => {
            await setFakeResult();
            const res = await makeFakeQuery('source', {filters: 'a,b,c', addFilters: 'c,g,h', removeFilters: 'b,c,d,h'})
                .expect('Content-Type', /json/)
                .expect(200);
            expect(res.body.input.options).toEqual([]);
            expect(res.body.input.filters).toEqual({a: true, g: true});
        });
    });

    describe('Multi language', () => {
        async function setFakeCompilers() {
            await compileHandler.setCompilers([
                {
                    compilerType: 'fake-for-test',
                    id: 'a',
                    lang: 'a',
                    exe: 'fake',
                    fakeResult: {code: 0, stdout: [], stderr: [], asm: [{text: 'LANG A'}]},
                },
                {
                    compilerType: 'fake-for-test',
                    id: 'b',
                    lang: 'b',
                    exe: 'fake',
                    fakeResult: {code: 0, stdout: [], stderr: [], asm: [{text: 'LANG B'}]},
                },
                {
                    compilerType: 'fake-for-test',
                    id: 'a',
                    lang: 'b',
                    exe: 'fake',
                    fakeResult: {code: 0, stdout: [], stderr: [], asm: [{text: 'LANG B but A'}]},
                },
            ]);
        }

        function makeFakeJson(compiler: string, lang: any) {
            return request(app).post(`/${compiler}/compile`).set('Accept', 'application/json').send({
                lang: lang,
                options: {},
                source: '',
            });
        }

        it('finds without language', async () => {
            await setFakeCompilers();
            const res = await makeFakeJson('b', {}).expect('Content-Type', /json/).expect(200);
            expect(res.body.asm).toEqual([{text: 'LANG B'}]);
        });

        it('disambiguates by language, choosing A', async () => {
            await setFakeCompilers();
            const res = await makeFakeJson('b', 'a').expect('Content-Type', /json/).expect(200);
            expect(res.body.asm).toEqual([{text: 'LANG B'}]);
        });

        it('disambiguates by language, choosing B', async () => {
            await setFakeCompilers();
            const res = await makeFakeJson('a', 'b');
            expect(res.body.asm).toEqual([{text: 'LANG B but A'}]);
        });
    });
});
