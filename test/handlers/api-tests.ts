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

import express from 'express';
import request from 'supertest';
import {beforeAll, describe, expect, it} from 'vitest';

import {CompilationEnvironment} from '../../lib/compilation-env.js';
import {ApiHandler} from '../../lib/handlers/api.js';
import {CompileHandler} from '../../lib/handlers/compile.js';
import {CompilerProps, fakeProps} from '../../lib/properties.js';
import {StorageNull} from '../../lib/storage/index.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import {Language, LanguageKey} from '../../types/languages.interfaces.js';
import {makeFakeCompilerInfo, makeFakeLanguage} from '../utils.js';

const languages: Partial<Record<LanguageKey, Language>> = {
    'c++': makeFakeLanguage({
        id: 'c++',
        name: 'C++',
        monaco: 'cppp',
        extensions: ['.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c'],
    }),
    haskell: makeFakeLanguage({
        id: 'haskell',
        name: 'Haskell',
        monaco: 'haskell',
        extensions: ['.hs', '.haskell'],
    }),
    pascal: makeFakeLanguage({
        id: 'pascal',
        name: 'Pascal',
        monaco: 'pascal',
        extensions: ['.pas'],
    }),
};
const compilers: CompilerInfo[] = [
    makeFakeCompilerInfo({
        id: 'gcc900',
        name: 'GCC 9.0.0',
        lang: 'c++',
    }),
    makeFakeCompilerInfo({
        id: 'fpc302',
        name: 'FPC 3.0.2',
        lang: 'pascal',
    }),
    makeFakeCompilerInfo({
        id: 'clangtrunk',
        name: 'Clang trunk',
        lang: 'c++',
    }),
];

describe('API handling', () => {
    let app;

    beforeAll(() => {
        app = express();
        const apiHandler = new ApiHandler(
            {
                handle: res => res.send('compile'),
                handleCmake: res => res.send('cmake'),
                handlePopularArguments: res => res.send('ok'),
                handleOptimizationArguments: res => res.send('ok'),
            } as unknown as CompileHandler, // TODO(mrg) ideally fake this out or make it a higher-level interface
            fakeProps({
                formatters: 'formatt:badformatt',
                'formatter.formatt.exe': process.platform === 'win32' ? 'cmd' : 'echo',
                'formatter.formatt.type': 'clangformat',
                'formatter.formatt.version': 'Release',
                'formatter.formatt.name': 'FormatT',
            }),
            new StorageNull('/', new CompilerProps(languages, fakeProps({}))),
            'default',
            {ceProps: (key, def) => def} as CompilationEnvironment,
        );
        app.use(express.json());
        app.use('/api', apiHandler.handle);
        apiHandler.setCompilers(compilers);
        apiHandler.setLanguages(languages);
    });

    it('should respond to plain text compiler requests', async () => {
        const res = await request(app).get('/api/compilers').expect(200).expect('Content-Type', /text/);
        expect(res.text).toContain('Compiler Name');
        expect(res.text).toContain('gcc900');
        expect(res.text).toContain('GCC 9.0.0');
    });
    it('should respond to JSON compiler requests', async () => {
        await request(app)
            .get('/api/compilers')
            .set('Accept', 'application/json')
            .expect('Content-Type', /json/)
            .expect(200, compilers);
    });
    it('should respond to JSON compiler requests with all fields', async () => {
        await request(app)
            .get('/api/compilers?fields=all')
            .set('Accept', 'application/json')
            .expect('Content-Type', /json/)
            .expect(compilers);
    });
    it('should respond to JSON compiler requests with limited fields', async () => {
        await request(app)
            .get('/api/compilers?fields=id,name')
            .set('Accept', 'application/json')
            .expect('Content-Type', /json/)
            .expect(
                200,
                compilers.map(c => {
                    return {id: c.id, name: c.name};
                }),
            );
    });
    it('should respond to JSON compilers requests with c++ filter', async () => {
        await request(app)
            .get('/api/compilers/c++')
            .set('Accept', 'application/json')
            .expect('Content-Type', /json/)
            .expect(200, [compilers[0], compilers[2]]);
    });
    it('should respond to JSON compilers requests with pascal filter', async () => {
        await request(app)
            .get('/api/compilers/pascal')
            .set('Accept', 'application/json')
            .expect('Content-Type', /json/)
            .expect(200, [compilers[1]]);
    });
    it('should respond to plain text language requests', async () => {
        const res = await request(app).get('/api/languages').expect(200).expect('Content-Type', /text/);
        expect(res.text).toContain('Name');
        expect(res.text).toContain('c++');
        expect(res.text).toContain('c++');
        // We should not list languages for which there are no compilers
        expect(res.text).not.toContain('haskell');
    });
    it('should respond to JSON languages requests', async () => {
        await request(app)
            .get('/api/languages')
            .set('Accept', 'application/json')
            .expect('Content-Type', /json/)
            .expect(200, [languages['c++'], languages.pascal]);
    });
});
