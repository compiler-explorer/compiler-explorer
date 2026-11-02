// Copyright (c) 2026, Compiler Explorer Authors
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

import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {beforeAll, describe, expect, it} from 'vitest';

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {PythonCompiler} from '../lib/compilers/python.js';
import {resolvePathFromAppRoot} from '../lib/utils.js';
import {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {
    python: {id: 'python' as LanguageKey},
};

describe('Python', () => {
    let ce: CompilationEnvironment;

    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
    });

    function makeCompiler(version: string) {
        return new PythonCompiler(makeFakeCompilerInfo({exe: '/dev/null', lang: languages.python.id, version}), ce);
    }

    it('runs Python 3 interpreters in isolated mode', () => {
        const compiler = makeCompiler('Python 3.11.13 (abc, Jun 01 2026, 12:00:00)');
        const options = compiler.optionsForFilter({}, 'output.txt');
        expect(options[0]).toEqual('-I');
        expect(options.slice(-3)).toEqual(['--outputfile', 'output.txt', '--inputfile']);
    });

    it('uses -E -s for Python 2 interpreters which lack -I', () => {
        const compiler = makeCompiler('Python 2.7.18 (d3d629bf20c9, May 26 2026, 06:07:47)');
        const options = compiler.optionsForFilter({}, 'output.txt');
        expect(options.slice(0, 2)).toEqual(['-E', '-s']);
        expect(options).not.toContain('-I');
    });

    describe('staging helper scripts', () => {
        function makeStagingCompiler() {
            const env = makeCompilationEnvironment({languages, props: {stageHelperScripts: true}});
            return new PythonCompiler(
                makeFakeCompilerInfo({exe: '/dev/null', lang: languages.python.id, version: 'Python 3.12.0'}),
                env,
            );
        }

        it('runs the helper scripts from where CE is installed by default', () => {
            const options = makeCompiler('Python 3.12.0').optionsForFilter({}, '/tmp/build/output.txt');
            expect(options[1]).toEqual(resolvePathFromAppRoot('etc', 'scripts', 'disasms', 'dis_all.py'));
        });

        it('runs the helper scripts from the compilation directory when staging them', () => {
            const options = makeStagingCompiler().optionsForFilter({}, '/tmp/build/output.txt');
            expect(options[1]).toEqual(path.join('/tmp/build', '.ce-python-disasm.py'));
        });

        it('copies the helper scripts into the compilation directory when staging them', async () => {
            const dirPath = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-python-stage-test-'));
            try {
                await (makeStagingCompiler() as any).writeAllFiles(dirPath, 'print(1)', []);

                const staged = await fs.readFile(path.join(dirPath, '.ce-python-disasm.py'), 'utf8');
                const original = await fs.readFile(path.resolve('etc/scripts/disasms/dis_all.py'), 'utf8');
                expect(staged).toEqual(original);
                await expect(fs.stat(path.join(dirPath, '.ce-python-ast.py'))).resolves.toBeTruthy();
            } finally {
                await fs.rm(dirPath, {recursive: true, force: true});
            }
        });
    });
});
