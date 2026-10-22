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

import {beforeAll, describe, expect, it, vi} from 'vitest';

import type {CompilationEnvironment} from '../lib/compilation-env.js';
import {SolidityCompiler} from '../lib/compilers/index.js';
import type {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {
    solidity: {id: 'solidity' as LanguageKey},
};

describe('Solidity', () => {
    let env: CompilationEnvironment;

    beforeAll(() => {
        env = makeCompilationEnvironment({languages});
    });

    async function compileWithOutput(stdout: string, stderr: string) {
        const compiler = new SolidityCompiler(
            makeFakeCompilerInfo({exe: '/opt/solc', lang: languages.solidity.id, semver: '0.8.30'}),
            env,
        );
        vi.spyOn(compiler, 'exec').mockResolvedValue({
            code: 0,
            okToCache: true,
            filenameTransform: (fn: string) => fn,
            stdout,
            stderr,
            execTime: 0,
            timedOut: false,
            truncated: false,
        });
        return compiler.runCompiler('/opt/solc', [], '/tmp/example.sol', compiler.getDefaultExecOptions());
    }

    it('should drop the successful compilation message from stdout', async () => {
        const result = await compileWithOutput(
            'Compiler run successful. Artifact(s) can be found in directory "contracts".\n',
            '',
        );
        expect(result.stdout).toEqual([]);
        expect(result.stderr).toEqual([]);
    });

    it('should drop the successful compilation message from stderr', async () => {
        const result = await compileWithOutput('', 'Compiler run successful, no output requested.\n');
        expect(result.stderr).toEqual([]);
    });

    it('should keep warnings', async () => {
        const result = await compileWithOutput(
            'Compiler run successful. Artifact(s) can be found in directory "contracts".\n',
            'Warning: SPDX license identifier not provided in source file.\n',
        );
        expect(result.stdout).toEqual([]);
        expect(result.stderr.map(line => line.text)).toEqual([
            'Warning: SPDX license identifier not provided in source file.',
        ]);
    });
});
