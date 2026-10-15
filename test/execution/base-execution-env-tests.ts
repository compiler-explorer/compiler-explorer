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
import path from 'node:path';

import {afterAll, beforeAll, describe, expect, it, vi} from 'vitest';

import {CompilationEnvironment} from '../../lib/compilation-env.js';
import * as exec from '../../lib/exec.js';
import {LocalExecutionEnvironment} from '../../lib/execution/base-execution-env.js';
import {Packager} from '../../lib/packager.js';
import * as props from '../../lib/properties.js';
import * as utils from '../../lib/utils.js';
import {newTempDir} from '../utils.js';

describe('LocalExecutionEnvironment', () => {
    beforeAll(() => {
        props.initialize(path.resolve('./test/test-properties/execution'), ['test']);
    });
    afterAll(() => {
        props.reset();
    });

    it('executes a downloaded package by its unpacked path', async () => {
        // Package as storePackageWithExecutable does: executableFilename is masked to be relative.
        const buildDir = newTempDir();
        await fs.writeFile(path.join(buildDir, 'output.s'), '');
        await fs.writeFile(
            path.join(buildDir, 'compilation-result.json'),
            JSON.stringify({executableFilename: 'output.s'}),
        );
        const packageFile = path.join(newTempDir(), 'package.tgz');
        await new Packager().package(buildDir, packageFile);
        const data = await fs.readFile(packageFile);

        const environment = {
            ceProps: (_key: string, defaultValue: any) => defaultValue,
            executableCache: {get: async () => ({hit: true, data})},
        } as unknown as CompilationEnvironment;
        const execEnv = new LocalExecutionEnvironment(environment);
        const execBinary = vi.spyOn(execEnv, 'execBinary').mockResolvedValue(utils.getEmptyExecutionResult());

        await execEnv.downloadExecutablePackage('somehash');
        await execEnv.execute({});

        const [executable, , homeDir] = execBinary.mock.calls[0];
        expect(executable).toEqual(path.join(homeDir, 'output.s'));
        await expect(utils.fileExists(executable)).resolves.toBe(true);
        // dumb-init searches PATH for a bare name, so nsjail must be handed a path.
        const {args} = exec.getSandboxNsjailOptions(executable, [], {customCwd: homeDir, appHome: homeDir});
        expect(args.slice(-2)).toEqual(['/usr/local/bin/dumb-init', './output.s']);
    });
});
