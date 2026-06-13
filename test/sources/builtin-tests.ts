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

import {afterEach, describe, expect, it, vi} from 'vitest';

const tempDirs: string[] = [];

async function makeTempDir() {
    const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-builtin-sources-'));
    tempDirs.push(tempDir);
    return tempDir;
}

describe('builtin source provider', () => {
    afterEach(async () => {
        const props = await import('../../lib/properties.js');
        props.reset();
        vi.resetModules();

        await Promise.all(tempDirs.splice(0).map(tempDir => fs.rm(tempDir, {recursive: true, force: true})));
    });

    it('uses sourcePath configured after the module has been imported', async () => {
        const tempDir = await makeTempDir();
        const examplesPath = path.join(tempDir, 'custom-examples');
        const languagePath = path.join(examplesPath, 'customlang');
        const configPath = path.join(tempDir, 'config');
        const source = 'int configured_example() { return 42; }\n';

        await fs.mkdir(languagePath, {recursive: true});
        await fs.mkdir(configPath, {recursive: true});
        await fs.writeFile(path.join(languagePath, 'configured_example.cpp'), source);
        await fs.writeFile(path.join(configPath, 'builtin.test.properties'), `sourcePath=${examplesPath}\n`);

        vi.resetModules();
        const {builtin} = await import('../../lib/sources/builtin.js');
        const props = await import('../../lib/properties.js');

        props.initialize(configPath, ['test']);

        await expect(builtin.list()).resolves.toEqual([
            {
                lang: 'customlang',
                name: 'configured example',
                file: 'configured_example',
            },
        ]);
        await expect(builtin.load('customlang', 'configured_example')).resolves.toEqual({file: source});
    });
});
