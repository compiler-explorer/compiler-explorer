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

import {afterEach, describe, expect, it} from 'vitest';

import * as props from '../../lib/properties.js';
import {BuiltinSource, createBuiltinSource} from '../../lib/sources/builtin.js';

const tempDirs: string[] = [];

async function makeTempDir() {
    const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-builtin-sources-'));
    tempDirs.push(tempDir);
    return tempDir;
}

async function makeExamplesDir(source: string) {
    const examplesPath = path.join(await makeTempDir(), 'custom-examples');
    const languagePath = path.join(examplesPath, 'customlang');
    await fs.mkdir(languagePath, {recursive: true});
    await fs.writeFile(path.join(languagePath, 'configured_example.cpp'), source);
    return examplesPath;
}

afterEach(async () => {
    props.reset();
    await Promise.all(tempDirs.splice(0).map(tempDir => fs.rm(tempDir, {recursive: true, force: true})));
});

describe('BuiltinSource', () => {
    it('lists and loads examples from the given directory', async () => {
        const source = 'int configured_example() { return 42; }\n';
        const builtin = new BuiltinSource(await makeExamplesDir(source));

        await expect(builtin.list()).resolves.toEqual([
            {lang: 'customlang', name: 'configured example', file: 'configured_example'},
        ]);
        await expect(builtin.load('customlang', 'configured_example')).resolves.toEqual({file: source});
    });

    it('returns "No path found" for an unknown example', async () => {
        const builtin = new BuiltinSource(await makeExamplesDir('int x;\n'));
        await expect(builtin.load('customlang', 'nope')).resolves.toEqual({file: 'No path found'});
    });

    it('fails fast when constructed with a non-existent sourcePath', async () => {
        const missingPath = path.join(await makeTempDir(), 'does-not-exist');
        expect(() => new BuiltinSource(missingPath)).toThrow();
    });
});

describe('createBuiltinSource', () => {
    it('reads sourcePath from properties at construction time', async () => {
        const source = 'int configured_example() { return 42; }\n';
        const examplesPath = await makeExamplesDir(source);
        const configPath = path.join(await makeTempDir(), 'config');
        await fs.mkdir(configPath, {recursive: true});
        await fs.writeFile(path.join(configPath, 'builtin.test.properties'), `sourcePath=${examplesPath}\n`);

        props.initialize(configPath, ['test']);

        const builtin = createBuiltinSource();
        await expect(builtin.list()).resolves.toEqual([
            {lang: 'customlang', name: 'configured example', file: 'configured_example'},
        ]);
    });
});
