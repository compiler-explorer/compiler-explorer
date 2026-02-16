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

import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {afterEach, beforeAll, beforeEach, describe, expect, it, vi} from 'vitest';

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {GolangCompiler} from '../lib/compilers/golang.js';
import * as utils from '../lib/utils.js';
import {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {
    go: {id: 'go' as LanguageKey},
};

let ce: CompilationEnvironment;
const info = {
    exe: '/dev/null',
    remote: {
        target: 'foo',
        path: 'bar',
        cmakePath: 'cmake',
        basePath: '/',
    },
    lang: languages.go.id,
};

async function testGoAsm(baseFilename: string) {
    const compiler = new GolangCompiler(makeFakeCompilerInfo(info), ce);

    const asmLines = utils.splitLines(fs.readFileSync(baseFilename + '.asm').toString());

    const result = {
        stderr: asmLines.map(line => {
            return {
                text: line,
            };
        }),
    };

    const [output] = await compiler.postProcess(result);
    const expectedOutput = utils.splitLines(fs.readFileSync(baseFilename + '.output.asm').toString());
    expect(utils.splitLines(output.asm)).toEqual(expectedOutput);
    expect(output).toEqual({
        asm: expectedOutput.join('\n'),
        stdout: [],
        stderr: [],
    });
}

describe('GO asm tests', () => {
    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
    });

    it('Handles unknown line number correctly', async () => {
        await testGoAsm('test/golang/bug-901');
    });
    it('Rewrites PC jumps to labels', async () => {
        await testGoAsm('test/golang/labels');
    });
});

describe('GO environment variables', () => {
    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
    });

    it('Derives GOROOT from compiler executable path', () => {
        const compilerInfo = makeFakeCompilerInfo({
            exe: '/opt/compiler-explorer/go1.20/bin/go',
            lang: languages.go.id,
        });
        const compiler = new GolangCompiler(compilerInfo, ce);
        const execOptions = compiler.getDefaultExecOptions();

        expect(execOptions.env.GOROOT).toBe('/opt/compiler-explorer/go1.20');
        // GOCACHE is not set in default exec options, it's set during runCompiler
        expect(execOptions.env.GOCACHE).toBeUndefined();
    });

    it('Uses explicit GOROOT from properties when set', () => {
        const compilerInfo = makeFakeCompilerInfo({
            exe: '/opt/compiler-explorer/go1.20/bin/go',
            lang: languages.go.id,
            id: 'go120',
        });
        // Override compilerProps to return a specific GOROOT
        const ceWithProps = makeCompilationEnvironment({
            languages,
            props: {
                goroot: '/custom/goroot/path',
            },
        });
        const compiler = new GolangCompiler(compilerInfo, ceWithProps);
        const execOptions = compiler.getDefaultExecOptions();

        expect(execOptions.env.GOROOT).toBe('/custom/goroot/path');
        // GOCACHE is not set in default exec options, it's set during runCompiler
        expect(execOptions.env.GOCACHE).toBeUndefined();
    });
});

describe('GO optionsForFilter', () => {
    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
    });

    it('Includes -trimpath for Go >= 1.13', () => {
        const compiler = new GolangCompiler(
            makeFakeCompilerInfo({exe: '/dev/null', lang: languages.go.id, semver: '1.21.0'}),
            ce,
        );
        const options = compiler.optionsForFilter({binary: false} as any, '/tmp/output', []);
        expect(options).toContain('-trimpath');
    });

    it('Excludes -trimpath for Go < 1.13', () => {
        const compiler = new GolangCompiler(
            makeFakeCompilerInfo({exe: '/dev/null', lang: languages.go.id, semver: '1.12.0'}),
            ce,
        );
        const options = compiler.optionsForFilter({binary: false} as any, '/tmp/output', []);
        expect(options).not.toContain('-trimpath');
    });

    it('Includes -trimpath for Go 1.13 exactly', () => {
        const compiler = new GolangCompiler(
            makeFakeCompilerInfo({exe: '/dev/null', lang: languages.go.id, semver: '1.13.0'}),
            ce,
        );
        const options = compiler.optionsForFilter({binary: false} as any, '/tmp/output', []);
        expect(options).toContain('-trimpath');
    });

    it('Excludes -trimpath for Go 1.12 in binary mode', () => {
        const compiler = new GolangCompiler(
            makeFakeCompilerInfo({exe: '/dev/null', lang: languages.go.id, semver: '1.12.0'}),
            ce,
        );
        const options = compiler.optionsForFilter({binary: true} as any, '/tmp/output', []);
        expect(options).not.toContain('-trimpath');
    });
});

describe('GO library support', () => {
    let tempDir: string;
    let compiler: GolangCompiler;

    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
    });

    beforeEach(async () => {
        tempDir = await fsp.mkdtemp(path.join(os.tmpdir(), 'ce-go-test-'));
        const compilerInfo = makeFakeCompilerInfo({
            exe: '/opt/compiler-explorer/go1.20/bin/go',
            lang: languages.go.id,
        });
        compiler = new GolangCompiler(compilerInfo, ce);
    });

    afterEach(async () => {
        await fsp.rm(tempDir, {recursive: true, force: true});
        vi.restoreAllMocks();
    });

    it('Returns empty array for getIncludeArguments', () => {
        const result = compiler.getIncludeArguments([{paths: ['/some/path']}]);
        expect(result).toEqual([]);
    });

    it('Finds downloaded libraries with metadata.json', async () => {
        // Create mock library directories
        const libDir = path.join(tempDir, 'uuid');
        await fsp.mkdir(libDir, {recursive: true});
        await fsp.writeFile(
            path.join(libDir, 'metadata.json'),
            JSON.stringify({
                module: 'github.com/google/uuid',
                version: 'v1.6.0',
                go_mod_require: 'github.com/google/uuid v1.6.0',
                go_sum: 'github.com/google/uuid v1.6.0 h1:abc123=',
            }),
        );

        // Also create a non-library directory
        const nonLibDir = path.join(tempDir, 'cache');
        await fsp.mkdir(nonLibDir, {recursive: true});

        const libraries = await (compiler as any).findDownloadedLibraries(tempDir);
        expect(libraries).toEqual(['uuid']);
    });

    it('Reads library metadata correctly', async () => {
        const libDir = path.join(tempDir, 'uuid');
        await fsp.mkdir(libDir, {recursive: true});
        const metadata = {
            module: 'github.com/google/uuid',
            version: 'v1.6.0',
            go_mod_require: 'github.com/google/uuid v1.6.0',
            go_sum: 'github.com/google/uuid v1.6.0 h1:abc123=',
        };
        await fsp.writeFile(path.join(libDir, 'metadata.json'), JSON.stringify(metadata));

        const result = await (compiler as any).readLibraryMetadata(tempDir, 'uuid');
        expect(result).toEqual(metadata);
    });

    it('Returns null for missing metadata', async () => {
        const result = await (compiler as any).readLibraryMetadata(tempDir, 'nonexistent');
        expect(result).toBeNull();
    });

    it('Generates go.mod with require statements', async () => {
        const libDir = path.join(tempDir, 'uuid');
        await fsp.mkdir(libDir, {recursive: true});
        await fsp.writeFile(
            path.join(libDir, 'metadata.json'),
            JSON.stringify({
                module: 'github.com/google/uuid',
                version: 'v1.6.0',
                go_mod_require: 'github.com/google/uuid v1.6.0',
                go_sum: 'github.com/google/uuid v1.6.0 h1:abc123=',
            }),
        );

        await (compiler as any).generateGoMod(tempDir, ['uuid'], tempDir);

        const goModContent = await fsp.readFile(path.join(tempDir, 'go.mod'), 'utf-8');
        expect(goModContent).toContain('module example');
        expect(goModContent).toContain('github.com/google/uuid v1.6.0');

        const goSumContent = await fsp.readFile(path.join(tempDir, 'go.sum'), 'utf-8');
        expect(goSumContent).toContain('github.com/google/uuid v1.6.0 h1:abc123=');
    });

    it('Appends to existing go.mod with require block', async () => {
        const existingGoMod = `module mymodule

go 1.20

require (
\texisting/module v1.0.0
)
`;
        await fsp.writeFile(path.join(tempDir, 'go.mod'), existingGoMod);

        const libDir = path.join(tempDir, 'uuid');
        await fsp.mkdir(libDir, {recursive: true});
        await fsp.writeFile(
            path.join(libDir, 'metadata.json'),
            JSON.stringify({
                module: 'github.com/google/uuid',
                version: 'v1.6.0',
                go_mod_require: 'github.com/google/uuid v1.6.0',
                go_sum: '',
            }),
        );

        await (compiler as any).generateGoMod(tempDir, ['uuid'], tempDir);

        const goModContent = await fsp.readFile(path.join(tempDir, 'go.mod'), 'utf-8');
        expect(goModContent).toContain('module mymodule');
        expect(goModContent).toContain('existing/module v1.0.0');
        expect(goModContent).toContain('github.com/google/uuid v1.6.0');
    });

    it('Merges cache delta into GOCACHE', async () => {
        const cachePath = path.join(tempDir, 'cache');
        const cacheDeltaPath = path.join(tempDir, 'lib', 'cache_delta');

        await fsp.mkdir(cachePath, {recursive: true});
        await fsp.mkdir(cacheDeltaPath, {recursive: true});

        // Create existing cache file
        await fsp.writeFile(path.join(cachePath, 'existing.txt'), 'existing content');

        // Create cache delta files
        await fsp.writeFile(path.join(cacheDeltaPath, 'new_file.txt'), 'new content');

        await (compiler as any).mergeGocache(cachePath, cacheDeltaPath);

        // Both files should exist
        expect(await utils.fileExists(path.join(cachePath, 'existing.txt'))).toBe(true);
        expect(await utils.fileExists(path.join(cachePath, 'new_file.txt'))).toBe(true);

        const newFileContent = await fsp.readFile(path.join(cachePath, 'new_file.txt'), 'utf-8');
        expect(newFileContent).toBe('new content');
    });

    it('Sets up module sources in GOPATH', async () => {
        const goPath = path.join(tempDir, 'gopath');
        const libPath = path.join(tempDir, 'uuid');
        const moduleSourcesPath = path.join(libPath, 'module_sources');

        await fsp.mkdir(goPath, {recursive: true});
        await fsp.mkdir(moduleSourcesPath, {recursive: true});

        // Create module source files
        const modulePath = path.join(moduleSourcesPath, 'github.com', 'google', 'uuid@v1.6.0');
        await fsp.mkdir(modulePath, {recursive: true});
        await fsp.writeFile(path.join(modulePath, 'uuid.go'), 'package uuid');

        await (compiler as any).setupModuleSources(goPath, libPath);

        const destPath = path.join(goPath, 'pkg', 'mod', 'github.com', 'google', 'uuid@v1.6.0', 'uuid.go');
        expect(await utils.fileExists(destPath)).toBe(true);
    });
});
