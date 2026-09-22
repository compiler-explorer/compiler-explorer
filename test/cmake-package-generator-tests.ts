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

import {afterEach, beforeEach, describe, expect, it} from 'vitest';

import {
    CMakePackageGenerator,
    type CMakePackageLibrary,
    escapeCMakeString,
    GENERATED_PACKAGE_DIRNAME,
} from '../lib/build-systems/cmake-package-generator.js';

function makeLib(id: string, overrides: Partial<CMakePackageLibrary['version']> = {}): CMakePackageLibrary {
    return {
        id,
        version: {
            version: '1.0.0',
            path: [],
            libpath: [],
            staticliblink: [],
            liblink: [],
            ...overrides,
        },
    };
}

/** Mirrors the real fmt entry: headers unpacked on disk, one static lib from the conan package. */
function fmtLib(): CMakePackageLibrary {
    return makeLib('fmt', {
        version: '12.0.0',
        path: ['/opt/compiler-explorer/libs/fmt/12.0.0/include'],
        staticliblink: ['fmtd'],
    });
}

describe('CMakePackageGenerator path and property resolution', () => {
    const dirPath = '/tmp/ce-build';

    it('exposes the on-disk include path', () => {
        const gen = new CMakePackageGenerator(dirPath);
        expect(gen.includeDirsFor(fmtLib())).toEqual(['/opt/compiler-explorer/libs/fmt/12.0.0/include']);
    });

    it('adds the extracted package include dir for packagedheaders libraries', () => {
        const gen = new CMakePackageGenerator(dirPath);
        const lib = makeLib('qt', {path: ['/opt/qt/include'], packagedheaders: true});
        expect(gen.includeDirsFor(lib)).toEqual(['/opt/qt/include', path.join(dirPath, 'qt', 'include')]);
    });

    it('always searches the conan extraction lib dir, which is where fmt binaries land', () => {
        const gen = new CMakePackageGenerator(dirPath);
        expect(gen.libraryDirsFor(fmtLib())).toEqual([path.join(dirPath, 'fmt', 'lib')]);
    });

    it('keeps configured libpath entries ahead of the extraction dir', () => {
        const gen = new CMakePackageGenerator(dirPath);
        const lib = makeLib('libuv', {libpath: ['/opt/libuv/x86_64/lib']});
        expect(gen.libraryDirsFor(lib)).toEqual(['/opt/libuv/x86_64/lib', path.join(dirPath, 'libuv', 'lib')]);
    });

    it('extracts to the root when packages are not per-library', () => {
        const gen = new CMakePackageGenerator(dirPath, {packagesExtractedPerLib: false});
        expect(gen.extractedPackageDir('fmt')).toEqual(dirPath);
        expect(gen.libraryDirsFor(fmtLib())).toEqual([path.join(dirPath, 'lib')]);
    });

    it('combines static and shared link names without duplicates', () => {
        const gen = new CMakePackageGenerator(dirPath);
        const lib = makeLib('mixed', {staticliblink: ['a', 'b'], liblink: ['b', 'c']});
        expect(gen.linkNamesFor(lib)).toEqual(['a', 'b', 'c']);
    });

    it('offers the extraction dir as a prefix only for packagedheaders libraries', () => {
        const gen = new CMakePackageGenerator(dirPath);
        expect(gen.installedPrefixFor(fmtLib())).toBeUndefined();
        expect(gen.installedPrefixFor(makeLib('qt', {packagedheaders: true}))).toEqual(path.join(dirPath, 'qt'));
    });
});

describe('CMakePackageGenerator generation policy', () => {
    const gen = new CMakePackageGenerator('/tmp/ce-build');

    it('generates for a library with headers', () => {
        expect(gen.shouldGenerate(fmtLib())).toBe(true);
    });

    it('generates for a library that only has link libraries', () => {
        expect(gen.shouldGenerate(makeLib('onlylibs', {staticliblink: ['thing']}))).toBe(true);
    });

    it('does not generate for a library with nothing to offer', () => {
        expect(gen.shouldGenerate(makeLib('empty'))).toBe(false);
    });

    it('honours skipcmakepackage', () => {
        const lib = fmtLib();
        lib.version.skipcmakepackage = true;
        expect(gen.shouldGenerate(lib)).toBe(false);
    });

    it('skips packagedheaders libraries, whose package ships its own config', () => {
        // Qt installs lib/cmake/Qt6/Qt6Config.cmake and callers write find_package(Qt6).
        const qt = makeLib('qt', {version: '6.11.0', packagedheaders: true, path: ['/opt/qt/include']});
        expect(gen.skipsGeneration(qt)).toBe(true);
        expect(gen.shouldGenerate(qt)).toBe(false);
    });

    it('treats packagedheaders as decisive, so skipcmakepackage=false cannot force generation', () => {
        const lib = makeLib('odd', {packagedheaders: true, skipcmakepackage: false, staticliblink: ['odd']});
        expect(gen.skipsGeneration(lib)).toBe(true);
        expect(gen.shouldGenerate(lib)).toBe(false);
    });

    it('skips on either flag alone', () => {
        expect(gen.skipsGeneration(makeLib('a', {skipcmakepackage: true}))).toBe(true);
        expect(gen.skipsGeneration(makeLib('b', {packagedheaders: true}))).toBe(true);
        expect(gen.skipsGeneration(makeLib('c', {}))).toBe(false);
    });

    it('generates for an ordinary library, which has no packagedheaders flag', () => {
        expect(gen.skipsGeneration(fmtLib())).toBe(false);
    });
});

describe('CMakePackageGenerator config rendering', () => {
    const gen = new CMakePackageGenerator('/tmp/ce-build');

    it('declares the package found, with its version and include dirs', () => {
        const config = gen.renderConfig(fmtLib());
        expect(config).toContain('set(fmt_VERSION "12.0.0")');
        expect(config).toContain('set(fmt_FOUND TRUE)');
        expect(config).toContain('set(fmt_INCLUDE_DIRS "/opt/compiler-explorer/libs/fmt/12.0.0/include")');
    });

    it('creates an imported target per link library and an aggregate target', () => {
        const config = gen.renderConfig(fmtLib());
        expect(config).toContain('add_library(fmt::fmtd UNKNOWN IMPORTED)');
        expect(config).toContain('add_library(fmt::fmt INTERFACE IMPORTED)');
        expect(config).toContain('set_property(TARGET fmt::fmt APPEND PROPERTY INTERFACE_LINK_LIBRARIES fmt::fmtd)');
    });

    it('also defines an unqualified target for target_link_libraries(x fmt)', () => {
        const config = gen.renderConfig(fmtLib());
        expect(config).toContain('if(NOT TARGET fmt)');
        expect(config).toContain('add_library(fmt INTERFACE IMPORTED)');
    });

    it('searches only the known library dirs', () => {
        const config = gen.renderConfig(fmtLib());
        expect(config).toContain('NAMES fmtd');
        expect(config).toContain('NO_DEFAULT_PATH');
    });

    it('emits each search path as its own argument, since PATHS "a;b" finds nothing', () => {
        const lib = makeLib('multi', {
            libpath: ['/opt/one/lib', '/opt/two/lib'],
            staticliblink: ['thing'],
        });
        const config = new CMakePackageGenerator('/tmp/ce-build').renderConfig(lib);
        expect(config).toContain('PATHS "/opt/one/lib" "/opt/two/lib" "/tmp/ce-build/multi/lib"');
        expect(config).not.toContain('"/opt/one/lib;/opt/two/lib"');
    });

    it('still uses a semicolon list for the cache variables, which are lists', () => {
        const lib = makeLib('multi', {path: ['/opt/one/include', '/opt/two/include']});
        const config = new CMakePackageGenerator('/tmp/ce-build').renderConfig(lib);
        expect(config).toContain('set(multi_INCLUDE_DIRS "/opt/one/include;/opt/two/include")');
    });

    it('fails loudly when a declared link library is missing rather than linking nothing', () => {
        const config = gen.renderConfig(fmtLib());
        expect(config).toContain('message(FATAL_ERROR');
    });

    it('guards against being included twice', () => {
        expect(gen.renderConfig(fmtLib())).toContain('if(TARGET fmt::fmt)');
    });

    it('omits find_library entirely for a header-only library', () => {
        const config = gen.renderConfig(makeLib('headeronly', {path: ['/opt/h/include']}));
        expect(config).not.toContain('find_library');
        expect(config).toContain('add_library(headeronly::headeronly INTERFACE IMPORTED)');
    });

    it('renders a version file that accepts an equal or lower requested version', () => {
        const versionFile = gen.renderConfigVersion(fmtLib());
        expect(versionFile).toContain('set(PACKAGE_VERSION "12.0.0")');
        expect(versionFile).toContain('PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION');
        expect(versionFile).toContain('set(PACKAGE_VERSION_EXACT TRUE)');
    });

    it('falls back to a placeholder version when none is configured', () => {
        const lib = makeLib('noversion', {path: ['/opt/x/include']});
        lib.version.version = undefined;
        expect(gen.renderConfigVersion(lib)).toContain('set(PACKAGE_VERSION "0.0.0")');
    });
});

describe('CMakePackageGenerator jailed builds', () => {
    // Under nsjail the compilation dir is bind-mounted at /app. exec rewrites command arguments
    // and environment variables, but nothing rewrites the contents of a file we generate, so an
    // embedded host path simply does not exist for the build that reads it.
    const hostDir = '/tmp/ce-compiler-12345';
    const gen = new CMakePackageGenerator(hostDir, {buildDirPath: '/app'});

    it('embeds the build-visible extraction dir, not the host one', () => {
        expect(gen.libraryDirsFor(fmtLib())).toEqual([path.join('/app', 'fmt', 'lib')]);
    });

    it('embeds the build-visible include dir for packagedheaders libraries', () => {
        const lib = makeLib('qt', {packagedheaders: true});
        expect(gen.includeDirsFor(lib)).toEqual([path.join('/app', 'qt', 'include')]);
    });

    it('never leaks the host compilation dir into the rendered config', () => {
        const config = gen.renderConfig(fmtLib());
        expect(config).toContain('PATHS "/app/fmt/lib"');
        expect(config).not.toContain(hostDir);
    });

    it('still writes the package to the host path, which is where it must land', async () => {
        const dirPath = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-cmake-jail-'));
        try {
            const jailed = new CMakePackageGenerator(dirPath, {buildDirPath: '/app'});
            const generated = await jailed.generateOne(fmtLib());
            expect(generated!.prefixPath).toEqual(path.join(dirPath, GENERATED_PACKAGE_DIRNAME, 'fmt'));
            const config = await fs.readFile(generated!.configFile, 'utf8');
            expect(config).toContain('PATHS "/app/fmt/lib"');
        } finally {
            await fs.rm(dirPath, {recursive: true, force: true});
        }
    });

    it('leaves paths alone when the build is not jailed', () => {
        const plain = new CMakePackageGenerator(hostDir);
        expect(plain.libraryDirsFor(fmtLib())).toEqual([path.join(hostDir, 'fmt', 'lib')]);
    });
});

describe('CMakePackageGenerator describing what find_package can resolve', () => {
    let dirPath: string;

    beforeEach(async () => {
        dirPath = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-cmake-describe-'));
    });

    afterEach(async () => {
        await fs.rm(dirPath, {recursive: true, force: true});
    });

    it('describes a generated package by the names it authored', async () => {
        const gen = new CMakePackageGenerator(dirPath);
        const [described] = await gen.describePackages([fmtLib()]);
        expect(described.generated).toBe(true);
        expect(described.packageNames).toEqual(['fmt']);
        expect(described.targets).toEqual(['fmt::fmt', 'fmt::fmtd', 'fmt']);
    });

    it('reads real package names and targets out of an install tree, as Qt has', async () => {
        const gen = new CMakePackageGenerator(dirPath);
        const qtDir = path.join(dirPath, 'qt', 'lib', 'cmake', 'Qt6Core');
        await fs.mkdir(qtDir, {recursive: true});
        await fs.writeFile(path.join(qtDir, 'Qt6CoreConfig.cmake'), '# config');
        await fs.writeFile(
            path.join(qtDir, 'Qt6CoreTargets.cmake'),
            'add_library(Qt6::Core SHARED IMPORTED)\nadd_library(Qt6::CorePrivate INTERFACE IMPORTED)\n',
        );

        const [described] = await gen.describePackages([makeLib('qt', {packagedheaders: true})]);
        expect(described.generated).toBe(false);
        expect(described.packageNames).toEqual(['Qt6Core']);
        expect(described.targets).toEqual(['Qt6::Core', 'Qt6::CorePrivate']);
    });

    it('accepts the lowercase -config.cmake spelling', async () => {
        const gen = new CMakePackageGenerator(dirPath);
        const dir = path.join(dirPath, 'thing', 'lib', 'cmake', 'thing');
        await fs.mkdir(dir, {recursive: true});
        await fs.writeFile(path.join(dir, 'thing-config.cmake'), '# config');
        const [described] = await gen.describePackages([makeLib('thing', {packagedheaders: true})]);
        expect(described.packageNames).toEqual(['thing']);
    });

    it('also looks under share/cmake', async () => {
        const gen = new CMakePackageGenerator(dirPath);
        const dir = path.join(dirPath, 'shared', 'share', 'cmake', 'Shared');
        await fs.mkdir(dir, {recursive: true});
        await fs.writeFile(path.join(dir, 'SharedConfig.cmake'), 'add_library(Shared::Shared INTERFACE IMPORTED)');
        const [described] = await gen.describePackages([makeLib('shared', {packagedheaders: true})]);
        expect(described.packageNames).toEqual(['Shared']);
        expect(described.targets).toEqual(['Shared::Shared']);
    });

    it('says nothing about a skipped library with no install tree to read', async () => {
        const gen = new CMakePackageGenerator(dirPath);
        expect(await gen.describePackages([makeLib('qt', {packagedheaders: true})])).toEqual([]);
    });
});

describe('escapeCMakeString', () => {
    it('escapes quotes, backslashes and variable expansion', () => {
        expect(escapeCMakeString('a"b')).toBe('a\\"b');
        expect(escapeCMakeString('a\\b')).toBe('a\\\\b');
        // A path containing a CMake variable reference must not be expanded when the config runs.
        const variableReference = `$${'{'}EVIL}`;
        expect(escapeCMakeString(variableReference)).toBe(`\\${variableReference}`);
    });

    it('keeps a windows path from being read as escapes', () => {
        const gen = new CMakePackageGenerator('C:\\build');
        const config = gen.renderConfig(makeLib('winlib', {path: ['C:\\libs\\winlib\\include']}));
        expect(config).toContain('C:/libs/winlib/include');
        expect(config).not.toContain('C:\\libs');
    });
});

describe('CMakePackageGenerator writing packages', () => {
    let dirPath: string;

    beforeEach(async () => {
        dirPath = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-cmake-pkg-'));
    });

    afterEach(async () => {
        await fs.rm(dirPath, {recursive: true, force: true});
    });

    it('writes a config and version file in a findable layout', async () => {
        const gen = new CMakePackageGenerator(dirPath);
        const generated = await gen.generateOne(fmtLib());
        expect(generated).toBeDefined();
        expect(generated!.prefixPath).toEqual(path.join(dirPath, GENERATED_PACKAGE_DIRNAME, 'fmt'));
        // find_package(fmt CONFIG) searches <prefix>/lib/cmake/fmt/
        expect(generated!.configFile).toEqual(
            path.join(generated!.prefixPath, 'lib', 'cmake', 'fmt', 'fmtConfig.cmake'),
        );
        await expect(fs.readFile(generated!.configFile, 'utf8')).resolves.toContain('set(fmt_FOUND TRUE)');
        await expect(fs.readFile(generated!.versionFile, 'utf8')).resolves.toContain('set(PACKAGE_VERSION "12.0.0")');
    });

    it('writes nothing for a skipped library', async () => {
        const gen = new CMakePackageGenerator(dirPath);
        const lib = fmtLib();
        lib.version.skipcmakepackage = true;
        expect(await gen.generateOne(lib)).toBeUndefined();
        await expect(fs.readdir(path.join(dirPath, GENERATED_PACKAGE_DIRNAME))).rejects.toThrow();
    });

    it('writes nothing for a packagedheaders library, as Qt is', async () => {
        const gen = new CMakePackageGenerator(dirPath);
        const qt = makeLib('qt', {version: '6.11.0', packagedheaders: true, path: ['/opt/qt/include']});
        expect(await gen.generateOne(qt)).toBeUndefined();
        await expect(fs.readdir(path.join(dirPath, GENERATED_PACKAGE_DIRNAME))).rejects.toThrow();
    });

    it('writes nothing for a packagedheaders library even with skipcmakepackage=false', async () => {
        const gen = new CMakePackageGenerator(dirPath);
        const lib = makeLib('plain', {packagedheaders: true, skipcmakepackage: false, staticliblink: ['plain']});
        expect(await gen.generateOne(lib)).toBeUndefined();
    });

    it('generates one package per selected library', async () => {
        const gen = new CMakePackageGenerator(dirPath);
        const generated = await gen.generate([fmtLib(), makeLib('re2', {path: ['/opt/re2/include']})]);
        expect(generated.map(g => g.libId)).toEqual(['fmt', 're2']);
    });

    it('orders real install prefixes ahead of generated ones', async () => {
        const gen = new CMakePackageGenerator(dirPath);
        const qt = makeLib('qt', {version: '6.11.0', packagedheaders: true});
        const prefixes = await gen.prefixPaths([qt, fmtLib()]);
        expect(prefixes).toEqual([path.join(dirPath, 'qt'), path.join(dirPath, GENERATED_PACKAGE_DIRNAME, 'fmt')]);
    });

    it('returns no prefixes when nothing is selected', async () => {
        const gen = new CMakePackageGenerator(dirPath);
        expect(await gen.prefixPaths([])).toEqual([]);
    });
});
