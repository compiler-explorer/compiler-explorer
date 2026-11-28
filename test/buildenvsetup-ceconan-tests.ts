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

import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import zlib from 'node:zlib';

import tar from 'tar-stream';
import {afterAll, afterEach, beforeAll, describe, expect, it, vi} from 'vitest';

import {BuildEnvSetupCeConanDirect} from '../lib/buildenvsetup/ceconan.js';
import {BuildEnvSetupCeConanCircleDirect} from '../lib/buildenvsetup/ceconan-circle.js';
import {BuildEnvSetupCeConanFortranDirect} from '../lib/buildenvsetup/ceconan-fortran.js';
import * as temp from '../lib/temp.js';
import type {LibraryBuildType} from '../types/libraries/libraries.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const languages = {
    'c++': {id: 'c++'},
} as const;

// Hand-built tar entry: tar-stream's packer won't produce malformed archives (e.g. a
// directory entry claiming a non-zero size), which we need to test robustness against.
function rawTarEntry(name: string, typeflag: string, size: number, content: Buffer): Buffer {
    const header = Buffer.alloc(512);
    header.write(name, 0);
    header.write('0000777\0', 100);
    header.write('0000000\0', 108);
    header.write('0000000\0', 116);
    header.write(`${size.toString(8).padStart(11, '0')}\0`, 124);
    header.write('00000000000\0', 136);
    header.fill(' ', 148, 156);
    header.write(typeflag, 156);
    header.write('ustar\0', 257);
    header.write('00', 263);
    let checksum = 0;
    for (const byte of header) checksum += byte;
    header.write(`${checksum.toString(8).padStart(6, '0')}\0 `, 148);
    const body = Buffer.alloc(Math.ceil(size / 512) * 512);
    content.copy(body);
    return Buffer.concat([header, body]);
}

async function makeTarGz(files: Record<string, string>): Promise<Buffer> {
    const pack = tar.pack();
    for (const [name, content] of Object.entries(files)) {
        pack.entry({name}, content);
    }
    pack.finalize();
    const chunks: Buffer[] = [];
    for await (const chunk of pack) {
        chunks.push(chunk as Buffer);
    }
    return zlib.gzipSync(Buffer.concat(chunks));
}

describe('BuildEnvSetupCeConanDirect.downloadAndExtractPackage', () => {
    let server: http.Server;
    let baseUrl: string;
    let setup: BuildEnvSetupCeConanDirect;
    let packageTarGz: Buffer;
    let zipSlipTarGz: Buffer;
    let sizedDirTarGz: Buffer;
    const fileContents = crypto.randomBytes(64 * 1024).toString('hex');

    beforeAll(async () => {
        packageTarGz = await makeTarGz({
            'include/somelib.h': fileContents,
            'empty.txt': '',
        });
        zipSlipTarGz = await makeTarGz({
            'good.txt': 'good',
            // A directory merely *named* with leading dots is legitimate, not a traversal.
            '..odd/inside.txt': 'inside',
            // Escapes the download path entirely.
            '../../escape.txt': 'escaped',
            // Stays within the download path but escapes this library's own subdirectory.
            '../sibling.txt': 'sneaky',
        });
        sizedDirTarGz = zlib.gzipSync(
            Buffer.concat([
                rawTarEntry('ok.txt', '0', 2, Buffer.from('ok')),
                rawTarEntry('evildir/', '5', 10, Buffer.from('whoops....')),
                Buffer.alloc(1024),
            ]),
        );

        server = http.createServer((req, res) => {
            if (req.url === '/package.tgz') {
                res.writeHead(200, {'Content-Type': 'application/octet-stream'});
                res.end(packageTarGz);
            } else if (req.url === '/zipslip.tgz') {
                res.writeHead(200, {'Content-Type': 'application/octet-stream'});
                res.end(zipSlipTarGz);
            } else if (req.url === '/sizeddir.tgz') {
                res.writeHead(200, {'Content-Type': 'application/octet-stream'});
                res.end(sizedDirTarGz);
            } else if (req.url === '/chunked.tgz') {
                // No Content-Length, so only the streaming byte count can enforce the cap.
                res.writeHead(200, {'Content-Type': 'application/octet-stream'});
                res.write(packageTarGz.subarray(0, 1024));
                res.end(packageTarGz.subarray(1024));
            } else if (req.url === '/stalled.tgz') {
                // Sends part of the body and then nothing more, without closing the connection.
                res.writeHead(200, {'Content-Type': 'application/octet-stream'});
                res.write(packageTarGz.subarray(0, 1024));
            } else if (req.url === '/severed.tgz') {
                // Send a valid gzip prefix then cut the connection mid-stream, as seen when a
                // download is interrupted: the client must reject, not hang forever.
                res.writeHead(200, {'Content-Type': 'application/octet-stream'});
                res.write(packageTarGz.subarray(0, Math.floor(packageTarGz.length / 2)));
                setTimeout(() => res.destroy(), 50);
            } else {
                res.writeHead(404);
                res.end();
            }
        });
        await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
        const address = server.address();
        if (address === null || typeof address !== 'object') throw new Error('no server address');
        baseUrl = `http://127.0.0.1:${address.port}`;

        const ce = makeCompilationEnvironment({languages});
        const compilerInfo = makeFakeCompilerInfo({
            lang: 'c++',
            exe: '/dev/null',
            options: '',
            buildenvsetup: {
                id: 'ceconan',
                props: (name: string, def?: any) => def,
            },
        });
        setup = new BuildEnvSetupCeConanDirect(compilerInfo, ce);
    });

    afterAll(async () => {
        server.closeAllConnections();
        await new Promise<void>(resolve => {
            server.close(() => resolve());
        });
    });

    it('downloads and extracts a package', async () => {
        const downloadPath = await temp.mkdir('ce-conan-test');
        const info = await setup.downloadAndExtractPackage('somelib', '1.0', downloadPath, `${baseUrl}/package.tgz`);
        expect(info.packageUrl).toEqual(`${baseUrl}/package.tgz`);
        const extracted = await fs.promises.readFile(path.join(downloadPath, 'somelib', 'include/somelib.h'), 'utf8');
        expect(extracted).toEqual(fileContents);
        const empty = await fs.promises.readFile(path.join(downloadPath, 'somelib', 'empty.txt'), 'utf8');
        expect(empty).toEqual('');
    });

    it('skips entries that try to escape the library extraction root', async () => {
        const downloadPath = await temp.mkdir('ce-conan-test');
        await setup.downloadAndExtractPackage('somelib', '1.0', downloadPath, `${baseUrl}/zipslip.tgz`);
        await expect(fs.promises.readFile(path.join(downloadPath, 'somelib', 'good.txt'), 'utf8')).resolves.toEqual(
            'good',
        );
        await expect(
            fs.promises.readFile(path.join(downloadPath, 'somelib', '..odd', 'inside.txt'), 'utf8'),
        ).resolves.toEqual('inside');
        await expect(fs.promises.access(path.resolve(downloadPath, '..', 'escape.txt'))).rejects.toThrow();
        await expect(fs.promises.access(path.join(downloadPath, 'sibling.txt'))).rejects.toThrow();
    });

    it('rejects when the server returns an error status', async () => {
        const downloadPath = await temp.mkdir('ce-conan-test');
        await expect(
            setup.downloadAndExtractPackage('somelib', '1.0', downloadPath, `${baseUrl}/missing.tgz`),
        ).rejects.toThrow('Unable to request library from conan: 404');
    });

    it('does not hang on a directory entry claiming a non-zero size', async () => {
        const downloadPath = await temp.mkdir('ce-conan-test');
        // Whether such a malformed archive extracts or rejects is tar-stream's business; ours is
        // that the promise settles rather than wedging a compilation queue slot forever.
        await setup
            .downloadAndExtractPackage('somelib', '1.0', downloadPath, `${baseUrl}/sizeddir.tgz`)
            .catch(() => {});
        // The good entry preceding the malformed one proves the archive was actually processed.
        await expect(fs.promises.readFile(path.join(downloadPath, 'somelib', 'ok.txt'), 'utf8')).resolves.toEqual('ok');
        await expect(fs.promises.access(path.join(downloadPath, 'somelib', 'evildir'))).rejects.toThrow();
    }, 10_000);

    it('rejects a package whose extracted size exceeds the cap', async () => {
        const downloadPath = await temp.mkdir('ce-conan-test');
        (setup as any).maxExtractedBytes = 1024;
        try {
            await expect(
                setup.downloadAndExtractPackage('somelib', '1.0', downloadPath, `${baseUrl}/package.tgz`),
            ).rejects.toThrow(/exceeds 1024 bytes/);
        } finally {
            (setup as any).maxExtractedBytes = 2 * 1024 * 1024 * 1024;
        }
    });

    it('rejects a non-http(s) package URL', async () => {
        const downloadPath = await temp.mkdir('ce-conan-test');
        await expect(
            setup.downloadAndExtractPackage('somelib', '1.0', downloadPath, 'file:///etc/passwd'),
        ).rejects.toThrow(/Unexpected protocol 'file:'/);
    });

    it('rejects when the download is severed mid-stream', async () => {
        const downloadPath = await temp.mkdir('ce-conan-test');
        await expect(
            setup.downloadAndExtractPackage('somelib', '1.0', downloadPath, `${baseUrl}/severed.tgz`),
        ).rejects.toThrow();
    }, 10_000);

    it('rejects a package bigger than the download cap', async () => {
        const downloadPath = await temp.mkdir('ce-conan-test');
        (setup as any).maxDownloadBytes = 1024;
        try {
            await expect(
                setup.downloadAndExtractPackage('somelib', '1.0', downloadPath, `${baseUrl}/package.tgz`),
            ).rejects.toThrow(/exceed 1024 bytes downloaded/);
            await expect(
                setup.downloadAndExtractPackage('somelib', '1.0', downloadPath, `${baseUrl}/chunked.tgz`),
            ).rejects.toThrow(/exceed 1024 bytes downloaded/);
        } finally {
            (setup as any).maxDownloadBytes = 1024 * 1024 * 1024;
        }
    });

    it('applies the download cap across all packages of one compilation', async () => {
        const downloadPath = await temp.mkdir('ce-conan-test');
        (setup as any).maxDownloadBytes = Math.floor(packageTarGz.length * 1.5);
        try {
            const budget = {downloadedBytes: 0};
            await setup.downloadAndExtractPackage('lib1', '1.0', downloadPath, `${baseUrl}/chunked.tgz`, budget);
            expect(budget.downloadedBytes).toEqual(packageTarGz.length);
            await expect(
                setup.downloadAndExtractPackage('lib2', '1.0', downloadPath, `${baseUrl}/chunked.tgz`, budget),
            ).rejects.toThrow(/bytes downloaded/);
            // A fresh compilation starts with a fresh budget.
            await setup.downloadAndExtractPackage('lib3', '1.0', downloadPath, `${baseUrl}/package.tgz`);
        } finally {
            (setup as any).maxDownloadBytes = 1024 * 1024 * 1024;
        }
    });

    it('gives up on a download that stalls', async () => {
        const downloadPath = await temp.mkdir('ce-conan-test');
        (setup as any).fetchTimeoutMs = 200;
        try {
            await expect(
                setup.downloadAndExtractPackage('somelib', '1.0', downloadPath, `${baseUrl}/stalled.tgz`),
            ).rejects.toThrow();
        } finally {
            (setup as any).fetchTimeoutMs = 120_000;
        }
    }, 10_000);
});

function makeLibraryDetails(count: number): Record<string, any> {
    const details: Record<string, any> = {};
    for (let i = 0; i < count; i++) {
        details[`lib${i}`] = {
            version: '1.0',
            packagedheaders: true,
            libpath: [],
            staticliblink: [],
            liblink: [],
        };
    }
    return details;
}

describe('BuildEnvSetupCeConanDirect library limits', () => {
    const fakeKey = {libraries: [], options: []} as any;

    function makeSetup(props: Record<string, any>) {
        const ce = makeCompilationEnvironment({languages});
        const compilerInfo = makeFakeCompilerInfo({
            lang: 'c++',
            exe: '/dev/null',
            options: '',
            buildenvsetup: {
                id: 'ceconan',
                props: (name: string, def?: any) => (name in props ? props[name] : def),
            },
        });
        return new BuildEnvSetupCeConanDirect(compilerInfo, ce);
    }

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('reads limits from properties', () => {
        const setup = makeSetup({fetchTimeoutMs: 1234, maxLibraries: 3, maxDownloadBytes: 5678, maxExtractedBytes: 91});
        expect((setup as any).fetchTimeoutMs).toEqual(1234);
        expect((setup as any).maxExtractedBytes).toEqual(91);
        expect((setup as any).maxLibraries).toEqual(3);
        expect((setup as any).maxDownloadBytes).toEqual(5678);
    });

    it('rejects more libraries than the cap without contacting the server', async () => {
        const fetchSpy = vi.spyOn(globalThis, 'fetch');
        const setup = makeSetup({host: 'http://conan.invalid', maxLibraries: 2});
        await expect(setup.download(fakeKey, '/nonexistent', makeLibraryDetails(3))).rejects.toThrow(
            /Too many libraries selected \(3\); at most 2/,
        );
        expect(fetchSpy).not.toHaveBeenCalled();
    });

    it('allows libraries up to the cap', async () => {
        const fetchSpy = vi
            .spyOn(globalThis, 'fetch')
            .mockImplementation(async () => new Response('{}', {status: 200}) as Response);
        const setup = makeSetup({host: 'http://conan.invalid', maxLibraries: 2});
        // No builds match the empty search results, so nothing is downloaded.
        await expect(setup.download(fakeKey, '/nonexistent', makeLibraryDetails(2))).resolves.toEqual([]);
        expect(fetchSpy).toHaveBeenCalledTimes(2);
    });

    it('passes a timeout signal to every conan request', async () => {
        const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(
            (_url, init) =>
                new Promise((_resolve, reject) => {
                    init?.signal?.addEventListener('abort', () => reject(init.signal?.reason));
                }),
        );
        const setup = makeSetup({host: 'http://conan.invalid', fetchTimeoutMs: 50});
        await expect(setup.getAllPossibleBuilds('somelib', '1.0')).rejects.toThrow(/timeout/i);
        await expect(setup.getPackageUrl('somelib', '1.0', 'abc')).rejects.toThrow(/timeout/i);
        expect(fetchSpy).toHaveBeenCalledTimes(2);
    });

    describe('download orchestration', () => {
        const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

        function stubLookups(setup: BuildEnvSetupCeConanDirect, searchDelays: Record<string, number>) {
            vi.spyOn(setup, 'getAllPossibleBuilds').mockImplementation(async (libid: string) => {
                await sleep(searchDelays[libid] ?? 0);
                return {[`hash-${libid}`]: {}};
            });
            vi.spyOn(setup, 'findBuildHash').mockImplementation(async (_key, builds: any) => ({
                hash: Object.keys(builds)[0],
                buildType: 'Debug',
            }));
            vi.spyOn(setup, 'getPackageUrl').mockImplementation(async (libid: string) => `http://pkg/${libid}`);
        }

        it('fails the compilation when a library search fails for a reason other than not found', async () => {
            const setup = makeSetup({host: 'http://conan.invalid'});
            vi.spyOn(setup, 'getAllPossibleBuilds').mockRejectedValue(new Error('conan unreachable'));
            await expect(setup.download(fakeKey, '/nonexistent', makeLibraryDetails(1))).rejects.toThrow(
                'conan unreachable',
            );
        });

        it('skips a library conan does not have', async () => {
            const setup = makeSetup({host: 'http://conan.invalid'});
            vi.spyOn(setup, 'getAllPossibleBuilds').mockRejectedValue('Not found (http://conan.invalid/x)');
            await expect(setup.download(fakeKey, '/nonexistent', makeLibraryDetails(1))).resolves.toEqual([]);
        });

        it('handles a failed download while later libraries are still being looked up', async () => {
            const setup = makeSetup({host: 'http://conan.invalid'});
            stubLookups(setup, {lib0: 0, lib1: 100});
            const extract = vi.spyOn(setup, 'downloadAndExtractPackage').mockImplementation(async () => {
                await sleep(10);
                throw new Error('package too big');
            });
            // Without a handler attached as the download starts, this rejection would go unhandled while
            // lib1's search is awaited, and vitest would fail the run.
            await expect(setup.download(fakeKey, '/nonexistent', makeLibraryDetails(2))).rejects.toThrow(
                'package too big',
            );
            // lib1's search was aborted with the rest, so its download never started.
            expect(extract).toHaveBeenCalledTimes(1);
        });

        it('aborts and waits for sibling downloads when one fails', async () => {
            const setup = makeSetup({host: 'http://conan.invalid'});
            stubLookups(setup, {});
            let siblingSettled = false;
            vi.spyOn(setup, 'downloadAndExtractPackage').mockImplementation(async (libId, _v, _d, _u, context) => {
                if (libId === 'lib0') {
                    await sleep(10);
                    throw new Error('package too big');
                }
                await new Promise<void>(resolve => context?.signal?.addEventListener('abort', () => resolve()));
                await sleep(50);
                siblingSettled = true;
                throw new Error('aborted');
            });
            await expect(setup.download(fakeKey, '/nonexistent', makeLibraryDetails(2))).rejects.toThrow(
                'package too big',
            );
            expect(siblingSettled).toBe(true);
        });
    });
});

describe('BuildEnvSetupCeConanDirect.findBuildHash', () => {
    const ce = makeCompilationEnvironment({languages});
    const compilerInfo = makeFakeCompilerInfo({
        id: 'g132',
        lang: 'c++',
        exe: '/dev/null',
        options: '',
        buildenvsetup: {
            id: 'ceconan',
            props: (name: string, def?: any) => def,
        },
    });
    const setup = new BuildEnvSetupCeConanDirect(compilerInfo, ce);
    const key = {options: [] as string[]} as any;

    async function settingsFor(buildType: LibraryBuildType) {
        return {settings: await setup.getConanBuildProperties(key, buildType)};
    }

    it('uses the Debug package by default even when a Release one exists', async () => {
        const possibleBuilds = {debughash: await settingsFor('Debug'), releasehash: await settingsFor('Release')};
        expect(await setup.findBuildHash(key, possibleBuilds, 'Debug')).toEqual({
            hash: 'debughash',
            buildType: 'Debug',
        });
    });

    it('uses the Release package when the request asks for it', async () => {
        const possibleBuilds = {debughash: await settingsFor('Debug'), releasehash: await settingsFor('Release')};
        expect(await setup.findBuildHash(key, possibleBuilds, 'Release')).toEqual({
            hash: 'releasehash',
            buildType: 'Release',
        });
    });

    it('falls back to Debug when Release is asked for but not built', async () => {
        const possibleBuilds = {debughash: await settingsFor('Debug')};
        expect(await setup.findBuildHash(key, possibleBuilds, 'Release')).toEqual({
            hash: 'debughash',
            buildType: 'Debug',
        });
    });

    it('returns undefined when nothing matches', async () => {
        const other = await settingsFor('Debug');
        const possibleBuilds = {otherhash: {settings: {...other.settings, 'compiler.version': 'clang1600'}}};
        expect(await setup.findBuildHash(key, possibleBuilds, 'Debug')).toBeUndefined();
    });
});

describe('Conan subclasses honour the requested build type', () => {
    const ce = makeCompilationEnvironment({languages});
    const key = {options: [] as string[]} as any;

    it('circle', async () => {
        const linked: Record<string, string> = {linkedCompilerId: 'g132', linkedCompilerType: 'gcc'};
        const compilerInfo = makeFakeCompilerInfo({
            lang: 'c++',
            exe: '/dev/null',
            options: '',
            buildenvsetup: {id: 'ceconan-circle', props: (name: string, def?: any) => linked[name] ?? def},
        });
        const setup = new BuildEnvSetupCeConanCircleDirect(compilerInfo, ce);
        const props = await setup.getConanBuildProperties(key, 'Release');
        expect(props.build_type).toEqual('Release');
        expect(props['compiler.version']).toEqual('g132');
    });

    it('fortran', async () => {
        const compilerInfo = makeFakeCompilerInfo({
            lang: 'c++',
            exe: '/dev/null',
            options: '',
            buildenvsetup: {id: 'ceconan-fortran', props: (name: string, def?: any) => def},
        });
        const setup = new BuildEnvSetupCeConanFortranDirect(compilerInfo, ce);
        expect((await setup.getConanBuildProperties(key, 'Release')).build_type).toEqual('Release');
    });
});
