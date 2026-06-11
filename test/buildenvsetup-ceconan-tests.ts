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
import {afterAll, beforeAll, describe, expect, it} from 'vitest';

import {BuildEnvSetupCeConanDirect} from '../lib/buildenvsetup/ceconan.js';
import * as temp from '../lib/temp.js';
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
        await expect(fs.promises.access(path.join(downloadPath, 'somelib', 'evildir'))).rejects.toThrow();
    }, 10_000);

    it('rejects when the download is severed mid-stream', async () => {
        const downloadPath = await temp.mkdir('ce-conan-test');
        await expect(
            setup.downloadAndExtractPackage('somelib', '1.0', downloadPath, `${baseUrl}/severed.tgz`),
        ).rejects.toThrow();
    }, 10_000);
});
