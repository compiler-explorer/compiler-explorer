// Copyright (c) 2019, Compiler Explorer Authors
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

import {createReadStream, createWriteStream} from 'node:fs';
import fs from 'node:fs/promises';
import type {Readable, Transform, Writable} from 'node:stream';
import {pipeline} from 'node:stream/promises';
import zlib from 'node:zlib';

import tar from 'tar-fs';

const DEFAULT_TIMEOUT_MS = 60_000;

export class Packager {
    constructor(private readonly timeoutMs: number = DEFAULT_TIMEOUT_MS) {}

    async package(directory: string, destination: string): Promise<string> {
        return await this.tarGzFiles(directory, destination);
    }

    async unpack(packageFile: string, destination: string): Promise<string> {
        await this.pipeWithTimeout(
            `Unpacking ${packageFile}`,
            createReadStream(packageFile),
            zlib.createGunzip(),
            tar.extract(destination) as unknown as Writable,
        );
        return destination;
    }

    async tarGzFiles(cwd: string, destination: string): Promise<string> {
        try {
            await this.pipeWithTimeout(
                `Packing ${cwd}`,
                tar.pack(cwd, {dereference: true}) as unknown as Readable,
                zlib.createGzip({level: 3, memLevel: 9}),
                createWriteStream(destination),
            );
        } catch (err) {
            // Not awaited: if the filesystem is what stalled, this could hang too.
            fs.rm(destination, {force: true}).catch(() => {});
            throw err;
        }
        return destination;
    }

    // The tar-fs streams are streamx ones, hence the casts in the callers: pipeline() takes them, but they aren't typed
    // as Node's Readable and Writable.
    private async pipeWithTimeout(
        description: string,
        source: Readable,
        transform: Transform,
        destination: Writable,
    ): Promise<void> {
        let timer: NodeJS.Timeout | undefined;
        // Raced rather than relying on destroy() to fail the pipeline: a stream stuck in a filesystem call doesn't
        // close until that call returns, and the pipeline waits for it.
        const timedOut = new Promise<never>((_, reject) => {
            timer = setTimeout(() => {
                const error = new Error(`${description} timed out after ${this.timeoutMs}ms`);
                for (const stream of [source, transform, destination]) stream.destroy(error);
                reject(error);
            }, this.timeoutMs);
        });
        try {
            await Promise.race([pipeline(source, transform, destination), timedOut]);
        } finally {
            clearTimeout(timer);
        }
    }
}
