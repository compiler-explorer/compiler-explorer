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

import {S3, NoSuchKey} from '@aws-sdk/client-s3';

import type {GetResult} from '../types/cache.interfaces.js';

import type {S3HandlerOptions} from './s3-handler.interfaces.js';
import {awsCredentials} from './aws.js';

const clientsByRegion: Map<string, S3> = new Map();

export class S3Bucket {
    private readonly instance: S3;
    readonly bucket: string;
    readonly region: string;

    constructor(bucket: string, region: string) {
        const maybeInstance = clientsByRegion.get(region);
        if (maybeInstance) {
            this.instance = maybeInstance;
        } else {
            this.instance = new S3({region, credentials: awsCredentials()});
            clientsByRegion.set(region, this.instance);
        }
        this.bucket = bucket;
        this.region = region;
    }

    async get(key: string, path: string): Promise<GetResult> {
        try {
            const result = await this.instance.getObject({Bucket: this.bucket, Key: `${path}/${key}`});
            if (!result.Body) return {hit: false};
            return {hit: true, data: Buffer.from(await result.Body.transformToByteArray())};
        } catch (x: any) {
            if (x instanceof NoSuchKey) return {hit: false};
            throw x;
        }
    }

    async delete(key, path): Promise<boolean> {
        try {
            await this.instance.deleteObject({Bucket: this.bucket, Key: `${path}/${key}`});
        } catch (x: any) {
            if (x instanceof NoSuchKey) return false;
            throw x;
        }
        return true;
    }

    async put(key: string, value: Buffer, path: string, options: S3HandlerOptions): Promise<void> {
        await this.instance.putObject({
            Bucket: this.bucket,
            Key: `${path}/${key}`,
            Body: value,
            StorageClass: options.redundancy || 'STANDARD',
            Metadata: options.metadata || {},
        });
    }
}
