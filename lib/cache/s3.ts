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

import * as Sentry from '@sentry/node';

import type {GetResult} from '../../types/cache.interfaces.js';
import {logger} from '../logger.js';
import {S3Bucket} from '../s3-handler.js';
import type {S3HandlerOptions} from '../s3-handler.interfaces.js';

import {BaseCache} from './base.js';
import {StorageClass} from '@aws-sdk/client-s3';

function messageFor(e) {
    return e.message || e.toString();
}

export class S3Cache extends BaseCache {
    private readonly s3: S3Bucket;
    readonly path: string;
    readonly region: string;
    private readonly onError: (Error, string) => void;

    constructor(cacheName: string, bucket: string, path: string, region: string, onError?: (Error, string) => void) {
        super(cacheName, `S3Cache(s3://${bucket}/${path} in ${region})`, 's3');
        this.path = path;
        this.region = region;
        this.s3 = new S3Bucket(bucket, region);
        this.onError =
            onError ||
            ((e, op) => {
                Sentry.captureException(e);
                logger.error(`Error while trying to ${op} S3 cache: ${messageFor(e)}`);
            });
    }

    override async getInternal(key: string): Promise<GetResult> {
        try {
            return await this.s3.get(key, this.path);
        } catch (e) {
            this.onError(e, 'read');
            return {hit: false};
        }
    }

    override async putInternal(key: string, value: Buffer, creator?: string): Promise<void> {
        const options: S3HandlerOptions = {
            metadata: creator ? {CreatedBy: creator} : {},
            redundancy: StorageClass.REDUCED_REDUNDANCY,
        };
        try {
            await this.s3.put(key, value, this.path, options);
        } catch (e) {
            this.onError(e, 'write');
        }
    }
}
