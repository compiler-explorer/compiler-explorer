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

import {logger} from '../logger';

import {Cache} from './base.interfaces';
import {InMemoryCache} from './in-memory';
import {MultiCache} from './multi';
import {NullCache} from './null';
import {OnDiskCache} from './on-disk';
import {S3Cache} from './s3';

function paramInt(config: string, param: string): number {
    const result = parseInt(param);
    if (isNaN(result)) throw new Error(`Bad params: ${config}`);
    return result;
}

function createInternal(name: string, config: string): Cache {
    if (!config) {
        return new NullCache(name);
    }
    const parts = config.split(';');
    if (parts.length > 1) {
        return new MultiCache(name, ...parts.map(part => createInternal(name, part)));
    }
    const match = config.match(/^([^(]+)\(([^)]+)\)$/);
    if (!match) throw new Error(`Unable to parse '${config}'`);
    const params = match[2].split(',');
    switch (match[1]) {
        case 'InMemory': {
            if (params.length !== 1) throw new Error(`Bad params: ${config}`);
            return new InMemoryCache(name, paramInt(config, params[0]));
        }

        case 'OnDisk': {
            if (params.length !== 2) throw new Error(`Bad params: ${config}`);
            return new OnDiskCache(name, params[0], paramInt(config, params[1]));
        }

        case 'S3': {
            if (params.length !== 3) throw new Error(`Bad params: ${config}`);
            return new S3Cache(name, params[0], params[1], params[2]);
        }

        default: {
            throw new Error(`Unrecognised cache type '${match[1]}'`);
        }
    }
}

export function createCacheFromConfig(name: string, config: string): Cache {
    const result = createInternal(name, config);
    logger.info(`Created cache ${name} of type ${result.details}`);
    return result;
}
