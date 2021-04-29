// Copyright (c) 2021, Compiler Explorer Authors
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

import { Cluster } from 'ioredis';

import { logger } from '../logger';

import { BaseCache } from './base';

const REDIS_EXPIRE_SECONDS = 60 * 60 * 24;
const REDIS_MAX_CACHE_ENTRY_SIZE = 1024 * 3;

export class RedisClusterCache extends BaseCache {
    constructor(cacheName, hostname, port) {
        super(cacheName, `RedisCluster(${hostname}:${port})`, 'redis-cluster');

        const members = [{ hostname: hostname, port: port }];
        this.redis = new Cluster(members, {
            scaleReads: 'all',
            enableOfflineQueue: false,
            commandTimeout: 100, // 100ms
            // use exponential backoff
            clusterRetryStrategy: (times) =>  Math.min(100 + Math.pow(2, times), 60000),
            lazyConnect: true,
        });

        // need to register an error listener to prevent regular console logging
        this.redis.on('error', err => logger.error('redis error:', err));
    }

    makeRedisKey(key) {
        return `${this.cacheName}:${key}`;
    }

    async getInternal(key) {
        const rkey = this.makeRedisKey(key);
        try {
            // TODO: use GETEX once available on elasticache
            const value = await this.redis.get(rkey);

            // redis returns null if key does not exist
            const miss = value === null;

            // empty string indicates the value was too large, so we didn't hit
            // but isn't an explicit miss, which allows the next cache to be checked
            const hit = !miss && value !== '';

            return {hit, miss, data: value};
        } catch (err) {
            logger.error(`redis cache read error. cache=${this.cacheName}`, err);
            return {hit: false};
        }
    }

    async putInternal(key, value) {
        // override value to empty string if it is too large
        // this allows redis to be used as a fast authoritative
        // cache in front of a slow backing cache
        if (value.length > REDIS_MAX_CACHE_ENTRY_SIZE)
            value = Buffer.from('');

        const rkey = this.makeRedisKey(key);
        try {
            await this.redis.setex(rkey, REDIS_EXPIRE_SECONDS, value);
        } catch (err) {
            logger.error(`redis cache write error. cache=${this.cacheName}`, err);
        }
    }
}
