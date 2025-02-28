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
import path from 'node:path';
import {Readable} from 'node:stream';

import {GetObjectCommand, NoSuchKey, PutObjectCommand, S3} from '@aws-sdk/client-s3';
import {sdkStreamMixin} from '@smithy/util-stream';
import {AwsClientStub, mockClient} from 'aws-sdk-client-mock';
import {beforeEach, describe, expect, it} from 'vitest';

import {BaseCache} from '../lib/cache/base.js';
import {createCacheFromConfig} from '../lib/cache/from-config.js';
import {InMemoryCache} from '../lib/cache/in-memory.js';
import {MultiCache} from '../lib/cache/multi.js';
import {NullCache} from '../lib/cache/null.js';
import {OnDiskCache} from '../lib/cache/on-disk.js';
import {S3Cache} from '../lib/cache/s3.js';

import {newTempDir} from './utils.js';

function basicTests(factory: () => BaseCache) {
    it('should start empty', async () => {
        const cache = factory();
        expect(cache.stats()).toEqual({hits: 0, puts: 0, gets: 0});
        await expect(cache.get('not a key')).resolves.toHaveProperty('hit', false);
        expect(cache.stats()).toEqual({hits: 0, puts: 0, gets: 1});
    });

    it('should store and retrieve strings', async () => {
        const cache = factory();
        await cache.put('a key', 'a value', 'bob');
        expect(cache.stats()).toEqual({hits: 0, puts: 1, gets: 0});
        await expect(cache.get('a key')).resolves.toEqual({
            hit: true,
            data: Buffer.from('a value'),
        });
        expect(cache.stats()).toEqual({hits: 1, puts: 1, gets: 1});
    });

    it('should store and retrieve binary buffers', async () => {
        const cache = factory();
        const buffer = Buffer.alloc(2 * 1024);
        buffer.fill('@');
        await cache.put('a key', buffer, 'bob');
        expect(cache.stats()).toEqual({hits: 0, puts: 1, gets: 0});
        await expect(cache.get('a key')).resolves.toEqual({
            hit: true,
            data: buffer,
        });
        expect(cache.stats()).toEqual({hits: 1, puts: 1, gets: 1});
    });
}

describe('In-memory caches', () => {
    basicTests(() => new InMemoryCache('test', 10));
    it('should give extra stats', () => {
        const cache = new InMemoryCache('test', 1);
        expect(cache.statString()).toEqual(
            '0 puts; 0 gets, 0 hits, 0 misses (0.00%), LRU has 0 item(s) totalling 0 bytes',
        );
    });

    it('should evict old objects', async () => {
        const cache = new InMemoryCache('test', 1);
        await cache.put('a key', 'a value', 'bob');
        const promises: Promise<void>[] = [];
        const oneK = ''.padEnd(1024);
        for (let i = 0; i < 1024; i++) {
            promises.push(cache.put(`key${i}`, oneK));
        }
        await Promise.all(promises);
        await expect(cache.get('a key')).resolves.toHaveProperty('hit', false);
    });
});

describe('Multi caches', () => {
    basicTests(
        () =>
            new MultiCache(
                'test',
                new InMemoryCache('test', 10),
                new InMemoryCache('test', 20),
                new InMemoryCache('test', 30),
            ),
    );

    it('should write through', async () => {
        const subCache1 = new InMemoryCache('test', 1);
        const subCache2 = new InMemoryCache('test', 1);
        const cache = new MultiCache('test', subCache1, subCache2);
        await cache.put('a key', 'a value', 'bob');
        await expect(cache.get('a key')).resolves.toEqual({hit: true, data: Buffer.from('a value')});
        await expect(subCache1.get('a key')).resolves.toEqual({hit: true, data: Buffer.from('a value')});
        await expect(subCache2.get('a key')).resolves.toEqual({hit: true, data: Buffer.from('a value')});
    });

    it('services from the first cache hit', async () => {
        const subCache1 = new InMemoryCache('test', 1);
        const subCache2 = new InMemoryCache('test', 1);
        // Set up caches with deliberately skew values for the same key.
        await subCache1.put('a key', 'cache1');
        await subCache2.put('a key', 'cache2');
        const cache = new MultiCache('test', subCache1, subCache2);
        await expect(cache.get('a key')).resolves.toEqual({hit: true, data: Buffer.from('cache1')});

        expect(subCache1.hits).toEqual(1);
        expect(subCache1.gets).toEqual(1);
        expect(subCache2.hits).toEqual(0);
        expect(subCache2.gets).toEqual(0);

        await expect(subCache1.get('a key')).resolves.toEqual({hit: true, data: Buffer.from('cache1')});
        await expect(subCache2.get('a key')).resolves.toEqual({hit: true, data: Buffer.from('cache2')});
    });
});

describe('On disk caches', () => {
    basicTests(() => new OnDiskCache('test', newTempDir(), 10));
    it('should evict old objects', async () => {
        const tempDir = newTempDir();
        const cache = new OnDiskCache('test', tempDir, 1);
        await cache.put('a key', 'a value', 'bob');
        const promises: Promise<void>[] = [];
        const oneHundredK = ''.padEnd(1024 * 100);
        for (let i = 0; i < 12; i++) {
            promises.push(cache.put(`key${i}`, oneHundredK));
        }
        await Promise.all(promises);
        await expect(cache.get('a key')).resolves.toHaveProperty('hit', false);
    });

    it('should handle existing data', async () => {
        const tempDir = newTempDir();
        fs.writeFileSync(path.join(tempDir, 'abcdef'), 'this is abcdef');
        fs.mkdirSync(path.join(tempDir, 'path'));
        fs.writeFileSync(path.join(tempDir, 'path', 'test'), 'this is path/test');
        const cache = new OnDiskCache('test', tempDir, 1);
        await expect(cache.get('abcdef')).resolves.toEqual({hit: true, data: Buffer.from('this is abcdef')});
        await expect(cache.get(path.join('path', 'test'))).resolves.toEqual({
            hit: true,
            data: Buffer.from('this is path/test'),
        });
    });

    // MRG ideally handle the case of pre-populated stuff overflowing the size
    // and test sorting by mtime, but that might be too tricky.
});

function setup(mockS3: AwsClientStub<S3>) {
    const S3FS = {};
    mockS3.on(GetObjectCommand, {Bucket: 'test.bucket'}).callsFake(params => {
        const result = S3FS[params.Key];
        if (result) {
            const stream = new Readable();
            stream.push(result);
            stream.push(null);
            return {Body: sdkStreamMixin(stream)};
        }
        throw new NoSuchKey({message: 'No such key', $metadata: {}});
    });
    mockS3.on(PutObjectCommand, {Bucket: 'test.bucket'}).callsFake(params => {
        S3FS[params.Key] = params.Body;
        return {};
    });
}

describe('S3 tests', () => {
    const mockS3 = mockClient(S3);
    beforeEach(() => {
        mockS3.reset();
        setup(mockS3);
    });
    basicTests(() => new S3Cache('test', 'test.bucket', 'cache', 'uk-north-1'));

    it('should correctly handle errors', async () => {
        mockS3.on(GetObjectCommand, {Bucket: 'test.bucket'}).rejects('Some s3 error');
        let err: Error = new Error('not an error');
        const cache = new S3Cache('test', 'test.bucket', 'cache', 'uk-north-1', (e: Error, op: string) => {
            err = e;
            expect(op).toEqual('read');
        });
        await expect(cache.get('doesntmatter')).resolves.toHaveProperty('hit', false);
        expect(cache.stats()).toEqual({hits: 0, puts: 0, gets: 1});
        expect(err.toString()).toEqual('Error: Some s3 error');
    });

    // BE VERY CAREFUL - the below can be used with sufficient permissions to test on prod (With mocks off)...
    // basicTests(() => new S3Cache(''test', storage.godbolt.org', 'cache', 'us-east-1'));
});

describe('Config tests', () => {
    const mockS3 = mockClient(S3);
    beforeEach(() => {
        mockS3.reset();
        setup(mockS3);
    });
    it('should create null cache on empty config', () => {
        const cache = createCacheFromConfig('name', '');
        expect(cache.constructor).toEqual(NullCache);
        expect(cache.cacheName).toEqual('name');
    });
    it('should throw on bad types', () => {
        expect(() => createCacheFromConfig('test', 'InMemory')).toThrow();
        expect(() => createCacheFromConfig('test', 'NotAType()')).toThrow();
    });
    it('should create in memory caches', () => {
        const cache = createCacheFromConfig<InMemoryCache>('test', 'InMemory(123)');
        expect(cache.constructor).toEqual(InMemoryCache);
        expect(cache.cacheMb).toEqual(123);
        expect(() => createCacheFromConfig('test', 'InMemory()')).toThrow();
        expect(() => createCacheFromConfig('test', 'InMemory(argh)')).toThrow();
        expect(() => createCacheFromConfig('test', 'InMemory(123,yibble)')).toThrow();
    });
    it('should create on disk caches', () => {
        const tempDir = newTempDir();
        const cache = createCacheFromConfig<OnDiskCache>('test', `OnDisk(${tempDir},456)`);
        expect(cache.constructor).toEqual(OnDiskCache);
        expect(cache.path).toEqual(tempDir);
        expect(cache.cacheMb).toEqual(456);
        expect(() => createCacheFromConfig('test', 'OnDisk()')).toThrow();
        expect(() => createCacheFromConfig('test', 'OnDisk(argh,yibble)')).toThrow();
        expect(() => createCacheFromConfig('test', 'OnDisk(/tmp/moo,456,blah)')).toThrow();
    });
    it('should create S3 caches', () => {
        const cache = createCacheFromConfig<S3Cache>('test', 'S3(test.bucket,cache,uk-north-1)');
        expect(cache.constructor).toEqual(S3Cache);
        expect(cache.path).toEqual('cache');
        expect(cache.region).toEqual('uk-north-1');
        expect(() => createCacheFromConfig('test', 'S3()')).toThrow();
        expect(() => createCacheFromConfig('test', 'S3(argh,yibble)')).toThrow();
        expect(() => createCacheFromConfig('test', 'S3(/tmp/moo,456,blah,nork)')).toThrow();
    });
    it('should create multi caches', () => {
        const tempDir = newTempDir();
        const cache = createCacheFromConfig<MultiCache>(
            'multi',
            `InMemory(123);OnDisk(${tempDir},456);S3(test.bucket,cache,uk-north-1)`,
        );
        expect(cache.constructor).toEqual(MultiCache);

        const upstream: BaseCache[] = (cache as any).upstream; // This isn't pretty. upstream is private.
        expect(upstream.length).toEqual(3);
        expect(upstream[0].constructor).toEqual(InMemoryCache);
        expect(upstream[1].constructor).toEqual(OnDiskCache);
        expect(upstream[2].constructor).toEqual(S3Cache);
        expect(upstream[0].cacheName).toEqual('multi');
        expect(upstream[1].cacheName).toEqual('multi');
        expect(upstream[2].cacheName).toEqual('multi');
    });
});
