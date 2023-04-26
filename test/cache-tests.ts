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

import {Readable} from 'stream';

import {GetObjectCommand, NoSuchKey, PutObjectCommand, S3} from '@aws-sdk/client-s3';
import {sdkStreamMixin} from '@aws-sdk/util-stream-node';
import {AwsClientStub, AwsError, AwsStub, mockClient} from 'aws-sdk-client-mock';
import temp from 'temp';

import {BaseCache} from '../lib/cache/base.js';
import {createCacheFromConfig} from '../lib/cache/from-config.js';
import {InMemoryCache} from '../lib/cache/in-memory.js';
import {MultiCache} from '../lib/cache/multi.js';
import {NullCache} from '../lib/cache/null.js';
import {OnDiskCache} from '../lib/cache/on-disk.js';
import {S3Cache} from '../lib/cache/s3.js';

import {fs, path, shouldExist} from './utils.js';

function newTempDir() {
    temp.track(true);
    return temp.mkdirSync({prefix: 'compiler-explorer-cache-tests', dir: process.env.tmpDir});
}

function basicTests(factory: () => BaseCache) {
    it('should start empty', () => {
        const cache = factory();
        cache.stats().should.eql({hits: 0, puts: 0, gets: 0});
        return cache
            .get('not a key')
            .should.eventually.contain({hit: false})
            .then(x => {
                cache.stats().should.eql({hits: 0, puts: 0, gets: 1});
                return x;
            });
    });

    it('should store and retrieve strings', () => {
        const cache = factory();
        return cache
            .put('a key', 'a value', 'bob')
            .then(() => {
                cache.stats().should.eql({hits: 0, puts: 1, gets: 0});
                return cache.get('a key').should.eventually.eql({
                    hit: true,
                    data: Buffer.from('a value'),
                });
            })
            .then(x => {
                cache.stats().should.eql({hits: 1, puts: 1, gets: 1});
                return x;
            });
    });

    it('should store and retrieve binary buffers', () => {
        const cache = factory();
        const buffer = Buffer.alloc(2 * 1024);
        buffer.fill('@');
        return cache
            .put('a key', buffer, 'bob')
            .then(() => {
                cache.stats().should.eql({hits: 0, puts: 1, gets: 0});
                return cache.get('a key').should.eventually.eql({
                    hit: true,
                    data: buffer,
                });
            })
            .then(x => {
                cache.stats().should.eql({hits: 1, puts: 1, gets: 1});
                return x;
            });
    });
}

describe('In-memory caches', () => {
    basicTests(() => new InMemoryCache('test', 10));
    it('should give extra stats', () => {
        const cache = new InMemoryCache('test', 1);
        cache
            .statString()
            .should.equal('0 puts; 0 gets, 0 hits, 0 misses (0.00%), LRU has 0 item(s) totalling 0 bytes');
    });

    it('should evict old objects', () => {
        const cache = new InMemoryCache('test', 1);
        return cache
            .put('a key', 'a value', 'bob')
            .then(() => {
                const promises: Promise<void>[] = [];
                const oneK = ''.padEnd(1024);
                for (let i = 0; i < 1024; i++) {
                    promises.push(cache.put(`key${i}`, oneK));
                }
                return Promise.all(promises);
            })
            .then(() => {
                return cache.get('a key').should.eventually.contain({hit: false});
            });
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

    it('should write through', () => {
        const subCache1 = new InMemoryCache('test', 1);
        const subCache2 = new InMemoryCache('test', 1);
        const cache = new MultiCache('test', subCache1, subCache2);
        return cache.put('a key', 'a value', 'bob').then(() => {
            return Promise.all([
                cache.get('a key').should.eventually.eql({hit: true, data: Buffer.from('a value')}),
                subCache1.get('a key').should.eventually.eql({hit: true, data: Buffer.from('a value')}),
                subCache2.get('a key').should.eventually.eql({hit: true, data: Buffer.from('a value')}),
            ]);
        });
    });

    it('services from the first cache hit', async () => {
        const subCache1 = new InMemoryCache('test', 1);
        const subCache2 = new InMemoryCache('test', 1);
        // Set up caches with deliberately skew values for the same key.
        await subCache1.put('a key', 'cache1');
        await subCache2.put('a key', 'cache2');
        const cache = new MultiCache('test', subCache1, subCache2);
        await cache.get('a key').should.eventually.eql({hit: true, data: Buffer.from('cache1')});

        subCache1.hits.should.equal(1);
        subCache1.gets.should.equal(1);
        subCache2.hits.should.equal(0);
        subCache2.gets.should.equal(0);

        await subCache1.get('a key').should.eventually.eql({hit: true, data: Buffer.from('cache1')});
        await subCache2.get('a key').should.eventually.eql({hit: true, data: Buffer.from('cache2')});
    });
});

describe('On disk caches', () => {
    basicTests(() => new OnDiskCache('test', newTempDir(), 10));
    it('should evict old objects', () => {
        const tempDir = newTempDir();
        const cache = new OnDiskCache('test', tempDir, 1);
        return cache
            .put('a key', 'a value', 'bob')
            .then(() => {
                const promises: Promise<void>[] = [];
                const oneHundredK = ''.padEnd(1024 * 100);
                for (let i = 0; i < 12; i++) {
                    promises.push(cache.put(`key${i}`, oneHundredK));
                }
                return Promise.all(promises);
            })
            .then(() => {
                return cache.get('a key').should.eventually.contain({hit: false});
            });
    });

    it('should handle existing data', () => {
        const tempDir = newTempDir();
        fs.writeFileSync(path.join(tempDir, 'abcdef'), 'this is abcdef');
        fs.mkdirSync(path.join(tempDir, 'path'));
        fs.writeFileSync(path.join(tempDir, 'path', 'test'), 'this is path/test');
        const cache = new OnDiskCache('test', tempDir, 1);
        return Promise.all([
            cache.get('abcdef').should.eventually.eql({hit: true, data: Buffer.from('this is abcdef')}),
            cache.get(path.join('path', 'test')).should.eventually.eql({
                hit: true,
                data: Buffer.from('this is path/test'),
            }),
        ]);
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

    it('should correctly handle errors', () => {
        mockS3.on(GetObjectCommand, {Bucket: 'test.bucket'}).rejects('Some s3 error');
        let err: Error | null = null;
        const cache = new S3Cache('test', 'test.bucket', 'cache', 'uk-north-1', (e: Error, op: string) => {
            err = e;
            op.should.equal('read');
        });
        return cache
            .get('doesntmatter')
            .should.eventually.contain({hit: false})
            .then(x => {
                cache.stats().should.eql({hits: 0, puts: 0, gets: 1});
                if (shouldExist(err)) {
                    err.toString().should.equal('Error: Some s3 error');
                }
                return x;
            });
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
        cache.constructor.should.eql(NullCache);
        cache.cacheName.should.eql('name');
    });
    it('should throw on bad types', () => {
        (() => createCacheFromConfig('test', 'InMemory')).should.throw();
        (() => createCacheFromConfig('test', 'NotAType()')).should.throw();
    });
    it('should create in memory caches', () => {
        const cache = createCacheFromConfig<InMemoryCache>('test', 'InMemory(123)');
        cache.constructor.should.eql(InMemoryCache);
        cache.cacheMb.should.equal(123);
        (() => createCacheFromConfig('test', 'InMemory()')).should.throw();
        (() => createCacheFromConfig('test', 'InMemory(argh)')).should.throw();
        (() => createCacheFromConfig('test', 'InMemory(123,yibble)')).should.throw();
    });
    it('should create on disk caches', () => {
        const tempDir = newTempDir();
        const cache = createCacheFromConfig<OnDiskCache>('test', `OnDisk(${tempDir},456)`);
        cache.constructor.should.eql(OnDiskCache);
        cache.path.should.equal(tempDir);
        cache.cacheMb.should.equal(456);
        (() => createCacheFromConfig('test', 'OnDisk()')).should.throw();
        (() => createCacheFromConfig('test', 'OnDisk(argh,yibble)')).should.throw();
        (() => createCacheFromConfig('test', 'OnDisk(/tmp/moo,456,blah)')).should.throw();
    });
    it('should create S3 caches', () => {
        const cache = createCacheFromConfig<S3Cache>('test', `S3(test.bucket,cache,uk-north-1)`);
        cache.constructor.should.eql(S3Cache);
        cache.path.should.equal('cache');
        cache.region.should.equal('uk-north-1');
        (() => createCacheFromConfig('test', 'S3()')).should.throw();
        (() => createCacheFromConfig('test', 'S3(argh,yibble)')).should.throw();
        (() => createCacheFromConfig('test', 'S3(/tmp/moo,456,blah,nork)')).should.throw();
    });
    it('should create multi caches', () => {
        const tempDir = newTempDir();
        const cache = createCacheFromConfig<MultiCache>(
            'multi',
            `InMemory(123);OnDisk(${tempDir},456);S3(test.bucket,cache,uk-north-1)`,
        );
        cache.constructor.should.eql(MultiCache);

        const upstream: BaseCache[] = (cache as any).upstream; // This isn't pretty. upstream is private.
        upstream.length.should.equal(3);
        upstream[0].constructor.should.eql(InMemoryCache);
        upstream[1].constructor.should.eql(OnDiskCache);
        upstream[2].constructor.should.eql(S3Cache);
        upstream[0].cacheName.should.eql('multi');
        upstream[1].cacheName.should.eql('multi');
        upstream[2].cacheName.should.eql('multi');
    });
});
