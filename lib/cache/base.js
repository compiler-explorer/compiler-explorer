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

import { promisify } from 'util';

import { logger } from '../logger';
import { getHash } from '../utils';

const HashVersion = 'Compiler Explorer Cache Version 1';
const sleep = promisify(setTimeout);

export class BaseCache {
    constructor(name) {
        this.name = name;
        this.gets = 0;
        this.hits = 0;
        this.puts = 0;

        this.anticipatedPromises = [];
        this.promiseCache = {};
    }

    stats() {
        return {hits: this.hits, puts: this.puts, gets: this.gets};
    }

    statString() {
        const pc = this.gets ? (100 * this.hits) / this.gets : 0;
        const misses = this.gets - this.hits;
        return `${this.puts} puts; ${this.gets} gets, ${this.hits} hits, ${misses} misses (${pc.toFixed(2)}%)`;
    }

    report() {
        logger.info(`${this.name}: cache stats: ${this.statString()}`);
    }

    static hash(object) {
        return getHash(object, HashVersion);
    }

    async cleanupNamedPromises() {
        this.anticipatedPromises = [];

        for (let key in this.promiseCache) {
            await this.promiseCache;
            delete this.promiseCache[key];
        }
    }

    async get(key) {
        this.gets++;
        const result = await this.getInternal(key);
        if (result.hit) this.hits++;
        return result;
    }

    async waitForNamedPromiseToAppear(key) {
        while (!Object.keys(this.promiseCache).includes(key)) {
            await sleep(10);
        }
    }

    async getNamedPromise(key) {
        if (this.anticipatedPromises.includes(key)) {
            await this.waitForNamedPromiseToAppear(key);
        }

        if (Object.keys(this.promiseCache).includes(key)) {
            const result = await this.promiseCache[key];
            const jsonResult = JSON.stringify(result);
            this.put(key, jsonResult);
            return Promise.resolve({
                hit: !!result,
                data: jsonResult,
            });
        } else {
            this.anticipatedPromises.push(key);
            return this.get(key);
        }
    }

    async put(key, value, creator) {
        if (!(value instanceof Buffer))
            value = Buffer.from(value);
        this.puts++;
        return this.putInternal(key, value, creator);
    }

    async putNamedPromise(key, value) {
        this.promiseCache[key] = value;
        const result = await this.getNamedPromise(key);
        return JSON.parse(result.data);
    }

    // eslint-disable-next-line no-unused-vars
    async getInternal(key) {
        throw new Error('should be implemented in subclass');
    }

    // eslint-disable-next-line no-unused-vars
    async putInternal(key, value, creator) {
        throw new Error('should be implemented in subclass');
    }
}
