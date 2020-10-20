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

import { Counter } from 'prom-client';

import { logger } from '../logger';
import { getHash } from '../utils';

const HashVersion = 'Compiler Explorer Cache Version 1';

const GetCounter = new Counter({
    name: 'ce_cache_get_total',
    help: 'Total number of cache gets',
    labelNames: ['result', 'name', 'type'],
});

const PutCounter = new Counter({
    name: 'ce_cache_put_total',
    help: 'Total number of cache puts',
    labelNames: ['type', 'name'],
});

export class BaseCache {
    constructor(cacheName, details, type) {
        this.cacheName = cacheName;
        this.details = details;
        this.type = type;
        this.gets = 0;
        this.hits = 0;
        this.puts = 0;
        this.countersEnabled = true;
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
        logger.info(`${this.cacheName} ${this.details}: cache stats: ${this.statString()}`);
    }

    static hash(object) {
        return getHash(object, HashVersion);
    }

    async get(key) {
        this.gets++;
        const result = await this.getInternal(key);
        if (result.hit) this.hits++;
        if (this.countersEnabled)
            GetCounter.inc({type: this.type, name: this.cacheName, result: result.hit ? 'hit' : 'miss'});
        return result;
    }

    async put(key, value, creator) {
        if (!(value instanceof Buffer))
            value = Buffer.from(value);
        this.puts++;
        if (this.countersEnabled)
            PutCounter.inc({type: this.type, name: this.cacheName});
        return this.putInternal(key, value, creator);
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
