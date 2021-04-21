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

import { BaseCache } from './base';

// A write-through multiple cache.
// Writes get pushed to all caches, but reads are serviced from the first cache that returns
// a hit.
export class MultiCache extends BaseCache {
    constructor(cacheName, ...upstream) {
        super(cacheName, 'Multi', 'multi');
        this.countersEnabled = false;
        this.upstream = upstream;
    }

    statString() {
        return `${super.statString()}. ${this.upstream.map(c => `${c.details}: ${c.statString()}`).join('. ')}`;
    }

    propagateHitToMissedCaches(caches, key, data) {
        return Promise.all(caches.map(c => c.put(key, data)));
    }

    async getInternal(key) {
        const misses = [];

        for (const cache of this.upstream) {
            const result = await cache.get(key);

            if (result.hit) {
                await this.propagateHitToMissedCaches(misses, key, result.data);
                return result;
            }

            // keep track of caches that missed so we can write to them if we do hit
            misses.push(cache);
        }

        // all caches missed
        return {hit: false};
    }

    putInternal(object, value, creator) {
        return Promise.all(this.upstream.map(cache => {
            return cache.put(object, value, creator);
        }));
    }
}
