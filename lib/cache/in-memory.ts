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

import LRU from 'lru-cache';

import type {GetResult} from '../../types/cache.interfaces.js';

import {BaseCache} from './base.js';

export class InMemoryCache extends BaseCache {
    readonly cacheMb: number;
    private readonly cache: LRU;

    constructor(cacheName: string, cacheMb: number) {
        super(cacheName, `InMemoryCache(${cacheMb}Mb)`, 'memory');
        this.cacheMb = cacheMb;
        this.cache = new LRU({
            max: cacheMb * 1024 * 1024,
            length: n => n.length,
        });
    }

    override statString(): string {
        return `${super.statString()}, LRU has ${this.cache.itemCount} item(s) totalling ${this.cache.length} bytes`;
    }

    override async getInternal(key: string): Promise<GetResult> {
        const cached = this.cache.get(key);
        return {hit: !!cached, data: cached};
    }

    override async putInternal(key: string, value: Buffer): Promise<void> {
        this.cache.set(key, value);
    }
}
