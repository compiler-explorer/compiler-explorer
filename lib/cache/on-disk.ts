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

import path from 'path';

import fs from 'fs-extra';
import LRU from 'lru-cache';

import { logger } from '../logger';

import { BaseCache } from './base';

// With thanks to https://gist.github.com/kethinov/6658166
function getAllFiles(root, dir) {
    dir = dir || root;
    return fs.readdirSync(dir).reduce((files, file) => {
        const fullPath = path.join(dir, file);
        const name = path.relative(root, fullPath);
        const isDirectory = fs.statSync(fullPath).isDirectory();
        return isDirectory ? [...files, ...getAllFiles(root, fullPath)] : [...files, {name, fullPath}];
    }, []);
}

export class OnDiskCache extends BaseCache {
    constructor(path, cacheMb) {
        super(`OnDiskCache(${path}, ${cacheMb}mb)`);
        this.path = path;
        this.cacheMb = cacheMb;
        this.cache = new LRU({
            max: cacheMb * 1024 * 1024,
            length: n => n.size,
            noDisposeOnSet: true,
            dispose: (key, n) => {
                fs.unlink(n.path, () => {});
            },
        });
        fs.mkdirSync(path, { recursive: true });
        const info = getAllFiles(path).map(({name, fullPath}) => {
            const stat = fs.statSync(fullPath);
            return {
                key: name,
                sort: stat.ctimeMs,
                data: {
                    path: fullPath,
                    size: stat.size,
                },
            };
        });
        // Sort oldest first
        info.sort((x, y) => x.sort - y.sort);
        for (let i of info) {
            this.cache.set(i.key, i.data);
        }
    }

    statString() {
        return `${super.statString()}, LRU has ${this.cache.itemCount} item(s) ` +
            `totalling ${this.cache.length} bytes on disk`;
    }

    async getInternal(key) {
        const cached = this.cache.get(key);
        if (!cached) return {hit: false};

        try {
            const data = await fs.readFile(cached.path);
            return {hit: true, data: data};
        } catch (err) {
            logger.error(`error reading '${key}' from disk cache: `, err);
            return {hit: false};
        }
    }

    async putInternal(key, value) {
        const info = {
            path: path.join(this.path, key),
            size: value.length,
        };
        await fs.writeFile(info.path, value);
        return this.cache.set(key, info);
    }
}
