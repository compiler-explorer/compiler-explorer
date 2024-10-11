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

import * as express from 'express';

import {logger} from '../logger.js';

import {ExpandedShortLink, StorageBase} from './base.js';

export class StorageRemote extends StorageBase {
    static get key() {
        return 'remote';
    }

    protected readonly baseUrl: string;
    protected readonly get: (uri: string, options?: RequestInit) => Promise<Response>;
    protected readonly post: (uri: string, options?: RequestInit) => Promise<Response>;

    constructor(httpRootDir, compilerProps) {
        super(httpRootDir, compilerProps);

        this.baseUrl = compilerProps.ceProps('remoteStorageServer');
        this.get = (uri: string, options?: RequestInit) => fetch(new URL(uri, this.baseUrl).href, options);
        this.post = (uri: string, options?: RequestInit) => {
            return fetch(new URL(uri, this.baseUrl).href, {
                ...options,
                method: 'POST',
            });
        };
    }

    override async handler(req: express.Request, res: express.Response) {
        let resp, responseBody;
        try {
            resp = await this.post('/api/shortener', {
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(req.body),
            });
            responseBody = await resp.json();
        } catch (err: any) {
            logger.error(err);
            res.status(500);
            res.end(err.message);
            return;
        }

        const url = responseBody.url;
        if (!url) {
            res.status(resp.statusCode);
            res.send(resp.body);
            return;
        }

        const relativeUrl = url.substring(url.lastIndexOf('/z/') + 1);
        const shortlink = `${req.protocol}://${req.get('host')}${this.httpRootDir}${relativeUrl}`;

        res.send({url: shortlink});
    }

    async expandId(id: string): Promise<ExpandedShortLink> {
        const resp = await this.get(`/api/shortlinkinfo/${id}`);

        if (resp.status !== 200) throw new Error(`ID ${id} not present in remote storage`);

        return {
            config: await resp.text(),
            specialMetadata: null,
        };
    }

    async incrementViewCount() {}

    async findUniqueSubhash() {}

    async storeItem() {}
}
