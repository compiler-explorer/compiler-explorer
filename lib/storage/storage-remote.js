// Copyright (c) 2018, Compiler Explorer Authors
//
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

const logger = require('../logger').logger,
    request = require('request'),
    StorageBase = require('./storage').StorageBase,
    util = require('util');

class StorageRemote extends StorageBase {
    constructor(httpRootDir, compilerProps) {
        super(httpRootDir, compilerProps);

        this.baseUrl = compilerProps.ceProps('remoteStorageServer');

        const req = request.defaults({
            baseUrl: this.baseUrl,
        });

        this.get = util.promisify(req.get);
        this.post = util.promisify(req.post);
    }

    async handler(req, res) {
        let resp;
        try {
            resp = await this.post('/shortener', {
                json: true,
                body: req.body,
            });
        } catch (err) {
            logger.error(err);
            res.status(500);
            res.end(err.message);
            return;
        }

        const url = resp.body.url;
        if (!url) {
            res.status(resp.statusCode);
            res.send(resp.body);
            return;
        }

        const relativeUrl = url.substring(url.lastIndexOf('/z/') + 1);
        const shortlink = `${req.protocol}://${req.get('host')}${this.httpRootDir}${relativeUrl}`;

        res.send({url: shortlink});
    }

    async expandId(id) {
        const resp = await this.get(`/api/shortlinkinfo/${id}`);

        if (resp.statusCode !== 200) throw new Error(`ID ${id} not present in remote storage`);

        return {
            config: resp.body,
            specialMetadata: null,
        };
    }

    async incrementViewCount() {
    }
}

module.exports = StorageRemote;
