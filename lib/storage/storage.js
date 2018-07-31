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
    hash = require('../utils').getHash,
    fs = require('fs-extra');

const FILE_HASH_VERSION = 'Compiler Explorer Config Hasher';

class Storage {
    constructor(compilerProps) {
        this.storageSolution = compilerProps.ceProps('storageSolution', 'local');
        logger.info(`Using storage solution: ${this.storageSolution}`);
    }

    handler(req, res) {
        const config = JSON.stringify(req.body.config);
        const configHash = hash(config, FILE_HASH_VERSION);
        const subHash = this.findUniqueSubhash(configHash);
        const storedObject = {
            id: subHash,
            hash: configHash,
            config: config
        };

        this.storeItem(storedObject);

        res.setHeader('Content-Type', 'application/json');
        res.send(JSON.stringify({ storedId: subHash }));
    }

    storeItem(item) {
        const fileName = `${item.id}.json`;
        const filePath = `./lib/storage/data/${fileName}`;
        const stringified = JSON.stringify(item);
        fs.writeFile(filePath, stringified, 'utf8');
    }

    findUniqueSubhash(hash) {
        return hash;
    }

    expandId(id, cb) {
        const expectedPath = `./lib/storage/data/${id}.json`;
        // Move to promises once on node 10
        fs.readFile(expectedPath, 'utf8', (err, data) => {
            if (!err) {
                const item = JSON.parse(data);
                cb(item.config);
            } else {
                cb({});
            }
        });
    }
}

module.exports = Storage;
