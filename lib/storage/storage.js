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
    hash = require('../utils').getBinaryHash;

const FILE_HASH_VERSION = 'Compiler Explorer Config Hasher';


class StorageBase {
    constructor(compilerProps) {
        this.compilerProps = compilerProps;
    }

    /**
     * Encode a buffer as a URL-safe string.
     * @param {Buffer} buffer
     * @returns {string}
     */
    static safe64Encoded(buffer) {
        return buffer.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    }

    handler(req, res) {
        let config = null;
        if (req.body.config) {
            config = JSON.stringify(req.body.config);
        } else if (req.body.sessions) {
            config = JSON.stringify(req.body);
        }

        const configHash = StorageBase.safe64Encoded(hash(config, FILE_HASH_VERSION));
        this.findUniqueSubhash(configHash)
            .then(result => {
                logger.info(`Unique subhash '${result.uniqueSubHash}' ` +
                    `(${result.alreadyPresent ? "was already present" : "newly-created"})`);
                if (!result.alreadyPresent) {
                    const storedObject = {
                        prefix: result.prefix,
                        uniqueSubHash: result.uniqueSubHash,
                        fullHash: configHash,
                        config: config
                    };

                    return this.storeItem(storedObject, req);
                } else {
                    return result;
                }
            })
            .then(result => {
                res.set('Content-Type', 'application/json');
                res.send(JSON.stringify({storedId: result.uniqueSubHash}));
            })
            .catch(err => {
                logger.error(err);
                res.status(500);
                res.end(err.message);
            });
    }

    storeItem(item) {
        logger.error('Trying to store item from base storage' + item);
        return Promise.reject();
    }

    findUniqueSubhash(hash) {
        logger.error(`Trying to find unique subhash from base storage ${hash}`);
        return Promise.reject();
    }

    expandId(id) {
        logger.error(`Trying to expand from base storage ${id}`);
        return Promise.reject();
    }
}

function storageFactory(compilerProps, awsProps) {
    const storageSolution = compilerProps.ceProps('storageSolution', 'local');
    const storage = require(`./storage-${storageSolution}`);
    return new storage(compilerProps, awsProps);
}

module.exports = {
    StorageBase: StorageBase,
    storageFactory: storageFactory
};
