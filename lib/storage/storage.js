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
    hash = require('../utils').getHash;

const FILE_HASH_VERSION = 'Compiler Explorer Config Hasher';


class StorageBase {
    constructor(compilerProps) {
        this.compilerProps = compilerProps;
    }

    handler(req, res) {
        const config = JSON.stringify(req.body.config);
        const configHash = Buffer.from(hash(config, FILE_HASH_VERSION)).toString('base64');
        this.findUniqueSubhash(configHash)
            .then(result => {
                logger.info(`File present ${result.alreadyPresent}. SubHas ${result.id}`);
                if (!result.alreadyPresent) {
                    const storedObject = {
                        id: result.id,
                        hash: configHash,
                        config: config
                    };

                    this.storeItem(storedObject);
                }
                res.set('Content-Type', 'application/json');
                res.send(JSON.stringify({ storedId: result.id }));
            })
            .catch((err) => {
                logger.error(err);
            });
    }

    storeItem(item) {
        logger.error(`Trying to store item from base storage ${item}`);
    }

    findUniqueSubhash(hash) {
        logger.error(`Trying to find unique subhash from base storage ${hash}`);
    }

    expandId(id) {
        logger.error(`Trying to expand from base storage ${id}`);
    }
}

function storageFactory(compilerProps) {
    const storageSolution = compilerProps.ceProps('storageSolution', 'local');
    const storage = require(`./storage-${storageSolution}`);
    return new storage(compilerProps);
}

module.exports = {
    StorageBase: StorageBase,
    storageFactory: storageFactory
};
