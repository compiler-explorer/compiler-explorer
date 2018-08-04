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
    fs = require('fs-extra'),
    _ = require('underscore');

const FILE_HASH_VERSION = 'Compiler Explorer Config Hasher';

class Storage {
    constructor(compilerProps) {
        this.storageSolution = compilerProps.ceProps('storageSolution', 'local');
        logger.info(`Using storage solution: ${this.storageSolution}`);
    }

    handler(req, res) {
        const config = JSON.stringify(req.body.config);
        const configHash = hash(config, FILE_HASH_VERSION);
        this.findUniqueSubhash(configHash)
            .then((alreadyPresent, subHash) => {
                const storedObject = {
                    id: subHash,
                    hash: configHash,
                    config: config
                };

                if (!alreadyPresent) {
                    this.storeItem(storedObject);
                }

                res.setHeader('Content-Type', 'application/json');
                res.send(JSON.stringify({ storedId: subHash }));
            });
    }

    storeItem(item) {
        const fileName = `${item.id}.json`;
        const filePath = `./lib/storage/data/${fileName}`;
        const stringified = JSON.stringify(item);
        fs.writeFile(filePath, stringified, 'utf8');
    }

    findUniqueSubhash(hash) {
        // This currently works on a hardcoded, local directory.
        // TODO: Expand to s3, make paths configurable
        return new Promise((resolve, reject) => {
            fs.readdir('./lib/storage/data/', (err, files) => {
                if (!err) {
                    let base = hash.substring(0, 8);
                    const filenames = _.chain(files)
                        .map(files, file => {
                            /* Check for possible files that match this id. Much of time, this will filter all but one
                             * Except when the first initial chars coincide in some cases
                             */
                            if (file.startsWith(base)) {
                                return file.substring(0, file.length - 5);
                            }
                            return null;
                        })
                        .compact()
                        .sort()
                        .value();
                    // Shortcircuit if somehow the full hash is there already (VERY unlikely)
                    if (_.indexOf(filenames, hash, true) !== -1) {
                        return resolve(true, hash);
                    }
                    for (let i = 8;i < hash.length - 1;i++) {
                        let base = hash.substring(0, i);
                        // Check if the current base is present in the array
                        const index = _.indexOf(filenames, base, true);
                        if (index === -1) {
                            // Current base is not present, we have a new config in our hands
                            return resolve(false, base);
                        } else {
                            const fileContents = fs.readFileSync(`./lib/storage/data/${base}.json`, 'utf8');
                            const fileData = JSON.parse(fileContents);
                            /* If the hashes coincide, it means this config has already been stored.
                             * TODO: What happens on diff config? (Collision)
                             * Else, keep looking
                             */
                            if (fileData.hash === hash) {
                                return resolve(true, base);
                            }
                        }
                    }
                    // If we reach here, it means we did not find a valid unique id, which could mean
                    return reject();
                } else {
                    // IO error, we have no way to store this right now. Please try again? What to do here?
                    return reject();
                }
            });
        });
    }

    expandId(id) {
        return new Promise((resolve, reject) => {
            const expectedPath = `./lib/storage/data/${id}.json`;
            // Move to promises once on node 10
            fs.readFile(expectedPath, 'utf8', (err, data) => {
                if (!err) {
                    const item = JSON.parse(data);
                    return resolve(item.config);
                } else {
                    return reject();
                }
            });
        });
    }
}

module.exports = Storage;
