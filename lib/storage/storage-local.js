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

const StorageBase = require('./storage').StorageBase,
    logger = require('../logger').logger,
    fs = require('fs-extra'),
    _ = require('underscore'),
    path = require('path');

const MIN_STORED_ID_LENGTH = 8;

class StorageLocal extends StorageBase {
    constructor(compilerProps) {
        super(compilerProps);
        this.storageFolder = path.normalize(compilerProps.ceProps('localStorageFolder', './lib/storage/data/'));
        // Ensure we have a working storage dir before we have a chance to process anything
        fs.ensureDirSync(this.storageFolder);
        logger.info(`Using local storage solution on ${this.storageFolder}`);
    }

    storeItem(item) {
        const filePath = path.join(this.storageFolder, item.id + '.json');
        fs.writeJson(filePath, item, {encoding: 'utf8'});
    }

    findUniqueSubhash(hash) {
        logger.info(`Finding local unique subhash for ${hash}`);
        // This currently works on a hardcoded, local directory.
        return fs.readdir(this.storageFolder)
            .then(files => {
                let base = hash.substring(0, MIN_STORED_ID_LENGTH);
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
                    return {
                        id: hash,
                        alreadyPresent: true
                    };
                }
                for (let i = MIN_STORED_ID_LENGTH; i < hash.length - 1; i++) {
                    let base = hash.substring(0, i);
                    // Check if the current base is present in the array
                    const index = _.indexOf(filenames, base, true);
                    if (index === -1) {
                        // Current base is not present, we have a new config in our hands
                        return {
                            id: base,
                            alreadyPresent: false
                        };
                    } else {
                        const expectedPath = path.join(this.storageFolder, base, '.json');
                        const item = fs.readJsonSync(expectedPath);
                        /* If the hashes coincide, it means this config has already been stored.
                         * TODO: What would happen on different config? (EXTRA UNLIKELY Collision)
                         * Else, keep looking
                         */
                        if (item.hash === hash) {
                            return {
                                id: base,
                                alreadyPresent: true
                            };
                        }
                    }
                }
                // Causes why we're here: MIN_STORED_ID_LENGTH < hash.length. Something else?
                throw 'Hash too small';
            })
            .catch(err => {
                // IO error/Logic error, we have no way to store this right now. Please try again? What to do here?
                logger.error(`Error when looking for unique local subhash for ${hash}: ${err}`);
                return err;
            });
    }

    expandId(id) {
        const expectedPath = path.join(this.storageFolder, id + '.json');
        logger.info(`Expanding local id ${id} to ${expectedPath}`);
        return fs.readJson(expectedPath)
            .then(item => item.config);
    }
}

module.exports = StorageLocal;
