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
import _ from 'underscore';

import {logger} from '../logger';

import {StorageBase} from './base';

const MIN_STORED_ID_LENGTH = 6;

export class StorageLocal extends StorageBase {
    static get key() {
        return 'local';
    }

    protected readonly storageFolder: string;

    constructor(httpRootDir, compilerProps) {
        super(httpRootDir, compilerProps);
        this.storageFolder = path.normalize(compilerProps.ceProps('localStorageFolder', './lib/storage/data/'));
        // Ensure we have a working storage dir before we have a chance to process anything
        fs.ensureDirSync(this.storageFolder);
        logger.info(`Using local storage solution on ${this.storageFolder}`);
    }

    async storeItem(item) {
        const filePath = path.join(this.storageFolder, item.uniqueSubHash);
        try {
            await fs.writeJson(filePath, item, {encoding: 'utf8'});
        } catch (err) {
            logger.error(`Caught exception while trying to store to ${path}`, err);
            throw err;
        }
        return item;
    }

    async findUniqueSubhash(hash: string) {
        logger.info(`Finding local unique subhash for ${hash}`);
        // This currently works on a hardcoded, local directory.
        try {
            const files = await fs.readdir(this.storageFolder);
            const prefix = hash.substring(0, MIN_STORED_ID_LENGTH);
            const filenames = _.chain(files)
                .filter(filename => filename.startsWith(prefix))
                .sort()
                .value();
            for (let i = MIN_STORED_ID_LENGTH; i < hash.length - 1; i++) {
                const subHash = hash.substring(0, i);
                // Check if the current subHash is present in the array
                const index = _.indexOf(filenames, subHash, true);
                if (index === -1) {
                    // Current base is not present, we have a new config in our hands
                    return {
                        prefix: prefix,
                        uniqueSubHash: subHash,
                        alreadyPresent: false,
                    };
                } else {
                    const expectedPath = path.join(this.storageFolder, subHash);
                    const item = await fs.readJson(expectedPath);
                    /* If the hashes coincide, it means this config has already been stored.
                     * Else, keep looking
                     */
                    if (item.fullHash === hash) {
                        return {
                            prefix: prefix,
                            uniqueSubHash: subHash,
                            alreadyPresent: true,
                        };
                    }
                }
            }
        } catch (err) {
            // IO error/Logic error, we have no way to store this right now. Please try again? What to do here?
            logger.error(`Error when looking for unique local subhash for ${hash}`, err);
            throw err;
        }
        // Causes why we're here: MIN_STORED_ID_LENGTH < hash.length. Something else?
        logger.error(`Hash too small when looking for unique local subhash for ${hash}`);
        throw new Error('Hash too small');
    }

    async expandId(id) {
        const expectedPath = path.join(this.storageFolder, id);
        logger.info(`Expanding local id ${id} to ${expectedPath}`);
        try {
            const item = await fs.readJson(expectedPath);
            return {
                config: item.config,
                specialMetadata: null,
            };
        } catch (err) {
            // IO error/Logic error, we have no way to store this right now. Please try again? What to do here?
            logger.error(`Error when expanding id ${id}`, err);
            throw err;
        }
    }

    async incrementViewCount() {
        // Nothing to do here, we don't store stats for local storage
    }
}
