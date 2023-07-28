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
import {profanities} from 'profanities';

import {logger} from '../logger.js';
import {CompilerProps} from '../properties.js';
import * as utils from '../utils.js';

const FILE_HASH_VERSION = 'Compiler Explorer Config Hasher 2';
/* How long a string to check for possible unusable hashes (Profanities or confusing text)
Note that a Hash might end up being longer than this!
 */
const USABLE_HASH_CHECK_LENGTH = 9; // Quite generous
const MAX_TRIES = 4;

export type ExpandedShortLink = {
    config: string;
    specialMetadata?: any;
    created?: Date;
};

export abstract class StorageBase {
    constructor(
        protected readonly httpRootDir: string,
        protected readonly compilerProps: CompilerProps,
    ) {}

    /**
     * Encode a buffer as a URL-safe string.
     */
    static encodeBuffer(buffer: Buffer): string {
        return utils.base32Encode(buffer);
    }

    static isCleanText(text: string) {
        const lowercased = text.toLowerCase();
        return !profanities.some(badWord => lowercased.includes(badWord));
    }

    static getRawConfigHash(config) {
        return StorageBase.encodeBuffer(utils.getBinaryHash(JSON.stringify(config), FILE_HASH_VERSION));
    }

    static getSafeHash(config) {
        // Keep rehashing until a usable text is found
        let configHash = StorageBase.getRawConfigHash(config);
        let tries = 1;
        while (!StorageBase.isCleanText(configHash.substr(0, USABLE_HASH_CHECK_LENGTH))) {
            // Shake up the hash a bit by adding, or incrementing a nonce value.
            config.nonce = tries;
            logger.info(`Unusable text found in full hash ${configHash} - Trying again (${tries})`);
            if (tries <= MAX_TRIES) {
                configHash = StorageBase.getRawConfigHash(config);
                ++tries;
            } else {
                logger.warn(`Gave up trying to find clean text for ${configHash}`);
                break;
            }
        }
        // And stringify it for the rest of the request
        config = JSON.stringify(config);
        return {config, configHash};
    }

    static configFor(req: express.Request) {
        if (req.body.config) {
            return req.body.config;
        } else if (req.body.sessions) {
            return req.body;
        }
        return null;
    }

    handler(req: express.Request, res: express.Response) {
        // Get the desired config and check for profanities in its hash
        const origConfig = StorageBase.configFor(req);
        if (!origConfig) {
            logger.error('No configuration found');
            res.status(500);
            res.send('Missing config parameter');
            return;
        }
        const {config, configHash} = StorageBase.getSafeHash(origConfig);
        this.findUniqueSubhash(configHash)
            .then(result => {
                logger.info(
                    `Unique subhash '${result.uniqueSubHash}' ` +
                        `(${result.alreadyPresent ? 'was already present' : 'newly-created'})`,
                );
                if (result.alreadyPresent) {
                    return result;
                } else {
                    const storedObject = {
                        prefix: result.prefix,
                        uniqueSubHash: result.uniqueSubHash,
                        fullHash: configHash,
                        config: config,
                    };

                    return this.storeItem(storedObject, req);
                }
            })
            .then(result => {
                res.send({url: `${req.protocol}://${req.get('host')}${this.httpRootDir}z/${result.uniqueSubHash}`});
            })
            .catch(err => {
                logger.error(err);
                res.status(500);
                res.send(err.message);
            });
    }

    abstract storeItem(item, req: express.Request): Promise<any>;

    abstract findUniqueSubhash(hash: string): Promise<any>;

    abstract expandId(id: string): Promise<ExpandedShortLink>;

    abstract incrementViewCount(id): Promise<any>;
}
