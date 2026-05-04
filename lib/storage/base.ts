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
import {PropertyGetter} from '../properties.interfaces.js';
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

export type StoredObject = {
    prefix: string;
    uniqueSubHash: string;
    fullHash: string;
    config: string;
};
export function encodeBuffer(buffer: Buffer): string {
    return utils.base32Encode(buffer);
}

export function isCleanText(text: string) {
    const lowercased = text.toLowerCase();
    return !profanities.some(badWord => lowercased.includes(badWord));
}

function getRawConfigHash(config: any) {
    return encodeBuffer(utils.getBinaryHash(JSON.stringify(config), FILE_HASH_VERSION));
}

export function getSafeHash(inputConfig: any) {
    // Shallow-clone so the nonce-rehashing loop doesn't mutate the caller's
    // object. The nonce is added at the top level only, so a shallow copy is
    // enough; the returned `config` string includes it via JSON.stringify.
    let config: any = {...inputConfig};
    let configHash = getRawConfigHash(config);
    let tries = 1;
    while (!isCleanText(configHash.substring(0, USABLE_HASH_CHECK_LENGTH))) {
        // Shake up the hash a bit by adding, or incrementing a nonce value.
        config.nonce = tries;
        logger.info(`Unusable text found in full hash ${configHash} - Trying again (${tries})`);
        if (tries <= MAX_TRIES) {
            configHash = getRawConfigHash(config);
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

function configFor(req: express.Request) {
    if (req.body.config) {
        return req.body.config;
    }
    if (req.body.sessions) {
        return req.body;
    }
    return null;
}

export abstract class StorageBase {
    constructor(
        public readonly httpRootDir: string,
        protected readonly compilerProps: CompilerProps | PropertyGetter,
    ) {}

    handler(req: express.Request, res: express.Response) {
        // Get the desired config and check for profanities in its hash
        const origConfig = configFor(req);
        if (!origConfig) {
            logger.error('No configuration found');
            res.status(500);
            res.send('Missing config parameter');
            return;
        }
        const {config, configHash} = getSafeHash(origConfig);
        this.findUniqueSubhash(configHash)
            .then(result => {
                logger.info(
                    `Unique subhash '${result.uniqueSubHash}' ` +
                        `(${result.alreadyPresent ? 'was already present' : 'newly-created'})`,
                );
                if (result.alreadyPresent) {
                    return result;
                }
                const storedObject: StoredObject = {
                    prefix: result.prefix,
                    uniqueSubHash: result.uniqueSubHash,
                    fullHash: configHash,
                    config: config,
                };

                return this.storeItem(storedObject, req);
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

    abstract storeItem(item: StoredObject, req: express.Request): Promise<any>;

    abstract findUniqueSubhash(hash: string): Promise<any>;

    abstract expandId(id: string): Promise<ExpandedShortLink>;

    abstract incrementViewCount(id: string): Promise<any>;
}
