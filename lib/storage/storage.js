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
    hash = require('../utils').getBinaryHash,
    /***
     * @type {string[]}
     */
    profanities = require('profanities');

const FILE_HASH_VERSION = 'Compiler Explorer Config Hasher';
const USABLE_NAME_MAX_LENGTH = 9; // Quite generous
const ILLEGIBLE_LETERS = ['l', 'I', 'O', '0', '1']; // L i and o are allowed. Readable if the others are not present
const MAX_TRIES = 4;

class StorageBase {
    constructor(httpRootDir, compilerProps) {
        this.compilerProps = compilerProps;
        this.httpRootDir = httpRootDir;
    }

    /**
     * Encode a buffer as a URL-safe string.
     * @param {Buffer} buffer
     * @returns {string}
     */
    static safe64Encoded(buffer) {
        return buffer.toString('base64')
            .replace(/\+/g, '-')
            .replace(/\//g, '_')
            .replace(/=+$/, '');
    }

    static isLegibleText(text) {
        return !ILLEGIBLE_LETERS.some(letter => text.includes(letter));
    }

    static isCleanText(text) {
        return !profanities.some(badWord => text.includes(badWord));
    }

    static isUsableText(text) {
        const lowercasedText = text.toLowerCase();
        return StorageBase.isLegibleText(text) && StorageBase.isCleanText(lowercasedText);
    }

    static getSafeHash(config) {
        // Keep rehashing until a usable text is found
        let configHash = StorageBase.safe64Encoded(hash(JSON.stringify(config), FILE_HASH_VERSION));
        let tries = 1;
        while (!StorageBase.isUsableText(configHash.substr(0, USABLE_NAME_MAX_LENGTH))) {
            // Shake up the hash a bit by adding, or incrementing a nonce value.
            config.nonce = tries;
            logger.info(`Unusable text found in full hash ${configHash} - Trying again (${tries})`);
            if (tries <= MAX_TRIES) {
                configHash = StorageBase.safe64Encoded(hash(JSON.stringify(config), FILE_HASH_VERSION));
                ++tries;
            } else {
                logger.warn(`Gave up trying to find usable text for ${configHash}`);
                break;
            }
        }
        // And stringify it for the rest of the request
        config = JSON.stringify(config);
        return {config, configHash};
    }

    static configFor(req) {
        if (req.body.config) {
            return req.body.config;
        } else if (req.body.sessions) {
            return req.body;
        }
        return null;
    }

    handler(req, res) {
        // Get the desired config and check for profanities in its hash
        const origConfig = StorageBase.configFor(req);
        if (!origConfig) {
            logger.error("No configuration found");
            res.status(500);
            res.end("Missing config parameter");
            return;
        }
        const {config, configHash} = StorageBase.getSafeHash(origConfig);
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
                const shortlink = `${req.protocol}://${req.get('host')}${this.httpRootDir}z/${result.uniqueSubHash}`;
                res.send(JSON.stringify({url: shortlink}));
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

    incrementViewCount(id) {
        logger.error(`Trying to increment view count from base storage ${id}`);
        return Promise.reject();
    }
}

function storageFactory(storageSolution, compilerProps, awsProps, httpRootDir) {
    const storage = require(`./storage-${storageSolution}`);
    return new storage(httpRootDir, compilerProps, awsProps);
}

module.exports = {
    StorageBase: StorageBase,
    storageFactory: storageFactory
};
