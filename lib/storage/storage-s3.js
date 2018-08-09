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
    AWS = require('aws-sdk'),
    _ = require('underscore'),
    S3Bucket = require('../s3-handler');


const MIN_STORED_ID_LENGTH = 6;
const TABLE_NAME = 'links2';

class StorageS3 extends StorageBase {
    constructor(compilerProps) {
        super(compilerProps);
        logger.info('Using s3 storage solution');
        // Hardcoded for now, needs to change
        AWS.config.update({region: 'us-east-1'});
        this.s3 = new S3Bucket('compiler-explorer', 'us-east-1');
        this.dynamoDb = new AWS.DynamoDB();
    }

    storeItem(item) {
        logger.info(`Storing item ${item.id}`);
        this.dynamoDb.putItem({
            TableName: TABLE_NAME,
            Item: {
                prefix: {
                    S: item.id
                },
                full_hash: {
                    S: item.hash
                }
            }
        }).promise()
            .then(() => {
                this.s3.put(item.id, item.config, 'links', {});
            })
            .catch(error => {
                logger.error(error);
            });
    }

    findUniqueSubhash(hash) {
        logger.info(`Finding s3 unique subhash for ${hash}`);
        const base = hash.substring(0, 6);
        return this.dynamoDb.scan({
            TableName: TABLE_NAME,
            ProjectionExpression: 'prefix, full_hash',
            FilterExpression: 'begins_with(prefix, :base)',
            ExpressionAttributeValues: {
                ':base': {S: base}
            }
        }).promise()
            .then(data => {
                const prefixes = _.pluck(data.Items, 'prefix');
                const fullHashes = _.pluck(data.Items, 'full_hash');
                for (let i = MIN_STORED_ID_LENGTH; i < hash.length - 1; i++) {
                    let base = hash.substring(0, i);
                    // Check if the current base is present in the array
                    const index = _.indexOf(prefixes, base, true);
                    if (index === -1) {
                        // Current base is not present, we have a new config in our hands
                        return {
                            id: base,
                            alreadyPresent: false
                        };
                    } else {
                        const itemHash = fullHashes[index];
                        /* If the hashes coincide, it means this config has already been stored.
                         * Else, keep looking
                         */
                        if (itemHash === hash) {
                            return {
                                id: base,
                                alreadyPresent: true
                            };
                        }
                    }
                }
                throw 'Could not find unique subhash';
            });
    }

    expandId(id) {
        logger.info(`Expanding s3 id ${id}`);
        return this.s3.get(id, 'links')
            .then(result => {
                if (result.hit) {
                    return result.data.toString();
                } else {
                    throw 'ID not present';
                }
            });
    }
}

module.exports = StorageS3;
