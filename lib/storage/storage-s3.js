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
const TABLE_NAME = 'links';

class StorageS3 extends StorageBase {
    constructor(compilerProps, awsProps) {
        super(compilerProps);
        const region = awsProps('region');
        logger.info(awsProps);
        logger.info(`Using s3 storage solution on ${region}`);
        // Hardcoded for now, needs to change
        AWS.config.update({region: region});
        this.s3 = new S3Bucket('compiler-explorer', region);
        this.dynamoDb = new AWS.DynamoDB();
    }

    storeItem(item) {
        logger.info(`Storing item ${item.prefix}`);
        this.dynamoDb.putItem({
            TableName: TABLE_NAME,
            Item: {
                prefix: {
                    S: item.prefix
                },
                unique_subhash: {
                    S: item.uniqueSubHash,
                },
                full_hash: {
                    S: item.fullHash
                },
                clicks: {
                    N: "0"
                }
            }
        }).promise()
            .then(() => {
                this.s3.put(item.fullHash, item.config, 'links', {});
            })
            .catch(error => {
                logger.error(error);
            });
    }

    findUniqueSubhash(hash) {
        logger.info(`Finding s3 unique subhash for ${hash}`);
        const prefix = hash.substring(0, MIN_STORED_ID_LENGTH);
        return this.dynamoDb.query({
            TableName: TABLE_NAME,
            ProjectionExpression: 'unique_subhash, full_hash',
            KeyConditionExpression: 'prefix = :prefix',
            ExpressionAttributeValues: {
                ':prefix': {S: prefix}
            }
        }).promise()
            .then(data => {
                const subHashes = _.chain(data.Items)
                    .pluck('unique_subhash')
                    .pluck('S')
                    .value();
                const fullHashes = _.chain(data.Items)
                    .pluck('full_hash')
                    .pluck('S')
                    .value();
                for (let i = MIN_STORED_ID_LENGTH; i < hash.length - 1; i++) {
                    let subHash = hash.substring(0, i);
                    // Check if the current base is present in the subHashes array
                    const index = _.indexOf(subHashes, subHash, true);
                    if (index === -1) {
                        // Current base is not present, we have a new config in our hands
                        return {
                            prefix: prefix,
                            uniqueSubHash: subHash,
                            alreadyPresent: false
                        };
                    } else {
                        const itemHash = fullHashes[index];
                        /* If the hashes coincide, it means this config has already been stored.
                         * Else, keep looking
                         */
                        if (itemHash === hash) {
                            return {
                                prefix: prefix,
                                uniqueSubHash: subHash,
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
        return this.dynamoDb.updateItem({
            TableName: TABLE_NAME,
            Key: {
                prefix: {
                    S: id.substring(0, 6)
                },
                unique_subhash: {
                    S: id
                }
            },
            UpdateExpression: 'SET clicks = clicks + :inc',
            ExpressionAttributeValues: {
                ':inc': {N: '1'}
            },
            ReturnValues: 'ALL_NEW'
        }).promise()
            .then(item => {
                const full_hash = item.Attributes.full_hash.S;
                return this.s3.get(full_hash, 'links')
                    .then(result => {
                        if (result.hit) {
                            // If we're here, we are pretty confident there is a match. But never hurts to double check
                            return result.data.toString();
                        } else {
                            throw 'ID not present';
                        }
                    });
            })
            .catch(err => {
                logger.error(err);
                throw err;
            });
    }
}

module.exports = StorageS3;
