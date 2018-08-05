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
    AWS = require('aws-sdk');

const MIN_STORED_ID_LENGTH = 8;

class StorageS3 extends StorageBase {
    constructor(compilerProps) {
        super(compilerProps);
        logger.info('Using s3 storage solution');
        // Hardcoded for now, will change
        this.dynamoDb = new AWS.DynamoDB({
            region: 'us-east-1',
            endpoint: "https://dynamodb.us-east-1.amazonaws.com"
        });
    }

    storeItem(item) {
        // TODO: Store item in s3
        this.dynamoDb.putItem({
            TableName: 'links',
            Item: {
                id: {S: item.id},
                sha: {S: item.hash},
                // Storing config in db for testing.
                // TODO: Move to s3
                config: {S: item.config},
                views: {N: '0'}
            }
        }, (err, data) => {
            if (err) {
                logger.error("Error", err);
            } else {
                logger.info("Success", data);
            }
        });
    }

    findUniqueSubhash(hash) {
        logger.info(`Finding s3 unique subhash for ${hash}`);
        const base = hash.substring(0, MIN_STORED_ID_LENGTH);
        // TODO: Make use of db to fetch metadata, goal is to only make 1 I/O request to S3 after all
        return new Promise((resolve, reject) => {
            this.dynamoDb.query({
                TableName: 'links',
                ProjectionExpression: 'id, sha',
                // Main key ID can't be begins_with compared, but id needs to be present on the condition, so invalid rn
                // We could scan, but it would get slower the more links were created
                // We could look for the ids one by one, and that would solve the issue, but it's better if we find
                // a solution not involving that
                KeyConditionExpression: 'begins_with(sha, :base)',
                ExpressionAttributeValues: {
                    ':base': {S: base}
                }
            }, (err, data) => {
                if (err) {
                    logger.error('Unable to query: ', JSON.stringify(err, null, 2));
                    reject();
                } else {
                    logger.info('Query successful: ', data);

                    resolve({
                        id: hash,
                        alreadyPresent: false
                    });
                }
            });
        });
    }

    expandId(id) {
        logger.info(`Expanding s3 id ${id}`);
        // TODO: Expand id based on db metadata
        return new Promise((resolve, reject) => {
            this.dynamoDb.getItem({
                TableName: 'links',
                Key: {
                    id: {S: id}
                }
            }, (err, data) => {
                if (err) {
                    logger.error("Error", err);
                    reject();
                } else {
                    logger.info("Success", data);
                    // Any way to update automatically on successful get?
                    this.dynamoDb.updateItem({
                        TableName: 'links',
                        Key: {
                            id: {S: id}
                        },
                        UpdateExpression: 'set views = views + 1'
                    });
                    resolve(data.Item.sha);
                }
            });
        });
    }
}

module.exports = StorageS3;
