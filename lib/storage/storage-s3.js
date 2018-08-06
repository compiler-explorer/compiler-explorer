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
    // AWS = require('aws-sdk'),
    // _ = require('underscore'),
    S3Bucket = require('../s3-handler');


// const MIN_STORED_ID_LENGTH = 6;
// const TABLE_NAME = 'links2';

class StorageS3 extends StorageBase {
    constructor(compilerProps) {
        super(compilerProps);
        logger.info('Using s3 storage solution');
        // Hardcoded for now, needs to change
        this.s3 = new S3Bucket('compiler-explorer', 'us-east-1');
    }

    storeItem(item) {
        // TODO: Store item in a db. Adjust redundancy
        this.s3.put(item.id, item.config, 'links', {});
    }

    findUniqueSubhash(hash) {
        logger.info(`Finding s3 unique subhash for ${hash}`);

        return Promise.resolve({
            id: hash,
            alreadyPresent: false
        });
        /*
        let base = hash.substring(0, MIN_STORED_ID_LENGTH);
        // TODO: Make use of db to fetch metadata, goal is to only make 1 I/O request to S3 after all
        return new Promise((resolve, reject) => {
            this.dynamoDb.query({
                TableName: TABLE_NAME,
                ProjectionExpression: 'id, prefix',
                KeyConditionExpression: 'prefix = :base and begins_with(full_hash, :base)',
                //FilterExpression: 'begins_with(full_hash, :base)',
                ExpressionAttributeValues: {
                    ':base': {S: base}
                }
            }, (err, data) => {
                if (err) {
                    logger.error('Unable to query: ', JSON.stringify(err, null, 2));
                    reject();
                } else {
                    const orderToindex = [];
                    let index = 0;
                    const ids = _.chain(data.Items)
                        .map(item => item.id)
                        .sort()
                        .each(item => {
                            for (let i = 0;i < data.Count;i++) {
                                if (data.Items[i][0] === item) {
                                    orderToindex[index] = i;
                                    index++;
                                    break;
                                }
                            }
                        })
                        .value();
                    logger.info(ids);
                    for (let i = MIN_STORED_ID_LENGTH; i < hash.length - 1; i++) {
                        let base = hash.substring(0, i);
                        // Check if the current base is present in the array
                        const index = _.indexOf(ids, base, true);
                        if (index === -1) {
                            // Current base is not present, we have a new config in our hands
                            resolve({
                                id: base,
                                alreadyPresent: false
                            });
                            return;
                        } else {
                            const item = data.Items[orderToindex[index]];
                            if (item[1] === hash) {
                                resolve({
                                    id: base,
                                    alreadyPresent: true
                                });
                                return;
                            }
                        }
                    }
                }
            });

        });
        */
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
        // TODO: Expand id based on db metadata
        /*return new Promise((resolve, reject) => {
            this.dynamoDb.getItem({
                TableName: TABLE_NAME,
                Key: {
                    prefix: {S: id}
                }
            }, (err, data) => {
                if (err) {
                    logger.error("Error", err);
                    reject();
                } else {
                    logger.info("Success", data);
                    // Any way to update automatically on successful get?
                    this.dynamoDb.updateItem({
                        TableName: TABLE_NAME,
                        Key: {
                            prefix: {S: id}
                        },
                        UpdateExpression: 'set views = views + 1'
                    });
                    resolve(data.Item.sha);
                }
            });
        });
        */
    }
}

module.exports = StorageS3;
