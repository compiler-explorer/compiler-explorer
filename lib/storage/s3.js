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

import AWS from 'aws-sdk';
import _ from 'underscore';

import { logger } from '../logger';
import { S3Bucket } from '../s3-handler';
import { anonymizeIp } from '../utils';

import { StorageBase } from './base';

const MIN_STORED_ID_LENGTH = 9; // If you change this, add the previous number to the list below.
// To deal with backward compatibility (for now), we need to check shorter prefixes for older "vanity" link IDs. e.g.
// originally things were stored as 6 characters long. (This is also true of any collision link ID that was 7 or more
// longer). A vanity URL of /z/acm19_count4 was stored originally with a prefix of six chars, and a full ID of
// acm19_count4, so { prefix: 'acm19_', unique_subhash: 'acm19_count4' }. This was ok as when we came to look up the
// 'acm19_count4' we truncated it to the _current_ MIN_STORED_ID_LENGTH (then also 6). However, when we bumped up to
// nine, now we look for { prefix: 'acm19_cou', unique_subhash: 'acm_count4' } and thus don't find it. At some point we
// will want to "fix" this by _not_ truncating the look up at all, and by doing a one-off rewrite of all vanity links to
// be prefix==unique_subhash (or something cleverer). For now, we will horribly try one after another. There aren't that
// many vanity links. Regular 6-char /z/abc123 short links don't suffer from this - they're already less than 9-chars.
const HISTORICAL_MIN_STORED_ID_LENGTHS = [
    MIN_STORED_ID_LENGTH, // check current first
    6, // then the older ones
];

export class StorageS3 extends StorageBase {
    static get key() {
        return 's3';
    }

    constructor(httpRootDir, compilerProps, awsProps) {
        super(httpRootDir, compilerProps);
        const region = awsProps('region');
        const bucket = awsProps('storageBucket');
        this.prefix = awsProps('storagePrefix');
        this.table = awsProps('storageDynamoTable');
        logger.info(`Using s3 storage solution on ${region}, bucket ${bucket}, ` +
            `prefix ${this.prefix}, dynamo table ${this.table}`);
        AWS.config.update({region: region});
        this.s3 = new S3Bucket(bucket, region);
        this.dynamoDb = new AWS.DynamoDB();
    }

    async storeItem(item, req) {
        logger.info(`Storing item ${item.prefix}`);
        const now = new Date();
        let ip = req.get('X-Forwarded-For') || anonymizeIp(req.ip);
        const commaIndex = ip.indexOf(',');
        if (commaIndex > 0) {
            // Anonymize only client IP
            ip = `${anonymizeIp(ip.substring(0, commaIndex))}${ip.substring(commaIndex, ip.length)}`;
        }
        now.setSeconds(0, 0);
        try {
            await Promise.all([
                this.dynamoDb.putItem({
                    TableName: this.table,
                    Item: {
                        prefix: {S: item.prefix},
                        unique_subhash: {S: item.uniqueSubHash},
                        full_hash: {S: item.fullHash},
                        stats: {
                            M: {clicks: {N: '0'}},
                        },
                        creation_ip: {S: ip},
                        creation_date: {S: now.toISOString()},
                    },
                }).promise(),
                this.s3.put(item.fullHash, item.config, this.prefix, {}),
            ]);
            return item;
        } catch (err) {
            logger.error('Unable to store item', {item, err});
            throw err;
        }
    }

    async findUniqueSubhash(hash) {
        const prefix = hash.substring(0, MIN_STORED_ID_LENGTH);
        const data = await this.dynamoDb.query({
            TableName: this.table,
            ProjectionExpression: 'unique_subhash, full_hash',
            KeyConditionExpression: 'prefix = :prefix',
            ExpressionAttributeValues: {':prefix': {S: prefix}},
        }).promise();
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
                    alreadyPresent: false,
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
                        alreadyPresent: true,
                    };
                }
            }
        }
        throw new Error(`Could not find unique subhash for hash "${hash}"`);
    }

    getKeyStructs(id) {
        return _.uniq(
            HISTORICAL_MIN_STORED_ID_LENGTHS.map(len => id.substring(0, len)))
            .map(prefix => ({
                prefix: {S: prefix},
                unique_subhash: {S: id},
            }));
    }

    async findFirstMatchingAttributes(id) {
        const structs = this.getKeyStructs(id);
        for (const key of structs) {
            const item = await this.dynamoDb.getItem({
                TableName: this.table,
                Key: key,
            }).promise();
            if (item.Item)
                return item.Item;
        }
        throw new Error(`Unable to find short id ${id} (tried ${JSON.stringify(structs)})`);
    }

    async expandId(id) {
        // By just getting the item and not trying to update it, we save an update when the link does not exist
        // for which we have less resources allocated, but get one extra read.
        const attributes = await this.findFirstMatchingAttributes(id);
        const result = await this.s3.get(attributes.full_hash.S, this.prefix);
        // If we're here, we are pretty confident there is a match. But never hurts to double check
        if (!result.hit)
            throw new Error(`ID ${id} not present in storage`);
        const metadata = attributes.named_metadata ? attributes.named_metadata.M : null;
        return {
            config: result.data.toString(),
            specialMetadata: metadata,
        };
    }

    async incrementViewCount(id) {
        for (const key of this.getKeyStructs(id)) {
            try {
                await this.dynamoDb.updateItem({
                    TableName: this.table,
                    Key: key,
                    UpdateExpression: 'SET stats.clicks = stats.clicks + :inc',
                    ExpressionAttributeValues: {
                        ':inc': {N: '1'},
                    },
                    ReturnValues: 'NONE',
                }).promise();
                return;
            } catch (err) {
                if (err.code === 'ValidationException') {
                    // Expected if we are trying to up the count on the wrong key struct
                    // (e.g. a vanity URL).
                } else {
                    logger.error(`Error when incrementing view count for ${id}`, err);
                    return;
                }
            }
        }
        logger.error(`Unable to to increment view count for ${id} (no key struct worked)`);
    }
}
