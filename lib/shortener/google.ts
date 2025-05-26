// Copyright (c) 2017, Compiler Explorer Authors
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

import {DynamoDB} from '@aws-sdk/client-dynamodb';
import {Counter} from 'prom-client';

import {awsCredentials} from '../aws.js';
import {logger} from '../logger.js';
import {PropertyGetter} from '../properties.interfaces.js';

const GoogleLinkDynamoDbHitCounter = new Counter({
    name: 'ce_google_link_dynamodb_hits_total',
    help: 'Total number of successful Google short link lookups from DynamoDB',
});

const GoogleLinkDynamoDbMissCounter = new Counter({
    name: 'ce_google_link_dynamodb_misses_total',
    help: 'Total number of Google short link lookups not found in DynamoDB',
});

// This will stop working in August 2025.
// https://developers.googleblog.com/en/google-url-shortener-links-will-no-longer-be-available/
export class ShortLinkResolver {
    private readonly dynamoDb?: DynamoDB;
    private readonly tableName?: string;

    constructor(awsProps?: PropertyGetter) {
        if (awsProps) {
            const tableName = awsProps('googleLinksDynamoTable', '');
            if (tableName) {
                const region = awsProps('region') as string;
                this.tableName = tableName;
                this.dynamoDb = new DynamoDB({region: region, credentials: awsCredentials()});
                logger.info(`Using DynamoDB table ${tableName} in region ${region} for Google link resolution`);
            }
        }
    }

    async resolve(url: string): Promise<{longUrl: string}> {
        const fragment = this.extractFragment(url);

        if (this.hasDynamoDbConfigured() && fragment) {
            const dynamoResult = await this.tryDynamoDbLookup(fragment);
            if (dynamoResult) {
                return dynamoResult;
            }
        }

        // In August 2025, the Google URL shortener will stop working. At that point, use this code:
        //     throw new Error('404: Not Found');

        return this.fallbackToGoogleShortener(url);
    }

    // Exported for testing
    extractFragment(url: string): string | undefined {
        return url.split('/').pop()?.split('?')[0];
    }

    // Exported for testing
    hasDynamoDbConfigured(): boolean {
        return !!(this.dynamoDb && this.tableName);
    }

    private async tryDynamoDbLookup(fragment: string): Promise<{longUrl: string} | null> {
        try {
            const result = await this.dynamoDb!.getItem({
                TableName: this.tableName!,
                Key: {
                    fragment: {S: fragment},
                },
            });

            const expandedUrl = result.Item?.expanded_url?.S;
            if (expandedUrl) {
                GoogleLinkDynamoDbHitCounter.inc();
                return {longUrl: expandedUrl};
            }

            GoogleLinkDynamoDbMissCounter.inc();
            logger.warn(`Google short link '${fragment}' not found in DynamoDB, falling back to Google URL shortener`);
            return null;
        } catch (err) {
            logger.error('DynamoDB lookup failed:', err);
            return null;
        }
    }

    private async fallbackToGoogleShortener(url: string): Promise<{longUrl: string}> {
        if (this.hasDynamoDbConfigured()) {
            logger.warn(`Falling back to Google URL shortener for '${url}' - this indicates missing data in DynamoDB`);
        }

        const res = await fetch(url + '?si=1', {
            method: 'HEAD',
            redirect: 'manual',
        });

        if (res.status !== 302) {
            throw new Error(`Got response ${res.status}`);
        }

        const targetLocation = res.headers.get('Location');
        if (!targetLocation) {
            throw new Error('Missing location url');
        }

        return {longUrl: targetLocation};
    }
}
