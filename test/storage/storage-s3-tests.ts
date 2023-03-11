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

import {Readable} from 'stream';

import {DynamoDB, GetItemCommand, PutItemCommand, QueryCommand, UpdateItemCommand} from '@aws-sdk/client-dynamodb';
import {GetObjectCommand, PutObjectCommand, S3} from '@aws-sdk/client-s3';
import {sdkStreamMixin} from '@aws-sdk/util-stream-node';
import {mockClient} from 'aws-sdk-client-mock';

import * as properties from '../../lib/properties.js';
import {StorageS3} from '../../lib/storage/index.js';

describe('Find unique subhash tests', () => {
    const mockDynamoDb = mockClient(DynamoDB);
    const mockS3 = mockClient(S3);
    beforeEach(() => {
        mockDynamoDb.reset();
        mockS3.reset();
    });
    const compilerProps = properties.fakeProps({});
    const httpRootDir = '/';
    const awsProps = properties.fakeProps({
        region: 'not-a-region',
        storageBucket: 'bucket',
        storagePrefix: 'prefix',
        storageDynamoTable: 'table',
    });
    it('works when empty', () => {
        const storage = new StorageS3(httpRootDir, compilerProps, awsProps);
        mockDynamoDb.on(QueryCommand, {TableName: 'table'}).resolves({});
        return storage.findUniqueSubhash('ABCDEFGHIJKLMNOPQRSTUV').should.eventually.deep.equal({
            alreadyPresent: false,
            prefix: 'ABCDEF',
            uniqueSubHash: 'ABCDEFGHI',
        });
    });
    it('works when not empty', () => {
        const storage = new StorageS3(httpRootDir, compilerProps, awsProps);
        mockDynamoDb.on(QueryCommand, {TableName: 'table'}).resolves({
            Items: [
                {
                    full_hash: {S: 'ZZVZT'},
                    unique_subhash: {S: 'ZZVZT'},
                },
            ],
        });

        return storage.findUniqueSubhash('ABCDEFGHIJKLMNOPQRSTUV').should.eventually.deep.equal({
            alreadyPresent: false,
            prefix: 'ABCDEF',
            uniqueSubHash: 'ABCDEFGHI',
        });
    });
    it("works when there's a collision", () => {
        const storage = new StorageS3(httpRootDir, compilerProps, awsProps);
        mockDynamoDb.on(QueryCommand, {TableName: 'table'}).resolves({
            Items: [
                {
                    full_hash: {S: 'ABCDEFGHIZZ'},
                    unique_subhash: {S: 'ABCDEFGHI'},
                },
            ],
        });
        return storage.findUniqueSubhash('ABCDEFGHIJKLMNOPQRSTUV').should.eventually.deep.equal({
            alreadyPresent: false,
            prefix: 'ABCDEF',
            uniqueSubHash: 'ABCDEFGHIJ',
        });
    });
    it('finds an existing match', () => {
        const storage = new StorageS3(httpRootDir, compilerProps, awsProps);
        mockDynamoDb.on(QueryCommand, {TableName: 'table'}).resolves({
            Items: [
                {
                    full_hash: {S: 'ABCDEFGHIJKLMNOPQRSTUV'},
                    unique_subhash: {S: 'ABCDEFGHI'},
                },
            ],
        });
        return storage.findUniqueSubhash('ABCDEFGHIJKLMNOPQRSTUV').should.eventually.deep.equal({
            alreadyPresent: true,
            prefix: 'ABCDEF',
            uniqueSubHash: 'ABCDEFGHI',
        });
    });
});

describe('Stores to s3', () => {
    const mockDynamoDb = mockClient(DynamoDB);
    const mockS3 = mockClient(S3);
    beforeEach(() => {
        mockDynamoDb.reset();
        mockS3.reset();
    });
    const httpRootDir = '/';
    const compilerProps = properties.fakeProps({});
    const awsProps = properties.fakeProps({
        region: 'not-a-region',
        storageBucket: 'bucket',
        storagePrefix: 'prefix',
        storageDynamoTable: 'table',
    });
    it('and works ok', async () => {
        const storage = new StorageS3(httpRootDir, compilerProps, awsProps);
        const object = {
            prefix: 'ABCDEF',
            uniqueSubHash: 'ABCDEFG',
            fullHash: 'ABCDEFGHIJKLMNOP',
            config: 'yo',
        };
        await storage.storeItem(object, {get: () => 'localhost'});
        mockS3
            .commandCalls(PutObjectCommand, {
                Bucket: 'bucket',
                Key: 'prefix/ABCDEFGHIJKLMNOP',
                Body: 'yo',
            })
            .should.have.lengthOf(1);
        mockDynamoDb
            .commandCalls(PutItemCommand, {
                TableName: 'table',
                Item: {
                    prefix: {S: 'ABCDEF'},
                    unique_subhash: {S: 'ABCDEFG'},
                    full_hash: {S: 'ABCDEFGHIJKLMNOP'},
                    stats: {M: {clicks: {N: '0'}}},
                    creation_ip: {S: 'localhost'},
                },
            })
            .should.have.lengthOf(1);
    });
});

describe('Retrieves from s3', () => {
    const mockDynamoDb = mockClient(DynamoDB);
    const mockS3 = mockClient(S3);
    beforeEach(() => {
        mockDynamoDb.reset();
        mockS3.reset();
    });
    const httpRootDir = '/';
    const compilerProps = properties.fakeProps({});
    const awsProps = properties.fakeProps({
        region: 'not-a-region',
        storageBucket: 'bucket',
        storagePrefix: 'prefix',
        storageDynamoTable: 'table',
    });
    it('fetches in the happy path', async () => {
        const storage = new StorageS3(httpRootDir, compilerProps, awsProps);
        mockDynamoDb
            .on(GetItemCommand, {
                TableName: 'table',
                Key: {
                    prefix: {S: 'ABCDEF'},
                    unique_subhash: {S: 'ABCDEF'},
                },
            })
            .resolves({Item: {full_hash: {S: 'ABCDEFGHIJKLMNOP'}}});
        const stream = new Readable();
        stream.push('I am a monkey');
        stream.push(null);
        mockS3
            .on(GetObjectCommand, {
                Bucket: 'bucket',
                Key: 'prefix/ABCDEFGHIJKLMNOP',
            })
            .resolves({Body: sdkStreamMixin(stream)});

        const result = await storage.expandId('ABCDEF');
        result.should.deep.equal({config: 'I am a monkey', specialMetadata: null});
    });
    it('should handle failures', async () => {
        const storage = new StorageS3(httpRootDir, compilerProps, awsProps);
        mockDynamoDb.on(GetItemCommand).resolves({});
        return storage.expandId('ABCDEF').should.be.rejectedWith(Error, 'ID ABCDEF not present in links table');
    });
});

describe('Updates counts in s3', async () => {
    const mockDynamoDb = mockClient(DynamoDB);
    const mockS3 = mockClient(S3);
    beforeEach(() => {
        mockDynamoDb.reset();
        mockS3.reset();
    });
    const httpRootDir = '/';
    const compilerProps = properties.fakeProps({});
    const awsProps = properties.fakeProps({
        region: 'not-a-region',
        storageBucket: 'bucket',
        storagePrefix: 'prefix',
        storageDynamoTable: 'table',
    });
    it('should increment for simple cases', async () => {
        const storage = new StorageS3(httpRootDir, compilerProps, awsProps);
        await storage.incrementViewCount('ABCDEF');
        mockDynamoDb
            .commandCalls(UpdateItemCommand, {
                ExpressionAttributeValues: {':inc': {N: '1'}},
                Key: {
                    prefix: {S: 'ABCDEF'},
                    unique_subhash: {S: 'ABCDEF'},
                },
                ReturnValues: 'NONE',
                TableName: 'table',
                UpdateExpression: 'SET stats.clicks = stats.clicks + :inc',
            })
            .should.have.lengthOf(1);
    });
});
