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

const chai = require('chai'),
    chaiAsPromised = require("chai-as-promised"),
    properties = require('../../lib/properties'),
    s3s = require('../../lib/storage/storage-s3'),
    AWS = require('aws-sdk-mock');

chai.use(chaiAsPromised);
const should = chai.should();

// NB!!! Anything using mocked AWS calls needs to be initialised in the `it(...)` block! If you initialise it in the
// `describe()` top level code then it won't be mocked in time. We only mock and de-mock before/after else we end up
// fighting over the global AWS mocking stuff. I hate mocha...there's probably a better way...
function mockerise(service, method) {
    const handlers = [];

    beforeEach(() => {
        AWS.mock(service, method, (q, callback) => {
            const qh = handlers.shift();
            should.exist(qh);
            try {
                callback(null, qh(q));
            } catch (e) {
                callback(e, null);
            }
        });
    });
    afterEach(() => {
        AWS.restore(service, method);
    });
    return handlers;
}


////////////////


describe('Find unique subhash tests', () => {
    const dynamoDbQueryHandlers = mockerise('DynamoDB', 'query');
    const compilerProps = properties.fakeProps({});
    const httpRootDir = '/';
    const awsProps = properties.fakeProps({
        region: 'not-a-region',
        storageBucket: 'bucket',
        storagePrefix: 'prefix',
        storageDynamoTable: 'table'
    });
    it('works when empty', () => {
        const storage = new s3s(httpRootDir, compilerProps, awsProps);
        dynamoDbQueryHandlers.push((q) => {
            q.TableName.should.equal('table');
            return {};
        });
        return storage.findUniqueSubhash("ABCDEFGHIJKLMNOPQRSTUV").should.eventually.deep.equal(
            {
                alreadyPresent: false,
                prefix: "ABCDEF",
                uniqueSubHash: "ABCDEF"
            }
        );
    });
    it('works when not empty', () => {
        const storage = new s3s(httpRootDir, compilerProps, awsProps);
        dynamoDbQueryHandlers.push(() => {
            return {
                Items: [
                    {
                        full_hash: {S: 'ZZVZT'},
                        unique_subhash: {S: 'ZZVZT'}
                    }
                ]
            };
        });
        return storage.findUniqueSubhash("ABCDEFGHIJKLMNOPQRSTUV").should.eventually.deep.equal(
            {
                alreadyPresent: false,
                prefix: "ABCDEF",
                uniqueSubHash: "ABCDEF"
            }
        );
    });
    it('works when there\' a collision', () => {
        const storage = new s3s(httpRootDir, compilerProps, awsProps);
        dynamoDbQueryHandlers.push(() => {
            return {
                Items: [
                    {
                        full_hash: {S: 'ABCDEFZZ'},
                        unique_subhash: {S: 'ABCDEF'}
                    }
                ]
            };
        });
        return storage.findUniqueSubhash("ABCDEFGHIJKLMNOPQRSTUV").should.eventually.deep.equal(
            {
                alreadyPresent: false,
                prefix: "ABCDEF",
                uniqueSubHash: "ABCDEFG"
            }
        );
    });
    it('finds an existing match', () => {
        const storage = new s3s(httpRootDir, compilerProps, awsProps);
        dynamoDbQueryHandlers.push(() => {
            return {
                Items: [
                    {
                        full_hash: {S: 'ABCDEFGHIJKLMNOPQRSTUV'},
                        unique_subhash: {S: 'ABCDEF'}
                    }
                ]
            };
        });
        return storage.findUniqueSubhash("ABCDEFGHIJKLMNOPQRSTUV").should.eventually.deep.equal(
            {
                alreadyPresent: true,
                prefix: "ABCDEF",
                uniqueSubHash: "ABCDEF"
            }
        );
    });
});


describe('Stores to s3', () => {
    const dynamoDbPutItemHandlers = mockerise('DynamoDB', 'putItem');
    const s3PutObjectHandlers = mockerise('S3', 'putObject');
    const httpRootDir = '/';
    const compilerProps = properties.fakeProps({});
    const awsProps = properties.fakeProps({
        region: 'not-a-region',
        storageBucket: 'bucket',
        storagePrefix: 'prefix',
        storageDynamoTable: 'table'
    });
    it('and works ok', () => {
        const storage = new s3s(httpRootDir, compilerProps, awsProps);
        const object = {
            prefix: "ABCDEF",
            uniqueSubHash: "ABCDEFG",
            fullHash: "ABCDEFGHIJKLMNOP",
            config: 'yo'
        };

        const ran = {s3: false, dynamo: false};
        s3PutObjectHandlers.push((q) => {
            q.Bucket.should.equal('bucket');
            q.Key.should.equal('prefix/ABCDEFGHIJKLMNOP');
            q.Body.should.equal('yo');
            ran.s3 = true;
            return {};
        });
        dynamoDbPutItemHandlers.push((q) => {
            q.TableName.should.equals('table');
            q.Item.should.deep.equals({
                prefix: {S: 'ABCDEF'},
                unique_subhash: {S: 'ABCDEFG'},
                full_hash: {S: 'ABCDEFGHIJKLMNOP'},
                stats: {M: {clicks: {N: '0'}}},
                creation_ip: {S: 'localhost'},
                // Cheat the date
                creation_date: {S: q.Item.creation_date.S}
            });
            ran.dynamo = true;
            return {};
        });
        return storage.storeItem(object, {get: () => 'localhost'}).then(() => {
            ran.should.deep.equal({s3: true, dynamo: true});
        });
    });
});
