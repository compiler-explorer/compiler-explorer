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

const BaseCache = require('./base.js');
const AWS = require('aws-sdk');

class S3Cache extends BaseCache {
    constructor(bucket, path, region) {
        super(`S3Cache(s3://${bucket}/${path} in ${region})`);
        this.bucket = bucket;
        this.path = path;
        this.s3 = new AWS.S3({region});
    }

    statString() {
        return `${super.statString()}, LRU has ${this.cache.itemCount} item(s) ` +
            `totalling ${this.cache.length} bytes on disk`;
    }

    getInternal(key) {
        return this.s3.getObject({Bucket: this.bucket, Key: `${this.path}/${key}`})
            .promise()
            .then((result) => {
                return {hit: true, data: result.Body};
            })
            .catch((x) => {
                if (x.code === 'NoSuchKey') return {hit: false};
                throw x;
            });
    }

    putInternal(key, value, creator) {
        return this.s3.putObject({
            Bucket: this.bucket, Key: `${this.path}/${key}`, Body: value,
            StorageClass: "REDUCED_REDUNDANCY",
            Metadata: {CreatedBy: creator}
        })
            .promise();
    }
}

module.exports = S3Cache;
