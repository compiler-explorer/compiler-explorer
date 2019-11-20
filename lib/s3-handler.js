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


const AWS = require('aws-sdk');

class S3Bucket {
    constructor(bucket, region) {
        this.instance = new AWS.S3({region});
        this.bucket = bucket;
        this.region = region;
    }

    get(key, path) {
        return this.instance.getObject({Bucket: this.bucket, Key: `${path}/${key}`})
            .promise()
            .then(result => {
                return {hit: true, data: result.Body};
            })
            .catch(x => {
                if (x.code === 'NoSuchKey') return {hit: false};
                throw x;
            });
    }

    delete(key, path) {
        return this.instance.deleteObject({Bucket: this.bucket, Key: `${path}/${key}`})
            .promise()
            .then(() => {
                return true;
            })
            .catch(x => {
                if (x.code === 'NoSuchKey') return false;
                throw x;
            });
    }

    put(key, value, path, options) {
        const metadata = options.metadata || {};
        const redundancy = options.redundancy || "STANDARD";
        return this.instance.putObject({
            Bucket: this.bucket, Key: `${path}/${key}`, Body: value,
            StorageClass: redundancy,
            Metadata: metadata
        }).promise();
    }
}

module.exports = S3Bucket;
