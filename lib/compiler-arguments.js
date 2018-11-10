// Copyright (c) 2018, Compiler Explorer Team
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
"use strict";

const
    _ = require('underscore'),
    quote = require('shell-quote'),
    fs = require('fs-extra'),
    AWS = require('aws-sdk'),
    S3Bucket = require('./s3-handler');

class CompilerArguments {
    constructor(compilerId) {
        this.compilerId = compilerId;
        this.possibleArguments = [];
    }

    loadFromStorage(awsProps) {
        const region = awsProps('region');
        const bucket = awsProps('storageBucketArgStats');
        const prefix = awsProps('storagePrefixArgStats');
        if (region && bucket && this.compilerId) {
            AWS.config.update({region: region});

            const s3 = new S3Bucket(bucket, region);
            s3.get(this.compilerId + ".json", prefix)
                .then(result => {
                    if (result.hit) {
                        const stats = JSON.parse(result.data.toString());
                        stats.forEach(stat => {
                            this.addOptionsToStatistics([stat.option], stat.count);
                        });
                    } else {
                        throw new Error(`${this.compilerId}.json not present in storage`);
                    }
                });
        }
    }

    getPopularArguments() {
        return _.pick(this.possibleArguments, (value) => {
            return (value.timesused >= 0);
        });
    }

    populateOptions(options) {
        this.possibleArguments = options;

        if (this.compilerId) {
            this.loadStatisticsFromJSONFile("argumentstats.json");
        }
    }

    loadStatisticsFromJSONFile(file) {
        fs.readFile(file).then(buffer => {
            const data = JSON.parse(buffer.toString());
            data.rows.forEach(row => {
                if (row[0] === this.compilerId) {
                    const args = _.chain(quote.parse(row[1] || '')
                        .map(x => typeof(x) === "string" ? x : x.pattern))
                        .compact()
                        .value();
                    this.addOptionsToStatistics(args, row[2]);
                }
            });
        });
    }

    addOptionsToStatistics(options, timesUsed) {
        if (!timesUsed) timesUsed = 1;

        options.forEach(option => {
            if (this.possibleArguments[option]) {
                this.possibleArguments[option].timesused += timesUsed;
            } else {
                const numre = /[0-9]*$/i;
                const genOption = option.replace(numre, "<number>");
                if ((genOption !== option) && this.possibleArguments[genOption]) {
                    this.possibleArguments[genOption].timesused += timesUsed;
                }
            }
        });
    }
}

module.exports = CompilerArguments;
