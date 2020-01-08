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
"use strict";

const
    _ = require('underscore'),
    AWS = require('aws-sdk'),
    logger = require('./logger').logger,
    S3Bucket = require('./s3-handler');

class CompilerArguments {
    constructor(compilerId) {
        this.compilerId = compilerId;
        this.possibleArguments = [];
        this.maxPopularArguments = 5;
        this.storeSpecificArguments = false;
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
                        _.each(stats, (times, arg) => {
                            this.addOptionToStatistics(arg, times);
                        });
                    } else {
                        logger.debug(`${this.compilerId}.json not present in storage`);
                    }
                });
        }
    }

    getOptimizationArguments(excludeUsedArguments) {
        let possibleArguments = {};
        _.forEach(this.possibleArguments, (obj, argKey) => {
            for (let argIdx in excludeUsedArguments) {
                if (this.match(argKey, excludeUsedArguments[argIdx])) return;
            }

            if (obj.description.includes("optimize") || obj.description.includes("optimization")) {
                possibleArguments[argKey] = {
                    description: obj.description
                };
            }
        });
        return possibleArguments;
    }

    getPopularArguments(excludeUsedArguments) {
        let possibleArguments = {};
        if (excludeUsedArguments && excludeUsedArguments.length > 0) {
            _.forEach(this.possibleArguments, (obj, argKey) => {
                for (let argIdx in excludeUsedArguments) {
                    if (this.match(argKey, excludeUsedArguments[argIdx])) return;
                }
                possibleArguments[argKey] = obj;
            });
        } else {
            possibleArguments = this.possibleArguments;
        }

        let arr = _.pairs(possibleArguments);
        arr.sort((a, b) => {
            if ((a[1].timesused === 0) && (b[1].timesused === 0)) {
                // prefer optimization flags or standard if statistics are not available
                if (a[1].description.includes("optimization")) {
                    return -1;
                } else if (b[1].description.includes("optimization")) {
                    return 1;
                } else if (a[1].description.includes("optimize")) {
                    return -1;
                } else if (b[1].description.includes("optimize")) {
                    return 1;
                } else if (a[1].description.includes("std")) {
                    return -1;
                } else if (b[1].description.includes("std")) {
                    return 1;
                }
            } else {
                return (b[1].timesused - a[1].timesused);
            }
        });
        arr = _.first(arr, this.maxPopularArguments);
        return _.object(arr);
    }

    populateOptions(options) {
        this.possibleArguments = options;
    }

    match(documentedOption, givenOption) {
        if (documentedOption.includes("<number>")
            || documentedOption.includes("<n>")
            || documentedOption.includes("=val")) {
            const numre = /[0-9]*$/i;
            if (documentedOption.indexOf(givenOption.replace(numre, "")) === 0) {
                return documentedOption;
            }
        }

        if (documentedOption.includes("=")) {
            const idx = documentedOption.indexOf("=");
            if (givenOption.indexOf("=") === idx) {
                if (documentedOption.substr(0, idx) === givenOption.substr(0, idx)) {
                    return documentedOption;
                }
            }
        }

        if (documentedOption.includes(":")) {
            const idx = documentedOption.indexOf(":");
            if (givenOption.indexOf(":") === idx) {
                if (documentedOption.substr(0, idx) === givenOption.substr(0, idx)) {
                    return documentedOption;
                }
            }
        }

        if (documentedOption.includes("[")) {
            const idx = documentedOption.indexOf("[") - 1;
            if (documentedOption.indexOf(givenOption.substr(0, idx)) === 0) {
                return documentedOption;
            }
        }

        if (documentedOption.indexOf(givenOption) === 0) {
            return documentedOption;
        }

        return false;
    }

    addOptionToStatistics(option, timesUsed) {
        if (!timesUsed) timesUsed = 1;

        const possibleKeys = _.compact(
            _.keys(this.possibleArguments).map((val) => this.match(val, option))
        );

        possibleKeys.forEach(key => {
            if (this.possibleArguments[key]) {
                this.possibleArguments[key].timesused += timesUsed;

                if (this.storeSpecificArguments && (key !== option)) {
                    if (!this.possibleArguments[key].specifically) {
                        this.possibleArguments[key].specifically = [];
                    }

                    this.possibleArguments[key].specifically.push({
                        arg: option,
                        timesused: timesUsed
                    });
                }
            }
        });
    }
}

module.exports = CompilerArguments;
