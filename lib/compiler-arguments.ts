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

import path from 'path';

import AWS from 'aws-sdk';
import fs from 'fs-extra';
import _ from 'underscore';

import {ICompilerArguments, PossibleArguments} from '../types/compiler-arguments.interfaces';

import {logger} from './logger';
import {PropertyGetter} from './properties.interfaces';
import {S3Bucket} from './s3-handler';
import {fileExists, resolvePathFromAppRoot} from './utils';

export class CompilerArguments implements ICompilerArguments {
    private readonly compilerId: string;
    public possibleArguments: PossibleArguments = {};
    private readonly maxPopularArguments = 5;
    private readonly storeSpecificArguments = false;
    private loadedFromFile = false;

    constructor(compilerId: string) {
        this.compilerId = compilerId;
    }

    async loadFromFile(awsProps: PropertyGetter): Promise<boolean> {
        let localfolder = awsProps('localfolderArgStats', '');
        if (localfolder) {
            if (localfolder.startsWith('./')) {
                localfolder = resolvePathFromAppRoot(localfolder);
            }

            const filepath = path.join(localfolder, this.compilerId + '.json');
            if (await fileExists(filepath)) {
                const contents = await fs.readFile(filepath);
                const stats = JSON.parse(contents.toString());
                _.each(stats, (times, arg) => {
                    this.addOptionToStatistics(arg, times);
                });
                logger.info(`${this.compilerId}.json loaded from file`);

                this.loadedFromFile = true;

                return true;
            }
        }

        return false;
    }

    async loadFromStorage(awsProps: PropertyGetter) {
        if (await this.loadFromFile(awsProps)) return;

        const region = awsProps('region', '');
        const bucket = awsProps('storageBucketArgStats', '');
        const prefix = awsProps('storagePrefixArgStats', '');

        if (region && bucket && prefix && this.compilerId) {
            AWS.config.update({region: region});

            const s3 = new S3Bucket(bucket, region);
            const result = await s3.get(this.compilerId + '.json', prefix);
            if (result.hit) {
                const stats = JSON.parse(result.data.toString());
                _.each(stats, (times, arg) => {
                    this.addOptionToStatistics(arg, times);
                });
                logger.debug(`${this.compilerId}.json has stats`, stats);
            } else {
                logger.debug(`${this.compilerId}.json not present in storage`);
            }
        }
    }

    getOptimizationArguments(excludeUsedArguments?: string[]): PossibleArguments {
        excludeUsedArguments = excludeUsedArguments || [];
        const possibleArguments: PossibleArguments = {};
        for (const [argKey, obj] of Object.entries(this.possibleArguments)) {
            if (!excludeUsedArguments.some(used => this.match(argKey, used))) {
                if (obj.description.includes('optimize') || obj.description.includes('optimization')) {
                    possibleArguments[argKey] = {
                        description: obj.description,
                        timesused: 0,
                    };
                }
            }
        }
        return possibleArguments;
    }

    getPopularArguments(excludeUsedArguments?: string[]): PossibleArguments {
        excludeUsedArguments = excludeUsedArguments || [];
        const possibleArguments: PossibleArguments = {};
        for (const [argKey, obj] of Object.entries(this.possibleArguments)) {
            if (!excludeUsedArguments.some(used => this.match(argKey, used))) {
                possibleArguments[argKey] = obj;
            }
        }

        let arr = _.pairs(possibleArguments);
        arr.sort((a, b) => {
            if (!this.loadedFromFile && a[1].timesused === 0 && b[1].timesused === 0) {
                // prefer optimization flags or standard if statistics are not available
                if (a[1].description.includes('optimization')) {
                    return -1;
                } else if (b[1].description.includes('optimization')) {
                    return 1;
                } else if (a[1].description.includes('optimize')) {
                    return -1;
                } else if (b[1].description.includes('optimize')) {
                    return 1;
                } else if (a[1].description.includes('std')) {
                    return -1;
                } else if (b[1].description.includes('std')) {
                    return 1;
                }
            }
            return b[1].timesused - a[1].timesused;
        });
        arr = _.first(arr, this.maxPopularArguments);
        return _.object(arr);
    }

    populateOptions(options: PossibleArguments) {
        this.possibleArguments = {
            ...this.possibleArguments,
            ...options,
        };
    }

    match(documentedOption: string, givenOption: string): string | boolean {
        if (
            documentedOption.includes('<number>') ||
            documentedOption.includes('<n>') ||
            documentedOption.includes('=val')
        ) {
            const numre = /\d*$/i;
            if (documentedOption.indexOf(givenOption.replace(numre, '')) === 0) {
                return documentedOption;
            }
        }

        if (documentedOption.includes('=')) {
            const idx = documentedOption.indexOf('=');
            if (givenOption.indexOf('=') === idx) {
                if (documentedOption.substr(0, idx) === givenOption.substr(0, idx)) {
                    return documentedOption;
                }
            }
        }

        if (documentedOption.includes(':')) {
            const idx = documentedOption.indexOf(':');
            if (givenOption.indexOf(':') === idx) {
                if (documentedOption.substr(0, idx) === givenOption.substr(0, idx)) {
                    return documentedOption;
                }
            }
        }

        if (documentedOption.includes('[')) {
            const idx = documentedOption.indexOf('[') - 1;
            if (documentedOption.indexOf(givenOption.substr(0, idx)) === 0) {
                return documentedOption;
            }
        }

        if (documentedOption.indexOf(givenOption) === 0) {
            return documentedOption;
        }

        return false;
    }

    addOptionToStatistics(option: string, timesUsed: number | undefined) {
        if (!timesUsed) timesUsed = 1;

        const possibleKeys = _.compact(_.keys(this.possibleArguments).map(val => this.match(val, option))) as string[];

        for (const key of possibleKeys) {
            const possibleArgument = this.possibleArguments[key];
            if (possibleArgument) {
                if (possibleKeys.length === 1 || option === key) {
                    possibleArgument.timesused += timesUsed;
                } else {
                    // non-exact match should be less valuable
                    possibleArgument.timesused += timesUsed - 1;
                }

                if (this.storeSpecificArguments && key !== option) {
                    if (!possibleArgument.specifically) {
                        possibleArgument.specifically = [];
                    }

                    possibleArgument.specifically.push({
                        arg: option,
                        timesused: timesUsed,
                    });
                }
            }
        }
    }
}
