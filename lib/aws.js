// Copyright (c) 2016, Compiler Explorer Authors
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

import { logger } from './logger';

export class InstanceFetcher {
    constructor(properties) {
        const region = properties('region');
        logger.info(`New instance fetcher for region ${region}`);
        this.ec2 = new AWS.EC2({region: region});
        this.tagKey = properties('tagKey');
        this.tagValue = properties('tagValue');
    }

    async getInstances() {
        return (await this.ec2.describeInstances().promise()).Reservations
            .flatMap(r => r.Instances)
            .filter(reservation => {
                if (reservation.State.Name !== 'running') return false;
                return reservation.Tags.some(t => t.Key === this.tagKey && t.Value === this.tagValue);
            });
    }
}

let awsConfigInit = false;
let awsConfig = {};
let awsProps = null;

async function loadAwsConfig(properties) {
    let region = properties('region');
    if (!region) return {};
    const ssm = new AWS.SSM({region: region});
    const path = '/compiler-explorer/';
    try {
        const response = await ssm.getParameters({Names: [path + 'sentryDsn']}).promise();
        const map = {};
        for (const param of response.Parameters) {
            map[param.Name.substr(path.length)] = param.Value;
        }
        logger.info('AWS info:', map);
        return map;
    } catch (err) {
        logger.error(`Failed to get AWS info: ${err}`);
        return {};
    }
}

export async function initConfig(properties) {
    awsConfigInit = true;
    awsProps = properties;
    awsConfig = await loadAwsConfig(properties);
}

export function getConfig(name) {
    if (!awsConfigInit)
        throw new Error("Reading AWS config before it's loaded");
    return awsConfig[name] || awsProps(name);
}
