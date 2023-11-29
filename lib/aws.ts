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

import {EC2, Instance} from '@aws-sdk/client-ec2';
import {SSM} from '@aws-sdk/client-ssm';
import {unwrap} from './assert.js';
import {logger} from './logger.js';
import type {PropertyGetter} from './properties.interfaces.js';
import {fromNodeProviderChain} from '@aws-sdk/credential-providers';
import {AwsCredentialIdentityProvider} from '@smithy/types/dist-types/identity/awsCredentialIdentity.js';

let cachedCredentials: AwsCredentialIdentityProvider | undefined;

export function awsCredentials(): AwsCredentialIdentityProvider {
    if (!cachedCredentials) throw new Error("Attempt to get AWS credentials before they've been initialised");
    return cachedCredentials;
}

export function fakeCredentialsForTest() {
    cachedCredentials = async () => {
        return {
            accessKeyId: 'not-a-real-key',
            secretAccessKey: 'not-a-real-secret',
        };
    };
}

async function initialiseAwsCredentials(region: string) {
    if (!cachedCredentials) {
        const provider = fromNodeProviderChain({
            logger: logger,
            timeout: 5000,
            maxRetries: 5,
            clientConfig: {region},
        });
        cachedCredentials = async (identityProperties?: Record<string, any>) => {
            logger.info(`Fetching AWS credentials for ${region}...`);
            const creds = await provider(identityProperties);
            logger.info(`Credentials: expiry:${creds.expiration}, keyId: ${creds.accessKeyId}`);
            return creds;
        };
    }
}

export class InstanceFetcher {
    ec2: EC2;
    tagKey: string;
    tagValue: string;

    constructor(properties: PropertyGetter) {
        const region = properties<string>('region');
        logger.info(`New instance fetcher for region ${region}`);
        this.ec2 = new EC2({region: region, credentials: awsCredentials()});
        this.tagKey = properties<string>('tagKey');
        this.tagValue = properties<string>('tagValue');
    }

    async getInstances() {
        const result = await this.ec2.describeInstances({});
        return unwrap(result.Reservations)
            .flatMap(r => r.Instances)
            .filter(reservation => {
                if (unwrap(unwrap(reservation).State).Name !== 'running') return false;
                return unwrap(unwrap(reservation).Tags).some(t => t.Key === this.tagKey && t.Value === this.tagValue);
            }) as Instance[];
    }
}

let awsConfigInit = false;
let awsConfig: Record<string, string | undefined> = {};
let awsProps: PropertyGetter | null = null;

async function loadAwsConfig(properties: PropertyGetter) {
    const region = properties<string>('region');
    if (!region) return {};
    await initialiseAwsCredentials(region);
    const ssm = new SSM({region: region, credentials: awsCredentials()});
    const path = '/compiler-explorer/';
    try {
        const response = await ssm.getParameters({Names: [path + 'sentryDsn']});
        const map: Record<string, string | undefined> = {};
        for (const param of unwrap(response.Parameters)) {
            map[unwrap(param.Name).substring(path.length)] = param.Value;
        }
        logger.info('AWS info:', map);
        return map;
    } catch (err) {
        logger.error(`Failed to get AWS info: ${err}`);
        return {};
    }
}

export async function initConfig(properties: PropertyGetter) {
    awsConfigInit = true;
    awsProps = properties;
    awsConfig = await loadAwsConfig(properties);
}

export function getConfig(name): string {
    if (!awsConfigInit) throw new Error("Reading AWS config before it's loaded");
    return awsConfig[name] || unwrap(awsProps)<string>(name);
}
