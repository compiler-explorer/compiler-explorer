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

import './utils.js';
import {DescribeInstancesCommand, EC2, Instance} from '@aws-sdk/client-ec2';
import {GetParametersCommand, SSM} from '@aws-sdk/client-ssm';
import {mockClient} from 'aws-sdk-client-mock';
import {beforeEach, describe, expect, it} from 'vitest';

import * as aws from '../lib/aws.js';

const instanceA = {
    State: {Name: 'running'},
    Id: 'A',
    Tags: [
        {Key: 'Name', Value: 'Alice'},
        {Key: 'Moose', Value: 'Bob'},
    ],
};
const instanceB = {
    State: {Name: 'sleeping'},
    Id: 'B',
    Tags: [
        {Key: 'Name', Value: 'Alice'},
        {Key: 'Moose', Value: 'Bob'},
    ],
};
const instanceC = {
    State: {Name: 'running'},
    Id: 'C',
    Tags: [
        {Key: 'Name', Value: 'Bob'},
        {Key: 'Moose', Value: 'Bob'},
    ],
};
const instanceD = {
    State: {Name: 'running'},
    Id: 'D',
    Tags: [
        {Key: 'Name', Value: 'Alice'},
        {Key: 'Moose', Value: 'Bob'},
    ],
};

describe('AWS instance fetcher tests', () => {
    const mockEC2 = mockClient(EC2);
    beforeEach(() => {
        mockEC2.reset();
        mockEC2.on(DescribeInstancesCommand).resolves({
            Reservations: [
                {
                    Instances: [instanceA, instanceB, instanceC, instanceD] as Instance[],
                },
            ],
        });
    });
    it('Fetches Bob', async () => {
        const fakeProps = {
            region: 'not-a-region',
            tagKey: 'Name',
            tagValue: 'Bob',
        };
        const fetcher = new aws.InstanceFetcher(prop => fakeProps[prop]);
        await expect(fetcher.getInstances()).resolves.toEqual([instanceC]);
    });

    it('Ignores sleeping nodes', async () => {
        const fakeProps = {
            region: 'not-a-region',
            tagKey: 'Name',
            tagValue: 'Alice',
        };
        const fetcher = new aws.InstanceFetcher(prop => fakeProps[prop]);
        await expect(fetcher.getInstances()).resolves.toEqual([instanceA, instanceD]);
    });
});

describe('AWS config tests', () => {
    const mockSSM = mockClient(SSM);
    beforeEach(() => {
        mockSSM.reset();
        mockSSM.on(GetParametersCommand).resolves({
            Parameters: [
                {
                    Name: '/compiler-explorer/configValue',
                    Value: 'fromAws',
                },
                {
                    Name: '/compiler-explorer/onlyOnAws',
                    Value: 'bibble',
                },
            ],
        });
    });
    it("Doesn't fetch unless region is configured", async () => {
        const fakeProps = {
            region: '',
            configValue: 'fromConfigFile',
        };
        await aws.initConfig(prop => fakeProps[prop]);
        expect(aws.getConfig('configValue')).toEqual('fromConfigFile');
    });

    it('Gets results from SSM, falling back to config if needed', async () => {
        const fakeProps = {
            region: 'a non-empty region',
            configValue: 'fromConfigFile',
            notInAmazon: 'yay',
        };
        await aws.initConfig(prop => fakeProps[prop]);
        expect(aws.getConfig('configValue')).toEqual('fromAws');
        expect(aws.getConfig('onlyOnAws')).toEqual('bibble');
        expect(aws.getConfig('notInAmazon')).toEqual('yay');
    });
});
