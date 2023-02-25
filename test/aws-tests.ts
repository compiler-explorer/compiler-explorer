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
import AWS from 'aws-sdk-mock';

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

function setup() {
    beforeEach(() => {
        AWS.mock('EC2', 'describeInstances', {
            Reservations: [
                {
                    Instances: [instanceA, instanceB, instanceC, instanceD],
                },
            ],
        });

        AWS.mock('SSM', 'getParameters', {
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
    afterEach(() => AWS.restore());
}

describe('AWS instance fetcher tests', () => {
    setup();
    it('Fetches Bob', () => {
        const fakeProps = {
            region: 'not-a-region',
            tagKey: 'Name',
            tagValue: 'Bob',
        };
        const fetcher = new aws.InstanceFetcher(prop => fakeProps[prop]);
        return fetcher.getInstances().should.eventually.deep.equal([instanceC]);
    });

    it('Ignores sleeping nodes', () => {
        const fakeProps = {
            region: 'not-a-region',
            tagKey: 'Name',
            tagValue: 'Alice',
        };
        const fetcher = new aws.InstanceFetcher(prop => fakeProps[prop]);
        return fetcher.getInstances().should.eventually.deep.equal([instanceA, instanceD]);
    });
});

describe('AWS config tests', () => {
    setup();
    it("Doesn't fetch unless region is configured", () => {
        const fakeProps = {
            region: '',
            configValue: 'fromConfigFile',
        };
        return aws
            .initConfig(prop => fakeProps[prop])
            .then(() => {
                aws.getConfig('configValue').should.equal('fromConfigFile');
            });
    });

    it('Gets results from SSM, falling back to config if needed', () => {
        const fakeProps = {
            region: 'a non-empty region',
            configValue: 'fromConfigFile',
            notInAmazon: 'yay',
        };
        return aws
            .initConfig(prop => fakeProps[prop])
            .then(() => {
                aws.getConfig('configValue').should.equal('fromAws');
                aws.getConfig('onlyOnAws').should.equal('bibble');
                aws.getConfig('notInAmazon').should.equal('yay');
            });
    });
});
