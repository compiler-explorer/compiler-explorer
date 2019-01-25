// Copyright (c) 2016, Matt Godbolt
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

const AWS = require('aws-sdk'),
    logger = require('./logger').logger;

class InstanceFetcher {
    constructor(properties) {
        const region = properties('region');
        logger.info(`New instance fetcher for region ${region}`);
        this.ec2 = new AWS.EC2({region: region});
        this.tagKey = properties('tagKey');
        this.tagValue = properties('tagValue');
    }

    onInstances(result) {
        let allInstances = [];
        result.Reservations.forEach(res => allInstances = allInstances.concat(res.Instances));
        return allInstances.filter(reservation => {
            if (reservation.State.Name !== "running") return false;
            return reservation.Tags.some(t => t.Key === this.tagKey && t.Value === this.tagValue);
        });
    }

    getInstances() {
        return this.ec2.describeInstances()
            .promise()
            .then(result => this.onInstances(result));
    }
}

let awsConfigInit = false;
let awsConfig = {};
let awsProps = null;

function loadAwsConfig(properties) {
    let region = properties('region');
    if (!region) {
        return Promise.resolve({});
    }
    let ssm = new AWS.SSM({region: region});
    const path = '/compiler-explorer/';
    return ssm.getParametersByPath({
        Path: path
    })
        .promise()
        .then((response) => {
            const map = {};
            response.Parameters.forEach((response) => {
                map[response.Name.substr(path.length)] = response.Value;
            });
            logger.info("AWS info:", map);
            return map;
        })
        .catch((error) => {
            logger.error("Failed to get AWS info: " + error);
            return {};
        });
}

function initConfig(properties) {
    awsConfigInit = true;
    awsProps = properties;
    return loadAwsConfig(properties).then(map => awsConfig = map);
}

function getConfig(name) {
    if (!awsConfigInit)
        throw new Error("Reading AWS config before it's loaded");
    return awsConfig[name] || awsProps(name);
}

module.exports = {
    InstanceFetcher: InstanceFetcher,
    initConfig: initConfig,
    getConfig: getConfig
};
