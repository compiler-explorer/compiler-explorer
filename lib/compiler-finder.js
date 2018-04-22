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

const logger = require('./logger').logger,
    _ = require('underscore-node'),
    fs = require('fs-extra'),
    http = require('http'),
    aws = require('./aws');

class CompilerFinder {
    constructor(compileHandler, compilerPropsL, compilerPropsAT, ceProps, awsProps, languages, args) {
        this.compilerPropsL = compilerPropsL;
        this.compilerPropsAT = compilerPropsAT;
        this.ceProps = ceProps;
        this.awsProps = awsProps;
        this.args = args;
        this.compileHandler = compileHandler;
        this.languages = languages;
        this.awsPoller = null;
    }

    awsInstances() {
        if (!this.awsPoller) this.awsPoller = new aws.InstanceFetcher(this.awsProps);
        return this.awsPoller.getInstances();
    }

    fetchRemote(host, port, props) {
        logger.info(`Fetching compilers from remote source ${host}:${port}`);
        return this.retryPromise(() => {
            return new Promise((resolve, reject) => {
                let request = http.get({
                    hostname: host,
                    port: port,
                    path: "/api/compilers",
                    headers: {
                        Accept: 'application/json'
                    }
                }, res => {
                    let str = '';
                    res.on('data', chunk => {
                        str += chunk;
                    });
                    res.on('end', () => {
                        let compilers = JSON.parse(str).map(compiler => {
                            compiler.exe = null;
                            compiler.remote = `http://${host}:${port}`;
                            return compiler;
                        });
                        resolve(compilers);
                    });
                })
                    .on('error', reject)
                    .on('timeout', () => reject("timeout"));
                request.setTimeout(this.awsProps('proxyTimeout', 1000));
            });
        },
        `${host}:${port}`,
        props('proxyRetries', 5),
        props('proxyRetryMs', 500)
        ).catch(() => {
            logger.warn(`Unable to contact ${host}:${port}; skipping`);
            return [];
        });
    }

    fetchAws() {
        logger.info("Fetching instances from AWS");
        return this.awsInstances().then(instances => {
            return Promise.all(instances.map(instance => {
                logger.info("Checking instance " + instance.InstanceId);
                let address = instance.PrivateDnsName;
                if (this.awsProps("externalTestMode", false)) {
                    address = instance.PublicDnsName;
                }
                return this.fetchRemote(address, this.args.port, this.awsProps);
            }));
        });
    }

    compilerConfigFor(langId, compilerName, parentProps) {
        const base = `compiler.${compilerName}.`;

        function props(propName, def) {
            let propsForCompiler = parentProps(langId, base + propName, undefined);
            if (propsForCompiler === undefined) {
                propsForCompiler = parentProps(langId, propName, def);
            }
            return propsForCompiler;
        }

        const supportsBinary = !!props("supportsBinary", true);
        const supportsExecute = supportsBinary && !!props("supportsExecute", true);
        const group = props("group", "");
        const demangler = props("demangler", "");
        const compilerInfo = {
            id: compilerName,
            exe: props("exe", compilerName),
            name: props("name", compilerName),
            alias: props("alias"),
            options: props("options"),
            versionFlag: props("versionFlag"),
            versionRe: props("versionRe"),
            compilerType: props("compilerType", ""),
            demangler: demangler,
            demanglerClassFile: props("demanglerClassFile", ""),
            objdumper: props("objdumper", ""),
            intelAsm: props("intelAsm", ""),
            needsMulti: !!props("needsMulti", true),
            supportsDemangle: !!demangler,
            supportsBinary: supportsBinary,
            supportsExecute: supportsExecute,
            postProcess: props("postProcess", "").split("|"),
            lang: langId,
            group: group,
            groupName: props("groupName", ""),
            includeFlag: props("includeFlag", "-isystem"),
            notification: props("notification", "")
        };
        logger.debug("Found compiler", compilerInfo);
        return Promise.resolve(compilerInfo);
    }

    recurseGetCompilers(langId, compilerName, parentProps) {
        // Don't treat @ in paths as remote addresses if requested
        if (this.args.fetchCompilersFromRemote && compilerName.indexOf("@") !== -1) {
            const bits = compilerName.split("@");
            const host = bits[0];
            const port = parseInt(bits[1]);
            return this.fetchRemote(host, port, this.ceProps);
        }
        if (compilerName.indexOf("&") === 0) {
            const groupName = compilerName.substr(1);

            const props = (langId, name, def) => {
                if (name === "group") {
                    return groupName;
                }
                return this.compilerPropsL(langId, `group.${groupName}.${name}`, parentProps(langId, name, def));
            };
            const compilerExes = _.compact(this.compilerPropsL(langId, `group.${groupName}.compilers`, '').split(":"));
            logger.debug(`Processing compilers from group ${groupName}`);
            return Promise.all(compilerExes.map(compiler => this.recurseGetCompilers(langId, compiler, props)));
        }
        if (compilerName === "AWS") return this.fetchAws();
        return this.compilerConfigFor(langId, compilerName, parentProps);
    }

    getCompilers() {
        this.getExes();
        let compilers = [];
        _.each(this.exes, (exs, langId) => {
            _.each(exs, exe => compilers.push(this.recurseGetCompilers(langId, exe, this.compilerPropsL)));
        });
        return compilers;
    }

    ensureDistinct(compilers) {
        let ids = {};
        _.each(compilers, compiler => {
            if (!ids[compiler.id]) ids[compiler.id] = [];
            ids[compiler.id].push(compiler);
        });
        _.each(ids, (list, id) => {
            if (list.length !== 1) {
                logger.error(`Compiler ID clash for '${id}' - used by ${
                    _.map(list, o => `lang:${o.lang} name:${o.name}`).join(', ')
                }`);
            }
        });
        return compilers;
    }

    retryPromise(promiseFunc, name, maxFails, retryMs) {
        return new Promise(function (resolve, reject) {
            let fails = 0;

            function doit() {
                const promise = promiseFunc();
                promise.then(function (arg) {
                    resolve(arg);
                }, function (e) {
                    fails++;
                    if (fails < maxFails) {
                        logger.warn(`Failed ${name} : ${e}, retrying`);
                        setTimeout(doit, retryMs);
                    } else {
                        logger.error(`Too many retries for ${name} : ${e}`);
                        reject(e);
                    }
                });
            }

            doit();
        });
    }

    getExes() {
        this.exes = this.compilerPropsAT(this.languages, exs => _.compact(exs.split(":")), "compilers", "");
        this.getNdkExes();
    }

    getNdkExes() {
        const ndkPaths = this.compilerPropsAT(this.languages, _.identity, 'androidNdk');
        _.each(ndkPaths, (ndkPath, langId) => {
            if (ndkPath) {
                let toolchains = fs.readdirSync(`${ndkPath}/toolchains`);
                toolchains.forEach((version, index, a) => {
                    const path = `${ndkPath}/toolchains/${version}/prebuilt/linux-x86_64/bin/`;
                    if (fs.existsSync(path)) {
                        const cc = fs.readdirSync(path).filter(filename => filename.indexOf("g++") !== -1);
                        a[index] = path + cc[0];
                    } else {
                        a[index] = null;
                    }
                });
                toolchains = toolchains.filter(x => x !== null);
                this.exes[langId].push(toolchains);
            }
        });
    }

    find() {
        return Promise.all(this.getCompilers())
            .then(_.flatten)
            .then(compilers => this.compileHandler.setCompilers(compilers))
            .then(compilers => _.compact(compilers))
            .then(this.ensureDistinct)
            .then(compilers => _.sortBy(compilers, "name"));
    }
}

module.exports = CompilerFinder;
