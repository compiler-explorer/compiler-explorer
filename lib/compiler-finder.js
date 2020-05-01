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
    _ = require('underscore'),
    fs = require('fs-extra'),
    path = require('path'),
    urljoin = require('url-join'),
    http = require('http'),
    https = require('https'),
    aws = require('./aws');

const sleep = require('util').promisify(setTimeout);

/***
 * Finds and initializes the compilers stored on the properties files
 */
class CompilerFinder {
    /***
     * @param {CompileHandler} compileHandler
     * @param {CompilerProps} compilerProps
     * @param {propsFor} awsProps
     * @param {Object} args
     */
    constructor(compileHandler, compilerProps, awsProps, args, optionsHandler) {
        this.compilerProps = compilerProps.get.bind(compilerProps);
        this.ceProps = compilerProps.ceProps;
        this.awsProps = awsProps;
        this.args = args;
        this.compileHandler = compileHandler;
        this.languages = compilerProps.languages;
        this.awsPoller = null;
        this.optionsHandler = optionsHandler;
    }

    awsInstances() {
        if (!this.awsPoller) this.awsPoller = new aws.InstanceFetcher(this.awsProps);
        return this.awsPoller.getInstances();
    }

    async fetchRemote(host, port, uriBase, props, langId) {
        const requestLib = port === 443 ? https : http;
        const uriSchema = port === 443 ? "https" : "http";
        const uri = urljoin(`${uriSchema}://${host}:${port}`, uriBase);
        const apiPath = urljoin('/', uriBase || '', 'api/compilers', langId || '');
        logger.info(`Fetching compilers from remote source ${uri}`);
        return this.retryPromise(
            () => {
                return new Promise((resolve, reject) => {
                    const request = requestLib.get({
                        hostname: host,
                        port: port,
                        path: apiPath,
                        headers: {
                            Accept: 'application/json'
                        }
                    }, res => {
                        let error;
                        const {statusCode} = res;
                        const contentType = res.headers['content-type'];

                        if (statusCode !== 200) {
                            error = new Error('Failed fetching remote compilers from ' +
                                `${uriSchema}://${host}:${port}${apiPath}\n` +
                                `Status Code: ${statusCode}`);
                        } else if (!/^application\/json/.test(contentType)) {
                            error = new Error('Invalid content-type.\n' +
                                `Expected application/json but received ${contentType}`);
                        }
                        if (error) {
                            logger.error(error.message);
                            // consume response data to free up memory
                            res.resume();
                            reject(error);
                            return;
                        }
                        let str = '';
                        res.on('data', chunk => {
                            str += chunk;
                        });
                        res.on('end', () => {
                            try {
                                const compilers = JSON.parse(str).map(compiler => {
                                    // Fix up old upstream implementations of Compiler Explorer
                                    // e.g. https://www.godbolt.ms
                                    // (see https://github.com/mattgodbolt/compiler-explorer/issues/1768)
                                    if (!compiler.alias) compiler.alias = [];
                                    if (typeof compiler.alias == "string") compiler.alias = [compiler.alias];
                                    // End fixup
                                    compiler.exe = null;
                                    compiler.remote = {
                                        target: `${uriSchema}://${host}:${port}`,
                                        path: urljoin('/', uriBase, 'api/compiler', compiler.id, 'compile')
                                    };
                                    return compiler;
                                });
                                resolve(compilers);
                            } catch (e) {
                                logger.error(`Error parsing response from ${uri} '${str}': ${e.message}`);
                                reject(e);
                            }
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

    async fetchAws() {
        logger.info("Fetching instances from AWS");
        const instances = await this.awsInstances();
        return Promise.all(instances.map(instance => {
            logger.info("Checking instance " + instance.InstanceId);
            const address = this.awsProps("externalTestMode", false) ? instance.PublicDnsName : instance.PrivateDnsName;
            return this.fetchRemote(address, this.args.port, '', this.awsProps, null);
        }));
    }

    copyAndFilterLibraries(allLibraries, filter) {
        const filterLibAndVersion = _.map(filter, (lib) => {
            const match = lib.match(/([a-z0-9_-]*)\.([a-z0-9_-]*)/i);
            if (match) {
                return {
                    id: match[1],
                    version: match[2]
                };
            } else {
                return {
                    id: lib,
                    version: false
                };
            }
        });

        const filterLibIds = new Set();
        _.each(filterLibAndVersion, (lib) => {
            filterLibIds.add(lib.id);
        });

        const copiedLibraries = {};
        _.each(allLibraries, (lib, libid) => {
            if (!filterLibIds.has(libid)) return;

            const libcopy = Object.assign({}, lib);
            libcopy.versions = _.omit(lib.versions, (version, versionid) => {
                for (let filterIdx = 0; filterIdx < filterLibAndVersion.length; filterIdx++) {
                    const filter = filterLibAndVersion[filterIdx];
                    if (filter.id === libid) {
                        if (!filter.version) return false;
                        if (filter.version === versionid) return false;
                    }
                }

                return true;
            });

            copiedLibraries[libid] = libcopy;
        });

        return copiedLibraries;
    }

    async compilerConfigFor(langId, compilerName, parentProps) {
        const base = `compiler.${compilerName}.`;

        function props(propName, def) {
            const propsForCompiler = parentProps(langId, base + propName, undefined);
            if (propsForCompiler !== undefined) return propsForCompiler;
            return parentProps(langId, propName, def);
        }

        const ceToolsPath = props("ceToolsPath", "./");

        const supportsBinary = !!props("supportsBinary", true);
        const supportsExecute = supportsBinary && !!props("supportsExecute", true);
        const supportsLibraryCodeFilter = !!props("supportsLibraryCodeFilter", true);

        const group = props("group", "");

        const demanglerProp = props("demangler", "");
        const demangler = demanglerProp ? path.normalize(demanglerProp.replace("${ceToolsPath}", ceToolsPath)) : "";

        const isSemVer = props("isSemVer", false);
        const baseName = props("baseName", null);
        const semverVer = props("semver", "");

        const name = props("name", compilerName);

        const baseOptions = props("baseOptions", "");
        const options = props("options", "");
        let actualOptions = _.compact([baseOptions, options]).join(' ');

        let supportedLibraries = null;

        const supportsLibrariesSetting = _.filter(props("supportsLibraries", "").split(":"), (a) => a !== "");
        if (supportsLibrariesSetting.length > 0) {
            const libs = this.optionsHandler.get().libs[langId];
            supportedLibraries = this.copyAndFilterLibraries(libs, supportsLibrariesSetting);
        } else {
            supportedLibraries = this.optionsHandler.get().libs[langId];
        }

        const envVars = (() => {
            const envVarsString = props("envVars", "");
            if (envVarsString === "") {
                return [];
            }
            const arr = [];
            for (const el of envVarsString.split(':')) {
                const [env, setting] = el.split('=');
                arr.push([env, setting]);
            }
            return arr;
        })();
        const exe = props("exe", compilerName);
        const exePath = path.dirname(exe);
        const compilerInfo = {
            id: compilerName,
            exe: exe,
            name: isSemVer && baseName ? `${baseName} ${semverVer}` : name,
            alias: _.filter(props("alias", "").split(":"), (a) => a !== ""),
            options: actualOptions,
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
            supportsLibraryCodeFilter: supportsLibraryCodeFilter,
            postProcess: props("postProcess", "").split("|"),
            lang: langId,
            group: group,
            groupName: props("groupName", ""),
            includeFlag: props("includeFlag", "-isystem"),
            includePath: props("includePath", ""),
            linkFlag: props("linkFlag", "-l"),
            rpathFlag: props("rpathFlag", "-Wl,-rpath,"),
            libpathFlag: props("libpathFlag", "-L"),
            libPath: props("libPath", ""),
            ldPath: props("ldPath", "")
                .split("|")
                .map(x => path.normalize(x.replace("${exePath}", exePath))),
            envVars: envVars,
            notification: props("notification", ""),
            isSemVer: isSemVer,
            semver: semverVer,
            libs: supportedLibraries,
            tools: _.omit(this.optionsHandler.get().tools[langId], tool => tool.isCompilerExcluded(compilerName)),
            unwiseOptions: props("unwiseOptions", "").split("|")
        };
        logger.debug("Found compiler", compilerInfo);
        return compilerInfo;
    }

    async recurseGetCompilers(langId, compilerName, parentProps) {
        // Don't treat @ in paths as remote addresses if requested
        if (this.args.fetchCompilersFromRemote && compilerName.indexOf("@") !== -1) {
            const bits = compilerName.split("@");
            const host = bits[0];
            const pathParts = bits[1].split('/');
            const port = parseInt(pathParts.shift());
            const path = pathParts.join('/');
            return this.fetchRemote(host, port, path, this.ceProps, langId);
        }
        if (compilerName.indexOf("&") === 0) {
            const groupName = compilerName.substr(1);

            const props = (langId, name, def) => {
                if (name === "group") {
                    return groupName;
                }
                return this.compilerProps(langId, `group.${groupName}.${name}`, parentProps(langId, name, def));
            };
            const exes = _.compact(this.compilerProps(langId, `group.${groupName}.compilers`, '').split(":"));
            logger.debug(`Processing compilers from group ${groupName}`);
            return Promise.all(exes.map(compiler => this.recurseGetCompilers(langId, compiler, props)));
        }
        if (compilerName === "AWS") return this.fetchAws();
        return this.compilerConfigFor(langId, compilerName, parentProps);
    }

    async getCompilers() {
        const compilers = [];
        _.each(this.getExes(), (exs, langId) => {
            _.each(exs, exe => compilers.push(this.recurseGetCompilers(langId, exe, this.compilerProps)));
        });
        return Promise.all(compilers);
    }

    ensureDistinct(compilers) {
        const ids = {};
        let foundClash = false;
        _.each(compilers, compiler => {
            if (!ids[compiler.id]) ids[compiler.id] = [];
            ids[compiler.id].push(compiler);
        });
        _.each(ids, (list, id) => {
            if (list.length !== 1) {
                foundClash = true;
                logger.error(`Compiler ID clash for '${id}' - used by ${
                    _.map(list, o => `lang:${o.lang} name:${o.name}`).join(', ')}`);
            }
        });
        return {compilers, foundClash};
    }

    async retryPromise(promiseFunc, name, maxFails, retryMs) {
        for (let fails = 0; fails < maxFails; ++fails) {
            try {
                return await promiseFunc();
            } catch (e) {
                if (fails < (maxFails - 1)) {
                    logger.warn(`Failed ${name} : ${e}, retrying`);
                    await sleep(retryMs);
                } else {
                    logger.error(`Too many retries for ${name} : ${e}`);
                    throw e;
                }
            }
        }
    }

    getExes() {
        const langToCompilers = this.compilerProps(this.languages, "compilers", "", exs => _.compact(exs.split(":")));
        this.addNdkExes(langToCompilers);
        logger.info('Exes found:', langToCompilers);
        return langToCompilers;
    }

    addNdkExes(langToCompilers) {
        const ndkPaths = this.compilerProps(this.languages, 'androidNdk');
        _.each(ndkPaths, (ndkPath, langId) => {
            if (ndkPath) {
                const toolchains = fs.readdirSync(`${ndkPath}/toolchains`);
                toolchains.forEach((version, index, a) => {
                    const path = `${ndkPath}/toolchains/${version}/prebuilt/linux-x86_64/bin/`;
                    if (fs.existsSync(path)) {
                        const cc = fs.readdirSync(path).filter(filename => filename.indexOf("g++") !== -1);
                        a[index] = path + cc[0];
                    } else {
                        a[index] = null;
                    }
                });
                langToCompilers[langId].push(toolchains.filter(x => x !== null));
            }
        });
    }

    async find() {
        let compilers = (await this.getCompilers()).flat(Infinity);
        compilers = await this.compileHandler.setCompilers(compilers);
        const result = this.ensureDistinct(_.compact(compilers));
        return {foundClash: result.foundClash, compilers: _.sortBy(result.compilers, "name")};
    }
}

module.exports = CompilerFinder;
