// Copyright (c) 2020, Compiler Explorer Authors
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
    BuildEnvSetupBase = require('./buildenvsetup-base'),
    logger = require('../logger').logger,
    request = require('request'),
    fs = require('fs-extra'),
    tar = require('tar-stream'),
    zlib = require('zlib'),
    path = require('path'),
    _ = require('underscore');

class BuildEnvSetupCeConanDirect extends BuildEnvSetupBase {
    constructor(compilerInfo, env, execCompilerCachedFunc) {
        super(compilerInfo, env, execCompilerCachedFunc);

        this.host = compilerInfo.buildenvsetup.props('host', false);
        this.onlyonstaticliblink = compilerInfo.buildenvsetup.props('onlyonstaticliblink', false);

        if (env.debug) request.debug = true;
    }

    async getAllPossibleBuilds(libid, version) {
        return new Promise((resolve, reject) => {
            const encLibid = encodeURIComponent(libid);
            const encVersion = encodeURIComponent(version);
            const url = `${this.host}/v1/conans/${encLibid}/${encVersion}/${encLibid}/${encVersion}/search`;
            const settings = {
                method: "GET",
                json: true
            };

            request(url, settings, (err, res, body) => {
                if (res.statusCode === 404) {
                    reject(`Not found (${url})`);
                    return;
                } else if (err) {
                    reject(err);
                    return;
                } else {
                    resolve(body);
                }
            });
        });
    }

    async getPackageUrl(libid, version, hash) {
        return new Promise((resolve, reject) => {
            const encLibid = encodeURIComponent(libid);
            const encVersion = encodeURIComponent(version);
            const libUrl = `${this.host}/v1/conans/${encLibid}/${encVersion}/${encLibid}/${encVersion}`;
            const url = `${libUrl}/packages/${hash}/download_urls`;

            const settings = {
                method: "GET",
                json: true
            };

            request(url, settings, (err, res, body) => {
                if (err) {
                    reject(err);
                    return;
                }

                resolve(body['conan_package.tgz']);
            });
        });
    }

    async downloadAndExtractPackage(downloadPath, packageUrl) {
        return new Promise((resolve) => {
            const extract = tar.extract();
            const gunzip = zlib.createGunzip();

            extract.on('entry', (header, stream, next) => {
                const filename = path.basename(header.name);
                const filestream = fs.createWriteStream(path.join(downloadPath, filename));
                stream.pipe(filestream);
                stream.on('end', next);
                stream.resume();
            });

            extract.on('finish', () => {
                resolve();
            });

            gunzip.pipe(extract);

            const settings = {
                method: 'GET',
                encoding: null
            };

            request(packageUrl, settings).pipe(gunzip);
        });
    }

    async getConanBuildProperties(key) {
        const arch = this.getTarget(key);
        const libcxx = this.getLibcxx(key);
        const stdver = "";
        const flagcollection = "";

        const buildProperties = {
            os: "Linux",
            build_type: "Debug",
            compiler: this.compilerTypeOrGCC,
            "compiler.version": this.compiler.id,
            "compiler.libcxx": libcxx,
            arch: arch,
            stdver: stdver,
            flagcollection: flagcollection
        };

        return buildProperties;
    }

    async findMatchingHash(buildProperties, possibleBuilds) {
        return _.findKey(possibleBuilds, (elem) => {
            return _.all(buildProperties, (val, key) => {
                return val === elem.settings[key];
            });
        });
    }

    async download(key, dirPath, libraryDetails) {
        const allDownloads = [];
        const allLibraryBuilds = [];

        _.each(libraryDetails, (details, libId) => {
            if (this.hasBinariesToLink(details)) {
                const lookupversion = details.lookupversion ? details.lookupversion : details.version;
                allLibraryBuilds.push({
                    id: libId,
                    version: details.version,
                    lookupversion: details.lookupversion,
                    possibleBuilds: this.getAllPossibleBuilds(libId, lookupversion).catch(() => false)
                });
            }
        });

        const buildProperties = await this.getConanBuildProperties(key);

        for (let idx = 0; idx < allLibraryBuilds.length; idx++) {
            const libVerBuilds = allLibraryBuilds[idx];
            const lookupversion = libVerBuilds.lookupversion ? libVerBuilds.lookupversion : libVerBuilds.version;
            const libVer = `${libVerBuilds.id}/${lookupversion}`;
            const possibleBuilds = await libVerBuilds.possibleBuilds;
            if (possibleBuilds) {
                const hash = await this.findMatchingHash(buildProperties, possibleBuilds);
                if (hash) {
                    logger.debug(`Found conan hash ${hash} for ${libVer}`);
                    allDownloads.push(
                        this.getPackageUrl(libVerBuilds.id, lookupversion, hash).then((downloadUrl) => {
                            return this.downloadAndExtractPackage(dirPath, downloadUrl);
                        })
                    );
                } else {
                    logger.info(`No build found for ${libVer} matching ${JSON.stringify(buildProperties)}`);
                }
            } else {
                logger.info(`Library ${libVer} not available`);
            }
        }

        return Promise.all(allDownloads);
    }

    async setup(key, dirPath, libraryDetails) {
        if (this.host && (!this.onlyonstaticliblink || this.hasAtLeastOneBinaryToLink(libraryDetails))) {
            return this.download(key, dirPath, libraryDetails);
        } else {
            return Promise.resolve();
        }
    }

    hasBinariesToLink(details) {
        return (details.libpath.length === 0) && ((details.staticliblink.length > 0) || (details.liblink.length > 0));
    }

    hasAtLeastOneBinaryToLink(libraryDetails) {
        return _.some(libraryDetails, (details) => this.hasBinariesToLink(details));
    }
}

module.exports = BuildEnvSetupCeConanDirect;
