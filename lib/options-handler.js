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

import https from 'https';

import fs from 'fs-extra';
import semverParser from 'semver';
import _ from 'underscore';

import {logger} from './logger';
import {getToolTypeByKey} from './tooling';
import {asSafeVer, getHash, splitArguments} from './utils';

/***
 * Handles the setup of the options object passed on each page request
 */
export class ClientOptionsHandler {
    /***
     *
     * @param {Object[]} fileSources - Files to show in the Load/Save pane
     * @param {string} fileSources.name - UI display name of the file
     * @param {string} fileSources.urlpart - Relative url path to fetch the file from
     * @param {CompilerProps} compilerProps
     * @param {Object} defArgs - Compiler Explorer arguments
     */
    constructor(fileSources, compilerProps, defArgs) {
        this.compilerProps = compilerProps.get.bind(compilerProps);
        this.ceProps = compilerProps.ceProps;
        const ceProps = compilerProps.ceProps;
        const sources = _.sortBy(
            fileSources.map(source => {
                return {name: source.name, urlpart: source.urlpart};
            }),
            'name',
        );

        /***
         * @type {CELanguages}
         */
        const languages = compilerProps.languages;

        this.supportsBinary = this.compilerProps(languages, 'supportsBinary', true, res => !!res);
        this.supportsExecutePerLanguage = this.compilerProps(languages, 'supportsExecute', true, res => !!res);
        this.supportsExecute = Object.values(this.supportsExecutePerLanguage).some(value => value);

        this.supportsLibraryCodeFilterPerLanguage = this.compilerProps(languages, 'supportsLibraryCodeFilter', false);
        this.supportsLibraryCodeFilter = Object.values(this.supportsLibraryCodeFilterPerLanguage).some(value => value);

        const libs = this.parseLibraries(this.compilerProps(languages, 'libs'));
        const tools = this.parseTools(this.compilerProps(languages, 'tools'));

        this.remoteLibs = {};

        const cookiePolicyEnabled = !!ceProps('cookiePolicyEnabled');
        const privacyPolicyEnabled = !!ceProps('privacyPolicyEnabled');
        const cookieDomainRe = ceProps('cookieDomainRe', '');
        this.options = {
            googleAnalyticsAccount: ceProps('clientGoogleAnalyticsAccount', 'UA-55180-6'),
            googleAnalyticsEnabled: ceProps('clientGoogleAnalyticsEnabled', false),
            sharingEnabled: ceProps('clientSharingEnabled', true),
            githubEnabled: ceProps('clientGitHubRibbonEnabled', true),
            showSponsors: ceProps('showSponsors', false),
            gapiKey: ceProps('googleApiKey', ''),
            googleShortLinkRewrite: ceProps('googleShortLinkRewrite', '').split('|'),
            urlShortenService: ceProps('urlShortenService', 'default'),
            defaultSource: ceProps('defaultSource', ''),
            compilers: [],
            libs: libs,
            remoteLibs: {},
            tools: tools,
            defaultLibs: this.compilerProps(languages, 'defaultLibs', ''),
            defaultCompiler: this.compilerProps(languages, 'defaultCompiler', ''),
            compileOptions: this.compilerProps(languages, 'defaultOptions', ''),
            supportsBinary: this.supportsBinary,
            supportsExecute: this.supportsExecute,
            supportsLibraryCodeFilter: this.supportsLibraryCodeFilter,
            languages: languages,
            sources: sources,
            sentryDsn: ceProps('sentryDsn', ''),
            sentryEnvironment: ceProps('sentryEnvironment') || defArgs.env[0],
            release: defArgs.releaseBuildNumber || defArgs.gitReleaseName,
            gitReleaseCommit: defArgs.gitReleaseName || '',
            cookieDomainRe: cookieDomainRe,
            localStoragePrefix: ceProps('localStoragePrefix'),
            cvCompilerCountMax: ceProps('cvCompilerCountMax', 6),
            defaultFontScale: ceProps('defaultFontScale', 14),
            doCache: defArgs.doCache,
            thirdPartyIntegrationEnabled: ceProps('thirdPartyIntegrationEnabled', true),
            statusTrackingEnabled: ceProps('statusTrackingEnabled', true),
            policies: {
                cookies: {
                    enabled: cookiePolicyEnabled,
                    key: 'cookie_status',
                },
                privacy: {
                    enabled: privacyPolicyEnabled,
                    key: 'privacy_status',
                },
            },
            motdUrl: ceProps('motdUrl', ''),
            pageloadUrl: ceProps('pageloadUrl', ''),
        };
        this._updateOptionsHash();
    }

    parseTools(baseTools) {
        const tools = {};
        for (const [lang, forLang] of Object.entries(baseTools)) {
            if (lang && forLang) {
                tools[lang] = {};
                for (const tool of forLang.split(':')) {
                    const toolBaseName = `tools.${tool}`;
                    const className = this.compilerProps(lang, toolBaseName + '.class');
                    const Tool = getToolTypeByKey(className);

                    const toolPath = this.compilerProps(lang, toolBaseName + '.exe');
                    if (fs.existsSync(toolPath)) {
                        tools[lang][tool] = new Tool(
                            {
                                id: tool,
                                name: this.compilerProps(lang, toolBaseName + '.name'),
                                type: this.compilerProps(lang, toolBaseName + '.type'),
                                exe: toolPath,
                                exclude: this.compilerProps(lang, toolBaseName + '.exclude'),
                                includeKey: this.compilerProps(lang, toolBaseName + '.includeKey'),
                                options: splitArguments(this.compilerProps(lang, toolBaseName + '.options')),
                                args: this.compilerProps(lang, toolBaseName + '.args'),
                                languageId: this.compilerProps(lang, toolBaseName + '.languageId'),
                                stdinHint: this.compilerProps(lang, toolBaseName + '.stdinHint'),
                                monacoStdin: this.compilerProps(lang, toolBaseName + '.monacoStdin'),
                                compilerLanguage: lang,
                            },
                            {
                                ceProps: this.ceProps,
                                compilerProps: propname => this.compilerProps(lang, propname),
                            },
                        );
                    } else {
                        logger.warn(`Unable to stat ${toolBaseName} tool binary`);
                    }
                }
            }
        }
        return tools;
    }

    splitIntoArray(value, defaultArr) {
        if (value) {
            return value.split(':');
        } else {
            return defaultArr;
        }
    }

    parseLibraries(baseLibs) {
        const libraries = {};
        for (const [lang, forLang] of Object.entries(baseLibs)) {
            if (lang && forLang) {
                libraries[lang] = {};
                for (const lib of forLang.split(':')) {
                    const libBaseName = `libs.${lib}`;
                    libraries[lang][lib] = {
                        name: this.compilerProps(lang, libBaseName + '.name'),
                        url: this.compilerProps(lang, libBaseName + '.url'),
                        description: this.compilerProps(lang, libBaseName + '.description'),
                        staticliblink: this.splitIntoArray(
                            this.compilerProps(lang, libBaseName + '.staticliblink'),
                            [],
                        ),
                        liblink: this.splitIntoArray(this.compilerProps(lang, libBaseName + '.liblink'), []),
                        dependencies: this.splitIntoArray(this.compilerProps(lang, libBaseName + '.dependencies'), []),
                        versions: {},
                        examples: this.splitIntoArray(this.compilerProps(lang, libBaseName + '.examples'), []),
                        options: splitArguments(this.compilerProps(lang, libBaseName + '.options', '')),
                    };
                    const listedVersions = `${this.compilerProps(lang, libBaseName + '.versions')}`;
                    if (listedVersions) {
                        for (const version of listedVersions.split(':')) {
                            const libVersionName = libBaseName + `.versions.${version}`;
                            const versionObject = {
                                version: this.compilerProps(lang, libVersionName + '.version'),
                                staticliblink: this.splitIntoArray(
                                    this.compilerProps(lang, libVersionName + '.staticliblink'),
                                    libraries[lang][lib].staticliblink,
                                ),
                                alias: this.splitIntoArray(this.compilerProps(lang, libVersionName + '.alias'), []),
                                dependencies: this.splitIntoArray(
                                    this.compilerProps(lang, libVersionName + '.dependencies'),
                                    libraries[lang][lib].dependencies,
                                ),
                                path: [],
                                libpath: [],
                                liblink: this.splitIntoArray(
                                    this.compilerProps(lang, libVersionName + '.liblink'),
                                    libraries[lang][lib].liblink,
                                ),
                                // Library options might get overridden later
                                options: libraries[lang][lib].options,
                                hidden: this.compilerProps(lang, libVersionName + '.hidden', false),
                            };

                            const lookupversion = this.compilerProps(lang, libVersionName + '.lookupversion');
                            if (lookupversion) {
                                versionObject.lookupversion = lookupversion;
                            }

                            const includes = this.compilerProps(lang, libVersionName + '.path');
                            if (includes && process.platform === 'win32') {
                                versionObject.path = includes.split(';');
                            } else if (includes) {
                                versionObject.path = includes.split(':');
                            } else {
                                logger.warn(`Library ${lib} ${version} (${lang}) has no include paths`);
                            }

                            const libpath = this.compilerProps(lang, libVersionName + '.libpath');
                            if (libpath && process.platform === 'win32') {
                                versionObject.libpath = libpath.split(';');
                            } else if (libpath) {
                                versionObject.libpath = libpath.split(':');
                            }

                            const options = this.compilerProps(lang, libVersionName + '.options');
                            if (options !== undefined) {
                                versionObject.options = splitArguments(options);
                            }

                            libraries[lang][lib].versions[version] = versionObject;
                        }
                    } else {
                        logger.warn(`No versions found for ${lib} library`);
                    }
                }
            }
        }
        return libraries;
    }

    getRemoteId(remoteUrl, language) {
        const url = new URL(remoteUrl);
        return url.host.replace(/\./g, '_') + '_' + language;
    }

    libArrayToObject(libsArr) {
        const libs = {};
        for (const lib of libsArr) {
            libs[lib.id] = lib;

            const versions = lib.versions;
            lib.versions = {};

            for (const version of versions) {
                lib.versions[version.id] = version;
            }
        }
        return libs;
    }

    async getRemoteLibraries(language, remoteUrl) {
        const remoteId = this.getRemoteId(remoteUrl, language);
        if (!this.remoteLibs[remoteId]) {
            return new Promise(resolve => {
                const url = remoteUrl + '/api/libraries/' + language;
                logger.info(`Fetching remote libraries from ${url}`);
                let fullData = '';
                https.get(url, res => {
                    res.on('data', data => {
                        fullData += data;
                    });
                    res.on('end', () => {
                        const libsArr = JSON.parse(fullData);

                        this.remoteLibs[remoteId] = this.libArrayToObject(libsArr);

                        resolve(this.remoteLibs[remoteId]);
                    });
                });
            });
        }
        return this.remoteLibs[remoteId];
    }

    async fetchRemoteLibrariesIfNeeded(language, remote) {
        await this.getRemoteLibraries(language, remote.target);
    }

    async setCompilers(compilers) {
        const forbiddenKeys = new Set([
            'exe',
            'versionFlag',
            'versionRe',
            'compilerType',
            'demangler',
            'objdumper',
            'postProcess',
            'demanglerType',
            'isSemVer',
        ]);
        const copiedCompilers = JSON.parse(JSON.stringify(compilers));
        let semverGroups = {};
        // Reset the supportsExecute flag in case critical compilers change

        for (const key of Object.keys(this.options.languages)) {
            this.options.languages[key].supportsExecute = false;
        }

        for (const [compilersKey, compiler] of copiedCompilers.entries()) {
            if (compiler.supportsExecute) {
                this.options.languages[compiler.lang].supportsExecute = true;
            }
            if (compiler.isSemVer) {
                if (!semverGroups[compiler.group]) semverGroups[compiler.group] = [];
                // Desired index which will keep the array in order
                semverGroups[compiler.group].push(compiler);
            }

            if (compiler.remote) {
                await this.fetchRemoteLibrariesIfNeeded(compiler.lang, compiler.remote);
            }

            for (const propKey of Object.keys(compiler)) {
                if (forbiddenKeys.has(propKey)) {
                    delete copiedCompilers[compilersKey][propKey];
                }
            }
        }

        for (const group of Object.values(semverGroups)) {
            group.sort((a, b) => semverParser.compare(asSafeVer(a.semver), asSafeVer(b.semver), true));
            let order = 0;
            // Set $order to -index on array. As group is an array, iteration order is guaranteed.
            for (const compiler of group) {
                compiler['$order'] = -order++;
            }
        }

        this.options.compilers = copiedCompilers;
        this.options.remoteLibs = this.remoteLibs;
        this._updateOptionsHash();
    }

    _updateOptionsHash() {
        this.optionsJSON = JSON.stringify(this.options);
        this.optionsHash = getHash(this.options, 'Options Hash V1');
        logger.info(`OPTIONS HASH: ${this.optionsHash}`);
    }

    get() {
        return this.options;
    }

    getJSON() {
        return this.optionsJSON;
    }

    getHash() {
        return this.optionsHash;
    }
}
