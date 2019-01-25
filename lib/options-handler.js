// Copyright (c) 2018, Compiler Explorer Team
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

const _ = require('underscore'),
    logger = require('./logger').logger,
    semverParser = require('semver'),
    path = require('path'),
    getHash = require('./utils').getHash,
    fs = require('fs-extra');

const HashVersion = 'Compiler Explorer Policies Version 1';

/***
 * Handles the setup of the options object passed on each page request
 */
class ClientOptionsHandler {
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
        const sources = _.sortBy(fileSources.map(source => {
            return {name: source.name, urlpart: source.urlpart};
        }), 'name');

        /***
         * @type {CELanguages}
         */
        const languages = compilerProps.languages;

        this.supportsBinary = this.compilerProps(languages, 'supportsBinary', true, res => !!res);
        this.supportsExecutePerLanguage = this.compilerProps(languages, 'supportsExecute', true, (res, lang) => {
            return this.supportsBinary[lang.id] && !!res;
        });
        this.supportsExecute = Object.values(this.supportsExecutePerLanguage).some(value => value);

        this.supportsLibraryCodeFilterPerLanguage = this.compilerProps(languages, 'supportsLibraryCodeFilter', false);
        this.supportsLibraryCodeFilter = Object.values(this.supportsLibraryCodeFilterPerLanguage).some(value => value);

        const libs = this.parseLibraries(this.compilerProps(languages, 'libs'));
        const tools = this.parseTools(this.compilerProps(languages, 'tools'));

        const cookiePolicyEnabled = !!ceProps('cookiePolicyEnabled');
        const privacyPolicyEnabled = !!ceProps('privacyPolicyEnabled');
        const cookieDomainRe = ceProps('cookieDomainRe', '');
        this.options = {
            googleAnalyticsAccount: ceProps('clientGoogleAnalyticsAccount', 'UA-55180-6'),
            googleAnalyticsEnabled: ceProps('clientGoogleAnalyticsEnabled', false),
            sharingEnabled: ceProps('clientSharingEnabled', true),
            githubEnabled: ceProps('clientGitHubRibbonEnabled', true),
            gapiKey: ceProps('googleApiKey', ''),
            googleShortLinkRewrite: ceProps('googleShortLinkRewrite', '').split('|'),
            urlShortenService: ceProps('urlShortenService', 'none'),
            defaultSource: ceProps('defaultSource', ''),
            compilers: [],
            libs: libs,
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
            release: defArgs.gitReleaseName,
            environment: defArgs.env,
            cookieDomainRe: cookieDomainRe,
            localStoragePrefix: ceProps('localStoragePrefix'),
            cvCompilerCountMax: ceProps('cvCompilerCountMax', 6),
            defaultFontScale: ceProps('defaultFontScale', 1.0),
            doCache: defArgs.doCache,
            policies: {
                cookies: {
                    enabled: cookiePolicyEnabled,
                    hash: cookiePolicyEnabled ? ClientOptionsHandler.getFileHash(
                        path.join(defArgs.staticDir, "policies", "cookies.html")) : null,
                    key: 'cookie_status'
                },
                privacy: {
                    enabled: privacyPolicyEnabled,
                    hash: privacyPolicyEnabled ? ClientOptionsHandler.getFileHash(
                        path.join(defArgs.staticDir, "policies", "privacy.html")) : null,
                    // How we store this privacy hash on the local storage
                    key: 'privacy_status'
                }
            },
            motdUrl: ceProps('motdUrl', '')
        };
    }

    parseTools(baseTools) {
        const tools = {};
        _.each(baseTools, (forLang, lang) => {
            if (lang && forLang) {
                tools[lang] = {};
                _.each(forLang.split(':'), tool => {
                    const toolBaseName = `tools.${tool}`;
                    const className = this.compilerProps(lang, toolBaseName + '.class');
                    const Tool = require("./tooling/" + className);
                    tools[lang][tool] = new Tool({
                        id: tool,
                        name: this.compilerProps(lang, toolBaseName + '.name'),
                        type: this.compilerProps(lang, toolBaseName + '.type'),
                        exe: this.compilerProps(lang, toolBaseName + '.exe'),
                        exclude: this.compilerProps(lang, toolBaseName + '.exclude'),
                        options: this.compilerProps(lang, toolBaseName + '.options')
                    },
                    {
                        ceProps: this.ceProps,
                        compilerProps: (propname) => this.compilerProps(lang, propname)
                    });
                });
            }
        });
        return tools;
    }

    parseLibraries(baseLibs) {
        const libraries = {};
        _.each(baseLibs, (forLang, lang) => {
            if (lang && forLang) {
                libraries[lang] = {};
                _.each(forLang.split(':'), lib => {
                    const libBaseName = `libs.${lib}`;
                    libraries[lang][lib] = {
                        name: this.compilerProps(lang, libBaseName + '.name'),
                        url: this.compilerProps(lang, libBaseName + '.url'),
                        description: this.compilerProps(lang, libBaseName + '.description')
                    };
                    libraries[lang][lib].versions = {};
                    const listedVersions = `${this.compilerProps(lang, libBaseName + '.versions')}`;
                    if (listedVersions) {
                        _.each(listedVersions.split(':'), version => {
                            const libVersionName = libBaseName + `.versions.${version}`;
                            libraries[lang][lib].versions[version] = {};
                            libraries[lang][lib].versions[version].version =
                                this.compilerProps(lang, libVersionName + '.version');
                            const includes = this.compilerProps(lang, libVersionName + '.path');
                            libraries[lang][lib].versions[version].path = [];
                            if (includes) {
                                libraries[lang][lib].versions[version].path = includes.split(':');
                            } else {
                                logger.warn(`Library ${lib} ${version} (${lang}) has no include paths`);
                            }

                        });
                    } else {
                        logger.warn(`No versions found for ${lib} library`);
                    }
                });
            }
        });
        return libraries;
    }

    _asSafeVer(semver) {
        return semverParser.valid(semver, true) || semverParser.valid(semver + '.0', true) || "9999999.99999.999";
    }

    setCompilers(compilers) {
        const blacklistedKeys = ['exe', 'versionFlag', 'versionRe', 'compilerType', 'demangler', 'objdumper',
            'postProcess', 'demanglerClassFile', 'isSemVer'];
        const copiedCompilers = JSON.parse(JSON.stringify(compilers));
        let semverGroups = {};
        _.each(copiedCompilers, (compiler, compilersKey) => {
            if (compiler.isSemVer) {
                if (!semverGroups[compiler.group]) semverGroups[compiler.group] = [];
                // Desired index which will keep the array in order
                const index = _.sortedIndex(semverGroups[compiler.group], compiler.semver, (lhg) => {
                    return semverParser.compare(this._asSafeVer(lhg.semver), this._asSafeVer(compiler.semver));
                });
                semverGroups[compiler.group].splice(index, 0, compiler);
            }
            _.each(compiler, (_, propKey) => {
                if (blacklistedKeys.includes(propKey)) {
                    delete copiedCompilers[compilersKey][propKey];
                }
            });
        });
        _.each(semverGroups, group => {
            let order = 0;
            // Set $order to -index on array. As group is an array, iteration order is guaranteed.
            _.each(group, compiler => compiler['$order'] = -order++);
        });

        this.options.compilers = copiedCompilers;
    }

    get() {
        return this.options;
    }

    static getFileHash(path) {
        if (!fs.existsSync(path)) {
            logger.error(`File ${path} requested for hashing not found`);
            // Should we throw? What should happen here?
        }

        return getHash(fs.readFileSync(path, 'utf-8'), HashVersion);
    }
}

module.exports = ClientOptionsHandler;
