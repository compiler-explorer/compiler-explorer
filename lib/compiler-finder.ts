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

import http from 'http';
import https from 'https';
import path from 'path';
import {promisify} from 'util';

import fs from 'fs-extra';
import _ from 'underscore';
import urljoin from 'url-join';

import {AppDefaultArguments} from '../app.js';
import {basic_comparator, remove} from '../shared/common-utils.js';
import type {CompilerInfo, PreliminaryCompilerInfo} from '../types/compiler.interfaces.js';
import {InstructionSet, InstructionSetsList} from '../types/instructionsets.js';
import type {Language, LanguageKey} from '../types/languages.interfaces.js';

import {assert, unwrap, unwrapString} from './assert.js';
import {InstanceFetcher} from './aws.js';
import {CompileHandler} from './handlers/compile.js';
import {logger} from './logger.js';
import {ClientOptionsHandler} from './options-handler.js';
import type {PropertyGetter} from './properties.interfaces.js';
import {CompilerProps, getRawProperties} from './properties.js';
import {getPossibleGccToolchainsFromCompilerInfo} from './toolchain-utils.js';

const sleep = promisify(setTimeout);

/***
 * Finds and initializes the compilers stored on the properties files
 */
export class CompilerFinder {
    compilerProps: CompilerProps['get'];
    ceProps: PropertyGetter;
    awsProps: PropertyGetter;
    args: AppDefaultArguments;
    compileHandler: CompileHandler;
    languages: Record<string, Language>;
    awsPoller: InstanceFetcher | null = null;
    optionsHandler: ClientOptionsHandler;
    //visitedCompilers = new Set<string>();

    constructor(
        compileHandler: CompileHandler,
        compilerProps: CompilerProps,
        awsProps: PropertyGetter,
        args: AppDefaultArguments,
        optionsHandler: ClientOptionsHandler,
    ) {
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
        if (!this.awsPoller) this.awsPoller = new InstanceFetcher(this.awsProps);
        return this.awsPoller.getInstances();
    }

    async fetchRemote(
        host: string,
        port: number,
        uriBase: string,
        props: PropertyGetter,
        langId: string | null,
    ): Promise<CompilerInfo[] | null> {
        const requestLib = port === 443 ? https : http;
        const uriSchema = port === 443 ? 'https' : 'http';
        const uri = urljoin(`${uriSchema}://${host}:${port}`, uriBase);
        const apiPath = urljoin('/', uriBase || '', 'api/compilers', langId || '', '?fields=all');
        logger.info(`Fetching compilers from remote source ${uri}`);
        return this.retryPromise(
            () => {
                return new Promise<CompilerInfo[]>((resolve, reject) => {
                    const request = requestLib
                        .get(
                            {
                                hostname: host,
                                port: port,
                                path: apiPath,
                                headers: {
                                    Accept: 'application/json',
                                },
                            },
                            res => {
                                let error;
                                const {
                                    statusCode,
                                    headers: {'content-type': contentType},
                                } = res;

                                if (statusCode !== 200) {
                                    error = new Error(
                                        'Failed fetching remote compilers from ' +
                                            `${uriSchema}://${host}:${port}${apiPath}\n` +
                                            `Status Code: ${statusCode}`,
                                    );
                                } else if (!contentType || !/^application\/json/.test(contentType)) {
                                    error = new Error(
                                        'Invalid content-type.\n' +
                                            `Expected application/json but received ${contentType}`,
                                    );
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
                                        const compilers = (JSON.parse(str) as CompilerInfo[]).map(compiler => {
                                            // Fix up old upstream implementations of Compiler Explorer
                                            // e.g. https://www.godbolt.ms
                                            // (see https://github.com/compiler-explorer/compiler-explorer/issues/1768)
                                            if (!compiler.alias) compiler.alias = [];
                                            if (typeof compiler.alias == 'string') compiler.alias = [compiler.alias];
                                            // End fixup
                                            compiler.exe = '/dev/null';
                                            compiler.remote = {
                                                target: `${uriSchema}://${host}:${port}`,
                                                path: urljoin('/', uriBase, 'api/compiler', compiler.id, 'compile'),
                                                cmakePath: urljoin('/', uriBase, 'api/compiler', compiler.id, 'cmake'),
                                            };
                                            return compiler;
                                        });
                                        resolve(compilers);
                                    } catch (e: any) {
                                        logger.error(`Error parsing response from ${uri} '${str}': ${e.message}`);
                                        reject(e);
                                    }
                                });
                            },
                        )
                        .on('error', reject)
                        .on('timeout', () => reject('timeout'));
                    request.setTimeout(this.awsProps('proxyTimeout', 1000));
                });
            },
            `${host}:${port}`,
            props('proxyRetries', 5),
            props('proxyRetryMs', 500),
        ).catch(() => {
            logger.warn(`Unable to contact ${host}:${port}; skipping`);
            return [];
        });
    }

    async fetchAws() {
        logger.info('Fetching instances from AWS');
        const instances = await this.awsInstances();
        const mapped = await Promise.all(
            instances.map(instance => {
                logger.info('Checking instance ' + instance.InstanceId);
                const address = this.awsProps('externalTestMode', false)
                    ? instance.PublicDnsName
                    : instance.PrivateDnsName;
                return this.fetchRemote(unwrap(address), this.args.port, '', this.awsProps, null);
            }),
        );

        return remove(mapped.flat(), null);
    }

    async compilerConfigFor(
        langId: string,
        compilerId: string,
        parentProps: CompilerProps['get'],
    ): Promise<PreliminaryCompilerInfo | null> {
        const base = `compiler.${compilerId}.`;

        const props: PropertyGetter = (propName: string, defaultValue?: any) => {
            const propsForCompiler = parentProps(langId, base + propName);
            if (propsForCompiler !== undefined) return propsForCompiler;
            return parentProps(langId, propName, defaultValue);
        };
        const splitArrayProps = (propName: string, split: string) => {
            return props<string | undefined>(propName)?.split(split);
        };
        const splitArrayPropsOrEmpty = (propName: string, split: string) => {
            const value = props<string>(propName, '');
            return value === '' ? [] : value.split(split);
        };

        const ceToolsPath = props('ceToolsPath', './');

        const supportsBinary = !!props('supportsBinary', true);
        const supportsBinaryObject = !!props('supportsBinaryObject', false);
        const interpreted = !!props('interpreted', false);
        const supportsExecute = (interpreted || supportsBinary) && !!props('supportsExecute', true);
        const executionWrapper = props('executionWrapper', '');
        const executionWrapperArgs = splitArrayPropsOrEmpty('executionWrapperArgs', '|');
        const supportsLibraryCodeFilter = !!props('supportsLibraryCodeFilter', true);

        const group = props('group', '');

        const demanglerProp = props('demangler', '');
        const demangler = demanglerProp ? path.normalize(demanglerProp.replace('${ceToolsPath}', ceToolsPath)) : '';

        const isSemVer = props('isSemVer', false);
        const baseName = props<string | undefined>('baseName');
        const semverVer = props('semver', '');

        const name = props<string>('name');

        // If name set, display that as the name
        // If not, check if we have a baseName + semver and display that
        // Else display compilerId as its name
        const displayName =
            name === undefined ? (isSemVer && baseName ? `${baseName} ${semverVer}` : compilerId) : name;

        const baseOptions = props('baseOptions', '');
        const options = props('options', '');
        const actualOptions = [baseOptions, options].filter(x => x.length > 0).join(' ');

        const envVars = (() => {
            const envVarsString = props('envVars', '');
            if (envVarsString === '') {
                return [];
            }
            const arr: [string, string][] = [];
            for (const el of envVarsString.split(':')) {
                const [env, setting] = el.split('=');
                arr.push([env, setting]);
            }
            return arr;
        })();
        const exe = props('exe', compilerId);
        const exePath = path.dirname(exe);
        const instructionSet = props<string | number>('instructionSet', '').toString() as InstructionSet | '';
        assert(
            instructionSet === '' || InstructionSetsList.includes(instructionSet),
            `Unexpected instruction set ${instructionSet} ${compilerId}`,
        );
        const compilerInfo: PreliminaryCompilerInfo = {
            id: compilerId,
            exe: exe,
            name: displayName,
            alias: props('alias', '')
                .split(':')
                .filter(a => a !== ''),
            options: actualOptions,
            versionFlag: splitArrayProps('versionFlag', '|'),
            versionRe: props<string>('versionRe'),
            explicitVersion: props<string>('explicitVersion'),
            compilerType: props('compilerType', ''),
            compilerCategories: splitArrayPropsOrEmpty('compilerCategories', ':'),
            debugPatched: props('debugPatched', false),
            demangler: demangler,
            demanglerType: props('demanglerType', ''),
            demanglerArgs: splitArrayPropsOrEmpty('demanglerArgs', '|'),
            nvdisasm: props('nvdisasm', ''),
            objdumper: props('objdumper', ''),
            objdumperType: props('objdumperType', ''),
            objdumperArgs: splitArrayPropsOrEmpty('objdumperArgs', '|'),
            intelAsm: props('intelAsm', ''),
            supportsAsmDocs: props('supportsAsmDocs', true),
            instructionSet: instructionSet === '' ? null : instructionSet,
            needsMulti: !!props('needsMulti', true),
            adarts: props('adarts', ''),
            supportsDemangle: !!demangler,
            supportsBinary,
            supportsBinaryObject,
            interpreted,
            supportsExecute,
            executionWrapper,
            executionWrapperArgs,
            supportsLibraryCodeFilter: supportsLibraryCodeFilter,
            postProcess: splitArrayPropsOrEmpty('postProcess', '|'),
            lang: langId as LanguageKey,
            group: group,
            groupName: props('groupName', ''),
            includeFlag: props('includeFlag', '-isystem'),
            includePath: props('includePath', ''),
            linkFlag: props('linkFlag', '-l'),
            rpathFlag: props('rpathFlag', '-Wl,-rpath,'),
            libpathFlag: props('libpathFlag', '-L'),
            libPath: props('libPath', '')
                .split(path.delimiter)
                .filter(p => p !== '')
                .map(x => path.normalize(x.replace('${exePath}', exePath))),
            ldPath: props('ldPath', '')
                .split('|')
                .map(x => path.normalize(x.replace('${exePath}', exePath))),
            extraPath: props('extraPath', '')
                .split(path.delimiter)
                .filter(p => p !== '')
                .map(x => path.normalize(x.replace('${exePath}', exePath))),
            envVars: envVars,
            notification: props('notification', ''),
            isSemVer: isSemVer,
            semver: semverVer,
            isNightly: props('isNightly', false),
            libsArr: this.getSupportedLibrariesArr(props),
            tools: _.omit(this.optionsHandler.get().tools[langId], tool => tool.isCompilerExcluded(compilerId, props)),
            unwiseOptions: splitArrayPropsOrEmpty('unwiseOptions', '|'),
            hidden: props('hidden', false),
            buildenvsetup: {
                id: props('buildenvsetup', ''),
                props: (name, def) => {
                    return props(`buildenvsetup.${name}`, def);
                },
            },
            externalparser: {
                id: props('externalparser', ''),
                props: (name, def) => {
                    return props(`externalparser.${name}`, def);
                },
            },
            license: {
                link: props<string>('licenseLink'),
                name: props<string>('licenseName'),
                preamble: props<string>('licensePreamble'),
                invasive: props<boolean>('licenseInvasive', false),
            },
            possibleOverrides: [],
            possibleRuntimeTools: [],
            $order: undefined as unknown as number, // TODO(jeremy-rifkin): Very dirty
        };

        if (props('demanglerClassFile') !== undefined) {
            logger.error(
                `Error in compiler.${compilerId}: ` +
                    'demanglerClassFile is no longer supported, please use demanglerType',
            );
            return null;
        }

        logger.debug('Found compiler', compilerInfo);
        return compilerInfo;
    }

    getSupportedLibrariesArr(props: PropertyGetter) {
        return props('supportsLibraries', '')
            .split(':')
            .filter(a => a !== '');
    }

    async recurseGetCompilers(
        langId: string,
        compilerName: string,
        parentProps: CompilerProps['get'],
    ): Promise<PreliminaryCompilerInfo[]> {
        // Don't treat @ in paths as remote addresses if requested
        if (this.args.fetchCompilersFromRemote && compilerName.includes('@')) {
            const bits = compilerName.split('@');
            const host = bits[0];
            const pathParts = bits[1].split('/');
            const port = parseInt(unwrap(pathParts.shift()));
            const path = pathParts.join('/');
            return (await this.fetchRemote(host, port, path, this.ceProps, langId)) || [];
        }
        if (compilerName.indexOf('&') === 0) {
            const groupName = compilerName.substring(1);

            const props: CompilerProps['get'] = (langId, name, def?): any => {
                if (name === 'group') {
                    return groupName;
                }
                return this.compilerProps(langId, `group.${groupName}.${name}`, parentProps(langId, name, def));
            };
            const exes = this.compilerProps(langId, `group.${groupName}.compilers`, '')
                .split(':')
                .filter(s => s !== '');
            logger.debug(`Processing compilers from group ${groupName}`);
            const allCompilers = await Promise.all(
                exes.map(compiler => this.recurseGetCompilers(langId, compiler, props)),
            );
            return allCompilers.flat();
        }
        if (compilerName === 'AWS') return this.fetchAws();
        const configs = [await this.compilerConfigFor(langId, compilerName, parentProps)];
        return remove(configs, null);
    }

    async getCompilers() {
        const compilers: Promise<PreliminaryCompilerInfo[]>[] = [];
        for (const [langId, exs] of Object.entries(this.getExes())) {
            for (const exe of exs) {
                compilers.push(this.recurseGetCompilers(langId, exe, this.compilerProps));
            }
        }
        const completeCompilers = await Promise.all(compilers);
        return completeCompilers.flat();
    }

    ensureDistinct(compilers: CompilerInfo[]) {
        const ids: Record<string, CompilerInfo[]> = {};
        let foundClash = false;
        for (const compiler of compilers) {
            if (!ids[compiler.id]) ids[compiler.id] = [];
            ids[compiler.id].push(compiler);
        }
        for (const [id, list] of Object.entries(ids)) {
            if (list.length !== 1) {
                foundClash = true;
                logger.error(
                    `Compiler ID clash for '${id}' - used by ${list
                        .map(o => `lang:${o.lang} name:${o.name}`)
                        .join(', ')}`,
                );
            }
        }
        return {compilers, foundClash};
    }

    async retryPromise<T>(promiseFunc: () => Promise<T>, name: string, maxFails: number, retryMs: number) {
        for (let fails = 0; fails < maxFails; ++fails) {
            try {
                return await promiseFunc();
            } catch (e) {
                if (fails < maxFails - 1) {
                    logger.warn(`Failed ${name} : ${e}, retrying`);
                    await sleep(retryMs);
                } else {
                    logger.error(`Too many retries for ${name} : ${e}`);
                    throw e;
                }
            }
        }
        return null;
    }

    getExes() {
        const langToCompilers = this.compilerProps(this.languages, 'compilers', '', exs =>
            exs.split(':').filter(s => s !== ''),
        );
        this.addNdkExes(langToCompilers);
        logger.info('Exes found:', langToCompilers);
        return langToCompilers;
    }

    addNdkExes(langToCompilers) {
        const ndkPaths = this.compilerProps(this.languages, 'androidNdk') as unknown as Record<string, string>;
        for (const [langId, ndkPath] of Object.entries(ndkPaths)) {
            if (ndkPath) {
                const toolchains = fs.readdirSync(`${ndkPath}/toolchains`);
                for (const version of toolchains) {
                    const path = `${ndkPath}/toolchains/${version}/prebuilt/linux-x86_64/bin`;
                    for (const exe of fs.readdirSync(path)) {
                        if (exe.endsWith('clang++') || exe.endsWith('g++')) {
                            langToCompilers[langId].push(`${path}/${exe}`);
                        }
                    }
                }
            }
        }
    }

    checkOrphanedProperties() {
        // Quickly check for any orphaned compilers
        let error = false;
        for (const domains of [
            ['amazon', 'amazonwin', 'gpu'],
            ['defaults', 'local'],
        ]) {
            const compilers = new Set<string>();
            // duplicate groups across languages is ok, so storing lang.group in the set
            const groups = new Set<string>();
            const reachableCompilers = new Set<string>();
            const reachableGroups = new Set<string>();
            const rawProps = getRawProperties();
            for (const [prop, props] of Object.entries(rawProps)) {
                const [lang, domain] = prop.split('.');
                if (domains.includes(domain)) {
                    if (domain === 'defaults' && `${lang}.local` in rawProps) {
                        continue; // let .local override
                    }
                    for (const [prop, value] of Object.entries(props)) {
                        const propParts = prop.split('.');
                        if (prop === 'compilers') {
                            for (const compiler of unwrapString(value).split(':')) {
                                if (compiler.startsWith('&')) {
                                    reachableGroups.add(`${lang}.${compiler.slice(1)}`);
                                } else {
                                    reachableCompilers.add(compiler);
                                }
                            }
                        }
                        if (propParts[0] === 'group') {
                            if (propParts[2] === 'compilers') {
                                // should appear exactly once
                                const fullGroup = `${lang}.${propParts[1]}`;
                                if (groups.has(fullGroup)) {
                                    const [lang, realGroup] = fullGroup.split('.');
                                    logger.error(
                                        `Duplicate group id ${realGroup} for ${lang} in domain ${domains.join(',')}`,
                                    );
                                    error = true;
                                }
                                groups.add(fullGroup);
                                for (const compiler of unwrapString(value).split(':')) {
                                    if (compiler.startsWith('&')) {
                                        reachableGroups.add(`${lang}.${compiler.slice(1)}`);
                                    } else {
                                        reachableCompilers.add(compiler);
                                    }
                                }
                            }
                        }
                        if (propParts[0] === 'compiler') {
                            if (propParts[2] === 'exe') {
                                // should appear exactly once
                                if (compilers.has(propParts[1])) {
                                    logger.error(
                                        `Duplicate compiler id ${propParts[1]} in domain ${domains.join(',')}`,
                                    );
                                    // android-java and android-kotlin are
                                    // expected to use the exact same compilers.
                                    if (lang !== 'android-java' && lang !== 'android-kotlin') {
                                        error = true;
                                    }
                                }
                                compilers.add(propParts[1]);
                            }
                        }
                    }
                }
            }
            for (const group of groups) {
                if (!reachableGroups.has(group)) {
                    const [lang, realGroup] = group.split('.');
                    logger.error(
                        `Group ${realGroup} is orphaned from the language compilers list for ` +
                            `${lang} in domain ${domains.join(',')}`,
                    );
                    error = true;
                }
            }
            for (const compiler of compilers) {
                if (!reachableCompilers.has(compiler)) {
                    logger.error(`Compiler ${compiler} is not part of any group in domain ${domains.join(',')}`);
                    error = true;
                }
            }
        }
        if (error) {
            assert(false);
        }
    }

    async find() {
        this.checkOrphanedProperties();

        const compilerList = await this.getCompilers();

        const toolchains = await getPossibleGccToolchainsFromCompilerInfo(compilerList);
        this.compileHandler.setPossibleToolchains(toolchains);

        const compilers = await this.compileHandler.setCompilers(compilerList, this.optionsHandler.get());
        if (!compilers) {
            logger.error('#### No compilers found: no compilation will be done!');
            throw new Error('No compilers found due to error or no configuration');
        }
        const result = this.ensureDistinct(compilers);
        return {
            foundClash: result.foundClash,
            compilers: result.compilers.sort((a, b) => basic_comparator(a.name, b.name)),
        };
    }

    async loadPrediscovered(compilers: CompilerInfo[]) {
        for (const compiler of compilers) {
            const langId = compiler.lang;

            if (compiler.buildenvsetup) {
                compiler.buildenvsetup.props = (propName, def) => {
                    return this.compilerProps(langId, 'buildenvsetup.' + propName, def);
                };
            }

            if (compiler.externalparser) {
                compiler.externalparser.props = (propName, def) => {
                    return this.compilerProps(langId, 'externalparser.' + propName, def);
                };
            }

            if (!compiler.remote && compiler.tools) {
                const fullOptions = this.optionsHandler.get();

                const toolinstances = {};
                for (const toolId in compiler.tools) {
                    if (fullOptions.tools[langId][toolId]) {
                        toolinstances[toolId] = fullOptions.tools[langId][toolId];
                    }
                }
                compiler.tools = toolinstances;
            }
        }
        return this.compileHandler.setCompilers(compilers, this.optionsHandler.get());
    }
}
