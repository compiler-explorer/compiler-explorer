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

import path from 'path';

import * as Sentry from '@sentry/node';
import express from 'express';
import fs from 'fs-extra';
import Server from 'http-proxy';
import PromClient, {Counter} from 'prom-client';
import temp from 'temp';
import _ from 'underscore';
import which from 'which';

import {remove} from '../../shared/common-utils.js';
import {
    BypassCache,
    CompileChildLibraries,
    ExecutionParams,
    FiledataPair,
} from '../../types/compilation/compilation.interfaces.js';
import {CompilerOverrideOptions} from '../../types/compilation/compiler-overrides.interfaces.js';
import {CompilerInfo, ICompiler, PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {getCompilerTypeByKey} from '../compilers/index.js';
import {logger} from '../logger.js';
import {ClientOptionsType} from '../options-handler.js';
import {PropertyGetter} from '../properties.interfaces.js';
import {SentryCapture} from '../sentry.js';
import {KnownBuildMethod} from '../stats.js';
import * as utils from '../utils.js';

import {
    CompileRequestJsonBody,
    CompileRequestQueryArgs,
    CompileRequestTextBody,
    ICompileHandler,
} from './compile.interfaces.js';

temp.track();

let hasSetUpAutoClean = false;

function initialise(compilerEnv: CompilationEnvironment) {
    if (hasSetUpAutoClean) return;
    hasSetUpAutoClean = true;
    const tempDirCleanupSecs = compilerEnv.ceProps('tempDirCleanupSecs', 600);
    logger.info(`Cleaning temp dirs every ${tempDirCleanupSecs} secs`);

    let cyclesBusy = 0;
    setInterval(() => {
        const status = compilerEnv.compilationQueue.status();
        if (status.busy) {
            cyclesBusy++;
            logger.warn(
                `temp cleanup skipped, pending: ${status.pending}, waiting: ${status.size}, cycles: ${cyclesBusy}`,
            );
            return;
        }

        cyclesBusy = 0;

        temp.cleanup((err, stats) => {
            if (err) logger.error('temp cleanup error', err);
            if (stats) logger.debug('temp cleanup stats', stats);
        });
    }, tempDirCleanupSecs * 1000);
}

export type ParsedRequest = {
    source: string;
    options: string[];
    backendOptions: Record<string, any>;
    filters: ParseFiltersAndOutputOptions;
    bypassCache: BypassCache;
    tools: any;
    executeParameters: ExecutionParams;
    libraries: CompileChildLibraries[];
};

export class CompileHandler implements ICompileHandler {
    private compilersById: Record<string, Record<string, BaseCompiler>> = {};
    private readonly compilerEnv: CompilationEnvironment;
    private readonly textBanner: string;
    private readonly proxy: Server;
    private readonly awsProps: PropertyGetter;
    private clientOptions: ClientOptionsType | null = null;
    private readonly compileCounter: Counter<string> = new PromClient.Counter({
        name: 'ce_compilations_total',
        help: 'Number of compilations',
        labelNames: ['language'],
    });
    private readonly executeCounter: Counter<string> = new PromClient.Counter({
        name: 'ce_executions_total',
        help: 'Number of executions',
        labelNames: ['language'],
    });
    private readonly cmakeCounter: Counter<string> = new PromClient.Counter({
        name: 'ce_cmake_compilations_total',
        help: 'Number of CMake compilations',
        labelNames: ['language'],
    });
    private readonly cmakeExecuteCounter: Counter<string> = new PromClient.Counter({
        name: 'ce_cmake_executions_total',
        help: 'Number of executions after CMake',
        labelNames: ['language'],
    });

    constructor(compilationEnvironment: CompilationEnvironment, awsProps: PropertyGetter) {
        this.compilerEnv = compilationEnvironment;
        this.textBanner = this.compilerEnv.ceProps<string>('textBanner');
        this.proxy = Server.createProxyServer({});
        this.awsProps = awsProps;
        initialise(this.compilerEnv);

        // Mostly cribbed from
        // https://github.com/nodejitsu/node-http-proxy/blob/master/examples/middleware/bodyDecoder-middleware.js
        // We just keep the body as-is though: no encoding using queryString.stringify(), as we don't use a form
        // decoding middleware.
        this.proxy.on('proxyReq', (proxyReq, req) => {
            // TODO ideally I'd work out if this is "ok" - IncomingMessage doesn't have a body, but pragmatically the
            //  object we get here does (introduced by body-parser).
            const body = (req as any).body;
            if (!body || Object.keys(body).length === 0) {
                return;
            }
            let contentType: string = proxyReq.getHeader('Content-Type') as string;
            let bodyData;

            if (contentType === 'application/json') {
                bodyData = JSON.stringify(body);
            } else if (contentType === 'application/x-www-form-urlencoded') {
                // Reshape the form body into what a json request looks like
                contentType = 'application/json';
                bodyData = JSON.stringify({
                    lang: body.lang,
                    compiler: body.compiler,
                    source: body.source,
                    options: {
                        userArguments: body.userArguments,
                        filters: Object.fromEntries(
                            [
                                'commentOnly',
                                'directives',
                                'libraryCode',
                                'labels',
                                'demangle',
                                'intel',
                                'execute',
                                'debugCalls',
                                'binary',
                                'binaryObject',
                                'trim',
                            ].map(key => [key, body[key] === 'true']),
                        ),
                    },
                });
            } else {
                SentryCapture(
                    new Error(`Unexpected Content-Type received by /compiler/:compiler/compile: ${contentType}`),
                    'lib/handlers/compile.ts proxyReq contentType',
                );
                proxyReq.write('Unexpected Content-Type');
            }

            try {
                if (bodyData) {
                    proxyReq.setHeader('Content-Length', Buffer.byteLength(bodyData));
                    proxyReq.setHeader('Content-Type', contentType);
                    proxyReq.write(bodyData);
                }
            } catch (e: any) {
                SentryCapture(e, 'lib/handlers/compile.ts proxyReq.write');
                let json = '<json stringify error>';
                try {
                    json = JSON.stringify(bodyData);
                } catch (e) {}
                Sentry.captureMessage(`Unknown proxy bodyData: ${typeof bodyData} ${json}`);
                proxyReq.write('Proxy error');
            }
        });
    }

    hasLanguages(): boolean {
        try {
            return Object.keys(this.compilersById).length > 0;
        } catch {
            return false;
        }
    }

    async create(compiler: PreliminaryCompilerInfo): Promise<ICompiler | null> {
        const isPrediscovered = !!compiler.version;

        const type = compiler.compilerType || 'default';
        let compilerClass: ReturnType<typeof getCompilerTypeByKey>;
        try {
            compilerClass = getCompilerTypeByKey(type);
        } catch (e: any) {
            logger.error(`Compiler ID: ${compiler.id}`);
            logger.error(e);
            logger.error(e.stack);
            process.exit(1);
        }

        // attempt to resolve non absolute exe paths
        if (compiler.exe && !path.isAbsolute(compiler.exe)) {
            const exe = await which(compiler.exe).catch(() => null);
            if (exe) {
                logger.debug(`Resolved '${compiler.exe}' to path '${exe}'`);
                compiler.exe = exe;
            } else {
                // errors resolving to absolute path are not fatal for backwards compatibility sake
                logger.error(`Unable to resolve '${compiler.exe}'`);
            }
        }

        if (compiler.exe && path.isAbsolute(compiler.exe)) {
            // Try stat'ing the compiler to cache its mtime and only re-run it if it
            // has changed since the last time.
            try {
                let modificationTime;
                if (isPrediscovered) {
                    modificationTime = compiler.mtime;
                } else {
                    const res = await fs.stat(compiler.exe);
                    modificationTime = res.mtime;
                    const cached = this.findCompiler(compiler.lang, compiler.id);
                    if (cached && cached.getModificationTime() === res.mtime.getTime()) {
                        logger.debug(`${compiler.id} is unchanged`);
                        return cached;
                    }
                }
                const compilerObj = new compilerClass(compiler, this.compilerEnv);
                return compilerObj.initialise(modificationTime, this.clientOptions, isPrediscovered);
            } catch (err) {
                logger.warn(`Unable to stat ${compiler.id} compiler binary: `, err);
                return null;
            }
        } else {
            return new compilerClass(compiler, this.compilerEnv);
        }
    }

    async setCompilers(
        compilers: PreliminaryCompilerInfo[],
        clientOptions: ClientOptionsType,
    ): Promise<CompilerInfo[]> {
        // Be careful not to update this.compilersById until we can replace it entirely.
        const compilersById = {};
        try {
            this.clientOptions = clientOptions;
            logger.info('Creating compilers: ' + compilers.length);
            let compilersCreated = 0;
            const createdCompilers = remove(await Promise.all(compilers.map(c => this.create(c))), null);
            for (const compiler of createdCompilers) {
                const langId = compiler.getInfo().lang;
                if (!compilersById[langId]) compilersById[langId] = {};
                compilersById[langId][compiler.getInfo().id] = compiler;
                compilersCreated++;
            }
            logger.info('Compilers created: ' + compilersCreated);
            if (this.awsProps) {
                logger.info('Fetching possible arguments from storage');
                await Promise.all(
                    createdCompilers.map(compiler => compiler.possibleArguments.loadFromStorage(this.awsProps)),
                );
            }
            this.compilersById = compilersById;
            return createdCompilers.map(compiler => compiler.getInfo());
        } catch (err) {
            logger.error('Exception while processing compilers:', err);
            return [];
        }
    }

    setPossibleToolchains(toolchains: CompilerOverrideOptions) {
        this.compilerEnv.setPossibleToolchains(toolchains);
    }

    compilerAliasMatch(compiler, compilerId): boolean {
        return compiler.compiler.alias && compiler.compiler.alias.includes(compilerId);
    }

    compilerIdOrAliasMatch(compiler, compilerId): boolean {
        return compiler.compiler.id === compilerId || this.compilerAliasMatch(compiler, compilerId);
    }

    findCompiler(langId, compilerId): BaseCompiler | undefined {
        if (!compilerId) return;

        const langCompilers: Record<string, BaseCompiler> | undefined = this.compilersById[langId];
        if (langCompilers) {
            if (langCompilers[compilerId]) {
                return langCompilers[compilerId];
            } else {
                const compiler = _.find(langCompilers, compiler => {
                    return this.compilerAliasMatch(compiler, compilerId);
                });

                if (compiler) return compiler;
            }
        }

        // If the lang is bad, try to find it in every language
        let response: BaseCompiler | undefined;
        _.each(this.compilersById, compilerInLang => {
            if (!response) {
                response = _.find(compilerInLang, compiler => {
                    return this.compilerIdOrAliasMatch(compiler, compilerId);
                });
            }
        });

        return response;
    }

    compilerFor(req): BaseCompiler | undefined {
        if (req.is('json')) {
            const lang = req.lang || req.body.lang;
            const compiler = this.findCompiler(lang, req.params.compiler);
            if (!compiler) {
                const withoutBody = _.extend({}, req.body, {source: '<removed>'});
                logger.warn(`Unable to find compiler with lang ${lang} for JSON request`, withoutBody);
            }
            return compiler;
        } else if (req.body && req.body.compiler) {
            const compiler = this.findCompiler(req.body.lang, req.body.compiler);
            if (!compiler) {
                const withoutBody = _.extend({}, req.body, {source: '<removed>'});
                logger.warn(`Unable to find compiler with lang ${req.body.lang} for request`, withoutBody);
            }
            return compiler;
        } else {
            const compiler = this.findCompiler(req.lang, req.params.compiler);
            if (!compiler) {
                logger.warn(`Unable to find compiler with lang ${req.lang} for request params`, req.params);
            }
            return compiler;
        }
    }

    checkRequestRequirements(req: express.Request): CompileRequestJsonBody {
        if (req.body.options === undefined) throw new Error('Missing options property');
        if (req.body.source === undefined) throw new Error('Missing source property');
        return req.body;
    }

    parseRequest(req: express.Request, compiler: BaseCompiler): ParsedRequest {
        let source: string,
            options: string,
            backendOptions: Record<string, any> = {},
            filters: ParseFiltersAndOutputOptions,
            bypassCache = BypassCache.None,
            tools;
        const execReqParams: ExecutionParams = {};
        let libraries: any[] = [];
        // IF YOU MODIFY ANYTHING HERE PLEASE UPDATE THE DOCUMENTATION!
        if (req.is('json')) {
            // JSON-style request
            const jsonRequest = this.checkRequestRequirements(req);
            const requestOptions = jsonRequest.options;
            source = jsonRequest.source;
            if (jsonRequest.bypassCache) bypassCache = jsonRequest.bypassCache;
            options = requestOptions.userArguments;
            const execParams = requestOptions.executeParameters || {};
            execReqParams.args = execParams.args;
            execReqParams.stdin = execParams.stdin;
            execReqParams.runtimeTools = execParams.runtimeTools;
            backendOptions = requestOptions.compilerOptions || {};
            filters = {...compiler.getDefaultFilters(), ...requestOptions.filters};
            tools = requestOptions.tools;
            libraries = requestOptions.libraries || [];
        } else if (req.body && req.body.compiler) {
            const textRequest = req.body as CompileRequestTextBody;
            source = textRequest.source;
            if (textRequest.bypassCache) bypassCache = textRequest.bypassCache;
            options = textRequest.userArguments;
            execReqParams.args = textRequest.executeParametersArgs;
            execReqParams.stdin = textRequest.executeParametersStdin;

            filters = compiler.getDefaultFilters();
            _.each(filters, (value, item) => {
                filters[item] = textRequest[item] === 'true';
            });

            backendOptions.filterAnsi = textRequest.filterAnsi === 'true';
            backendOptions.skipAsm = textRequest.skipAsm === 'true';
        } else {
            // API-style
            source = req.body;
            const query = req.query as CompileRequestQueryArgs;
            options = query.options || '';
            // By default we get the default filters.
            filters = compiler.getDefaultFilters();
            // If specified exactly, we'll take that with ?filters=a,b,c
            if (query.filters) {
                filters = _.object(
                    query.filters.split(',').map(filter => [filter, true]),
                ) as any as ParseFiltersAndOutputOptions;
            }
            // Add a filter. ?addFilters=binary
            _.each((query.addFilters || '').split(','), filter => {
                if (filter) filters[filter] = true;
            });
            // Remove a filter. ?removeFilter=intel
            _.each((query.removeFilters || '').split(','), filter => {
                if (filter) delete filters[filter];
            });
            // Ask for asm not to be returned
            backendOptions.skipAsm = query.skipAsm === 'true';
            backendOptions.skipPopArgs = query.skipPopArgs === 'true';
        }
        const executeParameters: ExecutionParams = {
            args: Array.isArray(execReqParams.args)
                ? execReqParams.args || ''
                : utils.splitArguments(execReqParams.args),
            stdin: execReqParams.stdin || '',
            runtimeTools: execReqParams.runtimeTools || [],
        };

        tools = tools || [];
        for (const tool of tools) {
            tool.args = utils.splitArguments(tool.args);
        }

        // Backwards compatibility: bypassCache used to be a boolean.
        // Convert a boolean input to an enum's underlying numeric value
        bypassCache = 1 * bypassCache;

        return {
            source,
            options: utils.splitArguments(options),
            backendOptions,
            filters,
            bypassCache,
            tools,
            executeParameters,
            libraries,
        };
    }

    handlePopularArguments(req: express.Request, res) {
        const compiler = this.compilerFor(req);
        if (!compiler) {
            return res.sendStatus(404);
        }
        res.send(compiler.possibleArguments.getPopularArguments(this.getUsedOptions(req)));
    }

    handleOptimizationArguments(req: express.Request, res) {
        const compiler = this.compilerFor(req);
        if (!compiler) {
            return res.sendStatus(404);
        }
        res.send(compiler.possibleArguments.getOptimizationArguments(this.getUsedOptions(req)));
    }

    getUsedOptions(req: express.Request) {
        if (req.body) {
            const data = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;

            if (data.presplit) {
                return data.usedOptions;
            } else {
                return utils.splitArguments(data.usedOptions);
            }
        }
        return false;
    }

    handleApiError(error, res: express.Response, next: express.NextFunction) {
        if (error.message) {
            return res.status(400).send({
                error: true,
                message: error.message,
            });
        } else {
            return next(error);
        }
    }

    handleCmake(req: express.Request, res: express.Response, next: express.NextFunction) {
        const compiler = this.compilerFor(req);
        if (!compiler) {
            return res.sendStatus(404);
        }

        const remote = compiler.getRemote();
        if (remote) {
            req.url = remote.cmakePath;
            this.proxy.web(req, res, {target: remote.target, changeOrigin: true}, e => {
                logger.error('Proxy error: ', e);
                next(e);
            });
            return;
        }

        try {
            if (req.body.files === undefined) throw new Error('Missing files property');

            this.cmakeCounter.inc({language: compiler.lang.id});
            const options = this.parseRequest(req, compiler);
            this.compilerEnv.statsNoter.noteCompilation(
                compiler.getInfo().id,
                options,
                req.body.files as FiledataPair[],
                KnownBuildMethod.CMake,
            );
            compiler
                // Backwards compatibility: bypassCache used to be a boolean.
                // Convert a boolean input to an enum's underlying numeric value
                .cmake(req.body.files, options, req.body.bypassCache * 1)
                .then(result => {
                    if (result.didExecute || (result.execResult && result.execResult.didExecute))
                        this.cmakeExecuteCounter.inc({language: compiler.lang.id});
                    res.send(result);
                })
                .catch(e => {
                    return this.handleApiError(e, res, next);
                });
        } catch (e) {
            return this.handleApiError(e, res, next);
        }
    }

    handle(req: express.Request, res: express.Response, next: express.NextFunction) {
        const compiler = this.compilerFor(req);
        if (!compiler) {
            return res.sendStatus(404);
        }

        const remote = compiler.getRemote();
        if (remote) {
            req.url = remote.path;
            this.proxy.web(req, res, {target: remote.target, changeOrigin: true}, e => {
                logger.error('Proxy error: ', e);
                next(e);
            });
            return;
        }

        let parsedRequest: ParsedRequest | undefined;
        try {
            parsedRequest = this.parseRequest(req, compiler);
        } catch (error) {
            return this.handleApiError(error, res, next);
        }

        const {source, options, backendOptions, filters, bypassCache, tools, executeParameters, libraries} =
            parsedRequest;

        let files;
        if (req.body.files) files = req.body.files;

        if (source === undefined || Object.keys(req.body).length === 0) {
            logger.warn('No body found in request', req);
            return next(new Error('Bad request'));
        }

        function textify(array: ResultLine[] | null, filterAnsi: boolean | undefined) {
            const text = (array || []).map(line => line.text).join('\n');
            if (filterAnsi) {
                // https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
                return text.replaceAll(/(\x9B|\x1B\[)[\d:;<=>?]*[ -/]*[@-~]/g, '');
            } else {
                return text;
            }
        }

        this.compileCounter.inc({language: compiler.lang.id});
        this.compilerEnv.statsNoter.noteCompilation(
            compiler.getInfo().id,
            parsedRequest,
            files as FiledataPair[],
            KnownBuildMethod.Compile,
        );
        // eslint-disable-next-line promise/catch-or-return
        compiler
            .compile(source, options, backendOptions, filters, bypassCache, tools, executeParameters, libraries, files)
            .then(
                result => {
                    if (result.didExecute || (result.execResult && result.execResult.didExecute))
                        this.executeCounter.inc({language: compiler.lang.id});
                    if (req.accepts(['text', 'json']) === 'json') {
                        res.send(result);
                    } else {
                        res.set('Content-Type', 'text/plain');
                        try {
                            if (!_.isEmpty(this.textBanner)) res.write('# ' + this.textBanner + '\n');
                            res.write(textify(result.asm, backendOptions.filterAnsi));
                            if (result.code !== 0) res.write('\n# Compiler exited with result code ' + result.code);
                            if (!_.isEmpty(result.stdout))
                                res.write('\nStandard out:\n' + textify(result.stdout, backendOptions.filterAnsi));
                            if (!_.isEmpty(result.stderr))
                                res.write('\nStandard error:\n' + textify(result.stderr, backendOptions.filterAnsi));

                            if (result.execResult) {
                                res.write('\n\n# Execution result with exit code ' + result.execResult.code + '\n');
                                if (!_.isEmpty(result.execResult.stdout)) {
                                    res.write(
                                        '# Standard out:\n' +
                                            textify(result.execResult.stdout, backendOptions.filterAnsi),
                                    );
                                }
                                if (!_.isEmpty(result.execResult.stderr)) {
                                    res.write(
                                        '\n# Standard error:\n' +
                                            textify(result.execResult.stderr, backendOptions.filterAnsi),
                                    );
                                }
                            }
                        } catch (ex) {
                            SentryCapture(ex, 'lib/handlers/compile.ts res.write');
                            res.write(`Error handling request: ${ex}`);
                        }
                        res.end('\n');
                    }
                },
                error => {
                    if (typeof error === 'string') {
                        logger.error('Error during compilation 1: ', {error});
                    } else {
                        if (error.stack) {
                            logger.error('Error during compilation 2: ', error);
                            SentryCapture(error, 'compile failed');
                        } else if (error.code) {
                            logger.error('Error during compilation 3: ', error.code);
                            if (typeof error.stderr === 'string') {
                                error.stdout = utils.parseOutput(error.stdout);
                                error.stderr = utils.parseOutput(error.stderr);
                            }
                            res.end(JSON.stringify(error));
                            return;
                        } else {
                            logger.error('Error during compilation 4: ', error);
                        }

                        error = `Internal Compiler Explorer error: ${error.stack || error}`;
                    }
                    res.end(JSON.stringify({code: -1, stdout: [], stderr: [{text: error}]}));
                },
            );
    }
}

export function SetTestMode() {
    hasSetUpAutoClean = true;
}
