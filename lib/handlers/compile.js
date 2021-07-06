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
import fs from 'fs-extra';
import httpProxy from 'http-proxy';
import PromClient from 'prom-client';
import temp from 'temp';
import _ from 'underscore';
import which from 'which';

import { getCompilerTypeByKey } from '../compilers';
import { logger } from '../logger';
import * as utils from '../utils';

temp.track();

let hasSetUpAutoClean = false;

function initialise(compilerEnv) {
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
                `temp cleanup skipped, pending: ${status.pending}, waiting: ${status.size}, cycles: ${cyclesBusy}`);
            return;
        }

        cyclesBusy = 0;

        temp.cleanup((err, stats) => {
            if (err) logger.error('temp cleanup error', err);
            if (stats) logger.debug('temp cleanup stats', stats);
        });
    }, tempDirCleanupSecs * 1000);
}

export class CompileHandler {
    constructor(compilationEnvironment, awsProps) {
        this.compilersById = {};
        this.compilerEnv = compilationEnvironment;
        this.textBanner = this.compilerEnv.ceProps('textBanner');
        this.proxy = httpProxy.createProxyServer({});
        this.awsProps = awsProps;
        initialise(this.compilerEnv);

        this.compileCounter = new PromClient.Counter({
            name: 'ce_compilations_total',
            help: 'Number of compilations',
            labelNames: ['language'],
        });

        this.executeCounter = new PromClient.Counter({
            name: 'ce_executions_total',
            help: 'Number of executions',
            labelNames: ['language'],
        });

        // Mostly cribbed from
        // https://github.com/nodejitsu/node-http-proxy/blob/master/examples/middleware/bodyDecoder-middleware.js
        // We just keep the body as-is though: no encoding using queryString.stringify(), as we don't use a form
        // decoding middleware.
        this.proxy.on('proxyReq', function (proxyReq, req) {
            if (!req.body || Object.keys(req.body).length === 0) {
                return;
            }

            const contentType = proxyReq.getHeader('Content-Type');
            let bodyData;

            if (contentType === 'application/json') {
                bodyData = JSON.stringify(req.body);
            }

            if (contentType === 'application/x-www-form-urlencoded') {
                bodyData = req.body;
            }

            if (bodyData) {
                proxyReq.setHeader('Content-Length', Buffer.byteLength(bodyData));
                proxyReq.write(bodyData);
            }
        });
    }

    async create(compiler) {
        const type = compiler.compilerType || 'default';
        const compilerClass = getCompilerTypeByKey(type);

        // attempt to resolve non absolute exe paths
        if (compiler.exe && !path.isAbsolute(compiler.exe)) {
            const exe = await which(compiler.exe).catch(() => null);
            if (exe) {
                logger.debug(`Resolved '${compiler.exe}' to path '${exe}'`);
            } else {
                // errors resolving to absolute path are not fatal for backwards compatibility sake
                logger.error(`Unable to resolve '${compiler.exe}'`);
            }
        }

        if (compiler.exe && path.isAbsolute(compiler.exe)) {
            // Try stat'ing the compiler to cache its mtime and only re-run it if it
            // has changed since the last time.
            try {
                const res = await fs.stat(compiler.exe);
                const cached = this.findCompiler(compiler.lang, compiler.id);
                if (cached && cached.mtime.getTime() === res.mtime.getTime()) {
                    logger.debug(`${compiler.id} is unchanged`);
                    return cached;
                }
                const compilerObj = new compilerClass(compiler, this.compilerEnv);
                return compilerObj.initialise(res.mtime);
            } catch (err) {
                logger.warn(`Unable to stat ${compiler.id} compiler binary: `, err);
                return null;
            }
        } else {
            return new compilerClass(compiler, this.compilerEnv);
        }
    }

    async setCompilers(compilers) {
        // Be careful not to update this.compilersById until we can replace it entirely.
        const compilersById = {};
        try {
            const createdCompilers = _.compact(await Promise.all(_.map(compilers, this.create, this)));
            for (const compiler of createdCompilers) {
                const langId = compiler.compiler.lang;
                if (!compilersById[langId]) compilersById[langId] = {};
                compilersById[langId][compiler.compiler.id] = compiler;
            }
            if (this.awsProps) {
                logger.info('Fetching possible arguments from storage');
                await Promise.all(createdCompilers.map(
                    compiler => compiler.possibleArguments.loadFromStorage(this.awsProps)));
            }
            this.compilersById = compilersById;
            return createdCompilers.map(compiler => compiler.getInfo());
        } catch (err) {
            logger.error('Exception while processing compilers:', err);
        }
    }

    compilerAliasMatch(compiler, compilerId) {
        return compiler.compiler.alias &&
            compiler.compiler.alias.includes(compilerId);
    }

    compilerIdOrAliasMatch(compiler, compilerId) {
        return (compiler.compiler.id === compilerId) ||
            this.compilerAliasMatch(compiler, compilerId);
    }

    findCompiler(langId, compilerId) {
        if (!compilerId) return;

        const langCompilers = this.compilersById[langId];
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
        let response;
        _.each(this.compilersById, compilerInLang => {
            if (!response) {
                response = _.find(compilerInLang, compiler => {
                    return this.compilerIdOrAliasMatch(compiler, compilerId);
                });
            }
        });

        return response;
    }

    compilerFor(req) {
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

    checkRequestRequirements(req) {
        if (req.body.options === undefined) throw new Error('Missing options property');
        if (req.body.source === undefined) throw new Error('Missing source property');
    }

    parseRequest(req, compiler) {
        let source, options, backendOptions = {}, filters, bypassCache = false, tools, executionParameters = {};
        let libraries = [];
        // IF YOU MODIFY ANYTHING HERE PLEASE UPDATE THE DOCUMENTATION!
        if (req.is('json')) {
            // JSON-style request
            this.checkRequestRequirements(req);
            const requestOptions = req.body.options;
            source = req.body.source;
            if (req.body.bypassCache)
                bypassCache = true;
            options = requestOptions.userArguments;
            const execParams = requestOptions.executeParameters || {};
            executionParameters.args = execParams.args;
            executionParameters.stdin = execParams.stdin;
            backendOptions = requestOptions.compilerOptions || {};
            filters = requestOptions.filters || compiler.getDefaultFilters();
            tools = requestOptions.tools;
            libraries = requestOptions.libraries || [];
        } else if (req.body && req.body.compiler) {
            source = req.body.source;
            if (req.body.bypassCache)
                bypassCache = true;
            options = req.body.userArguments;
            executionParameters.args = req.body.executeParametersArgs;
            executionParameters.stdin = req.body.executeParametersStdin;

            filters = compiler.getDefaultFilters();
            _.each(filters, (value, item) => {
                filters[item] = (req.body[item] === 'true');
            });

            backendOptions.skipAsm = req.body.skipAsm === 'true';
        } else {
            // API-style
            source = req.body;
            options = req.query.options;
            // By default we get the default filters.
            filters = compiler.getDefaultFilters();
            // If specified exactly, we'll take that with ?filters=a,b,c
            if (req.query.filters) {
                filters = _.object(_.map(req.query.filters.split(','), filter => [filter, true]));
            }
            // Add a filter. ?addFilters=binary
            _.each((req.query.addFilters || '').split(','), filter => {
                if (filter) filters[filter] = true;
            });
            // Remove a filter. ?removeFilter=intel
            _.each((req.query.removeFilters || '').split(','), filter => {
                if (filter) delete filters[filter];
            });
            // Ask for asm not to be returned
            backendOptions.skipAsm = req.query.skipAsm === 'true';

            backendOptions.skipPopArgs = req.query.skipPopArgs === 'true';
        }
        options = utils.splitArguments(options);
        if (!Array.isArray(executionParameters.args)) {
            executionParameters.args = utils.splitArguments(executionParameters.args);
        }

        tools = tools || [];
        for (const tool of tools) {
            tool.args = utils.splitArguments(tool.args);
        }
        return {source, options, backendOptions, filters, bypassCache, tools, executionParameters, libraries};
    }

    handlePopularArguments(req, res, next) {
        const compiler = this.compilerFor(req);
        if (!compiler) return next();
        res.send(compiler.possibleArguments.getPopularArguments(this.getUsedOptions(req)));
    }

    handleOptimizationArguments(req, res, next) {
        const compiler = this.compilerFor(req);
        if (!compiler) return next();
        res.send(compiler.possibleArguments.getOptimizationArguments(this.getUsedOptions(req)));
    }

    getUsedOptions(req) {
        if (req.body) {
            const data = (typeof req.body === 'string') ? JSON.parse(req.body) : req.body;

            if (data.presplit) {
                return data.usedOptions;
            } else {
                return utils.splitArguments(data.usedOptions);
            }
        }
        return false;
    }

    handleApiError(error, res, next) {
        if (error.message) {
            return res.status(404).send({
                error: true,
                message: error.message,
            });
        } else {
            return next(error);
        }
    }

    handleCmake(req, res, next) {
        const compiler = this.compilerFor(req);
        if (!compiler) {
            return next();
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

        try {
            if (req.body.files === undefined) throw new Error('Missing files property');

            const options = this.parseRequest(req, compiler);
            compiler.cmake(req.body.files, options).then(result => {
                res.send(result);
            }).catch(e => {
                return this.handleApiError(e, res, next);
            });
        } catch (e) {
            return this.handleApiError(e, res, next);
        }
    }

    handle(req, res, next) {
        const compiler = this.compilerFor(req);
        if (!compiler) {
            return next();
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

        let parsedRequest;
        try {
            parsedRequest = this.parseRequest(req, compiler);
        } catch (error) {
            return this.handleApiError(error, res, next);
        }

        const {
            source, options, backendOptions, filters,
            bypassCache, tools, executionParameters, libraries,
        } = parsedRequest;

        let files;
        if (req.body.files) files = req.body.files;

        if (source === undefined) {
            logger.warn('No body found in request', req);
            return next(new Error('Bad request'));
        }

        function textify(array) {
            return _.pluck(array || [], 'text').join('\n');
        }

        this.compileCounter.inc({language: compiler.lang.id});
        // eslint-disable-next-line promise/catch-or-return
        compiler.compile(source, options, backendOptions, filters, bypassCache, tools, executionParameters, libraries,
            files)
            .then(
                result => {
                    if (result.execResult)
                        this.executeCounter.inc({language: compiler.lang.id});
                    if (req.accepts(['text', 'json']) === 'json') {
                        res.send(result);
                    } else {
                        res.set('Content-Type', 'text/plain');
                        try {
                            if (!_.isEmpty(this.textBanner)) res.write('# ' + this.textBanner + '\n');
                            res.write(textify(result.asm));
                            if (result.code !== 0) res.write('\n# Compiler exited with result code ' + result.code);
                            if (!_.isEmpty(result.stdout)) res.write('\nStandard out:\n' + textify(result.stdout));
                            if (!_.isEmpty(result.stderr)) res.write('\nStandard error:\n' + textify(result.stderr));

                            if (result.execResult) {
                                res.write('\n\n# Execution result with exit code ' + result.execResult.code + '\n');
                                if (!_.isEmpty(result.execResult.stdout)) {
                                    res.write('# Standard out:\n' + textify(result.execResult.stdout));
                                }
                                if (!_.isEmpty(result.execResult.stderr)) {
                                    res.write('\n# Standard error:\n' + textify(result.execResult.stderr));
                                }
                            }
                        } catch (ex) {
                            Sentry.captureException(ex);
                            res.write(`Error handling request: ${ex}`);
                        }
                        res.end('\n');
                    }
                },
                error => {
                    if (typeof (error) !== 'string') {
                        if (error.stack) {
                            logger.error('Error during compilation: ', error);
                            Sentry.captureException(error);
                        } else if (error.code) {
                            logger.error('Error during compilation: ', error.code);
                            if (typeof (error.stderr) === 'string') {
                                error.stdout = utils.parseOutput(error.stdout);
                                error.stderr = utils.parseOutput(error.stderr);
                            }
                            res.end(JSON.stringify(error));
                            return;
                        } else {
                            logger.error('Error during compilation: ', error);
                        }

                        error = `Internal Compiler Explorer error: ${error.stack || error}`;
                    } else {
                        logger.error('Error during compilation: ', {error});
                    }
                    res.end(JSON.stringify({code: -1, stderr: [{text: error}]}));
                });
    }
}

export function SetTestMode() {
    hasSetUpAutoClean = true;
}
