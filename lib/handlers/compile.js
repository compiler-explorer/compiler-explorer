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

const temp = require('temp'),
    fs = require('fs-extra'),
    path = require('path'),
    httpProxy = require('http-proxy'),
    quote = require('shell-quote'),
    _ = require('underscore'),
    logger = require('../logger').logger,
    utils = require('../utils'),
    which = require('which'),
    Sentry = require('@sentry/node');

temp.track();

let hasSetUpAutoClean = false;

function initialise(compilerEnv) {
    if (hasSetUpAutoClean) return;
    hasSetUpAutoClean = true;
    const tempDirCleanupSecs = compilerEnv.ceProps("tempDirCleanupSecs", 600);
    logger.info(`Cleaning temp dirs every ${tempDirCleanupSecs} secs`);
    setInterval(() => {
        if (compilerEnv.isBusy()) {
            logger.warn("Skipping temporary file clean up as compiler environment is busy");
            return;
        }
        temp.cleanup((err, stats) => {
            if (err) logger.error("Error cleaning directories: ", err);
            if (stats) logger.debug("Directory cleanup stats:", stats);
        });
    }, tempDirCleanupSecs * 1000);
}

class CompileHandler {
    constructor(compilationEnvironment, awsProps) {
        this.compilersById = {};
        this.compilerEnv = compilationEnvironment;
        this.factories = {};
        this.textBanner = this.compilerEnv.ceProps('textBanner');
        this.proxy = httpProxy.createProxyServer({});
        this.awsProps = awsProps;
        initialise(this.compilerEnv);

        // Mostly cribbed from
        // https://github.com/nodejitsu/node-http-proxy/blob/master/examples/middleware/bodyDecoder-middleware.js
        // We just keep the body as-is though: no encoding using queryString.stringify(), as we don't use a form
        // decoding middleware.
        this.proxy.on('proxyReq', function (proxyReq, req) {
            if (!req.body || !Object.keys(req.body).length) {
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
        const type = compiler.compilerType || "default";
        if (this.factories[type] === undefined) {
            const compilerPath = '../compilers/' + type;
            logger.info(`Loading compiler from ${compilerPath}`);
            this.factories[type] = require(compilerPath);
        }

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
                const compilerObj = new this.factories[type](compiler, this.compilerEnv);
                compilerObj.mtime = res.mtime;
                return compilerObj.initialise();
            } catch (err) {
                logger.warn(`Unable to stat ${compiler.id} compiler binary: `, err);
                return null;
            }
        } else {
            return new this.factories[type](compiler, this.compilerEnv);
        }
    }

    setCompilers(compilers) {
        // Be careful not to update this.compilersById until we can replace it entirely.
        const compilersById = {};
        return Promise.all(_.map(compilers, this.create, this))
            .then(_.compact)
            .then(compilers => {
                _.each(compilers, compiler => {
                    const langId = compiler.compiler.lang;
                    if (!compilersById[langId]) compilersById[langId] = {};
                    compilersById[langId][compiler.compiler.id] = compiler;

                    if (this.awsProps) compiler.possibleArguments.loadFromStorage(this.awsProps);
                });
                this.compilersById = compilersById;
                return _.map(compilers, compiler => compiler.getInfo());
            })
            .catch(err => logger.error(err));
    }

    findCompiler(langId, compilerId) {
        const langCompilers = this.compilersById[langId];
        if (langCompilers && langCompilers[compilerId]) {
            return langCompilers[compilerId];
        }
        // If the lang is bad, try to find it in every language
        let response = undefined;
        _.each(this.compilersById, compilerInLang => {
            if (response === undefined) {
                _.each(compilerInLang, compiler => {
                    if (response === undefined &&
                        (compiler.compiler.id === compilerId || compiler.compiler.alias === compilerId)) {

                        response = compiler;
                    }
                });
            }
        });
        return response;
    }

    compilerFor(req) {
        if (req.is('json')) {
            const compiler = this.findCompiler(req.lang || req.body.lang, req.params.compiler || req.body.compiler);
            if (!compiler) {
                const withoutBody = _.extend({}, req.body, {source: '<removed>'});
                logger.warn(`Unable to find compiler for request body ${withoutBody} with lang ${req.lang}`);
            }
            return compiler;
        } else {
            const compiler = this.findCompiler(req.lang, req.params.compiler);
            if (!compiler) {
                logger.warn(`Unable to find compiler for request params ${req.params} with lang ${req.lang}`);
            }
            return compiler;
        }
    }

    parseRequest(req, compiler) {
        let source, options, backendOptions, filters, bypassCache = false, tools, executionParameters = {};
        let libraries = [];
        // IF YOU MODIFY ANYTHING HERE PLEASE UPDATE THE DOCUMENTATION!
        if (req.is('json')) {
            // JSON-style request
            const requestOptions = req.body.options;
            source = req.body.source;
            if (req.body.bypassCache)
                bypassCache = true;
            options = requestOptions.userArguments;
            const execParams = requestOptions.executeParameters || {};
            executionParameters.args = execParams.args;
            executionParameters.stdin = execParams.stdin;
            backendOptions = requestOptions.compilerOptions;
            filters = requestOptions.filters || compiler.getDefaultFilters();
            tools = requestOptions.tools;
            libraries = requestOptions.libraries || [];
        } else {
            // API-style
            source = req.body;
            options = req.query.options;
            // By default we get the default filters.
            filters = compiler.getDefaultFilters();
            // If specified exactly, we'll take that with ?filters=a,b,c
            if (req.query.filters) {
                filters = _.object(_.map(req.query.filters.split(","), filter => [filter, true]));
            }
            // Add a filter. ?addFilters=binary
            _.each((req.query.addFilters || "").split(","), filter => {
                if (filter) filters[filter] = true;
            });
            // Remove a filter. ?removeFilter=intel
            _.each((req.query.removeFilters || "").split(","), filter => {
                if (filter) delete filters[filter];
            });
        }
        options = this.splitArguments(options);
        executionParameters.args = this.splitArguments(executionParameters.args);

        tools = tools || [];
        tools.forEach((tool) => {
            tool.args = this.splitArguments(tool.args);
        });
        return {source, options, backendOptions, filters, bypassCache, tools, executionParameters, libraries};
    }

    splitArguments(options) {
        return _.chain(quote.parse(options || '')
            .map(x => typeof (x) === "string" ? x : x.pattern))
            .compact()
            .value();
    }

    handlePopularArguments(req, res, next) {
        const compiler = this.compilerFor(req);
        if (!compiler) {
            return next();
        }

        let usedOptions = false;
        if (req.body) {
            let data = req.body;
            if (typeof req.body === 'string') {
                data = JSON.parse(req.body);
            }

            if (data.presplit) {
                usedOptions = data.usedOptions;
            } else {
                usedOptions = this.splitArguments(data.usedOptions);
            }
        }

        const popularArguments = compiler.possibleArguments.getPopularArguments(usedOptions);

        res.end(JSON.stringify(popularArguments));
    }

    handleOptimizationArguments(req, res, next) {
        const compiler = this.compilerFor(req);
        if (!compiler) {
            return next();
        }

        let usedOptions = false;
        if (req.body) {
            let data = req.body;
            if (typeof req.body === 'string') {
                data = JSON.parse(req.body);
            }

            if (data.presplit) {
                usedOptions = data.usedOptions;
            } else {
                usedOptions = this.splitArguments(data.usedOptions);
            }
        }

        const args = compiler.possibleArguments.getOptimizationArguments(usedOptions);

        res.end(JSON.stringify(args));
    }

    handle(req, res, next) {
        const compiler = this.compilerFor(req);
        if (!compiler) {
            return next();
        }
        const {source, options, backendOptions, filters,
            bypassCache, tools, executionParameters, libraries} = this.parseRequest(req, compiler);
        const remote = compiler.getRemote();
        if (remote) {
            req.url = remote.path;
            this.proxy.web(req, res, {target: remote.target, changeOrigin: true}, e => {
                logger.error("Proxy error: ", e);
                next(e);
            });
            return;
        }

        if (source === undefined) {
            logger.warn("No body found in request", req);
            return next(new Error("Bad request"));
        }

        function textify(array) {
            return _.pluck(array || [], 'text').join("\n");
        }

        compiler.compile(source, options, backendOptions, filters, bypassCache, tools, executionParameters, libraries)
            .then(result => {
                if (req.accepts(['text', 'json']) === 'json') {
                    res.set('Content-Type', 'application/json');
                    res.end(JSON.stringify(result));
                } else {
                    res.set('Content-Type', 'text/plain');
                    try {
                        if (!_.isEmpty(this.textBanner)) res.write('# ' + this.textBanner + "\n");
                        res.write(textify(result.asm));
                        if (result.code !== 0) res.write("\n# Compiler exited with result code " + result.code);
                        if (!_.isEmpty(result.stdout)) res.write("\nStandard out:\n" + textify(result.stdout));
                        if (!_.isEmpty(result.stderr)) res.write("\nStandard error:\n" + textify(result.stderr));
                    } catch (ex) {
                        Sentry.captureException(ex);
                        res.write(`Error handling request: ${ex}`);
                    }
                    res.end('\n');
                }
            },
            error => {
                if (typeof (error) !== "string") {
                    if (error.stack) {
                        logger.error("Error during compilation: ", error);
                    } else if (error.code) {
                        logger.error("Error during compilation: ", error.code);
                        if (typeof (error.stderr) === "string") {
                            error.stdout = utils.parseOutput(error.stdout);
                            error.stderr = utils.parseOutput(error.stderr);
                        }
                        res.end(JSON.stringify(error));
                        return;
                    } else {
                        logger.error("Error during compilation: ", error);
                    }

                    error = `Internal Compiler Explorer error: ${error.stack || error}`;
                } else {
                    logger.error("Error during compilation: ", {error});
                }
                res.end(JSON.stringify({code: -1, stderr: [{text: error}]}));
            });
    }
}

module.exports.Handler = CompileHandler;
module.exports.SetTestMode = function () {
    hasSetUpAutoClean = true;
};
