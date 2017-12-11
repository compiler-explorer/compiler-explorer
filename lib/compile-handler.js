// Copyright (c) 2012-2017, Matt Godbolt
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
    fs = require('fs'),
    path = require('path'),
    httpProxy = require('http-proxy'),
    Promise = require('promise'), // jshint ignore:line
    quote = require('shell-quote'),
    _ = require('underscore-node'),
    logger = require('./logger').logger,
    utils = require('./utils'),
    CompilationEnvironment = require('./compilation-env').CompilationEnvironment,
    Raven = require('raven');

temp.track();

let oneTimeInit = false;

function initialise(ceProps, compilerEnv) {
    if (oneTimeInit) return;
    oneTimeInit = true;
    const tempDirCleanupSecs = ceProps("tempDirCleanupSecs", 600);
    logger.info("Cleaning temp dirs every " + tempDirCleanupSecs + " secs");
    setInterval(function () {
        if (compilerEnv.isBusy()) {
            logger.warn("Skipping temporary file clean up as compiler environment is busy");
            return;
        }
        temp.cleanup(function (err, stats) {
            if (err) logger.error("Error cleaning directories: ", err);
            if (stats) logger.debug("Directory cleanup stats:", stats);
        });
    }, tempDirCleanupSecs * 1000);
}

function CompileHandler(ceProps, compilerPropsL) {
    const self = this;
    this.compilersById = {};
    this.compilerEnv = new CompilationEnvironment(ceProps, compilerPropsL);
    initialise(ceProps, this.compilerEnv);
    this.factories = {};
    this.stat = Promise.denodeify(fs.stat);

    this.proxy = httpProxy.createProxyServer({});
    this.textBanner = ceProps('textBanner');

    this.create = function (compiler) {
        const type = compiler.compilerType || "default";
        if (self.factories[type] === undefined) {
            const compilerPath = './compilers/' + type;
            logger.debug("Loading compiler from", compilerPath);
            self.factories[type] = require(compilerPath);
        }
        if (path.isAbsolute(compiler.exe)) {
            // Try stat'ing the compiler to cache its mtime and only re-run it if it
            // has changed since the last time.
            return self.stat(compiler.exe)
                .then(_.bind(res => {
                    const cached = self.findCompiler(compiler.lang, compiler.id);
                    if (cached && cached.mtime.getTime() === res.mtime.getTime()) {
                        logger.debug(compiler.id + " is unchanged");
                        return cached;
                    }
                    return self.factories[type](compiler, self.compilerEnv, compiler.lang).then(compiler => {
                        compiler.mtime = res.mtime;
                        return compiler;
                    });
                }, self))
                .catch(err => {
                    logger.warn("Unable to stat compiler binary", err);
                    return null;
                });
        } else {
            return self.factories[type](compiler, self.compilerEnv, compiler.lang);
        }
    };
    this.handler = function compile(req, res, next) {
        let source, options, backendOptions, filters,
            compiler = self.findCompiler(req.lang || req.body.lang, req.compiler || req.body.compiler);
        if (!compiler) return next();
        if (req.is('json')) {
            // JSON-style request
            const requestOptions = req.body.options;
            source = req.body.source;
            options = requestOptions.userArguments;
            backendOptions = requestOptions.compilerOptions;
            filters = requestOptions.filters || compiler.getDefaultFilters();
        } else {
            // API-style
            // Find the compiler the user is interested in...
            source = req.body;
            options = req.query.options;
            // By default we get the default filters.
            filters = compiler.getDefaultFilters();
            // If specified exactly, we'll take that with ?filters=a,b,c
            if (req.query.filters) {
                filters = _.object(_.map(req.query.filters.split(","), function (filter) {
                    return [filter, true];
                }));
            }
            // Add a filter. ?addFilters=binary
            _.each((req.query.addFilters || "").split(","), function (filter) {
                filters[filter] = true;
            });
            // Remove a filter. ?removeFilter=intel
            _.each((req.query.removeFilters || "").split(","), function (filter) {
                delete filters[filter];
            });
        }
        const remote = compiler.getRemote();
        if (remote) {
            req.url = req.originalUrl;  // Undo any routing that was done to get here (i.e. /api/* path has been removed)
            self.proxy.web(req, res, {target: remote}, function (e) {
                logger.error("Proxy error: ", e);
                next(e);
            });
            return;
        }

        if (source === undefined) {
            return next(new Error("Bad request"));
        }
        options = _.chain(quote.parse(options || '')
            .map(x => {
                if (typeof(x) === "string") return x;
                return x.pattern;
            }))
            .filter(_.identity)
            .value();

        function textify(array) {
            return _.pluck(array || [], 'text').join("\n");
        }

        compiler.compile(source, options, backendOptions, filters).then(
            function (result) {
                if (req.accepts(['text', 'json']) === 'json') {
                    res.set('Content-Type', 'application/json');
                    res.end(JSON.stringify(result));
                } else {
                    res.set('Content-Type', 'text/plain');
                    try {
                        if (!_.isEmpty(self.textBanner)) res.write('# ' + self.textBanner + "\n");
                        res.write(textify(result.asm));
                        if (result.code !== 0) res.write("\n# Compiler exited with result code " + result.code);
                        if (!_.isEmpty(result.stdout)) res.write("\nStandard out:\n" + textify(result.stdout));
                        if (!_.isEmpty(result.stderr)) res.write("\nStandard error:\n" + textify(result.stderr));
                    } catch (ex) {
                        Raven.captureException(ex, {req: req});
                        res.write("Error handling request: " + ex);
                    }
                    res.end('\n');
                }
            },
            function (error) {
                logger.error("Error during compilation", error);
                if (typeof(error) !== "string") {
                    if (error.code) {
                        if (typeof(error.stderr) === "string") {
                            error.stdout = utils.parseOutput(error.stdout);
                            error.stderr = utils.parseOutput(error.stderr);
                        }
                        res.end(JSON.stringify(error));
                        return;
                    }
                    error = "Internal Compiler Explorer error: " + (error.stack || error);
                }
                res.end(JSON.stringify({code: -1, stderr: [{text: error}]}));
            }
        );
    };
    this.setCompilers = function (newCompilers) {
        // Delete every compiler first...
        self.compilersById = {};
        return Promise.all(_.map(newCompilers, self.create))
            .then(compilers => _.filter(compilers, _.identity))
            .then(compilers => {
                _.each(compilers, compiler => {
                    const langId = compiler.compiler.lang;
                    if (!self.compilersById[langId]) self.compilersById[langId] = {};
                    self.compilersById[langId][compiler.compiler.id] = compiler;
                }, self);
                return _.map(compilers,compiler => compiler.getInfo());
            })
            .catch(logger.error);
    };
    this.findCompiler = function (langId, compilerId) {
        if (langId && self.compilersById[langId]) {
            return self.compilersById[langId][compilerId];
        }
        // If the lang is bad, try to find it in every language
        let response;
        _.each(self.compilersById, compilerInLang => {
            if (response === undefined) {
                _.each(compilerInLang, compiler => {
                    if (response === undefined && compiler.compiler.id === compilerId) {
                        response = compiler;
                    }
                });
            }
        });
        return response;
    };
}

module.exports = {
    CompileHandler: CompileHandler
};
