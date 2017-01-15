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

var child_process = require('child_process'),
    temp = require('temp'),
    httpProxy = require('http-proxy'),
    Promise = require('promise'), // jshint ignore:line
    quote = require('shell-quote'),
    _ = require('underscore-node'),
    logger = require('./logger').logger,
    CompilationEnvironment = require('./compilation-env').CompilationEnvironment;

temp.track();

var oneTimeInit = false;
function initialise(gccProps, compilerEnv) {
    if (oneTimeInit) return;
    oneTimeInit = true;
    var tempDirCleanupSecs = gccProps("tempDirCleanupSecs", 600);
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

function CompileHandler(gccProps, compilerProps) {
    this.compilersById = {};
    this.compilerEnv = new CompilationEnvironment(gccProps, compilerProps);
    initialise(gccProps, this.compilerEnv);
    this.factories = {};

    this.create = function (compiler) {
        var type = compiler.compilerType || "default";
        if (this.factories[type] === undefined) {
            var path = './compilers/' + type;
            logger.info("Loading compiler from", path);
            this.factories[type] = require(path);
        }
        return this.factories[type](compiler, this.compilerEnv);
    };

    this.setCompilers = function (compilers) {
        var initPromises = _.map(compilers, this.create, this);
        return Promise.all(initPromises)
            .then(function (compilers) {
                return _.filter(compilers, _.identity);
            })
            .then(_.bind(function (compilers) {
                _.each(compilers, function (compiler) {
                    this.compilersById[compiler.compiler.id] = compiler;
                }, this);
                return _.map(compilers, function (compiler) {
                    return compiler.getInfo();
                });
            }, this)).catch(function (err) {
                logger.error(err);
            });
    };
    var proxy = httpProxy.createProxyServer({});

    this.handler = _.bind(function compile(req, res, next) {
        var source, options, filters, compiler;
        if (req.is('json')) {
            // JSON-style request
            compiler = this.compilersById[req.compiler || req.body.compiler];
            if (!compiler) return next();
            source = req.body.source;
            options = req.body.options;
            filters = req.body.filters || compiler.getDefaultFilters();
        } else {
            // API-style
            compiler = this.compilersById[req.compiler];
            if (!compiler) return next();
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
        var remote = compiler.getRemote();
        if (remote) {
            proxy.web(req, res, {target: remote}, function (e) {
                logger.error("Proxy error: ", e);
                next(e);
            });
            return;
        }

        if (source === undefined) {
            return next(new Error("Bad request"));
        }
        options = _.chain(quote.parse(options || '')
            .map(function (x) {
                if (typeof(x) == "string") return x;
                return x.pattern;
            }))
            .filter(_.identity)
            .value();
        function textify(array) {
            return _.pluck(array || [], 'text').join("\n");
        }

        compiler.compile(source, options, filters).then(
            function (result) {
                if (req.accepts(['text', 'json']) === 'json') {
                    res.set('Content-Type', 'application/json');
                    res.end(JSON.stringify(result));
                } else {
                    res.set('Content-Type', 'text/plain');
                    try {
                        res.write("# Compilation provided by Compiler Explorer at " + req.get('Host') + "\n");
                        res.write(textify(result.asm));
                        if (result.code !== 0) res.write("\n# Compiler exited with result code " + result.code);
                        if (!_.isEmpty(result.stdout)) res.write("\nStandard out:\n" + textify(result.stdout));
                        if (!_.isEmpty(result.stderr)) res.write("\nStandard error:\n" + textify(result.stderr));
                    } catch (ex) {
                        re.write("Error handling request: " + ex);
                    }
                    res.end('\n');
                }
            },
            function (error) {
                logger.error("Error: " + error);
                if (typeof(error) !== "string") {
                    error = "Internal GCC explorer error: " + error.toString();
                }
                res.end(JSON.stringify({code: -1, stderr: [{text: error}]}));
            }
        );
    }, this);
}

module.exports = {
    CompileHandler: CompileHandler
};
