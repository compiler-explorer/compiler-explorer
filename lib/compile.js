// Copyright (c) 2012-2016, Matt Godbolt
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
    path = require('path'),
    httpProxy = require('http-proxy'),
    fs = require('fs-extra'),
    Promise = require('promise'), // jshint ignore:line
    asm = require('./asm'),
    utils = require('./utils'),
    quote = require('shell-quote'),
    _ = require('underscore-node'),
    logger = require('./logger').logger,
    CompilationEnvironment = require('./compilation-env').CompilationEnvironment;

temp.track();

function periodicCleanup() {
    temp.cleanup(function (err, stats) {
        if (err) logger.error("Error cleaning directories: ", err);
        if (stats) logger.debug("Directory cleanup stats:", stats);
    });
}
var gccProps = null;
var compilerProps = null;
var stubRe = null;
var stubText = null;

function identity(x) {
    return x;
}

function initialise(gccProps_, compilerProps_) {
    gccProps = gccProps_;
    compilerProps = compilerProps_;
    var tempDirCleanupSecs = gccProps("tempDirCleanupSecs", 600);
    logger.info("Cleaning temp dirs every " + tempDirCleanupSecs + " secs");
    setInterval(periodicCleanup, tempDirCleanupSecs * 1000);
    asm.initialise(compilerProps);
    stubRe = compilerProps("stubRe");
    stubText = compilerProps("stubText");
}

function Compile(compiler, env) {
    this.compiler = compiler;
    this.env = env;
}

Compile.prototype.newTempDir = function () {
    return new Promise(function (resolve, reject) {
        temp.mkdir('gcc-explorer-compiler', function (err, dirPath) {
            if (err)
                reject("Unable to open temp file: " + err);
            else
                resolve(dirPath);
        });
    });
};

Compile.prototype.writeFile = Promise.denodeify(fs.writeFile);
Compile.prototype.readFile = Promise.denodeify(fs.readFile);
Compile.prototype.stat = Promise.denodeify(fs.stat);

Compile.prototype.getRemote = function () {
    if (this.compiler.exe === null && this.compiler.remote)
        return this.compiler.remote;
    return false;
};

Compile.prototype.runCompiler = function (compiler, options, inputFilename) {
    return this.exec(compiler, options, {
        timeoutMs: gccProps("compileTimeoutMs", 100),
        maxErrorOutput: gccProps("max-error-output", 5000),
        env: this.env.getEnv(this.compiler.needsMulti),
        wrapper: compilerProps("compiler-wrapper")
    }).then(function (result) {
        result.stdout = utils.parseOutput(result.stdout, inputFilename);
        result.stderr = utils.parseOutput(result.stderr, inputFilename);
        return result;
    });
};

Compile.prototype.supportsObjdump = function () {
    return true;
};

Compile.prototype.objdump = function (outputFilename, result, maxSize, intelAsm) {
    var args = ["-d", "-C", outputFilename, "-l", "--insn-width=16"];
    if (intelAsm) args = args.concat(["-M", "intel"]);
    return this.exec("objdump", args, {maxOutput: maxSize})
        .then(function (objResult) {
            result.asm = objResult.stdout;
            if (objResult.code !== 0) {
                result.asm = "<No output: objdump returned " + objResult.code + ">";
            }
            return result;
        });
};

Compile.prototype.filename = function (fn) {
    return fn;
};

Compile.prototype.optionsForFilter = function (filters, outputFilename) {
    var options = ['-g', '-o', this.filename(outputFilename)];
    if (this.compiler.intelAsm && filters.intel && !filters.binary) {
        options = options.concat(this.compiler.intelAsm.split(" "));
    }
    if (!filters.binary) options = options.concat('-S');
    return options;
};

Compile.prototype.prepareArguments = function (userOptions, filters, inputFilename, outputFilename) {
    var options = this.optionsForFilter(filters, outputFilename);
    if (this.compiler.options) {
        options = options.concat(this.compiler.options.split(" "));
    }
    return options.concat(userOptions || []).concat([this.filename(inputFilename)]);
};

Compile.prototype.compile = function (source, options, filters) {
    var self = this;
    var optionsError = self.checkOptions(options);
    if (optionsError) return Promise.reject(optionsError);
    var sourceError = self.checkSource(source);
    if (sourceError) return Promise.reject(sourceError);

    // Don't run binary for unsupported compilers, even if we're asked.
    if (filters.binary && !self.compiler.supportsBinary) {
        delete filters.binary;
    }

    var key = JSON.stringify({compiler: this.compiler, source: source, options: options, filters: filters});
    var cached = this.env.cacheGet(key);
    if (cached) {
        return Promise.resolve(cached);
    }

    if (filters.binary && !source.match(stubRe)) {
        source += "\n" + stubText + "\n";
    }

    var tempFileAndDirPromise = Promise.resolve().then(function () {
        return self.newTempDir().then(function (dirPath) {
            var inputFilename = path.join(dirPath, compilerProps("compileFilename"));
            return self.writeFile(inputFilename, source).then(function () {
                return {inputFilename: inputFilename, dirPath: dirPath};
            });
        });
    });

    var compileToAsmPromise = tempFileAndDirPromise.then(function (info) {
        var inputFilename = info.inputFilename;
        var dirPath = info.dirPath;
        var outputFilename = path.join(dirPath, 'output.s'); // NB keep lower case as ldc compiler `tolower`s the output name
        options = self.prepareArguments(options, filters, inputFilename, outputFilename);

        options = options.filter(identity);
        return self.runCompiler(self.compiler.exe, options, self.filename(inputFilename))
            .then(function (result) {
                result.dirPath = dirPath;
                if (result.code !== 0) {
                    result.asm = "<Compilation failed>";
                    return result;
                }
                return self.postProcess(result, outputFilename, filters);
            });
    });

    return self.env.enqueue(function () {
        return compileToAsmPromise.then(function (result) {
            if (result.dirPath) {
                fs.remove(result.dirPath);
                result.dirPath = undefined;
            }
            if (result.okToCache) {
                result.asm = asm.processAsm(result.asm, filters);
                self.env.cachePut(key, result);
            } else {
                result.asm = {text: result.asm};
            }
            return result;
        });
    });
};

Compile.prototype.postProcess = function (result, outputFilename, filters) {
    var postProcess = this.compiler.postProcess.filter(identity);
    var maxSize = gccProps("max-asm-size", 8 * 1024 * 1024);
    if (filters.binary && this.supportsObjdump()) {
        return this.objdump(outputFilename, result, maxSize, filters.intel);
    }
    return this.stat(outputFilename).then(_.bind(function (stat) {
            if (stat.size >= maxSize) {
                result.asm = "<No output: generated assembly was too large (" + stat.size + " > " + maxSize + " bytes)>";
                return result;
            }
            if (postProcess.length) {
                var postCommand = 'cat "' + outputFilename + '" | ' + postProcess.join(" | ");
                return this.exec("bash", ["-c", postCommand], {maxOutput: maxSize})
                    .then(function (postResult) {
                        result.asm = postResult.stdout;
                        if (postResult.code !== 0) {
                            result.asm = "<Error during post processing: " + postResult.code + ">";
                        }
                        return result;
                    });
            } else {
                return this.readFile(outputFilename).then(function (contents) {
                    result.asm = contents.toString();
                    return Promise.resolve(result);
                });
            }
        }, this),
        function () {
            result.asm = "<No output file>";
            return result;
        }
    );
};

Compile.prototype.checkOptions = function (options) {
    var error = this.env.findBadOptions(options);
    if (error.length > 0) return "Bad options: " + error.join(", ");
    return null;
};

Compile.prototype.checkSource = function (source) {
    var re = /^\s*#\s*i(nclude|mport)(_next)?\s+["<"](\/|.*\.\.)/;
    var failed = [];
    utils.splitLines(source).forEach(function (line, index) {
        if (line.match(re)) {
            failed.push("<stdin>:" + (index + 1) + ":1: no absolute or relative includes please");
        }
    });
    if (failed.length > 0) return failed.join("\n");
    return null;
};

Compile.prototype.exec = function (command, args, options) {
    options = options || {};
    var maxOutput = options.maxOutput || 1024 * 1024;
    var timeoutMs = options.timeoutMs || 0;
    var env = options.env;

    if (options.wrapper) {
        args.unshift(command);
        command = options.wrapper;
    }

    var okToCache = true;
    var child = child_process.spawn(command, args, {
        env: env,
        detached: process.platform == 'linux'
    });
    var stderr = "";
    var stdout = "";
    var timeout;
    if (timeoutMs) timeout = setTimeout(function () {
        okToCache = false;
        child.kill();
        stderr += "\nKilled - processing time exceeded";
    }, timeoutMs);
    var truncated = false;
    child.stdout.on('data', function (data) {
        if (truncated) return;
        if (stdout.length > maxOutput) {
            stdout += "\n[Truncated]";
            truncated = true;
            child.kill();
            return;
        }
        stdout += data;
    });
    child.stderr.on('data', function (data) {
        if (truncated) return;
        if (stderr.length > maxOutput) {
            stderr += "\n[Truncated]";
            truncated = true;
            child.kill();
            return;
        }
        stderr += data;
    });
    return new Promise(function (resolve, reject) {
        child.on('error', function (e) {
            reject(e);
        });
        child.on('exit', function (code) {
            if (timeout !== undefined) clearTimeout(timeout);
            // Why is this apparently needed in some cases (e.g. when I used to use this to do getMultiarch)?
            // Without it, I apparently get stdout/stderr callbacks *after* the exit...
            setTimeout(function () {
                resolve({
                    code: code,
                    stdout: stdout,
                    stderr: stderr,
                    okToCache: okToCache
                });
            }, 0);
        });
        child.stdin.end();
    });
};

Compile.prototype.initialise = function () {
    if (this.getRemote()) return Promise.resolve(this);
    var compiler = this.compiler.exe;
    var versionFlag = this.compiler.versionFlag || '--version';
    var versionRe = new RegExp(this.compiler.versionRe || '.*');
    logger.info("Gathering version information on", compiler);
    return this.exec(compiler, [versionFlag])
        .then(_.bind(function (result) {
            if (result.code !== 0) {
                logger.error("Unable to get version for compiler '" + compiler + "' - non-zero result " + result.code);
                return null;
            }
            var version = "";
            _.each(utils.splitLines(result.stdout + result.stderr), function (line) {
                if (version) return;
                var match = line.match(versionRe);
                if (match) version = match[0];
            });
            if (!version) {
                logger.error("Unable to get compiler version for '" + compiler + "'");
                return null;
            }
            logger.info(compiler + " is version '" + version + "'");
            this.compiler.version = version;
            if (this.compiler.intelAsm) return this;
            return this.exec(compiler, ['--target-help'])
                .then(_.bind(function (result) {
                    var options = {};
                    if (result.code === 0) {
                        var splitness = /--?[-a-zA-Z]+( ?[-a-zA-Z]+)/;
                        utils.eachLine(result.stdout + result.stderr, function (line) {
                            var match = line.match(splitness);
                            if (!match) return;
                            options[match[0]] = true;
                        });
                    }
                    if (options['-masm']) {
                        this.compiler.intelAsm = "-masm=intel";
                    }

                    logger.debug("compiler options: ", options);

                    return this;
                }, this));
        }, this));
};

Compile.prototype.getInfo = function () {
    return this.compiler;
};

function compileCl(info, env) {
    var compile = new Compile(info, env);
    info.supportsFiltersInBinary = true;
    if (process.platform == "linux") {
        var wine = gccProps("wine");
        var origExec = compile.exec;
        compile.exec = function (command, args, options) {
            if (command.toLowerCase().endsWith(".exe")) {
                args.unshift(command);
                command = wine;
            }
            return origExec(command, args, options);
        };
        compile.filename = function (fn) {
            return 'Z:' + fn;
        };
    }
    compile.supportsObjdump = function () {
        return false;
    };
    compile.optionsForFilter = function (filters, outputFilename) {
        return [
            '/FAsc',
            '/c',
            '/Fa' + this.filename(outputFilename),
            '/Fo' + this.filename(outputFilename + '.obj')
        ];
    };
    return compile.initialise();
}

function compile6g(info, env) {
    function convert6g(code) {
        var re = /^[0-9]+\s*\(([^:]+):([0-9]+)\)\s*([A-Z]+)(.*)/;
        var prevLine = null;
        var file = null;
        return code.map(function (obj) {
            var line = obj.line;
            var match = line.match(re);
            if (match) {
                var res = "";
                if (file === null) {
                    res += "\t.file 1 \"" + match[1] + "\"\n";
                    file = match[1];
                }
                if (prevLine != match[2]) {
                    res += "\t.loc 1 " + match[2] + "\n";
                    prevLine = match[2];
                }
                return res + "\t" + match[3].toLowerCase() + match[4];
            } else
                return null;
        }).filter(identity).join("\n");
    }

    var compiler = new Compile(info, env);
    compiler.postProcess = function () {
        result.asm = this.convert6g(result.stdout);
        result.stdout = [];
        return Promise.resolve(result);
    };
    return compiler;
}

function compileRust(info, env) {
    var compiler = new Compile(info, env);
    // TODO this needs testing!
    compiler.optionsForFilter = function (filters, outputFilename) {
        var options = ['-g', '-o', this.filename(outputFilename)];
        // TODO: binary not supported(?)
        if (!filters.binary) options = options.concat('--emit', 'asm');
        options = options.concat(['--crate-type', 'staticlib']);
        return options;
    };
}

function compileLdc(info, env) {
    var compiler = new Compile(info, env);
    // TODO this needs testing!
    compiler.optionsForFilter = function (filters, outputFilename) {
        var options = ['-g', '-of', this.filename(outputFilename)];
        if (filters.intel && !filters.binary) options.concat('-x86-asm-syntax=intel');
        if (!filters.binary) options = options.concat('-output-s');
        return options;
    };
}

var compileFactories = {
    "": function (info, env) {
        var comp = new Compile(info, env);
        return comp.initialise();
    },
    "CL": compileCl,
    "6g": compile6g,
    "rust": compileRust,
    "ldc": compileLdc
};

function CompileHandler() {
    this.compilersById = {};
    this.compilerEnv = new CompilationEnvironment(gccProps);

    this.setCompilers = function (compilers) {
        var initPromises = _.map(compilers, function (compiler) {
            return compileFactories[compiler.compilerType](compiler, this.compilerEnv);
        }, this);
        return Promise.all(initPromises)
            .then(function (compilers) {
                return _.filter(compilers, identity);
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
        var compiler = this.compilersById[req.body.compiler];
        if (!compiler) return next();

        var remote = compiler.getRemote();
        if (remote) {
            proxy.web(req, res, {target: remote}, function (e) {
                logger.error("Proxy error: ", e);
                next(e);
            });
            return;
        }
        var source = req.body.source;
        var options = req.body.options || '';
        if (source === undefined) {
            return next(new Error("Bad request"));
        }
        options = quote.parse(options).filter(identity);
        var filters = req.body.filters;
        compiler.compile(source, options, filters).then(
            function (result) {
                res.set('Content-Type', 'application/json');
                res.end(JSON.stringify(result));
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
    CompileHandler: CompileHandler,
    initialise: initialise
};
