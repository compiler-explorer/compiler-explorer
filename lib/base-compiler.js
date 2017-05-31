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
    path = require('path'),
    fs = require('fs-extra'),
    Promise = require('promise'), // jshint ignore:line
    asm = require('./asm'),
    utils = require('./utils'),
    quote = require('shell-quote'),
    _ = require('underscore-node'),
    exec = require('./exec'),
    logger = require('./logger').logger,
    compilerOptInfo = require("compiler-opt-info");

function Compile(compiler, env) {
    this.compiler = compiler;
    this.env = env;
    this.asm = new asm.AsmParser(env.compilerProps);
    this.compiler.supportsIntel = !!this.compiler.intelAsm;
}

Compile.prototype.newTempDir = function () {
    return new Promise(function (resolve, reject) {
        temp.mkdir('compiler-explorer-compiler', function (err, dirPath) {
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
Compile.prototype.couldHaveOptOutput = function (options, version) {
    //TODO(jared): a compiler meta data gatherer could improve this
    return version.toLowerCase().indexOf("clang") > -1 &&
        options.some((x) => x == "-fsave-optimization-record");
};
Compile.prototype.couldHaveASTOutput = function (options, version) {
    return version.toLowerCase().indexOf("clang") > -1;
};

Compile.prototype.getRemote = function () {
    if (this.compiler.exe === null && this.compiler.remote)
        return this.compiler.remote;
    return false;
};

Compile.prototype.exec = function (compiler, args, options) {
    // Here only so can be overridden by compiler implementations.
    return exec.execute(compiler, args, options);
};

Compile.prototype.runCompiler = function (compiler, options, inputFilename, execOptions) {
    if (typeof(execOptions)==='undefined') {
        execOptions = {
            timeoutMs: this.env.gccProps("compileTimeoutMs", 100),
            maxErrorOutput: this.env.gccProps("max-error-output", 5000),
            env: this.env.getEnv(this.compiler.needsMulti),
            wrapper: this.env.compilerProps("compiler-wrapper")
        }
    }
    
    return this.exec(compiler, options, execOptions).then(function (result) {
        result.stdout = utils.parseOutput(result.stdout, inputFilename);
        result.stderr = utils.parseOutput(result.stderr, inputFilename);
        return result;
    });
};

Compile.prototype.supportsObjdump = function () {
    return this.compiler.objdumper !== "";
};

Compile.prototype.objdump = function (outputFilename, result, maxSize, intelAsm) {
    var args = ["-d", "-C", outputFilename, "-l", "--insn-width=16"];
    if (intelAsm) args = args.concat(["-M", "intel"]);
    return this.exec(this.compiler.objdumper, args, {maxOutput: maxSize})
        .then(function (objResult) {
            result.asm = objResult.stdout;
            if (objResult.code !== 0) {
                result.asm = "<No output: objdump returned " + objResult.code + ">";
            }
            return result;
        });
};

Compile.prototype.execBinary = function (executable, result, maxSize) {
    return exec.sandbox(executable, [], {
        maxOutput: maxSize,
        timeoutMs: 2000
    })  // TODO make config
        .then(function (execResult) {
            execResult.stdout = utils.parseOutput(execResult.stdout);
            execResult.stderr = utils.parseOutput(execResult.stderr);
            result.execResult = execResult;
            return result;
        }).catch(function (err) {
            // TODO: is this the best way? Perhaps failures in sandbox shouldn't reject
            // with "results", but instead should play on?
            result.execResult = {
                stdout: err.stdout ? utils.parseOutput(err.stdout) : [],
                stderr: err.stderr ? utils.parseOutput(err.stderr) : [],
                code: err.code !== undefined ? err.code : -1
            };
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

Compile.prototype.generateAST = function(inputFilename, options) {
    var newOptions = options.concat(["-Xclang", "-ast-dump"]);

    execOptions = {
        timeoutMs: this.env.gccProps("compileTimeoutMs", 100),
        maxErrorOutput: this.env.gccProps("max-error-output", 5000),
        env: this.env.getEnv(this.compiler.needsMulti),
        wrapper: this.env.compilerProps("compiler-wrapper"),
        maxOutput: 1024*1024*1024
    }
    
    return this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions);
}

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

    if (filters.binary && !source.match(this.env.compilerProps("stubRe"))) {
        source += "\n" + this.env.compilerProps("stubText") + "\n";
    }
    return self.env.enqueue(function () {
        var tempFileAndDirPromise = Promise.resolve().then(function () {
            return self.newTempDir().then(function (dirPath) {
                var inputFilename = path.join(dirPath, self.env.compilerProps("compileFilename"));
                return self.writeFile(inputFilename, source).then(function () {
                    return {inputFilename: inputFilename, dirPath: dirPath};
                });
            });
        });

        var compileToAsmPromise = tempFileAndDirPromise.then(function (info) {
            var inputFilename = info.inputFilename;
            var dirPath = info.dirPath;
            var outputFilebase = "output";
            var outputFilename = path.join(dirPath, outputFilebase + ".s"); // NB keep lower case as ldc compiler `tolower`s the output name
            options = self.prepareArguments(options, filters, inputFilename, outputFilename);

            options = options.filter(_.identity);

            var asmPromise = self.runCompiler(self.compiler.exe, options, self.filename(inputFilename))
            var astPromise = self.generateAST(inputFilename, options)
            
            return Promise.all([asmPromise, astPromise])
                .then(function (results) {
                    var asmResult = results[0];
                    var astResult = results[1];

                    asmResult.dirPath = dirPath;
                    if (asmResult.code !== 0) {
                        asmResult.asm = "<Compilation failed>";
                        return asmResult;
                    }
                    asmResult.hasOptOutput = false;
                    if (self.couldHaveOptOutput(options, self.compiler.version)) {
                        const optPath = path.join(dirPath, outputFilebase + ".opt.yaml");
                        if (fs.existsSync(optPath)) {
                            asmResult.hasOptOutput = true;
                            asmResult.optPath = optPath;
                        }
                    }
                    asmResult.hasAstOutput = true;
                    asmResult.astOutput = astResult.stdout;
                    
                    return self.postProcess(asmResult, outputFilename, filters);
                });
        });

        return compileToAsmPromise
            .then(function (results) {
                //TODO(jared): this isn't ideal. Rethink
                var result;
                var optOutput;
                if (results.length) {
                    result = results[0];
                    optOutput = results[1];
                } else {
                    result = results;
                }
                if (result.dirPath) {
                    fs.remove(result.dirPath);
                    result.dirPath = undefined;
                }
                if (result.okToCache) {
                    result.asm = self.asm.process(result.asm, filters);
                } else {
                    result.asm = {text: result.asm};
                }
                if (result.hasOptOutput) {
                    result.optPath = undefined;
                    result.optOutput = optOutput;
                }
                return result;
            })
            .then(_.bind(self.postProcessAsm, self))
            .then(function (result) {
                if (result.okToCache) self.env.cachePut(key, result);
                return result;
            });
    });
};

Compile.prototype.postProcessAsm = function (result) {
    if (!result.okToCache) return result;
    var demangler = this.compiler.demangler;
    if (!demangler) return result;
    return this.exec(demangler, [], {input: _.pluck(result.asm, 'text').join("\n")})
        .then(function (demangleResult) {
            var lines = utils.splitLines(demangleResult.stdout);
            for (var i = 0; i < result.asm.length; ++i)
                result.asm[i].text = lines[i];
            return result;
        });
};


Compile.prototype.processOptOutput = function (hasOptOutput, optPath) {
    var output = [];
    return new Promise(
        function (resolve, reject) {
            fs.createReadStream(optPath, {encoding: "utf-8"})
                .pipe(new compilerOptInfo.LLVMOptTransformer())
                .on("data", function (opt) {
                    output.push(opt);
                })
                .on("end", function () {
                    resolve(output);
                });
        });
};


Compile.prototype.processAstOutput = function (output) {
    output = output.map(function(x) { return x.text; });

    var topLevelRegex = /^(\||`)-/;
    var sourceRegex = /<source>/;
    var lineRegex = /<line:/;

    var mostRecentIsSource = false;

    console.log(output);

    for (var i = 0; i < output.length; ++i) {
        if (output[i].match(topLevelRegex)) {
            if (output[i].match(lineRegex) && mostRecentIsSource) {
                //do nothing
            }
            else if (!output[i].match(sourceRegex)) {
                var spliceMax = i+1;
                while (output[spliceMax] && !output[spliceMax].match(topLevelRegex)) {
                    spliceMax++;
                }
                output.splice(i, spliceMax-i);
                --i;
                mostRecentIsSource = false;
            }
            else {
                mostRecentIsSource = true;
            }
        }
    }

    output = output.join('\n');
    
    var addressRegex = /^([^A-Za-z]*[A-Za-z]+) 0x[a-z0-9]+/mg;
    output = output.replace(addressRegex, '$1');

    var slocRegex = / ?<?<invalid sloc>>?/g;
    output = output.replace(slocRegex, '');

    var sourceRegex = /<source>/g;
    output = output.replace(sourceRegex, 'line');    

    return output;
}

Compile.prototype.postProcess = function (result, outputFilename, filters) {
    var postProcess = this.compiler.postProcess.filter(_.identity);
    var maxSize = this.env.gccProps("max-asm-size", 8 * 1024 * 1024);
    var optPromise, asmPromise, execPromise;
    if (result.hasOptOutput) {
        optPromise = this.processOptOutput(result.hasOptOutput, result.optPath);
    } else {
        optPromise = Promise.resolve("");
    }

    if (result.hasAstOutput) {
        result.astOutput = this.processAstOutput(result.astOutput);
    }
    
    if (filters.binary && this.supportsObjdump()) {
        asmPromise = this.objdump(outputFilename, result, maxSize, filters.intel);
    } else {
        asmPromise = this.stat(outputFilename).then(_.bind(function (stat) {
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
    }
    if (filters.execute) {
        var maxExecOutputSize = this.env.gccProps("max-executable-output-size", 32 * 1024);
        execPromise = this.execBinary(outputFilename, result, maxExecOutputSize);
    } else {
        execPromise = Promise.resolve("");
    }

    return Promise.all([asmPromise, optPromise, execPromise]);
};

Compile.prototype.checkOptions = function (options) {
    var error = this.env.findBadOptions(options);
    if (error.length > 0) return "Bad options: " + error.join(", ");
    return null;
};

// This check for arbitrary user-controlled preprocessor inclusions
// can be circumvented in more than one way. The goal here is to respond
// to simple attempts with a clear diagnostic; the service still needs to
// assume that malicious actors can make the compiler open arbitrary files.
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
                    logger.error("Unable to find compiler version for '" + compiler + "':", result,
                        'with re', versionRe);
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
                            this.compiler.supportsIntel = true;
                        }

                        logger.debug("compiler options: ", options);

                        return this;
                    }, this));
            }, this),
            _.bind(function (err) {
                logger.error("Unable to get version for compiler '" + compiler + "' - " + err);
                return null;
            }, this));
};

Compile.prototype.getInfo = function () {
    return this.compiler;
};

Compile.prototype.getDefaultFilters = function () {
    // TODO; propagate to UI?
    return {
        intel: true,
        commentOnly: true,
        directives: true,
        labels: true
    };
};

module.exports = Compile;
