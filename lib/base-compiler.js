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

/*jslint node: true */
"use strict";

var child_process = require('child_process'),
    temp = require('temp'),
    path = require('path'),
    fs = require('fs-extra'),
    Promise = require('promise'), // jshint ignore:line
    asm = require('./asm'),
    utils = require('./utils'),
    _ = require('underscore-node'),
    exec = require('./exec'),
    logger = require('./logger').logger,
    compilerOptInfo = require("compiler-opt-info"),
    argumentParsers = require("./compilers/argument-parsers"),
    cfg = require('./cfg');

function Compile(compiler, env) {
    this.compiler = compiler;
    this.env = env;
    this.asm = new asm.AsmParser(env.compilerProps);
    this.compiler.supportsIntel = !!this.compiler.intelAsm;
}

// AP: Create directory in specified tmpDir. tmpDir on Windows volume is required for WSL execution.
// NPM temp package: https://www.npmjs.com/package/temp, see Affixes
Compile.prototype.newTempDir = function () {
    return new Promise(function (resolve, reject) {
        // temp.mkdir('compiler-explorer-compiler', function (err, dirPath) {
        temp.mkdir({prefix: 'compiler-explorer-compiler', dir: process.env.tmpDir}, function (err, dirPath) {
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

Compile.prototype.optOutputRequested = function (options) {
    return options.some((x) => x === "-fsave-optimization-record");
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

Compile.prototype.getDefaultExecOptions = function () {
    return {
        timeoutMs: this.env.gccProps("compileTimeoutMs", 100),
        maxErrorOutput: this.env.gccProps("max-error-output", 5000),
        env: this.env.getEnv(this.compiler.needsMulti),
        wrapper: this.env.compilerProps("compiler-wrapper")
    };
};

Compile.prototype.runCompiler = function (compiler, options, inputFilename, execOptions) {
    if (!execOptions) {
        execOptions = this.getDefaultExecOptions();
    }

    return this.exec(compiler, options, execOptions).then(function (result) {
        result.inputFilename = inputFilename;
        result.stdout = utils.parseOutput(result.stdout, inputFilename);
        result.stderr = utils.parseOutput(result.stderr, inputFilename);
        return result;
    });
};

Compile.prototype.supportsObjdump = function () {
    return this.compiler.objdumper !== "";
};

Compile.prototype.objdump = function (outputFilename, result, maxSize, intelAsm, demangle) {
    var args = ["-d", outputFilename, "-l", "--insn-width=16"];
    if (demangle) args = args.concat("-C");
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

Compile.prototype.optionsForFilter = function (filters, outputFilename, userOptions) {
    var options = ['-g', '-o', this.filename(outputFilename)];
    if (this.compiler.intelAsm && filters.intel && !filters.binary) {
        options = options.concat(this.compiler.intelAsm.split(" "));
    }
    if (!filters.binary) options = options.concat('-S');
    return options;
};

Compile.prototype.prepareArguments = function (userOptions, filters, backendOptions, inputFilename, outputFilename) {
    var options = this.optionsForFilter(filters, outputFilename, userOptions);
    backendOptions = backendOptions || {};

    if (this.compiler.options) {
        options = options.concat(this.compiler.options.split(" "));
    }

    if (this.compiler.supportsOptOutput && backendOptions.produceOptInfo) {
        options = options.concat(this.compiler.optArg);
    }

    return options.concat(userOptions || []).concat([this.filename(inputFilename)]);
};

Compile.prototype.generateAST = function (inputFilename, options) {
    // These options make Clang produce an AST dump
    var newOptions = options.concat(["-Xclang", "-ast-dump", "-fsyntax-only"]);

    let execOptions = this.getDefaultExecOptions();
    // A higher max output is needed for when the user includes headers
    execOptions.maxOutput = 1024 * 1024 * 1024;

    return this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions)
        .then(this.processAstOutput);
};

Compile.prototype.getOutputFilename = function (dirPath, outputFilebase) {
    return path.join(dirPath, outputFilebase + ".s"); // NB keep lower case as ldc compiler `tolower`s the output name
};

Compile.prototype.generateGccDump = function (inputFilename, options, gccDumpOptions) {
    // Maybe we should not force any RTL dump and let user hand-pick what he needs
    var addOpts = [];
    /* if not defined, consider it true */

    if (gccDumpOptions.treeDump !== false) {
        addOpts.push("-fdump-tree-all");
    }
    if (gccDumpOptions.rtlDump !== false) {
        addOpts.push("-fdump-rtl-all");
    }

    var newOptions = options.concat(addOpts);

    let execOptions = this.getDefaultExecOptions();
    // A higher max output is needed for when the user includes headers
    execOptions.maxOutput = 1024 * 1024 * 1024;

    return this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions)
        .then(_.bind(this.processGccDumpOutput, gccDumpOptions));
};

Compile.prototype.compile = function (source, options, backendOptions, filters) {
    var self = this;
    var optionsError = self.checkOptions(options);
    if (optionsError) return Promise.reject(optionsError);
    var sourceError = self.checkSource(source);
    if (sourceError) return Promise.reject(sourceError);

    // Don't run binary for unsupported compilers, even if we're asked.
    if (filters.binary && !self.compiler.supportsBinary) {
        delete filters.binary;
    }

    var key = JSON.stringify({
        compiler: this.compiler,
        source: source,
        options: options,
        backendOptions: backendOptions,
        filters: filters
    });

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
            var outputFilename = self.getOutputFilename(dirPath, outputFilebase);

            options = self.prepareArguments(options, filters, backendOptions, inputFilename, outputFilename);

            options = options.filter(_.identity);

            var execOptions = self.getDefaultExecOptions();

            var asmPromise = self.runCompiler(self.compiler.exe, options, self.filename(inputFilename), execOptions);

            var astPromise;
            if (backendOptions && backendOptions.produceAst) {
                if (self.couldSupportASTDump(options, self.compiler.version)) {
                    astPromise = self.generateAST(inputFilename, options);
                }
                else {
                    astPromise = Promise.resolve("AST output is only supported in Clang >= 3.3");
                }
            }
            else {
                astPromise = Promise.resolve("");
            }

            var gccDumpPromise;
            if (backendOptions && backendOptions.produceGccDump && backendOptions.produceGccDump.opened) {
                gccDumpPromise = self.generateGccDump(inputFilename, options, backendOptions.produceGccDump);
            }
            else {
                gccDumpPromise = Promise.resolve("");
            }

            return Promise.all([asmPromise, astPromise, gccDumpPromise])
                .then(function (results) {
                    var asmResult = results[0];
                    var astResult = results[1];
                    var gccDumpResult = results[2];

                    asmResult.dirPath = dirPath;
                    if (asmResult.code !== 0) {
                        asmResult.asm = "<Compilation failed>";
                        return asmResult;
                    }
                    asmResult.hasOptOutput = false;
                    if (self.compiler.supportsOptOutput && self.optOutputRequested(options)) {
                        asmResult.hasOptOutput = false;
                        const optPath = path.join(dirPath, outputFilebase + ".opt.yaml");
                        if (fs.existsSync(optPath)) {
                            asmResult.hasOptOutput = true;
                            asmResult.optPath = optPath;
                        }
                    }
                    if (astResult) {
                        asmResult.hasAstOutput = true;
                        asmResult.astOutput = astResult;
                    }

                    if (self.compiler.supportsGccDump && gccDumpResult) {
                        asmResult.hasGccDumpOutput = true;
                        asmResult.gccDumpOutput = gccDumpResult;
                    }

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
            .then(function (result) {
                return filters.demangle ? _.bind(self.postProcessAsm, self, result)() : result;
            })
            .then(function (result) {
                result.supportsCfg = false;
                if (result.code === 0 && !filters.binary && self.isCfgCompiler(self.compiler.version)) {
                    var cfg_ = new cfg.ControlFlowGraph(self.compiler.version);
                    result.cfg = cfg_.generateCfgStructure(result.asm);
                    result.supportsCfg = true;
                }
                return result;
            })
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
        .then(_.bind(function (demangleResult) {
            var lines = utils.splitLines(demangleResult.stdout);
            for (var i = 0; i < result.asm.length; ++i)
                result.asm[i].text = lines[i];
            return result;
        }, this));
};

Compile.prototype.processOptOutput = function (hasOptOutput, optPath) {
    var output = [],
        //if we have no compileFilename for whatever reason
        //let everything through
        inputFile = this.env.compilerProps("compileFilename", "");
    return new Promise(
        function (resolve) {
            fs.createReadStream(optPath, {encoding: "utf-8"})
                .pipe(new compilerOptInfo.LLVMOptTransformer())
                .on("data", function (opt) {
                    if (opt.DebugLoc &&
                        opt.DebugLoc.File &&
                        opt.DebugLoc.File.indexOf(inputFile) > -1) {

                        output.push(opt);
                    }
                }.bind(this))
                .on("end", function () {
                    if (this.compiler.demangler) {
                        var result = JSON.stringify(output, null, 4);
                        this.exec(this.compiler.demangler, ["-n", "-p"], {input: result})
                            .then(function (demangleResult) {
                                output = JSON.parse(demangleResult.stdout);
                                resolve(output);
                            }.bind(this))
                            .catch(function (exception) {
                                logger.warn("Caught exception " + exception + " during opt demangle parsing");
                                resolve(output);
                            }.bind(this));
                    } else {
                        resolve(output);
                    }
                }.bind(this));
        }.bind(this));
};

Compile.prototype.couldSupportASTDump = function (options, version) {
    var versionRegex = /version (\d.\d+)/;
    var versionMatch = versionRegex.exec(version);

    if (versionMatch) {
        var versionNum = parseFloat(versionMatch[1]);
        return version.toLowerCase().indexOf("clang") > -1 && versionNum >= 3.3;
    }

    return false;
};

Compile.prototype.isCfgCompiler = function (compilerVersion) {
    return compilerVersion.includes("clang") || compilerVersion.indexOf("g++") === 0;
};

Compile.prototype.processAstOutput = function (output) {
    output = output.stdout;
    output = output.map(function (x) {
        return x.text;
    });

    // Top level decls start with |- or `-
    var topLevelRegex = /^(\||`)-/;

    // Refers to the user's source file rather than a system header
    var sourceRegex = /<source>/g;

    // Refers to whatever the most recent file specified was
    var lineRegex = /<line:/;

    var mostRecentIsSource = false;

    // Remove all AST nodes which aren't directly from the user's source code
    for (var i = 0; i < output.length; ++i) {
        if (output[i].match(topLevelRegex)) {
            if (output[i].match(lineRegex) && mostRecentIsSource) {
                //do nothing
            }
            // This is a system header or implicit definition,
            // remove everything up to the next top level decl
            else if (!output[i].match(sourceRegex)) {
                // Top level decls with invalid sloc as the file don't change the most recent file
                let slocRegex = /<<invalid sloc>>/;
                if (!output[i].match(slocRegex)) {
                    mostRecentIsSource = false;
                }

                var spliceMax = i + 1;
                while (output[spliceMax] && !output[spliceMax].match(topLevelRegex)) {
                    spliceMax++;
                }
                output.splice(i, spliceMax - i);
                --i;
            }
            else {
                mostRecentIsSource = true;
            }
        }
    }

    output = output.join('\n');

    // Filter out the symbol addresses
    var addressRegex = /^([^A-Za-z]*[A-Za-z]+) 0x[a-z0-9]+/mg;
    output = output.replace(addressRegex, '$1');

    // Filter out <invalid sloc> and <<invalid sloc>>
    let slocRegex = / ?<?<invalid sloc>>?/g;
    output = output.replace(slocRegex, '');

    // Unify file references
    output = output.replace(sourceRegex, 'line');

    return output;
};

Compile.prototype.processGccDumpOutput = function (opts) {
    var rootDir = path.dirname(opts.inputFilename);
    var allFiles = fs.readdirSync(rootDir);
    var base = path.basename(opts.inputFilename);

    if (this.treeDump === false && this.rtlDump === false) {
        return {
            all: [],
            selectedPass: "",
            currentPassOutput: 'Nothing selected for dump:\nselect at least one of Tree/RTL filter',
            syntaxHighlight: false
        };
    }

    var allPasses = [];

    for (var i in allFiles) {
        var pass_str_idx = allFiles[i].indexOf(base + '.');
        if (pass_str_idx != -1) {
            allPasses.push(allFiles[i].substring(base.length + 1));
        }
    }

    var output = {
        all: allPasses,
        selectedPass: this.pass,
        currentPassOutput: '<No pass selected>',
        syntaxHighlight: false
    };

    if (this.pass) {
        var passDump = opts.inputFilename + "." + this.pass;

        if (fs.existsSync(passDump) && fs.statSync(passDump).isFile()) {
            output.currentPassOutput = fs.readFileSync(passDump, 'utf-8');
            if (output.currentPassOutput.match('^\s*$')) {
                output.currentPassOutput = 'File for selected pass is empty.';
            } else {
                output.syntaxHighlight = true;
            }
        } else {
            // most probably filter has changed and the request is outdated.
            output.currentPassOutput = "Pass '" + output.selectedPass + "' was requested\n";
            output.currentPassOutput += "but is not valid anymore with current filters.\n";
            output.currentPassOutput += "Please select another pass or change filters.\n";

            output.selectedPass = "";
        }
    }

    return output;
};

Compile.prototype.postProcess = function (result, outputFilename, filters) {
    var postProcess = this.compiler.postProcess.filter(_.identity);
    var maxSize = this.env.gccProps("max-asm-size", 8 * 1024 * 1024);
    var optPromise, asmPromise, execPromise;
    if (result.hasOptOutput) {
        optPromise = this.processOptOutput(result.hasOptOutput, result.optPath);
    } else {
        optPromise = Promise.resolve("");
    }

    if (filters.binary && this.supportsObjdump()) {
        asmPromise = this.objdump(outputFilename, result, maxSize, filters.intel, filters.demangle);
    } else {
        asmPromise = this.stat(outputFilename).then(_.bind(function (stat) {
                if (stat.size >= maxSize) {
                    result.asm = "<No output: generated assembly was too large (" + stat.size + " > " + maxSize + " bytes)>";
                    return result;
                }
                if (postProcess.length) {
                    const postCommand = 'cat "' + outputFilename + '" | ' + postProcess.join(" | ");
                    return this.exec("bash", ["-c", postCommand], {maxOutput: maxSize})
                        .then((postResult) => {
                            return this.handlePostProcessResult(result, postResult);
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

Compile.prototype.handlePostProcessResult = function (result, postResult) {
    result.asm = postResult.stdout;
    if (postResult.code !== 0) {
        result.asm = "<Error during post processing: " + postResult.code + ">";
        logger.error("Error during post-processing", result);
    }
    return result;
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

Compile.prototype.getArgumentParser = function () {
    let exe = this.compiler.exe.toLowerCase();
    if (exe.indexOf("clang") >= 0) {  // check this first as "clang++" matches "g++"
        return argumentParsers.clang;
    } else if (exe.indexOf("g++") >= 0 || exe.indexOf("gcc") >= 0) {
        return argumentParsers.gcc;
    }
    //there is a lot of code around that makes this assumption.
    //probably not the best thing to do :D
    return argumentParsers.gcc;
};

Compile.prototype.initialise = function () {
    if (this.getRemote()) return Promise.resolve(this);
    let argumentParser = this.getArgumentParser();
    let execOptions = this.getDefaultExecOptions();
    var compiler = this.compiler.exe;
    var versionFlag = this.compiler.versionFlag || '--version';
    var versionRe = new RegExp(this.compiler.versionRe || '.*');
    return this.env.enqueue(() => {
        logger.info("Gathering version information on", compiler);
        return this.exec(compiler, [versionFlag], execOptions);
    })
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
                return argumentParser(this).then(function (compiler) {
                    delete compiler.compiler.supportedOptions;
                    return compiler;
                });
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
        labels: true,
        optOutput: false
    };
};

module.exports = Compile;
