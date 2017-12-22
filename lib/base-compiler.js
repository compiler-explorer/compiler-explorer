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

const child_process = require('child_process'),
    temp = require('temp'),
    path = require('path'),
    fs = require('fs-extra'),
    denodeify = require('denodeify'),
    asm = require('./asm'),
    utils = require('./utils'),
    _ = require('underscore-node'),
    exec = require('./exec'),
    logger = require('./logger').logger,
    compilerOptInfo = require("compiler-opt-info"),
    argumentParsers = require("./compilers/argument-parsers"),
    cfg = require('./cfg'),
    languages = require('./languages').list;

function Compile(compiler, env, langId) {
    this.compiler = compiler;
    this.lang = langId;
    this.langInfo = languages[langId];
    if (!this.langInfo) {
        throw new Error("Missing language info for " + langId);
    }
    this.compileFilename = 'example' + this.langInfo.extensions[0];
    this.env = env;
    this.compilerProps = _.partial(this.env.compilerPropsL, this.lang);
    this.asm = new asm.AsmParser(this.compilerProps);
    this.compiler.supportsIntel = !!this.compiler.intelAsm;
}

Compile.prototype.newTempDir = function () {
    return new Promise(function (resolve, reject) {
        temp.mkdir({prefix: 'compiler-explorer-compiler', dir: process.env.tmpDir}, function (err, dirPath) {
            if (err)
                reject("Unable to open temp file: " + err);
            else
                resolve(dirPath);
        });
    });
};

Compile.prototype.writeFile = denodeify(fs.writeFile);
Compile.prototype.readFile = denodeify(fs.readFile);
Compile.prototype.stat = denodeify(fs.stat);

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
        timeoutMs: this.env.ceProps("compileTimeoutMs", 100),
        maxErrorOutput: this.env.ceProps("max-error-output", 5000),
        env: this.env.getEnv(this.compiler.needsMulti),
        wrapper: this.compilerProps("compiler-wrapper")
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
    let args = ["-d", outputFilename, "-l", "--insn-width=16"];
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
    let options = ['-g', '-o', this.filename(outputFilename)];
    if (this.compiler.intelAsm && filters.intel && !filters.binary) {
        options = options.concat(this.compiler.intelAsm.split(" "));
    }
    if (!filters.binary) options = options.concat('-S');
    return options;
};

Compile.prototype.prepareArguments = function (userOptions, filters, backendOptions, inputFilename, outputFilename) {
    let options = this.optionsForFilter(filters, outputFilename, userOptions);
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
    let newOptions = _.filter(options, option => option !== '-fcolor-diagnostics')
        .concat(["-Xclang", "-ast-dump", "-fsyntax-only"]);

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
    const addOpts = [];
    /* if not defined, consider it true */

    if (gccDumpOptions.treeDump !== false) {
        addOpts.push("-fdump-tree-all");
    }
    if (gccDumpOptions.rtlDump !== false) {
        addOpts.push("-fdump-rtl-all");
    }

    const newOptions = options.concat(addOpts);

    const execOptions = this.getDefaultExecOptions();
    // A higher max output is needed for when the user includes headers
    execOptions.maxOutput = 1024 * 1024 * 1024;

    return this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions)
        .then(result => this.processGccDumpOutput(gccDumpOptions, result));
};

Compile.prototype.compile = function (source, options, backendOptions, filters) {
    const optionsError = this.checkOptions(options);
    if (optionsError) return Promise.reject(optionsError);
    const sourceError = this.checkSource(source);
    if (sourceError) return Promise.reject(sourceError);

    // Don't run binary for unsupported compilers, even if we're asked.
    if (filters.binary && !this.compiler.supportsBinary) {
        delete filters.binary;
    }

    const key = JSON.stringify({
        compiler: this.compiler,
        source: source,
        options: options,
        backendOptions: backendOptions,
        filters: filters
    });

    const cached = this.env.cacheGet(key);
    if (cached) {
        return Promise.resolve(cached);
    }

    if (filters.binary && !source.match(this.compilerProps("stubRe"))) {
        source += "\n" + this.compilerProps("stubText") + "\n";
    }
    return this.env.enqueue(() => {
        const tempFileAndDirPromise = this.newTempDir()
            .then(dirPath => {
                const inputFilename = path.join(dirPath, this.compileFilename);
                return this.writeFile(inputFilename, source).then(() => ({
                    inputFilename: inputFilename,
                    dirPath: dirPath
                }));
            });

        const compileToAsmPromise = tempFileAndDirPromise.then(info => {
            const inputFilename = info.inputFilename;
            const dirPath = info.dirPath;
            const outputFilebase = "output";
            const outputFilename = this.getOutputFilename(dirPath, outputFilebase);

            options = _.compact(this.prepareArguments(options, filters, backendOptions, inputFilename, outputFilename));

            const execOptions = this.getDefaultExecOptions();

            const asmPromise = this.runCompiler(this.compiler.exe, options, this.filename(inputFilename), execOptions);

            let astPromise;
            if (backendOptions && backendOptions.produceAst) {
                if (this.couldSupportASTDump(options, this.compiler.version)) {
                    astPromise = this.generateAST(inputFilename, options);
                }
                else {
                    astPromise = Promise.resolve("AST output is only supported in Clang >= 3.3");
                }
            }
            else {
                astPromise = Promise.resolve("");
            }

            let gccDumpPromise;
            if (backendOptions && backendOptions.produceGccDump && backendOptions.produceGccDump.opened) {
                gccDumpPromise = this.generateGccDump(inputFilename, options, backendOptions.produceGccDump);
            }
            else {
                gccDumpPromise = Promise.resolve("");
            }

            return Promise.all([asmPromise, astPromise, gccDumpPromise])
                .then(([asmResult, astResult, gccDumpResult]) => {
                    asmResult.dirPath = dirPath;
                    if (asmResult.code !== 0) {
                        asmResult.asm = "<Compilation failed>";
                        return asmResult;
                    }
                    asmResult.hasOptOutput = false;
                    if (this.compiler.supportsOptOutput && this.optOutputRequested(options)) {
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

                    if (this.compiler.supportsGccDump && gccDumpResult) {
                        asmResult.hasGccDumpOutput = true;
                        asmResult.gccDumpOutput = gccDumpResult;
                    }
                    return this.postProcess(asmResult, outputFilename, filters);
                });
        });

        return compileToAsmPromise
            .then(results => {
                //TODO(jared): this isn't ideal. Rethink
                let result;
                let optOutput;
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
                    result.asm = this.asm.process(result.asm, filters);
                } else {
                    result.asm = {text: result.asm};
                }
                if (result.hasOptOutput) {
                    delete result.optPath;
                    result.optOutput = optOutput;
                }
                return result;
            })
            .then(result => filters.demangle ? this.postProcessAsm(result) : result)
            .then(result => {
                result.supportsCfg = false;
                if (result.code === 0 && !filters.binary && this.isCfgCompiler(this.compiler.version)) {
                    const cfg_ = new cfg.ControlFlowGraph(this.compiler.version);
                    result.cfg = cfg_.generateCfgStructure(result.asm);
                    result.supportsCfg = true;
                }
                return result;
            })
            .then(result => {
                if (result.okToCache) this.env.cachePut(key, result);
                return result;
            });
    });
};

Compile.prototype.postProcessAsm = function (result) {
    if (!result.okToCache) return result;
    const demangler = this.compiler.demangler;
    if (!demangler) return result;
    return this.exec(demangler, [], {input: _.pluck(result.asm, 'text').join("\n")})
        .then((demangleResult) => {
            const lines = utils.splitLines(demangleResult.stdout);
            for (let i = 0; i < result.asm.length; ++i)
                result.asm[i].text = lines[i];
            return result;
        });
};

Compile.prototype.processOptOutput = function (hasOptOutput, optPath) {
    let output = [];
    return new Promise(
        resolve => {
            fs.createReadStream(optPath, {encoding: "utf-8"})
                .pipe(new compilerOptInfo.LLVMOptTransformer())
                .on("data", opt => {
                    if (opt.DebugLoc &&
                        opt.DebugLoc.File &&
                        opt.DebugLoc.File.indexOf(this.compileFilename) > -1) {

                        output.push(opt);
                    }
                })
                .on("end", () => {
                    if (this.compiler.demangler) {
                        const result = JSON.stringify(output, null, 4);
                        this.exec(this.compiler.demangler, ["-n", "-p"], {input: result})
                            .then(demangleResult => resolve(JSON.parse(demangleResult.stdout)))
                            .catch(exception => {
                                logger.warn("Caught exception " + exception + " during opt demangle parsing");
                                resolve(output);
                            });
                    } else {
                        resolve(output);
                    }
                });
        });
};

Compile.prototype.couldSupportASTDump = function (options, version) {
    const versionRegex = /version (\d.\d+)/;
    const versionMatch = versionRegex.exec(version);

    if (versionMatch) {
        const versionNum = parseFloat(versionMatch[1]);
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
    const topLevelRegex = /^(\||`)-/;

    // Refers to the user's source file rather than a system header
    const sourceRegex = /<source>/g;

    // Refers to whatever the most recent file specified was
    const lineRegex = /<line:/;

    let mostRecentIsSource = false;

    // Remove all AST nodes which aren't directly from the user's source code
    for (let i = 0; i < output.length; ++i) {
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

                let spliceMax = i + 1;
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
    const addressRegex = /^([^A-Za-z]*[A-Za-z]+) 0x[a-z0-9]+/mg;
    output = output.replace(addressRegex, '$1');

    // Filter out <invalid sloc> and <<invalid sloc>>
    let slocRegex = / ?<?<invalid sloc>>?/g;
    output = output.replace(slocRegex, '');

    // Unify file references
    output = output.replace(sourceRegex, 'line');

    return output;
};

Compile.prototype.processGccDumpOutput = function (opts, result) {
    const rootDir = path.dirname(result.inputFilename);
    const allFiles = fs.readdirSync(rootDir);
    const base = path.basename(result.inputFilename);

    if (opts.treeDump === false && opts.rtlDump === false) {
        return {
            all: [],
            selectedPass: "",
            currentPassOutput: 'Nothing selected for dump:\nselect at least one of Tree/RTL filter',
            syntaxHighlight: false
        };
    }

    const allPasses = [];

    for (let i in allFiles) {
        const pass_str_idx = allFiles[i].indexOf(base + '.');
        if (pass_str_idx != -1) {
            allPasses.push(allFiles[i].substring(base.length + 1));
        }
    }

    const output = {
        all: allPasses,
        selectedPass: opts.pass,
        currentPassOutput: '<No pass selected>',
        syntaxHighlight: false
    };

    if (opts.pass) {
        const passDump = result.inputFilename + "." + opts.pass;

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
    const postProcess = _.compact(this.compiler.postProcess);
    const maxSize = this.env.ceProps("max-asm-size", 8 * 1024 * 1024);
    let optPromise, asmPromise, execPromise;
    if (result.hasOptOutput) {
        optPromise = this.processOptOutput(result.hasOptOutput, result.optPath);
    } else {
        optPromise = Promise.resolve("");
    }

    if (filters.binary && this.supportsObjdump()) {
        asmPromise = this.objdump(outputFilename, result, maxSize, filters.intel, filters.demangle);
    } else {
        asmPromise = this.stat(outputFilename).then(stat => {
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
            },
            () => {
                result.asm = "<No output file>";
                return result;
            }
        );
    }
    if (filters.execute) {
        const maxExecOutputSize = this.env.ceProps("max-executable-output-size", 32 * 1024);
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
    const error = this.env.findBadOptions(options);
    if (error.length > 0) return "Bad options: " + error.join(", ");
    return null;
};

// This check for arbitrary user-controlled preprocessor inclusions
// can be circumvented in more than one way. The goal here is to respond
// to simple attempts with a clear diagnostic; the service still needs to
// assume that malicious actors can make the compiler open arbitrary files.
Compile.prototype.checkSource = function (source) {
    const re = /^\s*#\s*i(nclude|mport)(_next)?\s+["<"](\/|.*\.\.)/;
    const failed = [];
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
    const argumentParser = this.getArgumentParser();
    const compiler = this.compiler.exe;
    const versionRe = new RegExp(this.compiler.versionRe || '.*');
    return this.env.enqueue(() => {
        logger.info("Gathering version information on", compiler);
        const execOptions = this.getDefaultExecOptions();
        const versionFlag = this.compiler.versionFlag || '--version';
        execOptions.timeoutMs = 0; // No timeout for --version. A sort of workaround for slow EFS/NFS on the prod site
        return this.exec(compiler, [versionFlag], execOptions);
    })
        .then(result => {
                if (result.code !== 0) {
                    logger.error("Unable to get version for compiler '" + compiler + "' - non-zero result " + result.code);
                    return null;
                }
                let version = "";
                _.each(utils.splitLines(result.stdout + result.stderr), line => {
                    if (version) return;
                    const match = line.match(versionRe);
                    if (match) version = match[0];
                });
                if (!version) {
                    logger.error("Unable to find compiler version for '" + compiler + "':", result,
                        'with re', versionRe);
                    return null;
                }
                logger.debug(compiler + " is version '" + version + "'");
                this.compiler.version = version;
                return argumentParser(this);
            },
            err => {
                logger.error("Unable to get version for compiler '" + compiler + "' - " + err);
                return null;
            });
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
