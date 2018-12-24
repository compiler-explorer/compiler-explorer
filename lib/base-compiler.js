// Copyright (c) 2015, Matt Godbolt
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
    path = require('path'),
    fs = require('fs-extra'),
    denodeify = require('denodeify'),
    AsmParser = require('./asm-parser'),
    utils = require('./utils'),
    _ = require('underscore'),
    exec = require('./exec'),
    logger = require('./logger').logger,
    compilerOptInfo = require("compiler-opt-info"),
    argumentParsers = require("./compilers/argument-parsers"),
    cfg = require('./cfg'),
    languages = require('./languages').list;

class BaseCompiler {
    constructor(compilerInfo, env) {
        // Information about our compiler
        this.compiler = compilerInfo;
        this.lang = languages[compilerInfo.lang];
        if (!this.lang) {
            throw new Error(`Missing language info for ${compilerInfo.lang}`);
        }
        this.compileFilename = `example${this.lang.extensions[0]}`;
        this.env = env;
        // Partial application of compilerProps with the proper language id applied to it
        this.compilerProps = _.partial(this.env.compilerProps, this.lang.id);
        this.compiler.supportsIntel = !!this.compiler.intelAsm;

        if (!this.compiler.options) this.compiler.options = "";
        if (!this.compiler.optArg) this.compiler.optArg = "";
        if (!this.compiler.supportsOptOutput) this.compiler.supportsOptOutput = false;

        if (!this.compiler.disabledFilters)
            this.compiler.disabledFilters = [];
        else if (typeof this.compiler.disabledFilters === "string")
            this.compiler.disabledFilters = this.compiler.disabledFilters.split(',');

        this.writeFile = denodeify(fs.writeFile);
        this.readFile = denodeify(fs.readFile);
        this.stat = denodeify(fs.stat);
        this.asm = new AsmParser(this.compilerProps);

        this.possibleTools = _.values(compilerInfo.tools);
        this.possibleLibs = compilerInfo.libs;
    }

    newTempDir() {
        return new Promise((resolve, reject) => {
            temp.mkdir({prefix: 'compiler-explorer-compiler', dir: process.env.tmpDir}, (err, dirPath) => {
                if (err)
                    reject(`Unable to open temp file: ${err}`);
                else
                    resolve(dirPath);
            });
        });
    }

    optOutputRequested(options) {
        return options.some(x => x === "-fsave-optimization-record");
    }

    getRemote() {
        if (this.compiler.exe === null && this.compiler.remote)
            return this.compiler.remote;
        return false;
    }

    exec(compiler, args, options) {
        // Here only so can be overridden by compiler implementations.
        return exec.execute(compiler, args, options);
    }

    getDefaultExecOptions() {
        return {
            timeoutMs: this.env.ceProps("compileTimeoutMs", 7500),
            maxErrorOutput: this.env.ceProps("max-error-output", 5000),
            env: this.env.getEnv(this.compiler.needsMulti),
            wrapper: this.compilerProps("compiler-wrapper")
        };
    }

    runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        return this.exec(compiler, options, execOptions).then(result => {
            result.inputFilename = inputFilename;
            result.stdout = utils.parseOutput(result.stdout, inputFilename);
            result.stderr = utils.parseOutput(result.stderr, inputFilename);
            return result;
        });
    }

    supportsObjdump() {
        return this.compiler.objdumper !== "";
    }

    objdump(outputFilename, result, maxSize, intelAsm, demangle) {
        let args = ["-d", outputFilename, "-l", "--insn-width=16"];
        if (demangle) args = args.concat("-C");
        if (intelAsm) args = args.concat(["-M", "intel"]);
        return this.exec(this.compiler.objdumper, args, {maxOutput: maxSize})
            .then(objResult => {
                result.asm = objResult.stdout;
                if (objResult.code !== 0) {
                    result.asm = `<No output: objdump returned ${objResult.code}>`;
                }
                return result;
            });
    }

    execBinary(executable, maxSize, executeParameters) {
        // We might want to save this in the compilation environment once execution is made available
        const timeoutMs = this.env.ceProps('binaryExecTimeoutMs', 2000);
        return exec.sandbox(executable, executeParameters, {
            maxOutput: maxSize,
            timeoutMs: timeoutMs
        })  // TODO make config
            .then(execResult => {
                execResult.stdout = utils.parseOutput(execResult.stdout);
                execResult.stderr = utils.parseOutput(execResult.stderr);
                return execResult;
            }).catch(err => {
                // TODO: is this the best way? Perhaps failures in sandbox shouldn't reject
                // with "results", but instead should play on?
                return {
                    stdout: err.stdout ? utils.parseOutput(err.stdout) : [],
                    stderr: err.stderr ? utils.parseOutput(err.stderr) : [],
                    code: err.code !== undefined ? err.code : -1
                };
            });
    }

    filename(fn) {
        return fn;
    }

    optionsForFilter(filters, outputFilename) {
        let options = ['-g', '-o', this.filename(outputFilename)];
        if (this.compiler.intelAsm && filters.intel && !filters.binary) {
            options = options.concat(this.compiler.intelAsm.split(" "));
        }
        if (!filters.binary) options = options.concat('-S');
        return options;
    }

    prepareArguments(userOptions, filters, backendOptions, inputFilename, outputFilename) {
        let options = this.optionsForFilter(filters, outputFilename, userOptions);
        backendOptions = backendOptions || {};

        if (this.compiler.options) {
            options = options.concat(this.compiler.options.split(" "));
        }

        if (this.compiler.supportsOptOutput && backendOptions.produceOptInfo) {
            options = options.concat(this.compiler.optArg);
        }

        userOptions = this.filterUserOptions(userOptions);
        return options.concat(userOptions || []).concat([this.filename(inputFilename)]);
    }


    filterUserOptions(userOptions) {
        return userOptions;
    }

    generateAST(inputFilename, options) {
        // These options make Clang produce an AST dump
        let newOptions = _.filter(options, option => option !== '-fcolor-diagnostics')
            .concat(["-Xclang", "-ast-dump", "-fsyntax-only"]);

        let execOptions = this.getDefaultExecOptions();
        // A higher max output is needed for when the user includes headers
        execOptions.maxOutput = 1024 * 1024 * 1024;

        return this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions)
            .then(this.processAstOutput);
    }

    getOutputFilename(dirPath, outputFilebase) {
        // NB keep lower case as ldc compiler `tolower`s the output name
        return path.join(dirPath, `${outputFilebase}.s`);
    }

    getExecutableFilename(dirPath, outputFilebase) {
        return this.getOutputFilename(dirPath, outputFilebase);
    }

    generateGccDump(inputFilename, options, gccDumpOptions) {
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
    }

    checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters) {
        return this.stat(outputFilename)
            .then(stat => asmResult.asmSize = stat.size)
            .catch(() => {
            })
            .then(() => this.postProcess(asmResult, outputFilename, filters));
    }

    runToolsOfType(sourcefile, tools, type, compilerExe, outputFilename, options, filters, asm) {
        let tooling = [];
        if (tools) {
            tools.forEach((tool) => {
                const matches = this.possibleTools.filter(possibleTool => {
                    return possibleTool.getId() === tool.id &&
                           possibleTool.getType() === type;
                });

                if (matches[0]) {
                    const toolPromise = matches[0].runTool(sourcefile, tool.args,
                        compilerExe, outputFilename, options, filters, asm);
                    tooling.push(toolPromise);
                }
            });
        }

        return tooling;
    }

    buildExecutable(compiler, options, inputFilename, execOptions) {
        // default implementation, but should be overridden by compilers
        return this.runCompiler(compiler, options, inputFilename, execOptions);
    }

    getOrBuildExecutable(key) {
        // todo: ask cache for executable if it already exists

        return this.newTempDir()
            .then(dirPath => {
                const inputFilename = path.join(dirPath, this.compileFilename);
                return this.writeFile(inputFilename, key.source).then(() => ({
                    inputFilename: inputFilename,
                    dirPath: dirPath
                }));
            }).then((dirResult) => {
                const outputFilebase = "output";
                const outputFilename = this.getExecutableFilename(dirResult.dirPath, outputFilebase);

                let buildFilters = Object.assign({}, key.filters);
                buildFilters.binary = true;
                buildFilters.execute = true;

                const compilerArguments = _.compact(
                    this.prepareArguments(key.options, buildFilters, key.backendOptions,
                        dirResult.inputFilename, outputFilename)
                );

                return this.buildExecutable(key.compiler.exe, compilerArguments,
                    dirResult.inputFilename, this.getDefaultExecOptions()).then(result => {
                    result.executableFilename = outputFilename;
                    return result;
                });
            });
    }

    runExecutable(executable, executeParameters) {
        const maxExecOutputSize = this.env.ceProps("max-executable-output-size", 32 * 1024);

        return this.execBinary(executable, maxExecOutputSize, executeParameters);
    }

    handleExecution(key, executeParameters) {
        return this.getOrBuildExecutable(key)
            .then(buildResult => {
                if (buildResult.code !== 0) {
                    return buildResult;
                } else {
                    return this.runExecutable(buildResult.executableFilename,
                        executeParameters);
                }
            });
    }

    getCacheKey(source, options, backendOptions, filters, tools) {
        return {compiler: this.compiler, source, options, backendOptions, filters, tools};
    }

    compile(source, options, backendOptions, filters, bypassCache, tools) {
        const optionsError = this.checkOptions(options);
        if (optionsError) return Promise.reject(optionsError);
        const sourceError = this.checkSource(source);
        if (sourceError) return Promise.reject(sourceError);

        // Don't run binary for unsupported compilers, even if we're asked.
        if (filters.binary && !this.compiler.supportsBinary) {
            delete filters.binary;
        }

        const executeParameters = options.executeParameters || [];
        delete options.executeParameters;

        const key = this.getCacheKey(source, options, backendOptions, filters, tools);

        const doExecute = filters.execute;
        filters = Object.assign({}, filters);
        filters.execute = false;

        const cacheGet = bypassCache ? Promise.resolve(null) : this.env.cacheGet(key);
        return cacheGet
            .then((result) => {
                if (result) {
                    if (doExecute) {
                        return this.handleExecution(key, executeParameters).then((execResult) => {
                            result.execResult = execResult;

                            return result;
                        });
                    }

                    return result;
                }

                source = this.preProcess(source);
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
                        const inputFilenameSafe = this.filename(inputFilename);
                        const dirPath = info.dirPath;
                        const outputFilebase = "output";
                        const outputFilename = this.getOutputFilename(dirPath, outputFilebase);

                        options = _.compact(
                            this.prepareArguments(options, filters, backendOptions, inputFilename, outputFilename)
                        );

                        const toolsPromise = this.runToolsOfType(
                            inputFilename, tools, "independent",
                            this.compiler.exe, outputFilename, options, filters);

                        const execOptions = this.getDefaultExecOptions();

                        const asmPromise = this.runCompiler(this.compiler.exe, options, inputFilenameSafe, execOptions);

                        let astPromise;
                        if (backendOptions && backendOptions.produceAst && this.compiler.supportsAstView) {
                            astPromise = this.generateAST(inputFilename, options);
                        } else {
                            astPromise = Promise.resolve("");
                        }

                        let gccDumpPromise;
                        if (backendOptions && backendOptions.produceGccDump && backendOptions.produceGccDump.opened &&
                            this.compiler.supportsGccDump) {

                            gccDumpPromise = this.generateGccDump(
                                inputFilename, options, backendOptions.produceGccDump);
                        }
                        else {
                            gccDumpPromise = Promise.resolve("");
                        }

                        return Promise.all([asmPromise, astPromise, gccDumpPromise, Promise.all(toolsPromise)])
                            .then(([asmResult, astResult, gccDumpResult, toolsPromise]) => {
                                asmResult.dirPath = dirPath;
                                asmResult.compilationOptions = options;
                                // Here before the check to ensure dump reports even on failure cases
                                if (this.compiler.supportsGccDump && gccDumpResult) {
                                    asmResult.gccDumpOutput = gccDumpResult;
                                }

                                asmResult.tools = toolsPromise;

                                if (asmResult.code !== 0) {
                                    asmResult.asm = "<Compilation failed>";
                                    return asmResult;
                                }
                                asmResult.hasOptOutput = false;
                                if (this.compiler.supportsOptOutput && this.optOutputRequested(options)) {
                                    const optPath = path.join(dirPath, `${outputFilebase}.opt.yaml`);
                                    if (fs.existsSync(optPath)) {
                                        asmResult.hasOptOutput = true;
                                        asmResult.optPath = optPath;
                                    }
                                }
                                if (astResult) {
                                    asmResult.hasAstOutput = true;
                                    asmResult.astOutput = astResult;
                                }

                                return this.checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters);
                            });
                    });

                    let executionPromise;

                    return compileToAsmPromise
                        .then(results => {
                            let optOutput;
                            let result;
                            if (results.length) {
                                result = results[0];
                                optOutput = results[1];
                            } else {
                                result = results;
                            }
                            if (result.hasOptOutput) {
                                delete result.optPath;
                                result.optOutput = optOutput;
                            }
                            return result;
                        })
                        .then(result => {
                            if (doExecute) {
                                executionPromise = this.handleExecution(key, executeParameters);
                            } else {
                                executionPromise = Promise.resolve(false);
                            }
                            return result;
                        })
                        .then(result => {
                            const outputFilebase = "output";
                            const outputFilename = this.getOutputFilename(result.dirPath, outputFilebase);

                            const postToolsPromise = this.runToolsOfType(
                                result.inputFilename, tools, "postcompilation",
                                this.compiler.exe, outputFilename, options, filters, result.asm);

                            return Promise.all([Promise.all(postToolsPromise)]).then(([postToolsResult]) => {
                                result.tools = _.union(result.tools, postToolsResult);

                                return result;
                            });
                        })
                        .then(result => {
                            if (result.dirPath) {
                                fs.remove(result.dirPath);
                                result.dirPath = undefined;
                            }
                            if (result.okToCache) {
                                result.asm = this.asm.process(result.asm, filters);
                            } else {
                                result.asm = [{text: result.asm}];
                            }
                            return result;
                        })
                        .then(result => filters.demangle ? this.postProcessAsm(result, filters) : result)
                        .then(result => {
                            if (this.compiler.supportsCfg && backendOptions && backendOptions.produceCfg) {
                                result.cfg = cfg.generateStructure(this.compiler.version, result.asm);
                            }
                            return result;
                        })
                        .then(result => {
                            if (result.okToCache) {
                                this.env.cachePut(key, result);
                            }

                            return Promise.all([executionPromise]).then(([executionResult]) => {
                                if (executionResult) {
                                    result.execResult = executionResult;
                                }

                                return result;
                            });
                        });
                });
            });
    }

    postProcessAsm(result) {
        if (!result.okToCache) return result;
        const demanglerExe = this.compiler.demangler;
        if (!demanglerExe) return result;

        const demanglerClass = require(this.compiler.demanglerClassFile).Demangler;

        const demangler = new demanglerClass(demanglerExe, this);
        return demangler.process(result);
    }

    processOptOutput(hasOptOutput, optPath) {
        let output = [];
        return new Promise(resolve => {
            fs.createReadStream(optPath, {encoding: "utf-8"})
                .pipe(new compilerOptInfo.LLVMOptTransformer())
                .on("data", opt => {
                    if (opt.DebugLoc && opt.DebugLoc.File && opt.DebugLoc.File.indexOf(this.compileFilename) > -1) {
                        output.push(opt);
                    }
                })
                .on("end", () => {
                    if (this.compiler.demangler) {
                        const result = JSON.stringify(output, null, 4);
                        this.exec(this.compiler.demangler, ["-n", "-p"], {input: result})
                            .then(demangleResult => resolve(JSON.parse(demangleResult.stdout)))
                            .catch(exception => {
                                logger.warn(`Caught exception ${exception} during opt demangle parsing`);
                                resolve(output);
                            });
                    } else {
                        resolve(output);
                    }
                });
        });
    }

    couldSupportASTDump(version) {
        const versionRegex = /version (\d.\d+)/;
        const versionMatch = versionRegex.exec(version);

        if (versionMatch) {
            const versionNum = parseFloat(versionMatch[1]);
            return version.toLowerCase().indexOf("clang") > -1 && versionNum >= 3.3;
        }

        return false;
    }

    isCfgCompiler(compilerVersion) {
        return compilerVersion.includes("clang") ||
            compilerVersion.indexOf("g++") === 0 ||
            compilerVersion.indexOf("gdc") === 0;
    }

    processAstOutput(output) {
        output = output.stdout;
        output = output.map(x => x.text);

        // Top level decls start with |- or `-
        const topLevelRegex = /^([|`])-/;

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
    }

    processGccDumpOutput(opts, result) {
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
            const pass_str_idx = allFiles[i].indexOf(`${base}.`);
            if (pass_str_idx !== -1) {
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
            const passDump = `${result.inputFilename}.${opts.pass}`;

            if (fs.existsSync(passDump) && fs.statSync(passDump).isFile()) {
                output.currentPassOutput = fs.readFileSync(passDump, 'utf-8');
                if (output.currentPassOutput.match(/^\s*$/)) {
                    output.currentPassOutput = 'File for selected pass is empty.';
                } else {
                    output.syntaxHighlight = true;
                }
            } else {
                // most probably filter has changed and the request is outdated.
                output.currentPassOutput = `Pass '${output.selectedPass}' was requested
but is not valid anymore with current filters.
Please select another pass or change filters.`;
                output.selectedPass = "";
            }
        }

        return output;
    }

    execPostProcess(result, postProcesses, outputFilename, maxSize) {
        const postCommand = `cat "${outputFilename}" | ${postProcesses.join(" | ")}`;
        return this.exec("bash", ["-c", postCommand], {maxOutput: maxSize})
            .then(postResult => this.handlePostProcessResult(result, postResult));
    }

    preProcess(source) {
        return source;
    }

    postProcess(result, outputFilename, filters) {
        const postProcess = _.compact(this.compiler.postProcess);
        const maxSize = this.env.ceProps("max-asm-size", 8 * 1024 * 1024);
        let optPromise, asmPromise;
        if (result.hasOptOutput) {
            optPromise = this.processOptOutput(result.hasOptOutput, result.optPath);
        } else {
            optPromise = Promise.resolve("");
        }
        if (filters.binary && this.supportsObjdump()) {
            asmPromise = this.objdump(outputFilename, result, maxSize, filters.intel, filters.demangle);
        } else {
            asmPromise = (() => {
                if (result.asmSize === undefined) {
                    result.asm = "<No output file>";
                    return Promise.resolve(result);
                }
                if (result.asmSize >= maxSize) {
                    result.asm = `<No output: generated assembly was too large (${result.asmSize} > ${maxSize} bytes)>`;
                    return Promise.resolve(result);
                }
                if (postProcess.length) {
                    return this.execPostProcess(result, postProcess, outputFilename, maxSize);
                } else {
                    return this.readFile(outputFilename).then(contents => {
                        result.asm = contents.toString();
                        return Promise.resolve(result);
                    });
                }
            })();
        }

        return Promise.all([asmPromise, optPromise]);
    }

    handlePostProcessResult(result, postResult) {
        result.asm = postResult.stdout;
        if (postResult.code !== 0) {
            result.asm = `<Error during post processing: ${postResult.code}>`;
            logger.error("Error during post-processing", result);
        }
        return result;
    }

    checkOptions(options) {
        const error = this.env.findBadOptions(options);
        if (error.length > 0) return `Bad options: ${error.join(", ")}`;
        return null;
    }

    // This check for arbitrary user-controlled preprocessor inclusions
    // can be circumvented in more than one way. The goal here is to respond
    // to simple attempts with a clear diagnostic; the service still needs to
    // assume that malicious actors can make the compiler open arbitrary files.
    checkSource(source) {
        const re = /^\s*#\s*i(nclude|mport)(_next)?\s+["<](\/|.*\.\.)[">]/;
        const failed = [];
        utils.splitLines(source).forEach((line, index) => {
            if (line.match(re)) {
                failed.push(`<stdin>:${index + 1}:1: no absolute or relative includes please`);
            }
        });
        if (failed.length > 0) return failed.join("\n");
        return null;
    }

    getArgumentParser() {
        let exe = this.compiler.exe.toLowerCase();
        if (exe.indexOf("clang") >= 0) {  // check this first as "clang++" matches "g++"
            return argumentParsers.Clang;
        } else if (exe.indexOf("g++") >= 0 || exe.indexOf("gcc") >= 0) {
            return argumentParsers.GCC;
        }
        //there is a lot of code around that makes this assumption.
        //probably not the best thing to do :D
        return argumentParsers.GCC;
    }

    getVersion() {
        logger.info(`Gathering ${this.compiler.id} version information on ${this.compiler.exe}`);
        const execOptions = this.getDefaultExecOptions();
        const versionFlag = this.compiler.versionFlag || '--version';
        execOptions.timeoutMs = 0; // No timeout for --version. A sort of workaround for slow EFS/NFS on the prod site
        return this.exec(this.compiler.exe, [versionFlag], execOptions);
    }

    initialise() {
        if (this.getRemote()) return Promise.resolve(this);
        const compiler = this.compiler.exe;
        const versionRe = new RegExp(this.compiler.versionRe || '.*', 'i');
        return this.env.enqueue(() => {
            return this.getVersion();
        }).then(result => {
            if (result.code !== 0) {
                logger.warn(`Compiler '${compiler}' - non-zero result ${result.code}`);
            }
            let version = "";
            _.each(utils.splitLines(result.stdout + result.stderr), line => {
                if (version) return;
                const match = line.match(versionRe);
                if (match) version = match[0];
            });
            if (!version) {
                logger.error(`Unable to find compiler version for '${compiler}':`, result, 'with re', versionRe);
                return null;
            }
            logger.debug(`${compiler} is version '${version}'`);
            this.compiler.version = version;
            this.compiler.supportsCfg = this.isCfgCompiler(version);
            this.compiler.supportsAstView = this.couldSupportASTDump(version);
            return this.getArgumentParser().parse(this);
        }, err => {
            logger.error(`Unable to get version for compiler '${compiler}' - ${err}`);
            return null;
        });
    }

    getInfo() {
        return this.compiler;
    }

    getDefaultFilters() {
        // TODO; propagate to UI?
        return {
            intel: true,
            commentOnly: true,
            directives: true,
            labels: true,
            optOutput: false
        };
    }
}

module.exports = BaseCompiler;
