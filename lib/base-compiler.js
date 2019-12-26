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
    LlvmIrParser = require('./llvm-ir'),
    AsmParser = require('./asm-parser'),
    utils = require('./utils'),
    _ = require('underscore'),
    packager = require('./packager').Packager,
    exec = require('./exec'),
    logger = require('./logger').logger,
    compilerOptInfo = require("compiler-opt-info"),
    argumentParsers = require("./compilers/argument-parsers"),
    CompilerArguments = require("./compiler-arguments"),
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

        this.asm = new AsmParser(this.compilerProps);
        this.llvmIr = new LlvmIrParser(this.compilerProps);

        this.possibleArguments = new CompilerArguments(this.compiler.id);
        this.possibleTools = _.values(compilerInfo.tools);
        const demanglerExe = this.compiler.demangler;
        if (demanglerExe) {
            this.demanglerClass = require(this.compiler.demanglerClassFile).Demangler;
        }
        this.outputFilebase = "output";

        this.packager = new packager();
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

    getCompilerCacheKey(compiler, args, options) {
        return {mtime: this.mtime, compiler, args, options};
    }

    async execCompilerCached(compiler, args, options) {
        const key = this.getCompilerCacheKey(compiler, args, options);
        let result = await this.env.compilerCacheGet(key);
        if (!result) {
            result = await exec.execute(compiler, args, options);
            if (result.okToCache)
                this.env.compilerCachePut(key, result);
        }

        return result;
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

        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        return this.exec(compiler, options, execOptions).then(result => {
            result.inputFilename = inputFilename;
            const transformedInput = result.filenameTransform(inputFilename);
            result.stdout = utils.parseOutput(result.stdout, transformedInput);
            result.stderr = utils.parseOutput(result.stderr, transformedInput);
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
        const execOptions = {maxOutput: maxSize, customCwd: path.dirname(outputFilename)};

        return this.exec(this.compiler.objdumper, args, execOptions)
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
        return exec.sandbox(executable, executeParameters.args, {
            maxOutput: maxSize,
            timeoutMs: timeoutMs,
            ldPath: _.union(this.compiler.ldPath, executeParameters.ldPath).join(":"),
            input: executeParameters.stdin
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

    findLibVersion(selectedLib) {
        const foundLib = _.find(this.compiler.libs, (o, libId) => libId === selectedLib.id);
        if (!foundLib) return false;

        const foundVersion = _.find(foundLib.versions, (o, versionId) => versionId === selectedLib.version);
        return foundVersion;
    }

    findAutodetectStaticLibLink(linkname) {
        const foundLib = _.findKey(this.compiler.libs, (lib) => {
            return (lib.versions.autodetect && lib.versions.autodetect.staticliblink &&
                lib.versions.autodetect.staticliblink.includes(linkname));
        });
        if (!foundLib) return false;

        return {
            id: foundLib,
            version: "autodetect"
        };
    }

    getSortedStaticLibraries(libraries) {
        const dictionary = {};
        const links = _.uniq(_.flatten(_.map(libraries, (selectedLib) => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (!foundVersion) return false;

            return _.map(foundVersion.staticliblink, (lib) => {
                if (lib) {
                    dictionary[lib] = foundVersion;
                    return [lib, foundVersion.dependencies];
                } else {
                    return false;
                }
            });
        })));

        let sortedlinks = [];

        _.each(links, (libToInsertName) => {
            const libToInsertObj = dictionary[libToInsertName];

            let idxToInsert = sortedlinks.length;
            for (let idx = 0; idx < sortedlinks.length; idx++) {
                const libCompareName = sortedlinks[idx];
                const libCompareObj = dictionary[libCompareName];

                if (libToInsertObj && libCompareObj &&
                    _.intersection(libToInsertObj.dependencies, libCompareObj.staticliblink).length > 0) {
                    idxToInsert = idx;
                    break;
                } else if (libToInsertObj &&
                    libToInsertObj.dependencies.includes(libCompareName)) {
                    idxToInsert = idx;
                    break;
                } else if (libCompareObj &&
                    libCompareObj.dependencies.includes(libToInsertName)) {
                    continue;
                } else if (libToInsertObj &&
                    libToInsertObj.staticliblink.includes(libToInsertName) &&
                    libToInsertObj.staticliblink.includes(libCompareName)) {
                    if (libToInsertObj.staticliblink.indexOf(libToInsertName) >
                        libToInsertObj.staticliblink.indexOf(libCompareName)) {
                        continue;
                    } else {
                        idxToInsert = idx;
                    }
                    break;
                } else if (libCompareObj && libCompareObj.staticliblink.includes(libToInsertName) &&
                    libCompareObj.staticliblink.includes(libCompareName)) {
                    if (libCompareObj.staticliblink.indexOf(libToInsertName) >
                        libCompareObj.staticliblink.indexOf(libCompareName)) {
                        continue;
                    } else {
                        idxToInsert = idx;
                    }
                    break;
                }
            }

            if (idxToInsert < sortedlinks.length) {
                sortedlinks.splice(idxToInsert, 0, libToInsertName);
            } else {
                sortedlinks.push(libToInsertName);
            }
        });

        return sortedlinks;
    }

    getStaticLibraryLinks(libraries) {
        const linkFlag = this.compiler.linkFlag || "-l";

        return _.map(this.getSortedStaticLibraries(libraries), (lib) => {
            if (lib) {
                return linkFlag + lib;
            } else {
                return false;
            }
        });
    }

    getSharedLibraryLinks(libraries) {
        const linkFlag = this.compiler.linkFlag || "-l";

        return _.flatten(_.map(libraries, (selectedLib) => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (!foundVersion) return false;

            return _.map(foundVersion.liblink, (lib) => {
                if (lib) {
                    return linkFlag + lib;
                } else {
                    return false;
                }
            });
        }));
    }

    getSharedLibraryPaths(libraries) {
        return _.flatten(_.map(libraries, (selectedLib) => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (!foundVersion) return false;

            return foundVersion.libpath;
        }));
    }

    getSharedLibraryPathsAsArguments(libraries) {
        const pathFlag = this.compiler.rpathFlag || "-Wl,-rpath,";
        const libPathFlag = this.compiler.libpathFlag || "-L";

        return _.union(
            this.compiler.ldPath.map(path => pathFlag + path),
            this.getSharedLibraryPaths(libraries).map(path => pathFlag + path),
            this.getSharedLibraryPaths(libraries).map(path => libPathFlag + path));
    }

    getSharedLibraryPathsAsLdLibraryPaths(/*libraries*/) {
        return [];
    }

    getIncludeArguments(libraries) {
        const includeFlag = this.compiler.includeFlag || "-I";

        return _.flatten(_.map(libraries, (selectedLib) => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (!foundVersion) return false;

            return _.map(foundVersion.path, (path) => includeFlag + path);
        }));
    }

    prepareArguments(userOptions, filters, backendOptions, inputFilename, outputFilename, libraries) {
        let options = this.optionsForFilter(filters, outputFilename, userOptions);
        backendOptions = backendOptions || {};

        if (this.compiler.options) {
            options = options.concat(this.compiler.options.split(" "));
        }

        if (this.compiler.supportsOptOutput && backendOptions.produceOptInfo) {
            options = options.concat(this.compiler.optArg);
        }

        const libIncludes = this.getIncludeArguments(libraries);
        let libLinks = [];
        let libPaths = [];
        let staticLibLinks = [];

        if (filters.binary) {
            libLinks = this.getSharedLibraryLinks(libraries);
            libPaths = this.getSharedLibraryPathsAsArguments(libraries);
            staticLibLinks = this.getStaticLibraryLinks(libraries);
        }

        userOptions = this.filterUserOptions(userOptions) || [];
        return options.concat(libIncludes, libPaths, libLinks, userOptions,
            [this.filename(inputFilename)], staticLibLinks);
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

    generateIR(inputFilename, options, filters) {
        // These options make Clang produce an IR
        let newOptions = _.filter(options, option => option !== '-fcolor-diagnostics')
            .concat(this.compiler.irArg);

        let execOptions = this.getDefaultExecOptions();
        // A higher max output is needed for when the user includes headers
        execOptions.maxOutput = 1024 * 1024 * 1024;

        return this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions)
            .then((output) => {
                const ir = this.processIrOutput(output, filters);
                return ir.asm;
            });
    }

    processIrOutput(output, filters) {
        const irPath = this.getIrOutputFilename(output.inputFilename);
        if (fs.existsSync(irPath)) {
            const output = fs.readFileSync(irPath, 'utf-8');
            // uses same filters as main compiler
            return this.llvmIr.process(output, filters);
        }
        return this.llvmIr.process(output.stdout, filters);
    }

    getIrOutputFilename(inputFilename) {
        return inputFilename.replace(path.extname(inputFilename), '.ll');
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
        return fs.stat(outputFilename)
            .then(stat => asmResult.asmSize = stat.size)
            .catch(() => {
            })
            .then(() => this.postProcess(asmResult, outputFilename, filters));
    }

    runToolsOfType(tools, type, compilationInfo) {
        let tooling = [];
        if (tools) {
            tools.forEach((tool) => {
                const matches = this.possibleTools.filter(possibleTool => {
                    return possibleTool.getId() === tool.id &&
                        possibleTool.getType() === type;
                });

                if (matches[0]) {
                    const toolPromise = matches[0].runTool(compilationInfo,
                        compilationInfo.inputFilename, tool.args, tool.stdin);
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

    async buildExecutableInFolder(key, dirPath) {
        const inputFilename = path.join(dirPath, this.compileFilename);
        await fs.writeFile(inputFilename, key.source);

        const outputFilename = this.getExecutableFilename(dirPath, this.outputFilebase);

        const buildFilters = Object.assign({}, key.filters);
        buildFilters.binary = true;
        buildFilters.execute = true;

        const compilerArguments = _.compact(
            this.prepareArguments(key.options, buildFilters, key.backendOptions,
                inputFilename, outputFilename, key.libraries)
        );

        const execOptions = this.getDefaultExecOptions();
        execOptions.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths(key.libraries);

        const result = await this.buildExecutable(key.compiler.exe, compilerArguments, inputFilename,
            execOptions);

        result.executableFilename = outputFilename;
        result.compilationOptions = compilerArguments;
        return result;
    }

    async getOrBuildExecutable(key) {
        const dirPath = await this.newTempDir();
        const compilationResultFilename = "compilation-result.json";
        try {
            const outputFilename = await this.env.executableGet(key, dirPath);
            logger.debug(`Using cached package ${outputFilename}`);
            await this.packager.unpack(outputFilename, dirPath);
            const buildResults = JSON.parse(await fs.readFile(path.join(dirPath, compilationResultFilename)));
            return Object.assign({}, buildResults, {
                code: 0,
                inputFilename: path.join(dirPath, this.compileFilename),
                dirPath: dirPath,
                executableFilename: this.getExecutableFilename(dirPath, this.outputFilebase)
            });
        } catch (err) {
            logger.debug("Tried to get executable from cache, but got an error: ", {err});
        }
        const compilationResult = await this.buildExecutableInFolder(key, dirPath);
        if (compilationResult.code !== 0) {
            return compilationResult;
        }

        const packDir = await this.newTempDir();
        const packagedFile = path.join(packDir, "package.tgz");
        try {
            await fs.writeFile(path.join(dirPath, compilationResultFilename), JSON.stringify(compilationResult));
            await this.packager.package(dirPath, packagedFile);
            await this.env.executablePut(key, packagedFile);
        } catch (err) {
            logger.error("Caught an error trying to put to cache: ", {err});
        } finally {
            fs.remove(packDir);
        }
        return compilationResult;
    }

    runExecutable(executable, executeParameters) {
        const maxExecOutputSize = this.env.ceProps("max-executable-output-size", 32 * 1024);

        return this.execBinary(executable, maxExecOutputSize, executeParameters);
    }

    async handleExecution(key, executeParameters) {
        const buildResult = await this.getOrBuildExecutable(key);
        if (buildResult.code !== 0) {
            return {
                code: 0,
                didExecute: false,
                buildResult,
                stderr: [],
                stdout: []
            };
        }
        executeParameters.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths(key.libraries);
        const result = await this.runExecutable(buildResult.executableFilename, executeParameters);
        result.didExecute = true;
        result.buildResult = buildResult;
        return result;
    }

    getCacheKey(source, options, backendOptions, filters, tools, libraries) {
        return {compiler: this.compiler, source, options, backendOptions, filters, tools, libraries};
    }

    getCompilationInfo(key, result) {
        const compilationinfo = Object.assign({}, key, result);
        compilationinfo.outputFilename = this.getOutputFilename(result.dirPath, this.outputFilebase);
        compilationinfo.executableFilename = this.getExecutableFilename(result.dirPath, this.outputFilebase);

        return compilationinfo;
    }

    tryAutodetectLibraries(libsAndOptions) {
        const linkFlag = this.compiler.linkFlag || "-l";

        const detectedLibs = [];
        const foundlibOptions = [];
        _.each(libsAndOptions.options, (option) => {
            if (option.indexOf(linkFlag) === 0) {
                const libVersion = this.findAutodetectStaticLibLink(option.substr(linkFlag.length).trim());
                if (libVersion) {
                    foundlibOptions.push(option);
                    detectedLibs.push(libVersion);
                }
            }
        });

        if (detectedLibs.length > 0) {
            libsAndOptions.options = _.filter(libsAndOptions.options, (option) => !foundlibOptions.includes(option));
            libsAndOptions.libraries = _.union(libsAndOptions.libraries, detectedLibs);

            return true;
        } else {
            return false;
        }
    }

    compile(source, options, backendOptions, filters, bypassCache, tools, executionParameters, libraries) {
        const optionsError = this.checkOptions(options);
        if (optionsError) return Promise.reject(optionsError);
        const sourceError = this.checkSource(source);
        if (sourceError) return Promise.reject(sourceError);

        const libsAndOptions = {libraries, options};
        if (this.tryAutodetectLibraries(libsAndOptions)) {
            libraries = libsAndOptions.libraries;
            options = libsAndOptions.options;
        }

        // Don't run binary for unsupported compilers, even if we're asked.
        if (filters.binary && !this.compiler.supportsBinary) {
            delete filters.binary;
        }
        const executeParameters = {
            args: executionParameters.args || [],
            stdin: executionParameters.stdin || ""
        };
        const key = this.getCacheKey(source, options, backendOptions, filters, tools, libraries);

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
                        .then(async dirPath => {
                            const inputFilename = path.join(dirPath, this.compileFilename);
                            await fs.writeFile(inputFilename, source);
                            return {
                                inputFilename: inputFilename,
                                dirPath: dirPath
                            };
                        });
                    if (!backendOptions || !backendOptions.executorRequest) {
                        const compileToAsmPromise = tempFileAndDirPromise.then(info => {
                            const inputFilename = info.inputFilename;
                            const inputFilenameSafe = this.filename(inputFilename);
                            const dirPath = info.dirPath;
                            const outputFilename = this.getOutputFilename(dirPath, this.outputFilebase);

                            options = _.compact(
                                this.prepareArguments(options, filters, backendOptions,
                                    inputFilename, outputFilename, libraries)
                            );

                            const toolsPromise = this.runToolsOfType(tools, "independent",
                                this.getCompilationInfo(key, {inputFilename, dirPath, outputFilename}));

                            const execOptions = this.getDefaultExecOptions();
                            execOptions.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths(key.libraries);

                            const asmPromise = this.runCompiler(this.compiler.exe, options, inputFilenameSafe,
                                execOptions);

                            let astPromise;
                            if (backendOptions && backendOptions.produceAst && this.compiler.supportsAstView) {
                                astPromise = this.generateAST(inputFilename, options);
                            } else {
                                astPromise = Promise.resolve("");
                            }

                            let irPromise;
                            if (backendOptions && backendOptions.produceIr && this.compiler.supportsIrView) {
                                irPromise = this.generateIR(inputFilename, options, filters);
                            } else {
                                irPromise = Promise.resolve("");
                            }

                            let gccDumpPromise;
                            if (backendOptions && backendOptions.produceGccDump &&
                                backendOptions.produceGccDump.opened && this.compiler.supportsGccDump) {

                                gccDumpPromise = this.generateGccDump(
                                    inputFilename, options, backendOptions.produceGccDump);
                            } else {
                                gccDumpPromise = Promise.resolve("");
                            }

                            return Promise.all([
                                asmPromise,
                                astPromise,
                                gccDumpPromise,
                                irPromise,
                                Promise.all(toolsPromise)
                            ])
                                .then(([asmResult, astResult, gccDumpResult, irResult, toolsPromise]) => {
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

                                    if (this.compiler.supportsOptOutput && this.optOutputRequested(options)) {
                                        const optPath = path.join(dirPath, `${this.outputFilebase}.opt.yaml`);
                                        if (fs.existsSync(optPath)) {
                                            asmResult.hasOptOutput = true;
                                            asmResult.optPath = optPath;
                                        }
                                    }
                                    if (astResult) {
                                        asmResult.hasAstOutput = true;
                                        asmResult.astOutput = astResult;
                                    }
                                    if (irResult) {
                                        asmResult.hasIrOutput = true;
                                        asmResult.irOutput = irResult;
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
                                const postToolsPromise = this.runToolsOfType(tools, "postcompilation",
                                    this.getCompilationInfo(key, result));

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
                                    const res = this.processAsm(result, filters);
                                    result.asm = res.asm;
                                    result.labelDefinitions = res.labelDefinitions;
                                } else {
                                    result.asm = [{text: result.asm}];
                                }
                                return result;
                            })
                            .then(result => filters.demangle ? this.postProcessAsm(result, filters) : result)
                            .then(result => {
                                if (this.compiler.supportsCfg && backendOptions && backendOptions.produceCfg) {
                                    result.cfg = cfg.generateStructure(this.compiler.compilerType,
                                        this.compiler.version, result.asm);
                                }
                                return result;
                            })
                            .then(result => {
                                result.popularArguments = this.possibleArguments.getPopularArguments(options);

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
                    } else {
                        return this.handleExecution(key, executeParameters);
                    }
                });
            });
    }

    processAsm(result, filters) {
        if (this.llvmIr.isLlvmIr(result.asm)) {
            return this.llvmIr.process(result.asm, filters);
        }
        return this.asm.process(result.asm, filters);
    }

    postProcessAsm(result) {
        if (!result.okToCache || !this.demanglerClass || !result.asm) return result;
        const demangler = new this.demanglerClass(this.compiler.demangler, this);
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
        const versionRegex = /version (\d+.\d+)/;
        const versionMatch = versionRegex.exec(version);

        if (versionMatch) {
            const versionNum = parseFloat(versionMatch[1]);
            return version.toLowerCase().indexOf("clang") > -1 && versionNum >= 3.3;
        }

        return false;
    }

    isCfgCompiler(compilerVersion) {
        return compilerVersion.includes("clang") ||
            compilerVersion.match(/^([\w-]*-)?g((\+\+)|(cc)|(dc))/g) !== null;

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
                } else {
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

    async processGccDumpOutput(opts, result) {
        const rootDir = path.dirname(result.inputFilename);
        const allFiles = await fs.readdir(rootDir);
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

        for (let file of allFiles) {
            if (file.includes(`${base}.`)) {
                allPasses.push(file.substring(base.length + 1));
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

            if (fs.existsSync(passDump) && (await fs.stat(passDump)).isFile()) {
                output.currentPassOutput = await fs.readFile(passDump, 'utf-8');
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
        const maxSize = this.env.ceProps("max-asm-size", 64 * 1024 * 1024);
        let optPromise, asmPromise;
        if (result.hasOptOutput) {
            optPromise = this.processOptOutput(result.hasOptOutput, result.optPath);
        } else {
            optPromise = Promise.resolve("");
        }
        if (filters.binary && this.supportsObjdump()) {
            asmPromise = this.objdump(outputFilename, result, maxSize, filters.intel, filters.demangle);
        } else {
            asmPromise = (async () => {
                if (result.asmSize === undefined) {
                    result.asm = "<No output file>";
                    return result;
                }
                if (result.asmSize >= maxSize) {
                    result.asm = `<No output: generated assembly was too large (${result.asmSize} > ${maxSize} bytes)>`;
                    return result;
                }
                if (postProcess.length) {
                    return this.execPostProcess(result, postProcess, outputFilename, maxSize);
                } else {
                    const contents = await fs.readFile(outputFilename);
                    result.asm = contents.toString();
                    return result;
                }
            })();
        }

        return Promise.all([asmPromise, optPromise]);
    }

    handlePostProcessResult(result, postResult) {
        result.asm = postResult.stdout;
        if (postResult.code !== 0) {
            result.asm = `<Error during post processing: ${postResult.code}>`;
            logger.error("Error during post-processing: ", result);
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
        return this.execCompilerCached(this.compiler.exe, [versionFlag], execOptions);
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
                logger.error(`Unable to find compiler version for '${compiler}' with re ${versionRe}:`, result);
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
