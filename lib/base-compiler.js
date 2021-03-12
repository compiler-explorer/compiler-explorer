// Copyright (c) 2015, Compiler Explorer Authors
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

import * as compilerOptInfo from 'compiler-opt-info';
import fs from 'fs-extra';
import temp from 'temp';
import _ from 'underscore';

import { AsmParser } from './asm-parser';
import { getBuildEnvTypeByKey } from './buildenvsetup';
import * as cfg from './cfg';
import { CompilerArguments } from './compiler-arguments';
import { ClangParser, GCCParser } from './compilers/argument-parsers';
import { getDemanglerTypeByKey } from './demangler';
import * as exec from './exec';
import { languages } from './languages';
import { LlvmIrParser } from './llvm-ir';
import { logger } from './logger';
import { Packager } from './packager';
import { getToolchainPath } from './toolchain-utils';
import * as utils from './utils';

export class BaseCompiler {
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

        this.alwaysResetLdPath = this.env.ceProps('alwaysResetLdPath');
        this.delayCleanupTemp = this.env.ceProps('delayCleanupTemp', false);

        if (!this.compiler.options) this.compiler.options = '';
        if (!this.compiler.optArg) this.compiler.optArg = '';
        if (!this.compiler.supportsOptOutput) this.compiler.supportsOptOutput = false;

        if (!this.compiler.disabledFilters)
            this.compiler.disabledFilters = [];
        else if (typeof this.compiler.disabledFilters === 'string')
            this.compiler.disabledFilters = this.compiler.disabledFilters.split(',');

        this.asm = new AsmParser(this.compilerProps);
        this.llvmIr = new LlvmIrParser(this.compilerProps);

        this.toolchainPath = getToolchainPath(this.compiler.exe, this.compiler.options);

        this.possibleArguments = new CompilerArguments(this.compiler.id);
        this.possibleTools = _.values(compilerInfo.tools);
        const demanglerExe = this.compiler.demangler;
        if (demanglerExe && this.compiler.demanglerType) {
            this.demanglerClass = getDemanglerTypeByKey(this.compiler.demanglerType);
        }

        this.outputFilebase = 'output';

        this.mtime = null;
        this.buildenvsetup = null;
        if (this.compiler.buildenvsetup && this.compiler.buildenvsetup.id) {
            const buildenvsetupclass = getBuildEnvTypeByKey(this.compiler.buildenvsetup.id);
            this.buildenvsetup = new buildenvsetupclass(this.compiler, this.env, async (compiler, args, options) => {
                return this.execCompilerCached(compiler, args, options);
            });
        }

        this.packager = new Packager();
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
        return options.some(x => x === '-fsave-optimization-record');
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

    async execCompilerCached(compiler, args, options, useExecutionQueue) {
        if (!options) {
            options = this.getDefaultExecOptions();
            options.timeoutMs = 0;
            options.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths([]);
        }

        const key = this.getCompilerCacheKey(compiler, args, options);
        let result = await this.env.compilerCacheGet(key);
        if (!result) {
            const doExecute = async () => exec.execute(compiler, args, options);

            result = await (useExecutionQueue ? this.env.enqueue(doExecute) : doExecute());
            if (result.okToCache)
                this.env.compilerCachePut(key, result);
        }

        return result;
    }

    getDefaultExecOptions() {
        return {
            timeoutMs: this.env.ceProps('compileTimeoutMs', 7500),
            maxErrorOutput: this.env.ceProps('max-error-output', 5000),
            env: this.env.getEnv(this.compiler.needsMulti),
            wrapper: this.compilerProps('compiler-wrapper'),
        };
    }

    async runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        const result = await this.exec(compiler, options, execOptions);
        result.inputFilename = inputFilename;
        const transformedInput = result.filenameTransform(inputFilename);
        result.stdout = utils.parseOutput(result.stdout, transformedInput);
        result.stderr = utils.parseOutput(result.stderr, transformedInput);
        return result;
    }

    supportsObjdump() {
        return this.compiler.objdumper !== '';
    }

    async objdump(outputFilename, result, maxSize, intelAsm, demangle) {
        const args = ['-d', outputFilename, '-l', '--insn-width=16'];
        if (demangle) args.push('-C');
        if (intelAsm) args.push('-M', 'intel');
        const execOptions = {maxOutput: maxSize, customCwd: path.dirname(outputFilename)};

        const objResult = await this.exec(this.compiler.objdumper, args, execOptions);
        result.asm = objResult.stdout;
        if (objResult.code !== 0) {
            result.asm = `<No output: objdump returned ${objResult.code}>`;
        } else {
            result.objdumpTime = objResult.execTime;
        }
        return result;
    }

    async execBinary(executable, maxSize, executeParameters) {
        // We might want to save this in the compilation environment once execution is made available
        const timeoutMs = this.env.ceProps('binaryExecTimeoutMs', 2000);
        try {
            // TODO make config
            const execResult = await exec.sandbox(executable, executeParameters.args, {
                maxOutput: maxSize,
                timeoutMs: timeoutMs,
                ldPath: _.union(this.compiler.ldPath, executeParameters.ldPath).join(':'),
                input: executeParameters.stdin,
                additionalEnv: executeParameters.additionalEnv,
            });
            execResult.stdout = utils.parseOutput(execResult.stdout);
            execResult.stderr = utils.parseOutput(execResult.stderr);
            return execResult;
        } catch (err) {
            // TODO: is this the best way? Perhaps failures in sandbox shouldn't reject
            // with "results", but instead should play on?
            return {
                stdout: err.stdout ? utils.parseOutput(err.stdout) : [],
                stderr: err.stderr ? utils.parseOutput(err.stderr) : [],
                code: err.code !== undefined ? err.code : -1,
            };
        }
    }

    filename(fn) {
        return fn;
    }

    optionsForFilter(filters, outputFilename) {
        let options = ['-g', '-o', this.filename(outputFilename)];
        if (this.compiler.intelAsm && filters.intel && !filters.binary) {
            options = options.concat(this.compiler.intelAsm.split(' '));
        }
        if (!filters.binary) options = options.concat('-S');
        return options;
    }

    findLibVersion(selectedLib) {
        const foundLib = _.find(this.compiler.libs, (o, libId) => libId === selectedLib.id);
        if (!foundLib) return false;

        return _.find(foundLib.versions, (o, versionId) => (
            versionId === selectedLib.version ||
            (o.alias && o.alias.includes(selectedLib.version))));
    }

    findAutodetectStaticLibLink(linkname) {
        const foundLib = _.findKey(this.compiler.libs, (lib) => {
            return (lib.versions.autodetect && lib.versions.autodetect.staticliblink &&
                lib.versions.autodetect.staticliblink.includes(linkname));
        });
        if (!foundLib) return false;

        return {
            id: foundLib,
            version: 'autodetect',
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
            for (const [idx, libCompareName] of sortedlinks.entries()) {
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
        const linkFlag = this.compiler.linkFlag || '-l';

        return _.map(this.getSortedStaticLibraries(libraries), (lib) => {
            if (lib) {
                return linkFlag + lib;
            } else {
                return false;
            }
        });
    }

    getSharedLibraryLinks(libraries) {
        const linkFlag = this.compiler.linkFlag || '-l';

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
        const pathFlag = this.compiler.rpathFlag || '-Wl,-rpath,';
        const libPathFlag = this.compiler.libpathFlag || '-L';

        let toolchainLibraryPaths = [];
        if (this.toolchainPath) {
            toolchainLibraryPaths = [
                path.join(this.toolchainPath, '/lib64'),
                path.join(this.toolchainPath, '/lib32'),
            ];
        }

        return _.union(
            [libPathFlag + '.'],
            [pathFlag + '.'],
            this.compiler.libPath.map(path => pathFlag + path),
            toolchainLibraryPaths.map(path => pathFlag + path),
            this.getSharedLibraryPaths(libraries).map(path => pathFlag + path),
            this.getSharedLibraryPaths(libraries).map(path => libPathFlag + path));
    }

    getSharedLibraryPathsAsLdLibraryPaths(libraries) {
        let paths = [];
        if (!this.alwaysResetLdPath) {
            paths = process.env.LD_LIBRARY_PATH ? process.env.LD_LIBRARY_PATH : [];
        }
        return _.union(paths,
            this.compiler.ldPath,
            this.getSharedLibraryPaths(libraries));
    }

    getSharedLibraryPathsAsLdLibraryPathsForExecution(libraries) {
        let paths = [];
        if (!this.alwaysResetLdPath) {
            paths = process.env.LD_LIBRARY_PATH ? process.env.LD_LIBRARY_PATH : [];
        }
        return _.union(paths,
            this.compiler.ldPath,
            this.compiler.libPath,
            this.getSharedLibraryPaths(libraries));
    }

    getIncludeArguments(libraries) {
        const includeFlag = this.compiler.includeFlag || '-I';

        return _.flatten(_.map(libraries, (selectedLib) => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (!foundVersion) return false;

            return _.map(foundVersion.path, (path) => includeFlag + path);
        }));
    }

    getLibraryOptions(libraries) {
        return _.flatten(_.map(libraries, (selectedLib) => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (!foundVersion) return false;

            return foundVersion.options;
        }));
    }

    prepareArguments(userOptions, filters, backendOptions, inputFilename, outputFilename, libraries) {
        let options = this.optionsForFilter(filters, outputFilename, userOptions);
        backendOptions = backendOptions || {};

        if (this.compiler.options) {
            options = options.concat(utils.splitArguments(this.compiler.options));
        }

        if (this.compiler.supportsOptOutput && backendOptions.produceOptInfo) {
            options = options.concat(this.compiler.optArg);
        }

        const libIncludes = this.getIncludeArguments(libraries);
        const libOptions = this.getLibraryOptions(libraries);
        let libLinks = [];
        let libPaths = [];
        let staticLibLinks = [];

        if (filters.binary) {
            libLinks = this.getSharedLibraryLinks(libraries);
            libPaths = this.getSharedLibraryPathsAsArguments(libraries);
            staticLibLinks = this.getStaticLibraryLinks(libraries);
        }

        userOptions = this.filterUserOptions(userOptions) || [];
        return options.concat(libIncludes, libOptions, libPaths, libLinks, userOptions,
            [this.filename(inputFilename)], staticLibLinks);
    }

    filterUserOptions(userOptions) {
        return userOptions;
    }

    async generateAST(inputFilename, options) {
        // These options make Clang produce an AST dump
        const newOptions = _.filter(options, option => option !== '-fcolor-diagnostics')
            .concat(['-Xclang', '-ast-dump', '-fsyntax-only']);

        const execOptions = this.getDefaultExecOptions();
        // A higher max output is needed for when the user includes headers
        execOptions.maxOutput = 1024 * 1024 * 1024;

        return this.processAstOutput(
            await this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions));
    }

    async generateIR(inputFilename, options, filters) {
        // These options make Clang produce an IR
        const newOptions = _.filter(options, option => option !== '-fcolor-diagnostics')
            .concat(this.compiler.irArg);

        const execOptions = this.getDefaultExecOptions();
        // A higher max output is needed for when the user includes headers
        execOptions.maxOutput = 1024 * 1024 * 1024;

        const output = await this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions);
        if (output.code !== 0) {
            return [{text: 'Failed to run compiler to get IR code'}];
        }
        const ir = await this.processIrOutput(output, filters);
        return ir.asm;
    }

    async processIrOutput(output, filters) {
        const irPath = this.getIrOutputFilename(output.inputFilename);
        if (await fs.pathExists(irPath)) {
            const output = await fs.readFile(irPath, 'utf-8');
            // uses same filters as main compiler
            return this.llvmIr.process(output, filters);
        }
        return {
            asm: [{text: 'Internal error; unable to open output path'}],
            labelDefinitions: {},
        };
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

    async generateGccDump(inputFilename, options, gccDumpOptions) {
        // Maybe we should not force any RTL dump and let user hand-pick what he needs
        const addOpts = [];
        /* if not defined, consider it true */

        // Build dump options to append to the end of the -fdump command-line flag.
        // GCC accepts these options as a list of '-' separated names that may
        // appear in any order.
        var flags = '';
        if (gccDumpOptions.dumpFlags.address !== false) {
            flags += '-address';
        }
        if (gccDumpOptions.dumpFlags.slim !== false) {
            flags += '-slim';
        }
        if (gccDumpOptions.dumpFlags.raw !== false) {
            flags += '-raw';
        }
        if (gccDumpOptions.dumpFlags.details !== false) {
            flags += '-details';
        }
        if (gccDumpOptions.dumpFlags.stats !== false) {
            flags += '-stats';
        }
        if (gccDumpOptions.dumpFlags.blocks !== false) {
            flags += '-blocks';
        }
        if (gccDumpOptions.dumpFlags.vops !== false) {
            flags += '-vops';
        }
        if (gccDumpOptions.dumpFlags.lineno !== false) {
            flags += '-lineno';
        }
        if (gccDumpOptions.dumpFlags.uid !== false) {
            flags += '-uid';
        }
        if (gccDumpOptions.dumpFlags.all !== false) {
            flags += '-all';
        }

        if (gccDumpOptions.treeDump !== false) {
            addOpts.push('-fdump-tree-all' + flags);
        }
        if (gccDumpOptions.rtlDump !== false) {
            addOpts.push('-fdump-rtl-all' + flags);
        }
        if (gccDumpOptions.ipaDump !== false) {
            addOpts.push('-fdump-ipa-all' + flags);
        }

        const newOptions = options.concat(addOpts);

        const execOptions = this.getDefaultExecOptions();
        // A higher max output is needed for when the user includes headers
        execOptions.maxOutput = 1024 * 1024 * 1024;

        return this.processGccDumpOutput(
            gccDumpOptions,
            await this.runCompiler(this.compiler.exe, newOptions, this.filename(inputFilename), execOptions));
    }

    async checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters) {
        try {
            const stat = await fs.stat(outputFilename);
            asmResult.asmSize = stat.size;
        } catch (e) {
            // Ignore errors
        }
        return this.postProcess(asmResult, outputFilename, filters);
    }

    runToolsOfType(tools, type, compilationInfo) {
        let tooling = [];
        if (tools) {
            tools.forEach((tool) => {
                const matches = this.possibleTools.find(possibleTool => {
                    return possibleTool.getId() === tool.id &&
                        possibleTool.getType() === type;
                });

                if (matches) {
                    const toolPromise = matches.runTool(compilationInfo,
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

    async getRequiredLibraryVersions(libraries) {
        const libraryDetails = {};
        _.each(libraries, (selectedLib) => {
            const foundVersion = this.findLibVersion(selectedLib);
            if (foundVersion) libraryDetails[selectedLib.id] = foundVersion;
        });
        return libraryDetails;
    }

    async setupBuildEnvironment(key, dirPath) {
        if (this.buildenvsetup) {
            const libraryDetails = await this.getRequiredLibraryVersions(key.libraries);
            return this.buildenvsetup.setup(key, dirPath, libraryDetails);
        } else {
            return Promise.resolve();
        }
    }

    async buildExecutableInFolder(key, dirPath) {
        const buildEnvironment = this.setupBuildEnvironment(key, dirPath);

        const inputFilename = path.join(dirPath, this.compileFilename);
        const writerOfSource = fs.writeFile(inputFilename, key.source);

        const outputFilename = this.getExecutableFilename(dirPath, this.outputFilebase);

        const buildFilters = Object.assign({}, key.filters);
        buildFilters.binary = true;
        buildFilters.execute = true;

        const compilerArguments = _.compact(
            this.prepareArguments(key.options, buildFilters, key.backendOptions,
                inputFilename, outputFilename, key.libraries),
        );

        const execOptions = this.getDefaultExecOptions();
        execOptions.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths(key.libraries);

        await writerOfSource;
        const downloads = await buildEnvironment;
        const result = await this.buildExecutable(key.compiler.exe, compilerArguments, inputFilename,
            execOptions);

        result.downloads = downloads;

        result.executableFilename = outputFilename;
        result.compilationOptions = compilerArguments;
        return result;
    }

    async getOrBuildExecutable(key) {
        const dirPath = await this.newTempDir();

        const buildResults = await this.loadPackageWithExecutable(key, dirPath);
        if (buildResults) return buildResults;

        const compilationResult = await this.buildExecutableInFolder(key, dirPath);
        if (compilationResult.code !== 0) {
            return compilationResult;
        }

        await this.storePackageWithExecutable(key, dirPath, compilationResult);

        return compilationResult;
    }

    async loadPackageWithExecutable(key, dirPath) {
        const compilationResultFilename = 'compilation-result.json';
        try {
            const startTime = process.hrtime.bigint();
            const outputFilename = await this.env.executableGet(key, dirPath);
            if (outputFilename) {
                logger.debug(`Using cached package ${outputFilename}`);
                await this.packager.unpack(outputFilename, dirPath);
                const buildResults = JSON.parse(await fs.readFile(path.join(dirPath, compilationResultFilename)));
                const endTime = process.hrtime.bigint();
                return Object.assign({}, buildResults, {
                    code: 0,
                    inputFilename: path.join(dirPath, this.compileFilename),
                    dirPath: dirPath,
                    executableFilename: this.getExecutableFilename(dirPath, this.outputFilebase),
                    packageDownloadAndUnzipTime: ((endTime - startTime) / BigInt(1000000)).toString(),
                });
            }
            logger.debug('Tried to get executable from cache, but got a cache miss');
        } catch (err) {
            logger.error('Tried to get executable from cache, but got an error:', {err});
        }
        return false;
    }

    async storePackageWithExecutable(key, dirPath, compilationResult) {
        const compilationResultFilename = 'compilation-result.json';

        const packDir = await this.newTempDir();
        const packagedFile = path.join(packDir, 'package.tgz');
        try {
            await fs.writeFile(path.join(dirPath, compilationResultFilename), JSON.stringify(compilationResult));
            await this.packager.package(dirPath, packagedFile);
            await this.env.executablePut(key, packagedFile);
        } catch (err) {
            logger.error('Caught an error trying to put to cache: ', {err});
        } finally {
            fs.remove(packDir);
        }
    }

    runExecutable(executable, executeParameters) {
        const maxExecOutputSize = this.env.ceProps('max-executable-output-size', 32 * 1024);
        // Hardcoded fix for #2339. Ideally I'd have a config option for this, but for now this is plenty good enough.
        executeParameters.additionalEnv = {
            ASAN_OPTIONS: 'color=always',
            UBSAN_OPTIONS: 'color=always',
            MSAN_OPTIONS: 'color=always',
            LSAN_OPTIONS: 'color=always',
        };
        return this.execBinary(executable, maxExecOutputSize, executeParameters);
    }

    async handleExecution(key, executeParameters) {
        const buildResult = await this.getOrBuildExecutable(key);
        if (buildResult.code !== 0) {
            return {
                code: -1,
                didExecute: false,
                buildResult,
                stderr: [],
                stdout: [],
            };
        } else {
            if (!fs.existsSync(buildResult.executableFilename)) {
                const verboseResult = {
                    code: -1,
                    didExecute: false,
                    buildResult,
                    stderr: [],
                    stdout: [],
                };

                verboseResult.buildResult.stderr.push({text: 'Compiler did not produce an executable'});

                return verboseResult;
            }
        }

        if (!this.compiler.supportsExecute) {
            return {
                code: -1,
                didExecute: false,
                buildResult,
                stderr: [{text: 'Compiler does not support execution'}],
                stdout: [],
            };
        }

        executeParameters.ldPath = this.getSharedLibraryPathsAsLdLibraryPathsForExecution(key.libraries);
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
        compilationinfo.asmParser = this.asm;
        return compilationinfo;
    }

    tryAutodetectLibraries(libsAndOptions) {
        const linkFlag = this.compiler.linkFlag || '-l';

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

    async doCompilation(inputFilename, dirPath, key, options, filters, backendOptions, libraries, tools) {
        let buildEnvironment = Promise.resolve();
        if (filters.binary) {
            buildEnvironment = this.setupBuildEnvironment(key, dirPath);
        }

        const inputFilenameSafe = this.filename(inputFilename);
        const outputFilename = this.getOutputFilename(dirPath, this.outputFilebase);

        options = _.compact(
            this.prepareArguments(options, filters, backendOptions, inputFilename, outputFilename, libraries),
        );

        const execOptions = this.getDefaultExecOptions();
        execOptions.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths([]);

        const makeAst = backendOptions.produceAst && this.compiler.supportsAstView;
        const makeIr = backendOptions.produceIr && this.compiler.supportsIrView;
        const makeGccDump = backendOptions.produceGccDump && backendOptions.produceGccDump.opened
            && this.compiler.supportsGccDump;

        const downloads = await buildEnvironment;
        const [asmResult, astResult, gccDumpResult, irResult, toolsResult] = await Promise.all([
            this.runCompiler(this.compiler.exe, options, inputFilenameSafe, execOptions),
            (makeAst ? this.generateAST(inputFilename, options) : ''),
            (makeGccDump ? this.generateGccDump(inputFilename, options, backendOptions.produceGccDump) : ''),
            (makeIr ? this.generateIR(inputFilename, options, filters) : ''),
            Promise.all(this.runToolsOfType(tools, 'independent', this.getCompilationInfo(key, {
                inputFilename,
                dirPath,
                outputFilename,
            }))),
        ]);
        asmResult.dirPath = dirPath;
        asmResult.compilationOptions = options;
        asmResult.downloads = downloads;
        // Here before the check to ensure dump reports even on failure cases
        if (this.compiler.supportsGccDump && gccDumpResult) {
            asmResult.gccDumpOutput = gccDumpResult;
        }

        asmResult.tools = toolsResult;

        if (asmResult.code !== 0) {
            asmResult.asm = '<Compilation failed>';
            return [asmResult];
        }

        if (this.compiler.supportsOptOutput && this.optOutputRequested(options)) {
            const optPath = path.join(dirPath, `${this.outputFilebase}.opt.yaml`);
            if (await fs.pathExists(optPath)) {
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
    }

    doTempfolderCleanup(buildResult) {
        if (buildResult.dirPath && !this.delayCleanupTemp) {
            fs.remove(buildResult.dirPath);
        }
        buildResult.dirPath = undefined;
    }

    async compile(source, options, backendOptions, filters, bypassCache, tools, executionParameters, libraries) {
        const optionsError = this.checkOptions(options);
        if (optionsError) throw optionsError;
        const sourceError = this.checkSource(source);
        if (sourceError) throw sourceError;

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
            stdin: executionParameters.stdin || '',
        };
        const key = this.getCacheKey(source, options, backendOptions, filters, tools, libraries);

        const doExecute = filters.execute;
        filters = Object.assign({}, filters);
        filters.execute = false;

        if (!bypassCache) {
            const cacheRetreiveTimeStart = process.hrtime.bigint();
            const result = await this.env.cacheGet(key);
            if (result) {
                const cacheRetreiveTimeEnd = process.hrtime.bigint();
                result.retreivedFromCacheTime = ((cacheRetreiveTimeEnd - cacheRetreiveTimeStart) /
                    BigInt(1000000)).toString();
                result.retreivedFromCache = true;
                if (doExecute) {
                    result.execResult = await this.env.enqueue(async () => {
                        return this.handleExecution(key, executeParameters);
                    });

                    if (result.execResult && result.execResult.buildResult) {
                        this.doTempfolderCleanup(result.execResult.buildResult);
                    }
                }
                return result;
            }
        }

        return this.env.enqueue(async () => {
            source = this.preProcess(source);
            if (filters.binary && !source.match(this.compilerProps('stubRe'))) {
                source += '\n' + this.compilerProps('stubText') + '\n';
            }

            if (backendOptions.executorRequest) {
                const execResult = await this.handleExecution(key, executeParameters);
                if (execResult.buildResult) {
                    this.doTempfolderCleanup(execResult.buildResult);
                }
                return execResult;
            }

            const dirPath = await this.newTempDir();
            const inputFilename = path.join(dirPath, this.compileFilename);
            await fs.writeFile(inputFilename, source);

            // TODO make const when I can
            let [result, optOutput] = await this.doCompilation(
                inputFilename, dirPath, key, options, filters, backendOptions, libraries, tools);

            return await this.afterCompilation(result, doExecute, key, executeParameters, tools, backendOptions,
                filters, options, optOutput);
        });
    }

    async afterCompilation(result, doExecute, key, executeParameters, tools, backendOptions, filters, options,
        optOutput) {
        // Start the execution as soon as we can, but only await it at the end.
        const execPromise = doExecute ? this.handleExecution(key, executeParameters) : null;

        if (result.hasOptOutput) {
            delete result.optPath;
            result.optOutput = optOutput;
        }
        result.tools = _.union(result.tools, await Promise.all(this.runToolsOfType(tools, 'postcompilation',
            this.getCompilationInfo(key, result))));

        this.doTempfolderCleanup(result);
        if (result.buildResult) {
            this.doTempfolderCleanup(result.buildResult);
        }

        if (!backendOptions.skipAsm) {
            if (result.okToCache) {
                const res = this.processAsm(result, filters, options);
                result.asm = res.asm;
                result.labelDefinitions = res.labelDefinitions;
                result.parsingTime = res.parsingTime;
            } else {
                result.asm = [{text: result.asm}];
            }
            // TODO rephrase this so we don't need to reassign
            result = filters.demangle ? await this.postProcessAsm(result, filters) : result;
            if (this.compiler.supportsCfg && backendOptions.produceCfg) {
                result.cfg = cfg.generateStructure(this.compiler.compilerType,
                    this.compiler.version, result.asm);
            }
        } else {
            result.asm = [];
        }

        if (!backendOptions.skipPopArgs) result.popularArguments = this.possibleArguments.getPopularArguments(options);

        if (result.okToCache) {
            await this.env.cachePut(key, result);
        }

        if (doExecute) {
            result.execResult = await execPromise;

            if (result.execResult.buildResult) {
                this.doTempfolderCleanup(result.execResult.buildResult);
            }
        }
        return result;
    }

    processAsm(result, filters, options) {
        if ((options && options.includes('-emit-llvm')) || this.llvmIr.isLlvmIr(result.asm)) {
            return this.llvmIr.process(result.asm, filters);
        }

        return this.asm.process(result.asm, filters);
    }

    async postProcessAsm(result) {
        if (!result.okToCache || !this.demanglerClass || !result.asm) return result;
        const demangler = new this.demanglerClass(this.compiler.demangler, this);

        return demangler.process(result);
    }

    async processOptOutput(optPath) {
        const output = [];

        const optStream = fs.createReadStream(optPath, {encoding: 'utf-8'})
            .pipe(new compilerOptInfo.LLVMOptTransformer());

        for await (const opt of optStream) {
            if (opt.DebugLoc && opt.DebugLoc.File && opt.DebugLoc.File.includes(this.compileFilename)) {
                output.push(opt);
            }
        }

        if (this.compiler.demangler) {
            const result = JSON.stringify(output, null, 4);
            try {
                const demangleResult = await this.exec(
                    this.compiler.demangler, ['-n', '-p'], {input: result});
                return JSON.parse(demangleResult.stdout);
            } catch (exception) {
                // swallow exception and return non-demangled output
                logger.warn(`Caught exception ${exception} during opt demangle parsing`);
            }
        }

        return output;
    }

    couldSupportASTDump(version) {
        const versionRegex = /version (\d+.\d+)/;
        const versionMatch = versionRegex.exec(version);

        if (versionMatch) {
            const versionNum = parseFloat(versionMatch[1]);
            return version.toLowerCase().includes('clang') && versionNum >= 3.3;
        }

        return false;
    }

    isCfgCompiler(compilerVersion) {
        return compilerVersion.includes('clang') ||
            compilerVersion.match(/^([\w-]*-)?g((\+\+)|(cc)|(dc))/g) !== null;

    }

    processAstOutput(output) {
        output = output.stdout;
        output = output.map(x => x.text);

        // Top level decls start with |- or `-
        const topLevelRegex = /^([`|])-/;

        // Refers to the user's source file rather than a system header
        const sourceRegex = /<source>/g;

        // Refers to whatever the most recent file specified was
        const lineRegex = /<line:/;

        let mostRecentIsSource = false;

        // Remove all AST nodes which aren't directly from the user's source code
        for (let i = 0; i < output.length; ++i) {
            if (output[i].match(topLevelRegex)) {
                if (output[i].match(lineRegex) && mostRecentIsSource) {
                    // do nothing
                } else if (!output[i].match(sourceRegex)) {
                    // This is a system header or implicit definition,
                    // remove everything up to the next top level decl
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
        const addressRegex = /^([^A-Za-z]*[A-Za-z]+) 0x[\da-z]+/gm;
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

        if (opts.treeDump === false && opts.rtlDump === false && opts.ipaDump === false) {
            return {
                all: [],
                selectedPass: '',
                currentPassOutput: 'Nothing selected for dump:\nselect at least one of Tree/RTL filter',
                syntaxHighlight: false,
            };
        }

        const output = {
            all: [],
            selectedPass: opts.pass,
            currentPassOutput: '<No pass selected>',
            syntaxHighlight: false,
        };
        let passFound = false;
        // Phase letter is one of {i, l, r, t}
        // {outpufilename}.{extension}.{passNumber}{phaseLetter}.{phaseName}
        const dumpFilenameRegex = /^.+?\..+?\.(\d+?[ilrt]\..+)$/;
        for (let filename of allFiles) {
            const match = dumpFilenameRegex.exec(filename);
            if (match) {
                const pass = match[1];
                output.all.push(pass);
                const filePath = path.join(rootDir, filename);
                if (opts.pass === pass && (await fs.stat(filePath)).isFile()) {
                    passFound = true;
                    output.currentPassOutput = await fs.readFile(filePath, 'utf-8');
                    if (output.currentPassOutput.match(/^\s*$/)) {
                        output.currentPassOutput = 'File for selected pass is empty.';
                    } else {
                        output.syntaxHighlight = true;
                    }
                }
            }
        }

        if (opts.pass && !passFound) {
            output.currentPassOutput = `Pass '${opts.pass}' was requested
but is not valid anymore with current filters.
Please select another pass or change filters.`;
        }

        return output;
    }

    async execPostProcess(result, postProcesses, outputFilename, maxSize) {
        const postCommand = `cat "${outputFilename}" | ${postProcesses.join(' | ')}`;
        return this.handlePostProcessResult(
            result,
            await this.exec('bash', ['-c', postCommand], {maxOutput: maxSize}));
    }

    preProcess(source) {
        return source;
    }

    async postProcess(result, outputFilename, filters) {

        const postProcess = _.compact(this.compiler.postProcess);
        const maxSize = this.env.ceProps('max-asm-size', 64 * 1024 * 1024);
        const optPromise = result.hasOptOutput ? this.processOptOutput(result.optPath) : '';
        const asmPromise = (filters.binary && this.supportsObjdump())
            ? this.objdump(outputFilename, result, maxSize, filters.intel, filters.demangle)
            : (async () => {
                if (result.asmSize === undefined) {
                    result.asm = '<No output file>';
                    return result;
                }
                if (result.asmSize >= maxSize) {
                    result.asm = `<No output: generated assembly was too large (${result.asmSize} > ${maxSize} bytes)>`;
                    return result;
                }
                if (postProcess.length > 0) {
                    return this.execPostProcess(result, postProcess, outputFilename, maxSize);
                } else {
                    const contents = await fs.readFile(outputFilename);
                    result.asm = contents.toString();
                    return result;
                }
            })();
        return Promise.all([asmPromise, optPromise]);
    }

    handlePostProcessResult(result, postResult) {
        result.asm = postResult.stdout;
        if (postResult.code !== 0) {
            result.asm = `<Error during post processing: ${postResult.code}>`;
            logger.error('Error during post-processing: ', result);
        }
        return result;
    }

    checkOptions(options) {
        const error = this.env.findBadOptions(options);
        if (error.length > 0) return `Bad options: ${error.join(', ')}`;
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
        if (failed.length > 0) return failed.join('\n');
        return null;
    }

    getArgumentParser() {
        let exe = this.compiler.exe.toLowerCase();
        if (exe.includes('clang')
            || exe.includes('icpx')
            || exe.includes('icx')) {  // check this first as "clang++" matches "g++"
            return ClangParser;
        } else if (exe.includes('g++') || exe.includes('gcc')) {
            return GCCParser;
        }
        //there is a lot of code around that makes this assumption.
        //probably not the best thing to do :D
        return GCCParser;
    }

    getVersion() {
        logger.info(`Gathering ${this.compiler.id} version information on ${this.compiler.exe}`);
        const execOptions = this.getDefaultExecOptions();
        const versionFlag = this.compiler.versionFlag || '--version';
        execOptions.timeoutMs = 0; // No timeout for --version. A sort of workaround for slow EFS/NFS on the prod site
        execOptions.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths([]);

        try {
            return this.execCompilerCached(this.compiler.exe, [versionFlag], execOptions, true);
        } catch (err) {
            logger.error(`Unable to get version for compiler '${this.compiler.exe}' - ${err}`);
            return null;
        }
    }

    async initialise(mtime) {
        this.mtime = mtime;

        if (this.getRemote()) return this;
        const compiler = this.compiler.exe;
        const versionRe = new RegExp(this.compiler.versionRe || '.*', 'i');
        const result = await this.getVersion();
        if (!result) return null;
        if (result.code !== 0) {
            logger.warn(`Compiler '${compiler}' - non-zero result ${result.code}`);
        }
        let version = '';
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
    }

    getInfo() {
        return this.compiler;
    }

    getDefaultFilters() {
        return {
            binary: false,
            execute: false,
            demangle: true,
            intel: true,
            commentOnly: true,
            directives: true,
            labels: true,
            optOutput: false,
            libraryCode: false,
            trim: false,
        };
    }
}
