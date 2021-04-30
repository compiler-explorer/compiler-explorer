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
import { BuildEnvSetupBase } from './buildenvsetup/base';
import * as cfg from './cfg';
import { CompilerArguments } from './compiler-arguments';
import { ClangParser, GCCParser } from './compilers/argument-parsers';
import { getDemanglerTypeByKey } from './demangler';
import * as exec from './exec';
import { InstructionSets } from './instructionsets';
import { languages } from './languages';
import { LlvmAstParser } from './llvm-ast';
import { LlvmIrParser } from './llvm-ir';
import { logger } from './logger';
import { getObjdumperTypeByKey } from './objdumper';
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
        this.llvmAst = new LlvmAstParser(this.compilerProps);

        this.toolchainPath = getToolchainPath(this.compiler.exe, this.compiler.options);

        this.possibleArguments = new CompilerArguments(this.compiler.id);
        this.possibleTools = _.values(compilerInfo.tools);
        const demanglerExe = this.compiler.demangler;
        if (demanglerExe && this.compiler.demanglerType) {
            this.demanglerClass = getDemanglerTypeByKey(this.compiler.demanglerType);
        }
        const objdumperExe = this.compiler.objdumper;
        if (objdumperExe && this.compiler.objdumperType) {
            this.objdumperClass = getObjdumperTypeByKey(this.compiler.objdumperType);
        }

        this.outputFilebase = 'output';

        this.mtime = null;

        this.cmakeBaseEnv = {};

        this.buildenvsetup = null;
        if (this.compiler.buildenvsetup && this.compiler.buildenvsetup.id) {
            const buildenvsetupclass = getBuildEnvTypeByKey(this.compiler.buildenvsetup.id);
            this.buildenvsetup = new buildenvsetupclass(this.compiler, this.env, async (compiler, args, options) => {
                return this.execCompilerCached(compiler, args, options);
            });
        }

        if (!this.compiler.instructionSet) {
            const isets = new InstructionSets();
            if (this.buildenvsetup) {
                isets.getCompilerInstructionSetHint(this.buildenvsetup.compilerArch, this.compiler.exe).then(
                    (res) => this.compiler.instructionSet = res,
                ).catch(() => {});
            } else {
                const temp = new BuildEnvSetupBase(this.compiler, this.env);
                isets.getCompilerInstructionSetHint(temp.compilerArch, this.compiler.exe).then(
                    (res) => this.compiler.instructionSet = res,
                ).catch(() => {});
            }
        }

        this.packager = new Packager();
    }

    async getCmakeBaseEnv() {
        if (!this.compiler.exe) return {};

        const env = {};

        if (this.lang.id === 'c++') {
            env.CXX = this.compiler.exe;

            if (this.compiler.exe.endsWith('clang++')) {
                env.CC = this.compiler.exe.substr(0, this.compiler.exe.length - 2);
            } else if (this.compiler.exe.endsWith('g++')) {
                env.CC = this.compiler.exe.substr(0, this.compiler.exe.length - 2) + 'cc';
            }
        } else {
            env.CC = this.compiler.exe;
        }

        if (this.toolchainPath) {
            const ldPath = `${this.toolchainPath}/bin/ld`;
            const arPath = `${this.toolchainPath}/bin/ar`;
            const asPath = `${this.toolchainPath}/bin/as`;

            if (await utils.fileExists(ldPath)) env.LD = ldPath;
            if (await utils.fileExists(arPath)) env.AR = arPath;
            if (await utils.fileExists(asPath)) env.AS = asPath;
        }

        return env;
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
        return this.objdumperClass !== '';
    }

    getObjdumpOutputFilename(defaultOutputFilename) {
        return defaultOutputFilename;
    }

    postProcessObjdumpOutput(output) {
        return output;
    }

    async objdump(outputFilename, result, maxSize, intelAsm, demangle) {
        outputFilename = this.getObjdumpOutputFilename(outputFilename);
        const objdumper = new this.objdumperClass();
        const args = ['-d', outputFilename, '-l', ...objdumper.widthOptions];
        if (demangle) args.push('-C');
        if (intelAsm) args.push(...objdumper.intelAsmOptions);
        const execOptions = {maxOutput: maxSize, customCwd: path.dirname(outputFilename)};

        const objResult = await this.exec(this.compiler.objdumper, args, execOptions);
        if (objResult.code !== 0) {
            result.asm = `<No output: objdump returned ${objResult.code}>`;
        } else {
            result.objdumpTime = objResult.execTime;
            result.asm = this.postProcessObjdumpOutput(objResult.stdout);
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
                env: executeParameters.env,
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

        return this.llvmAst.processAst(
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

        if (key.files) {
            for (let file of key.files) {
                await fs.writeFile(this.getExtraFilepath(dirPath, file.filename), file.contents);
            }
        }

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
        executeParameters.env = {
            ASAN_OPTIONS: 'color=always',
            UBSAN_OPTIONS: 'color=always',
            MSAN_OPTIONS: 'color=always',
            LSAN_OPTIONS: 'color=always',
            ...executeParameters.env,
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
                stderr: [{text: 'Build failed'}],
                stdout: [],
            };
        } else {
            if (!await utils.fileExists(buildResult.executableFilename)) {
                const verboseResult = {
                    code: -1,
                    didExecute: false,
                    buildResult,
                    stderr: [{text: 'Executable not found'}],
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

    getCacheKey(source, options, backendOptions, filters, tools, libraries, files) {
        return {compiler: this.compiler, source, options, backendOptions, filters, tools, libraries, files};
    }

    getCmakeCacheKey(key, files) {
        const cacheKey = Object.assign({}, key);
        cacheKey.compiler = this.compiler;
        cacheKey.files = files;
        cacheKey.api = 'cmake';

        if (cacheKey.filters) delete cacheKey.filters.execute;
        delete cacheKey.executionParameters;
        delete cacheKey.tools;

        return cacheKey;
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

    getCompilerEnvironmentVariables(compilerflags) {
        if (this.lang.id === 'c++') {
            return { ...this.cmakeBaseEnv, CXXFLAGS: compilerflags };
        } else {
            return { ...this.cmakeBaseEnv, CCFLAGS: compilerflags };
        }
    }

    async doBuildstep(command, args, execParams) {
        const result = await this.exec(command, args, execParams);
        result.stdout = utils.parseOutput(result.stdout);
        result.stderr = utils.parseOutput(result.stderr);
        return result;
    }

    async doBuildstepAndAddToResult(result, name, command, args, execParams) {
        const stepResult = await this.doBuildstep(command, args, execParams);
        stepResult.step = name;
        logger.debug(name);
        result.buildsteps.push(stepResult);
        return stepResult;
    }

    createCmakeExecParams(execParams, dirPath, libsAndOptions) {
        const cmakeExecParams = Object.assign({}, execParams);

        const libIncludes = this.getIncludeArguments(libsAndOptions.libraries);
        const options = libsAndOptions.options.concat(libIncludes);

        _.extend(cmakeExecParams.env, this.getCompilerEnvironmentVariables(options.join(' ')));

        cmakeExecParams.env.LD_LIBRARY_PATH = dirPath;

        const libPaths = this.getSharedLibraryPathsAsArguments(libsAndOptions.libraries);
        cmakeExecParams.env.LDFLAGS = libPaths.join(' ');

        return cmakeExecParams;
    }

    createLibsAndOptions(key) {
        const libsAndOptions = {libraries: key.libraries, options: key.options};
        if (this.tryAutodetectLibraries(libsAndOptions)) {
            key.libraries = libsAndOptions.libraries;
            key.options = libsAndOptions.options;
        }
        return libsAndOptions;
    }

    async cmake(files, key) {
        // key = {source, options, backendOptions, filters, bypassCache, tools, executionParameters, libraries};

        if (!this.compiler.supportsBinary) {
            const errorResult = {
                code: -1,
                didExecute: false,
                stderr: [],
                stdout: [],
            };

            errorResult.stderr.push({text:'Compiler does not support compiling to binaries'});
            return errorResult;
        }

        _.defaults(key.filters, this.getDefaultFilters());
        key.filters.binary = true;

        const libsAndOptions = this.createLibsAndOptions(key);

        const doExecute = key.filters.execute;
        const executeParameters = {
            ldPath: this.getSharedLibraryPathsAsLdLibraryPaths(key.libraries),
            args: key.executionParameters.args || [],
            stdin: key.executionParameters.stdin || '',
        };

        const cacheKey = this.getCmakeCacheKey(key, files);

        const dirPath = await this.newTempDir();
        const outputFilename = this.getExecutableFilename(dirPath, this.outputFilebase);

        let fullResult = await this.loadPackageWithExecutable(cacheKey, dirPath);
        if (!fullResult) {
            const filesToWrite = [];
            filesToWrite.push(fs.writeFile(path.join(dirPath, 'CMakeLists.txt'), cacheKey.source));
            for (let file of files) {
                filesToWrite.push(fs.writeFile(this.getExtraFilepath(dirPath, file.filename), file.contents));
            }

            const execParams = this.getDefaultExecOptions();
            execParams.customCwd = dirPath; //path.join(dirPath, 'build');

            const makeExecParams = this.createCmakeExecParams(execParams, dirPath, libsAndOptions);

            await Promise.all(filesToWrite);

            fullResult = {
                buildsteps: [],
            };

            await this.setupBuildEnvironment(cacheKey, dirPath);

            let toolchainparam = '';
            if (this.toolchainPath) {
                toolchainparam = `-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=${this.toolchainPath}`;
            }

            await this.doBuildstepAndAddToResult(fullResult, 'cmake', this.env.ceProps('cmake'),
                [toolchainparam, '.'], makeExecParams);
            await this.doBuildstepAndAddToResult(fullResult, 'make', this.env.ceProps('make'),
                [], execParams);

            fullResult.result = {
                dirPath,
                okToCache: true,
            };

            const [asmResult] = await this.checkOutputFileAndDoPostProcess(
                fullResult.result, outputFilename, cacheKey.filters);
            fullResult.result = asmResult;

            await this.storePackageWithExecutable(cacheKey, dirPath, fullResult);
        } else {
            fullResult.fetchedFromCache = true;

            delete fullResult.code;
            delete fullResult.inputFilename;
            delete fullResult.dirPath;
            delete fullResult.executableFilename;
        }

        const optOutput = undefined;
        await this.afterCompilation(fullResult.result, false, cacheKey, [], cacheKey.tools, cacheKey.backendOptions,
            cacheKey.filters, libsAndOptions.options, optOutput);

        if (this.compiler.supportsExecute && doExecute) {
            fullResult.execResult = await this.runExecutable(outputFilename, executeParameters);
            fullResult.didExecute = true;
        }

        return fullResult;
    }

    getExtraFilepath(dirPath, filename) {
        // note: it's vitally important that the resulting path does not escape dirPath
        //       (filename is user input and thus unsafe)

        const sanere = /^[\s\w.-]+$/i;
        if (filename.match(sanere)) {
            const joined = path.join(dirPath, filename);
            const normalized = path.normalize(joined);
            if (process.platform === 'win32') {
                if (!normalized.replace(/\\/g, '/').startsWith(
                    dirPath.replace(/\\/g, '/'))
                ) {
                    throw new Error('Invalid filename');
                }
            } else {
                if (!normalized.startsWith(dirPath)) throw new Error('Invalid filename');
            }
            return normalized;
        } else {
            throw new Error('Invalid filename');
        }
    }

    async compile(source, options, backendOptions, filters, bypassCache, tools, executionParameters, libraries, files) {
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
        const key = this.getCacheKey(source, options, backendOptions, filters, tools, libraries, files);

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

            if (files) {
                for (let file of files) {
                    await fs.writeFile(this.getExtraFilepath(dirPath, file.filename), file.contents);
                }
            }

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
                result.filteredCount = res.filteredCount;
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

        try {
            this.cmakeBaseEnv = await this.getCmakeBaseEnv();
        } catch (e) {
            logger.error(e);
        }

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
