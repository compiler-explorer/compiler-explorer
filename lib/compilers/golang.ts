// Copyright (c) 2022, Compiler Explorer Authors
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

import fs from 'node:fs/promises';
import path from 'node:path';

import Semver from 'semver';
import _ from 'underscore';

import type {CacheKey, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import type {BuildEnvDownloadInfo} from '../buildenvsetup/buildenv.interfaces.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import * as utils from '../utils.js';
import {asSafeVer} from '../utils.js';
import {GolangParser} from './argument-parsers.js';

interface GoLibraryMetadata {
    module: string;
    version: string;
    go_mod_require: string;
    go_sum: string;
}

// Each arch has a list of jump instructions in
// Go source src/cmd/asm/internal/arch.
// x86 -> j, b
// arm -> cb, tb
// s390x -> cmpb, cmpub
const JUMP_RE = /^(j|b|cb|tb|cmpb|cmpub).*/i;
const LINE_RE = /^\s+(0[Xx]?[\dA-Za-z]+)?\s?(\d+)\s*\(([^:]+):(\d+)\)\s*([A-Z]+)(.*)/;
const UNKNOWN_RE = /^\s+(0[Xx]?[\dA-Za-z]+)?\s?(\d+)\s*\(<unknown line number>\)\s*([A-Z]+)(.*)/;
const FUNC_RE = /TEXT\s+[".]*(\S+)\(SB\)/;
const LOGGING_RE = /^[^:]+:\d+:(\d+:)?\s.*/;
const DECIMAL_RE = /(\s+)(\d+)(\s?)$/;

type GoEnv = {
    GOROOT?: string;
    GOARCH?: string;
    GOOS?: string;
};

export class GolangCompiler extends BaseCompiler {
    private readonly GOENV: GoEnv;
    private hasLibraries = false;
    private verboseBuild = false;

    static get key() {
        return 'golang';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);
        const group = this.compiler.group;

        let goroot = this.compilerProps<string | undefined>(
            'goroot',
            this.compilerProps<string | undefined>(`group.${group}.goroot`),
        );
        if (!goroot && compilerInfo.exe) {
            goroot = path.dirname(path.dirname(compilerInfo.exe));
        }

        const goarch = this.compilerProps<string | number | undefined>(
            'goarch',
            this.compilerProps<string | undefined>(`group.${group}.goarch`),
        );
        const goos = this.compilerProps<string | undefined>(
            'goos',
            this.compilerProps<string | undefined>(`group.${group}.goos`),
        );

        this.GOENV = {};
        if (goroot) {
            this.GOENV.GOROOT = goroot;
        }
        if (goarch) {
            this.GOENV.GOARCH = goarch.toString();
        }
        if (goos) {
            this.GOENV.GOOS = goos;
        }
    }

    async getSourceCachePath(): Promise<string | undefined> {
        if (!this.GOENV.GOROOT) return undefined;

        let sourceCachePath = path.join(this.GOENV.GOROOT, '..', 'cache');
        if (await utils.dirExists(sourceCachePath)) return sourceCachePath;

        sourceCachePath = path.join(this.GOENV.GOROOT, 'cache');
        if (await utils.dirExists(sourceCachePath)) return sourceCachePath;

        return undefined;
    }

    override async setupBuildEnvironment(
        key: CacheKey,
        dirPath: string,
        binary: boolean,
    ): Promise<BuildEnvDownloadInfo[]> {
        this.hasLibraries = key.libraries && key.libraries.length > 0;
        return super.setupBuildEnvironment(key, dirPath, binary);
    }

    protected async findDownloadedLibraries(dirPath: string): Promise<string[]> {
        const libraries: string[] = [];
        try {
            const entries = await fs.readdir(dirPath, {withFileTypes: true});
            for (const entry of entries) {
                if (entry.isDirectory()) {
                    const metadataPath = path.join(dirPath, entry.name, 'metadata.json');
                    if (await utils.fileExists(metadataPath)) {
                        libraries.push(entry.name);
                    }
                }
            }
        } catch {
            // Directory doesn't exist or can't be read
        }
        return libraries;
    }

    protected async readLibraryMetadata(dirPath: string, libId: string): Promise<GoLibraryMetadata | null> {
        try {
            const metadataPath = path.join(dirPath, libId, 'metadata.json');
            const content = await fs.readFile(metadataPath, 'utf-8');
            return JSON.parse(content) as GoLibraryMetadata;
        } catch (e) {
            logger.warn(`Failed to read metadata for Go library ${libId}: ${e}`);
            return null;
        }
    }

    protected async mergeGocache(cachePath: string, libCacheDeltaPath: string): Promise<void> {
        if (!(await utils.dirExists(libCacheDeltaPath))) return;
        await fs.cp(libCacheDeltaPath, cachePath, {recursive: true, force: false});
    }

    protected async setupModuleSources(goPath: string, libPath: string): Promise<void> {
        const moduleSourcesPath = path.join(libPath, 'module_sources');
        if (!(await utils.dirExists(moduleSourcesPath))) return;

        const pkgModPath = path.join(goPath, 'pkg', 'mod');
        await fs.mkdir(pkgModPath, {recursive: true});
        await fs.cp(moduleSourcesPath, pkgModPath, {recursive: true, force: false});
    }

    protected async generateGoMod(inputDir: string, libraries: string[], dirPath: string): Promise<void> {
        const goModPath = path.join(inputDir, 'go.mod');
        const goSumPath = path.join(inputDir, 'go.sum');

        let goModContent = '';
        let goSumContent = '';

        // Check if user provided their own go.mod
        const existingGoMod = await utils.fileExists(goModPath);
        if (existingGoMod) {
            goModContent = await fs.readFile(goModPath, 'utf-8');
        } else {
            goModContent = 'module example\n\ngo 1.21\n';
        }

        // Collect require statements and sum entries from all libraries
        const requireStatements: string[] = [];
        const sumEntries: string[] = [];

        for (const libId of libraries) {
            const metadata = await this.readLibraryMetadata(dirPath, libId);
            if (metadata) {
                if (metadata.go_mod_require) {
                    requireStatements.push(metadata.go_mod_require);
                }
                if (metadata.go_sum) {
                    sumEntries.push(metadata.go_sum);
                }
            }
        }

        // Append require statements to go.mod
        if (requireStatements.length > 0) {
            if (!goModContent.includes('require (')) {
                goModContent += '\nrequire (\n';
                goModContent += requireStatements.map(r => `\t${r}`).join('\n');
                goModContent += '\n)\n';
            } else {
                // Insert before the closing paren of require block
                goModContent = goModContent.replace(
                    /require \(([^)]*)\)/,
                    (match, inner) => `require (${inner}\n${requireStatements.map(r => `\t${r}`).join('\n')}\n)`,
                );
            }
        }

        await fs.writeFile(goModPath, goModContent);

        // Write go.sum
        if (sumEntries.length > 0) {
            goSumContent = sumEntries.join('\n') + '\n';
            await fs.writeFile(goSumPath, goSumContent);
        }
    }

    protected async setupGoLibraries(
        inputDir: string,
        cachePath: string,
        goPath: string,
        dirPath: string,
    ): Promise<void> {
        const libraries = await this.findDownloadedLibraries(dirPath);
        if (libraries.length === 0) return;

        for (const libId of libraries) {
            const libPath = path.join(dirPath, libId);

            // Merge cache_delta into GOCACHE
            const cacheDeltaPath = path.join(libPath, 'cache_delta');
            await this.mergeGocache(cachePath, cacheDeltaPath);

            // Copy module_sources to GOPATH/pkg/mod
            await this.setupModuleSources(goPath, libPath);
        }

        // Generate go.mod and go.sum
        await this.generateGoMod(inputDir, libraries, dirPath);
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        const inputDir = path.dirname(inputFilename);
        const dirPath = inputDir;
        const tempCachePath = path.join(inputDir, 'cache');
        const goPath = path.join(inputDir, 'gopath');

        execOptions.env = {
            ...execOptions.env,
            GOCACHE: tempCachePath,
            GOPATH: goPath,
        };

        this.verboseBuild = options.some(opt => opt === '-v' || opt === '-x');

        // Force offline compilation when libraries are selected
        if (this.hasLibraries) {
            execOptions.env.GOPROXY = 'off';
        }

        const sourceCachePath = await this.getSourceCachePath();
        if (sourceCachePath) {
            try {
                await fs.mkdir(tempCachePath, {recursive: true});
                await fs.cp(sourceCachePath, tempCachePath, {recursive: true, force: false});
            } catch {
                // Cache setup failed, continue without cache
            }
        }

        // Set up Go libraries if any were downloaded
        if (this.hasLibraries) {
            try {
                await fs.mkdir(goPath, {recursive: true});
                await this.setupGoLibraries(inputDir, tempCachePath, goPath, dirPath);
            } catch (e) {
                logger.warn(`Failed to set up Go libraries: ${e}`);
            }
        }

        return super.runCompiler(compiler, options, inputFilename, execOptions, filters);
    }

    convertNewGoL(code: ResultLine[]): string {
        let prevLine: string | null = null;
        let file: string | null = null;
        let fileCount = 0;
        let func: string | null = null;
        const funcCollisions: Record<string, number> = {};
        const labels: Record<string, boolean> = {};
        const usedLabels: Record<string, boolean> = {};
        const lines = code.map(obj => {
            let pcMatch: string | null = null;
            let fileMatch: string | null = null;
            let lineMatch: string | null = null;
            let ins: string | null = null;
            let args: string | null = null;

            const line = obj.text;
            let match = line.match(LINE_RE);
            if (match) {
                pcMatch = match[2];
                fileMatch = match[3];
                lineMatch = match[4];
                ins = match[5];
                args = match[6];
            } else {
                match = line.match(UNKNOWN_RE);
                if (match) {
                    pcMatch = match[2];
                    ins = match[3];
                    args = match[4];
                } else {
                    return [];
                }
            }

            match = line.match(FUNC_RE);
            if (match) {
                // Normalize function name.
                func = match[1].replaceAll(/[()*.]+/g, '_');

                // It's possible for normalized function names to collide.
                // Keep a count of collisions per function name. Labels get
                // suffixed with _[collisions] when collisions > 0.
                let collisions = funcCollisions[func];
                if (collisions == null) {
                    collisions = 0;
                } else {
                    collisions++;
                }

                funcCollisions[func] = collisions;
            }

            const res: string[] = [];
            if (pcMatch && !labels[pcMatch]) {
                // Create pseudo-label.
                let label = pcMatch.replace(/^0{0,4}/, '');
                let suffix = '';
                if (func && funcCollisions[func] > 0) {
                    suffix = `_${funcCollisions[func]}`;
                }

                label = `${func}_pc${label}${suffix}:`;
                if (!labels[label]) {
                    res.push(label);
                    labels[label] = true;
                }
            }

            if (fileMatch && file !== fileMatch) {
                fileCount++;
                res.push(`\t.file ${fileCount} "${fileMatch}"`);
                file = fileMatch;
            }

            if (lineMatch && prevLine !== lineMatch) {
                res.push(`\t.loc ${fileCount} ${lineMatch} 0`);
                prevLine = lineMatch;
            }

            if (func) {
                args = this.replaceJump(func, funcCollisions[func], ins, args, usedLabels);
                res.push(`\t${ins}${args}`);
            }
            return res;
        });

        // Find unused pseudo-labels so they can be filtered out.
        const unusedLabels = _.mapObject(labels, (val, key) => !usedLabels[key]);

        return lines
            .flat()
            .filter(line => !unusedLabels[line])
            .join('\n');
    }

    replaceJump(
        func: string,
        collisions: number,
        ins: string,
        args: string,
        usedLabels: Record<string, boolean>,
    ): string {
        // Check if last argument is a decimal number.
        const match = args.match(DECIMAL_RE);
        if (!match) {
            return args;
        }

        // Check instruction has a jump prefix
        if (JUMP_RE.test(ins)) {
            let label = `${func}_pc${match[2]}`;
            if (collisions > 0) {
                label += `_${collisions}`;
            }
            usedLabels[label + ':'] = true; // record label use for later filtering
            return `${match[1]}${label}${match[3]}`;
        }

        return args;
    }

    extractLogging(stdout: ResultLine[]): string {
        const filepath = `./${this.compileFilename}`;

        return stdout
            .filter(obj => obj.text.match(LOGGING_RE))
            .map(obj => obj.text.replace(filepath, '<source>'))
            .join('\n');
    }

    override async postProcess(result) {
        let out = result.stderr;
        if (this.compiler.id === '6g141') {
            out = result.stdout;
        }
        const logging = this.extractLogging(out);
        result.asm = this.convertNewGoL(out);
        result.stderr = this.verboseBuild
            ? out.filter(obj => !obj.text.match(LINE_RE) && !obj.text.match(UNKNOWN_RE) && !obj.text.match(LOGGING_RE))
            : [];
        result.stdout = utils.parseOutput(logging, result.inputFilename);
        return Promise.all([result, [], []]);
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override getIncludeArguments(libraries: object[]): string[] {
        // Go uses the module system, not include flags
        return [];
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        // If we're dealing with an older version...
        if (this.compiler.id === '6g141') {
            return ['tool', '6g', '-g', '-o', outputFilename, '-S'];
        }

        // -trimpath was introduced in Go 1.13
        const trimpath = Semver.gte(asSafeVer(this.compiler.semver), '1.13.0', true) ? ['-trimpath'] : [];

        if (filters.binary) {
            return ['build', ...trimpath, '-o', outputFilename, '-gcflags=' + unwrap(userOptions).join(' ')];
        }
        // Add userOptions to -gcflags to preserve previous behavior.
        return ['build', ...trimpath, '-o', outputFilename, '-gcflags=-S ' + unwrap(userOptions).join(' ')];
    }

    override filterUserOptions(userOptions: string[]) {
        if (this.compiler.id === '6g141') {
            return userOptions;
        }
        // userOptions are added to -gcflags in optionsForFilter
        return [];
    }

    override getDefaultExecOptions() {
        const options = {
            ...super.getDefaultExecOptions(),
        };

        options.env = {
            ...options.env,
            ...this.GOENV,
        };

        return options;
    }

    override getArgumentParserClass(): any {
        return GolangParser;
    }

    override isCfgCompiler() {
        // #6439: `gccgo` is ok, the default go compiler `gc` isn't
        return !this.compiler.version.includes('go version');
    }
}
