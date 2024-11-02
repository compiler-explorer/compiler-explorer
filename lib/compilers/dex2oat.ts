// Copyright (c) 2023, Compiler Explorer Authors
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
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS'
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

import fs from 'fs-extra';
import _ from 'underscore';

import type {ParsedAsmResult, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {
    OptPipelineBackendOptions,
    OptPipelineOutput,
} from '../../types/compilation/opt-pipeline-output.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler, SimpleOutputFilenameCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {Dex2OatPassDumpParser} from '../parsers/dex2oat-pass-dump-parser.js';
import * as utils from '../utils.js';

export class Dex2OatCompiler extends BaseCompiler {
    static get key() {
        return 'dex2oat';
    }

    passDumpParser: Dex2OatPassDumpParser;

    insnSetRegex: RegExp;
    insnSetFeaturesRegex: RegExp;

    compilerFilterRegex: RegExp;
    classRegex: RegExp;
    methodRegex: RegExp;
    methodSizeRegex: RegExp;
    insnRegex: RegExp;
    stackMapRegex: RegExp;

    insnSetArgRegex: RegExp;
    compilerFilterArgRegex: RegExp;
    fullOutputArgRegex: RegExp;

    versionPrefixRegex: RegExp;
    latestVersionRegex: RegExp;

    fullOutput: boolean;

    d8Id: string;
    artArtifactDir: string;
    profmanPath: string;

    libs: SelectedLibraryVersion[];

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super({...compilerInfo}, env);
        this.compiler.optPipeline = {
            arg: ['-print-after-all', '-print-before-all'],
            moduleScopeArg: ['-print-module-scope'],
            noDiscardValueNamesArg: [],
        };

        this.passDumpParser = new Dex2OatPassDumpParser();

        // These must be used before the output is split on newlines.
        this.insnSetRegex = /\s*INSTRUCTION\s+SET:\s*\n(.+?)\n/;
        this.insnSetFeaturesRegex = /\s*INSTRUCTION\s+SET\s+FEATURES:\s*\n(.+?)\n/;

        // These must be used after the output is split on newlines.
        this.compilerFilterRegex = /^compiler-filter\s=\s(.*)$/;
        this.classRegex = /^\s*\d+:\s+L(.*);\s+\(offset=0x\w+\)\s+\(type_idx=\d+\).*$/;
        this.methodRegex = /^\s+\d+:\s+(.*)\s+\(dex_method_idx=\d+\)$/;
        this.methodSizeRegex = /^\s+CODE:\s+\(code_offset=0x\w+\s+size=(\d+).*$/;
        this.insnRegex = /^\s+(0x\w+):\s+\w+\s+(.*)$/;
        // eslint-disable-next-line unicorn/better-regex
        this.stackMapRegex = /^\s+(StackMap\[\d+\])\s+\((.*)\).*$/;

        // ART version codes in CE are in the format of AABB, where AA is the
        // API level and BB is the number of months since the initial release.
        this.versionPrefixRegex = /^(java|kotlin)-dex2oat-(\d\d)\d+$/;
        this.latestVersionRegex = /^(java|kotlin)-dex2oat-latest$/;

        // User-provided arguments (with a default behavior if not provided).
        this.insnSetArgRegex = /^--instruction-set=.*$/;
        this.compilerFilterArgRegex = /^--compiler-filter=.*$/;

        // Whether the full dex2oat output should be displayed instead of just
        // the parsed and formatted methods.
        this.fullOutputArgRegex = /^--full-output$/;
        this.fullOutput = false;

        // The underlying D8 version+exe.
        this.d8Id = this.compilerProps<string>(`compiler.${this.compiler.id}.d8Id`);

        // The directory containing ART artifacts necessary for dex2oat to run.
        this.artArtifactDir = this.compilerProps<string>(`compiler.${this.compiler.id}.artArtifactDir`);

        // The path to the `profman` binary.
        this.profmanPath = this.compilerProps<string>(`compiler.${this.compiler.id}.profmanPath`);

        // Libraries that will flow to D8Compiler and Java/KotlinCompiler.
        this.libs = [];
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        // Make sure --full-output from previous invocations doesn't persist.
        this.fullOutput = false;

        // Instantiate D8 compiler, which will in turn instantiate a Java or
        // Kotlin compiler based on the current language.
        const d8Compiler = unwrap(
            global.handler_config.compileHandler.findCompiler(this.lang.id, this.d8Id),
        ) as BaseCompiler & SimpleOutputFilenameCompiler;
        if (!d8Compiler) {
            return {
                ...this.handleUserError(
                    {message: `Compiler ${this.lang.id} ${this.d8Id} not configured correctly`},
                    '',
                ),
                timedOut: false,
            };
        }
        const d8DirPath = path.dirname(inputFilename);
        const d8OutputFilename = d8Compiler.getOutputFilename(d8DirPath);
        const d8Options = _.compact(
            d8Compiler.prepareArguments(
                [''], //options
                d8Compiler.getDefaultFilters(),
                {}, // backendOptions
                inputFilename,
                d8OutputFilename,
                this.libs,
                [], // overrides
            ),
        );

        const compileResult = await d8Compiler.runCompiler(
            d8Compiler.getInfo().exe,
            d8Options,
            this.filename(inputFilename),
            d8Compiler.getDefaultExecOptions(),
        );

        if (compileResult.code !== 0) {
            return compileResult;
        }

        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        let useDefaultInsnSet = true;
        let useDefaultCompilerFilter = true;

        // The items in 'options' before the source file are user inputs.
        const sourceFileOptionIndex = options.findIndex(option => {
            return option.endsWith('.java') || option.endsWith('.kt');
        });
        let userOptions = options.slice(0, sourceFileOptionIndex);
        for (const option of userOptions) {
            if (this.insnSetArgRegex.test(option)) {
                useDefaultInsnSet = false;
            } else if (this.compilerFilterArgRegex.test(option)) {
                useDefaultCompilerFilter = false;
            } else if (this.fullOutputArgRegex.test(option)) {
                this.fullOutput = true;
            }
        }
        // Remove '--full-output' because it isn't a real dex2oat option.
        userOptions = userOptions.filter(option => !this.fullOutputArgRegex.test(option));

        const files = await fs.readdir(d8DirPath);
        const dexFile = files.find(f => f.endsWith('.dex'));
        if (!dexFile) {
            throw new Error('Generated dex file not found');
        }

        const profileAndResult = await this.generateProfile(d8DirPath, dexFile);
        if (profileAndResult && profileAndResult.result.code !== 0) {
            return {
                ...this.transformToCompilationResult(profileAndResult.result, inputFilename),
                languageId: this.getCompilerResultLanguageId(filters),
            };
        }

        const bootclassjars = [
            'bootjars/core-oj.jar',
            'bootjars/core-libart.jar',
            'bootjars/okhttp.jar',
            'bootjars/bouncycastle.jar',
            'bootjars/apache-xml.jar',
        ];

        let isLatest = false;
        let versionPrefix = 0;
        let match;
        if (this.versionPrefixRegex.test(this.compiler.id)) {
            match = this.compiler.id.match(this.versionPrefixRegex);
            versionPrefix = parseInt(match![2]);
        } else if (this.latestVersionRegex.test(this.compiler.id)) {
            isLatest = true;
        }

        const dex2oatOptions = [
            '--android-root=include',
            '--generate-debug-info',
            '--dex-location=/system/framework/classes.dex',
            `--dex-file=${d8DirPath}/${dexFile}`,
            '--copy-dex-files=always',
            ...(versionPrefix >= 34 || isLatest ? ['--runtime-arg', '-Xgc:CMC'] : []),
            '--runtime-arg',
            '-Xbootclasspath:' + bootclassjars.map(f => path.join(this.artArtifactDir, f)).join(':'),
            '--runtime-arg',
            '-Xbootclasspath-locations:/apex/com.android.art/javalib/core-oj.jar' +
                ':/apex/com.android.art/javalib/core-libart.jar' +
                ':/apex/com.android.art/javalib/okhttp.jar' +
                ':/apex/com.android.art/javalib/bouncycastle.jar' +
                ':/apex/com.android.art/javalib/apache-xml.jar',
            `--boot-image=${this.artArtifactDir}/app/system/framework/boot.art`,
            `--oat-file=${d8DirPath}/classes.odex`,
            `--app-image-file=${d8DirPath}/classes.art`,
            '--force-allow-oj-inlines',
            `--dump-cfg=${d8DirPath}/classes.cfg`,
            ...userOptions,
        ];
        if (useDefaultInsnSet) {
            dex2oatOptions.push('--instruction-set=arm64');
        }
        if (useDefaultCompilerFilter) {
            if (profileAndResult == null) {
                dex2oatOptions.push('--compiler-filter=speed');
            } else {
                dex2oatOptions.push('--compiler-filter=speed-profile');
            }
        }
        if (profileAndResult != null) {
            dex2oatOptions.push(`--profile-file=${profileAndResult.path}`);
        }

        execOptions.customCwd = d8DirPath;

        const result = await this.exec(this.compiler.exe, dex2oatOptions, execOptions);
        if (profileAndResult != null) {
            result.stdout = profileAndResult.result.stdout + result.stdout;
            result.stderr = profileAndResult.result.stderr + result.stderr;
        }
        return {
            ...this.transformToCompilationResult(result, d8OutputFilename),
            languageId: this.getCompilerResultLanguageId(filters),
        };
    }

    override getIncludeArguments(libraries: SelectedLibraryVersion[], dirPath: string): string[] {
        this.libs = libraries;
        return super.getIncludeArguments(libraries, dirPath);
    }

    private async generateProfile(
        d8DirPath: string,
        dexFile: string,
    ): Promise<{path: string; result: UnprocessedExecResult} | null> {
        const humanReadableFormatProfile = `${d8DirPath}/profile.prof.txt`;
        try {
            await fs.access(humanReadableFormatProfile);
        } catch (e) {
            // No profile. This is expected.
            return null;
        }

        const execOptions = this.getDefaultExecOptions();
        execOptions.customCwd = d8DirPath;
        const binaryFormatProfile = `${d8DirPath}/profile.prof`;
        const result = await this.exec(
            this.profmanPath,
            [
                `--create-profile-from=${humanReadableFormatProfile}`,
                `--apk=${d8DirPath}/${dexFile}`,
                '--dex-location=/system/framework/classes.dex',
                `--reference-profile-file=${binaryFormatProfile}`,
                '--output-profile-type=app',
            ],
            execOptions,
        );

        return {path: binaryFormatProfile, result: result};
    }

    override async objdump(outputFilename: string, result: any, maxSize: number) {
        const dirPath = path.dirname(outputFilename);
        const files = await fs.readdir(dirPath);
        const odexFile = files.find(f => f.endsWith('.odex'));
        const args = [...this.compiler.objdumperArgs, `--oat-file=${odexFile}`];
        const objResult = await this.exec(this.compiler.objdumper, args, {
            maxOutput: maxSize,
            customCwd: dirPath,
        });

        const asmResult: ParsedAsmResult = {
            asm: [
                {
                    text: objResult.stdout,
                },
            ],
        };
        if (objResult.code === 0) {
            result.objdumpTime = objResult.execTime;
        } else {
            asmResult.asm = [
                {
                    text: `<No output: oatdump returned ${objResult.code}>`,
                },
            ];
        }
        result.asm = asmResult.asm;
        return result;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions) {
        filters.binary = true;
        return [];
    }

    // dex2oat doesn't have --version, but artArtifactDir contains a file with
    // the build number.
    override async getVersion() {
        const versionFile = this.artArtifactDir + '/snapshot-creation-build-number.txt';
        const version = fs.readFileSync(versionFile, {encoding: 'utf8'});
        return {
            stdout: 'Android Build ' + version,
            stderr: '',
            code: 0,
        };
    }

    override async processAsm(result) {
        let asm: string = '';

        if (typeof result.asm === 'string') {
            const asmLines = utils.splitLines(result.asm);
            if (asmLines.length === 1 && asmLines[0][0] === '<') {
                return {
                    asm: [{text: asmLines[0], source: null}],
                };
            } else {
                return {
                    asm: [{text: JSON.stringify(asmLines), source: null}],
                };
            }
        } else {
            // result.asm is an array, but we only expect it to have one value.
            asm = result.asm[0].text;
        }

        const segments: ParsedAsmResultLine[] = [];
        if (this.fullOutput) {
            // Returns entire dex2oat output.
            segments.push({text: asm, source: null});
        } else {
            const {compileData, classNames, classToMethods, methodsToInstructions, methodsToSizes} = this.parseAsm(asm);

            segments.push(
                {
                    text: 'Instruction set:          ' + compileData.insnSet,
                    source: null,
                },
                {
                    text: 'Instruction set features: ' + compileData.insnSetFeatures,
                    source: null,
                },
                {
                    text: 'Compiler filter:          ' + compileData.compilerFilter,
                    source: null,
                },
                {text: '', source: null},
                {text: '', source: null},
            );

            for (const className of classNames) {
                for (const method of classToMethods[className]) {
                    segments.push({
                        text: method + ' [' + methodsToSizes[method] + ' bytes]',
                        source: null,
                    });
                    for (const instruction of methodsToInstructions[method]) {
                        segments.push({
                            text: '    ' + instruction,
                            source: null,
                        });
                    }
                    segments.push({text: '', source: null});
                }
            }
        }

        return {asm: segments};
    }

    parseAsm(oatdumpOut: string) {
        const compileData: {
            insnSet?: string;
            insnSetFeatures?: string;
            compilerFilter?: string;
        } = {};

        const classNames: string[] = [];
        const classToMethods: Record<string, string[]> = {};
        const methodsToInstructions: Record<string, string[]> = {};
        const methodsToSizes: Record<string, number> = {};

        let match;
        if (this.insnSetRegex.test(oatdumpOut)) {
            match = oatdumpOut.match(this.insnSetRegex);
            compileData.insnSet = match![1];
        }
        if (this.insnSetFeaturesRegex.test(oatdumpOut)) {
            match = oatdumpOut.match(this.insnSetFeaturesRegex);
            compileData.insnSetFeatures = match![1];
        }

        let inCode = false;
        let currentClass = '';
        let currentMethod = '';
        for (const l of oatdumpOut.split(/\n/)) {
            if (this.compilerFilterRegex.test(l)) {
                match = l.match(this.compilerFilterRegex);
                compileData.compilerFilter = match![1];
            } else if (this.classRegex.test(l)) {
                match = l.match(this.classRegex);
                currentClass = match![1];
                classNames.push(currentClass);
                classToMethods[currentClass] = [];
            } else if (this.methodRegex.test(l)) {
                match = l.match(this.methodRegex);
                currentMethod = match![1];
                classToMethods[currentClass].push(currentMethod);
                methodsToInstructions[currentMethod] = [];
                inCode = false;
            } else if (this.methodSizeRegex.test(l)) {
                match = l.match(this.methodSizeRegex);
                methodsToSizes[currentMethod] = Number.parseInt(match![1]);
                inCode = true;
            } else if (inCode && this.insnRegex.test(l)) {
                match = l.match(this.insnRegex);
                methodsToInstructions[currentMethod].push(match![1] + '    ' + match![2]);
            } else if (inCode && this.stackMapRegex.test(l)) {
                match = l.match(this.stackMapRegex);
                methodsToInstructions[currentMethod].push(' ' + match![1] + '   ' + match![2]);
            }
        }

        return {compileData, classNames, classToMethods, methodsToInstructions, methodsToSizes};
    }

    override async generateOptPipeline(
        inputFilename: string,
        options: string[],
        filters: ParseFiltersAndOutputOptions,
        OptPipelineOptions: OptPipelineBackendOptions,
    ): Promise<OptPipelineOutput | undefined> {
        const dirPath = path.dirname(inputFilename);
        const execOptions = this.getDefaultExecOptions();
        execOptions.maxOutput = 1024 * 1024 * 1024;

        const compileStart = performance.now();
        await this.runCompiler(this.compiler.exe, options, inputFilename, execOptions);
        const compileEnd = performance.now();

        try {
            const classesCfg = dirPath + '/classes.cfg';
            const rawText = fs.readFileSync(classesCfg, {encoding: 'utf8'});
            const parseStart = performance.now();
            const optPipeline = this.passDumpParser.process(rawText);
            const parseEnd = performance.now();
            return {
                results: optPipeline,
                parseTime: parseEnd - parseStart,
                compileTime: compileEnd - compileStart,
            };
        } catch (e: any) {
            return {
                error: e.toString(),
                results: {},
                compileTime: compileEnd - compileStart,
            };
        }
    }
}
