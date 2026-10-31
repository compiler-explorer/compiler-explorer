// Copyright (c) 2026, Compiler Explorer Authors
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

import {createHash} from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';

import type {ParsedAsmResult} from '../../types/asmresult/asmresult.interfaces.js';
import type {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import type {CompilationEnvironment} from '../compilation-env.js';
import {BaseParser} from './argument-parsers.js';

const producer = '83928fa4538245abe33b08549da47f8cbdc50ed2';
const version = '0.6.32-dev';
const schema = 'gooo/body-codegen-report/v3';
const maxOutput = 1024 * 1024;

function record(value: unknown): Record<string, unknown> {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        throw new Error('Expected a body-codegen JSON object');
    }
    return value as Record<string, unknown>;
}

function digest(value: string | Buffer): string {
    return `sha256:${createHash('sha256').update(value).digest('hex')}`;
}

function generatedSource(stdout: string, activity: string, input: Buffer): string {
    const envelope = record(JSON.parse(stdout));
    const report = record(envelope.report);
    const source = envelope.source;
    const completeness = record(report.completeness_receipt);
    const scope = record(completeness.scope);
    if (
        Object.keys(envelope).sort().join(',') !== 'report,source' ||
        typeof source !== 'string' ||
        source.trim() === '' ||
        report.schema !== schema ||
        report.decision !== 'PASS' ||
        report.activity !== activity ||
        report.compiler_source_sha !== producer ||
        report.typecheck_passed !== true ||
        report.deterministic_replay !== true ||
        report.repository_writes !== 0 ||
        report.error !== undefined ||
        report.source_digest !== digest(input) ||
        report.generated_digest !== digest(source) ||
        report.replay_digest !== digest(source) ||
        completeness.profile_id !== 'gooo/body-codegen-pure-v1' ||
        scope.compiler_source_sha !== producer ||
        ![report.plan_sha256, report.program_digest].every(
            value => typeof value === 'string' && /^sha256:[0-9a-f]{64}$/.test(value),
        )
    ) {
        throw new Error('Body-codegen report does not match the selected activity, pinned producer or source digests');
    }
    return source;
}

export class GoooCompiler extends BaseCompiler {
    static get key() {
        return 'gooo-body';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.outputFilebase = 'example';
        Object.assign(this.compiler, {
            supportsExecute: false,
            supportsBinary: false,
            supportsBinaryObject: false,
            supportsCfg: false,
            supportsIntel: false,
            supportsOptOutput: false,
            supportsStackUsageOutput: false,
            supportsAstView: false,
            supportsPpView: false,
            supportsIrView: false,
            supportsGccDump: false,
            supportsDemangle: false,
            instructionSet: null,
        });
        this.compiler.demangler = '';
        this.demanglerClass = null;
    }

    override getArgumentParserClass() {
        return BaseParser;
    }

    override isCfgCompiler() {
        return false;
    }

    override optionsForFilter(): string[] {
        return [];
    }

    override prepareArguments(userOptions: string[]): string[] {
        return [...userOptions];
    }

    override getCompilerResultLanguageId(): string {
        return 'go';
    }

    override getOutputFilename(dirPath: string, outputFilebase: string): string {
        return path.join(dirPath, `${outputFilebase}.go`);
    }

    override fixFiltersBeforeCacheKey(filters: ParseFiltersAndOutputOptions): void {
        delete filters.binary;
        delete filters.binaryObject;
        delete filters.execute;
    }

    override async handleExecution(): Promise<CompilationResult> {
        return {
            ...this.failure('This compiler displays Go function source only; execution is unavailable'),
            didExecute: false,
        };
    }

    override async processAsm(result: {asm: string}): Promise<ParsedAsmResult> {
        return {asm: result.asm.split('\n').map(text => ({text})), languageId: 'go'};
    }

    private failure(message: string): CompilationResult {
        return {
            code: 1,
            timedOut: false,
            okToCache: false,
            stdout: [],
            stderr: [{text: `Gooo: ${message.slice(0, 2048)}`}],
            asm: [],
            languageId: 'go',
        };
    }

    private executionOptions(options: ExecutionOptionsWithEnv): ExecutionOptionsWithEnv {
        const env = {...options.env, GOOO_LAYA_URL: '', GOOO_LAYA_API_KEY: ''};
        return {
            ...options,
            env,
            timeoutMs: options.timeoutMs && options.timeoutMs > 0 ? options.timeoutMs : 7500,
            maxOutput: Math.min(options.maxOutput || maxOutput, maxOutput),
        };
    }

    override async getVersion() {
        try {
            const result = await this.exec(
                this.compiler.exe,
                ['version', '--build', '--json'],
                this.executionOptions(this.getDefaultExecOptions()),
            );
            const identity = record(JSON.parse(result.stdout));
            if (
                result.code !== 0 ||
                result.timedOut ||
                result.truncated ||
                identity.schema !== 'gooo/build-identity/v1' ||
                identity.version !== version ||
                identity.compiler_source_sha !== producer ||
                identity.vcs_revision !== producer ||
                identity.build_info_observed !== true ||
                identity.go_version !== 'go1.27.2' ||
                identity.native_go_required !== 'go1.27.2' ||
                identity.source_status !== 'CLEAN_VCS' ||
                identity.vcs_modified !== 'false'
            )
                return null;
            return {stdout: result.stdout, stderr: result.stderr, code: result.code};
        } catch {
            return null;
        }
    }

    private failedResult(result: UnprocessedExecResult, inputFilename: string, message: string): CompilationResult {
        return {
            ...this.transformToCompilationResult(
                {
                    ...result,
                    code: result.code || 1,
                    okToCache: false,
                    stdout: '',
                    stderr: `${result.stderr}\nGooo: ${message.slice(0, 2048)}`,
                },
                inputFilename,
            ),
            asm: [],
            languageId: 'go',
        };
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ): Promise<CompilationResult> {
        const output = this.getOutputFilename(path.dirname(inputFilename), this.outputFilebase);
        const temporary = `${output}.validated`;
        try {
            await fs.rm(output, {force: true});
            await fs.rm(temporary, {force: true});
            if (
                options.length !== 2 ||
                options[0] !== '--activity' ||
                typeof options[1] !== 'string' ||
                !/^[A-Za-z_][A-Za-z0-9_]*$/.test(options[1])
            ) {
                return this.failure('Use exactly --activity <identifier>, for example --activity Double');
            }
            const input = await fs.readFile(inputFilename);
            const execution = this.executionOptions(execOptions || this.getDefaultExecOptions());
            execution.customCwd = path.dirname(inputFilename);
            const result = await this.exec(
                compiler,
                ['body-codegen', '--json', '--activity', options[1], inputFilename],
                execution,
            );
            if (result.code !== 0 || result.timedOut || result.truncated) {
                let message = 'Body-codegen failed, timed out or exceeded its output bound';
                try {
                    const report = record(JSON.parse(result.stdout));
                    if (
                        report.schema === schema &&
                        report.decision === 'FAIL_CLOSED' &&
                        report.source === '' &&
                        typeof report.error === 'string'
                    )
                        message = report.error;
                } catch {
                    /* Malformed failure output is never displayed as source. */
                }
                return this.failedResult(result, inputFilename, message);
            }
            let source: string;
            try {
                source = generatedSource(result.stdout, options[1], input);
            } catch (error) {
                return this.failedResult(
                    result,
                    inputFilename,
                    error instanceof Error ? error.message : 'Invalid report',
                );
            }
            await fs.writeFile(temporary, source, {flag: 'wx'});
            await fs.rename(temporary, output);
            return {...this.transformToCompilationResult({...result, stdout: ''}, inputFilename), languageId: 'go'};
        } catch (error) {
            await fs.rm(output, {force: true}).catch(() => undefined);
            await fs.rm(temporary, {force: true}).catch(() => undefined);
            return this.failure(error instanceof Error ? error.message : 'Body-codegen failed');
        }
    }
}
