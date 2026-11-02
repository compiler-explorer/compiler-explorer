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

import fs from 'node:fs/promises';
import path from 'node:path';

import type {ParsedAsmResult, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import type {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {CompilationEnvironment} from '../compilation-env.js';
import * as utils from '../utils.js';
import {JavaCompiler} from './java.js';

const TC_OPTIONS_WITH_VALUE = new Set(['--cc', '--fp16', '--fp64', '--arg']);
const KERNEL_HEADER = /^\/\/ ===== .* =====$/;
// TC ends each statement it can attribute with the Java line it came from: `x = y;  // Saxpy.java:9`.
const SOURCE_COMMENT = /\s*\/\/ [\w$]+\.java:(\d+)$/;
// TC reports refusals as a compiler does, naming the class's source file rather than CE's input file.
const DIAGNOSTIC = /^[\w$]+\.java:(\d+): /;

type Target = 'cuda' | 'opencl' | 'metal';

/**
 * TornadoVM's TC compiler: compiles the GPU kernels of a Java source file (static void methods
 * over primitives and TornadoVM types) to CUDA C++, OpenCL C or Metal Shading Language. javac
 * runs first, then `tc emit --all` over the class files; no GPU is involved.
 */
export class TornadoVmTcCompiler extends JavaCompiler {
    static override get key() {
        return 'tornadovm-tc';
    }

    tornadovmHome: string;
    target: Target;

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);
        this.tornadovmHome = this.compilerProps<string>(`compiler.${this.compiler.id}.tornadovmHome`);
        const target = this.compilerProps<string>(`compiler.${this.compiler.id}.tcTarget`, 'cuda');
        this.target = target === 'opencl' || target === 'metal' ? target : 'cuda';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions) {
        filters.binary = false;
        // -parameters: `--arg n=1024` and TC's notes name parameters as written.
        return ['-g', '-parameters', '-encoding', 'utf8'];
    }

    override getOutputFilename(dirPath: string) {
        const extension = {cuda: '.cu', opencl: '.cl', metal: '.metal'}[this.target];
        return path.join(dirPath, `${path.basename(this.compileFilename, this.lang.extensions[0])}${extension}`);
    }

    override getCompilerResultLanguageId(filters?: ParseFiltersAndOutputOptions): string | undefined {
        return this.target === 'opencl' ? 'openclc' : 'cuda';
    }

    async tornadoApiJars(): Promise<string[]> {
        const dir = path.join(this.tornadovmHome, 'share', 'java', 'tornado');
        const files = await fs.readdir(dir);
        return files.filter(f => f.startsWith('tornado-api') && f.endsWith('.jar')).map(f => path.join(dir, f));
    }

    async needsPreview(): Promise<boolean> {
        try {
            const jdk = await fs.readFile(path.join(this.tornadovmHome, 'etc', 'tornado.jdk'), 'utf8');
            return jdk.includes('tornado.jdk.preview=true');
        } catch {
            return false;
        }
    }

    /** Splits the user's options into TC's (target options) and javac's (the rest). */
    splitOptions(userOptions: string[]): {tc: string[]; javac: string[]} {
        const tc: string[] = [];
        const javac: string[] = [];
        for (let i = 0; i < userOptions.length; i++) {
            if (TC_OPTIONS_WITH_VALUE.has(userOptions[i]) && i + 1 < userOptions.length) {
                tc.push(userOptions[i], userOptions[++i]);
            } else {
                javac.push(userOptions[i]);
            }
        }
        return {tc, javac};
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        if (!execOptions) execOptions = this.getDefaultExecOptions();
        const dirPath = path.dirname(inputFilename);
        execOptions.customCwd = dirPath;
        execOptions.env = {...execOptions.env, TORNADOVM_HOME: this.tornadovmHome};

        const sourceIndex = options.findIndex(option => option.endsWith('.java'));
        const {tc: tcOptions, javac: userOptions} = this.splitOptions(options.slice(0, sourceIndex));
        const preview = (await this.needsPreview()) ? ['--release', '21', '--enable-preview'] : [];
        const classpath = (await this.tornadoApiJars()).join(':');
        const javacOptions = [
            ...preview,
            '-cp',
            classpath,
            '-d',
            dirPath,
            ...userOptions,
            ...options.slice(sourceIndex),
        ];
        const javac = await this.exec(compiler, javacOptions, execOptions);
        if (javac.code !== 0) {
            return {
                ...this.transformToCompilationResult(javac, inputFilename),
                languageId: this.getCompilerResultLanguageId(filters),
            };
        }

        const tcExe = path.join(this.tornadovmHome, 'bin', 'tc');
        const tcArgs = ['emit', '--target', this.target, ...tcOptions, '--cp', dirPath, '--all'];
        const tc = await this.exec(tcExe, tcArgs, execOptions);
        await fs.writeFile(this.getOutputFilename(dirPath), tc.stdout);
        // One refused kernel must not hide the others: their refusals stay in stderr, marking the editor.
        const code = this.compiledAnyKernel(tc.stdout) ? 0 : tc.code;
        const stderr = utils
            .splitLines(tc.stderr)
            .map(line => line.replace(DIAGNOSTIC, `${inputFilename}:$1: `))
            .join('\n');
        return {
            ...this.transformToCompilationResult(
                {...tc, code, stdout: javac.stdout, stderr: `${javac.stderr}${stderr}`},
                inputFilename,
            ),
            languageId: this.getCompilerResultLanguageId(filters),
        };
    }

    /** Whether `tc emit --all` printed code for a kernel, not only the comments of a refusal. */
    compiledAnyKernel(output: string): boolean {
        let inKernel = false;
        for (const line of utils.splitLines(output)) {
            if (KERNEL_HEADER.test(line)) {
                inKernel = true;
            } else if (inKernel && line.trim() !== '' && !line.startsWith('//')) {
                return true;
            }
        }
        return false;
    }

    override async processAsm(result: {asm: string}): Promise<ParsedAsmResult> {
        if (typeof result.asm !== 'string' || result.asm.startsWith('<')) {
            return {asm: [{text: String(result.asm), source: null}]};
        }
        const asm: ParsedAsmResultLine[] = utils.splitLines(result.asm).map(line => {
            const match = line.match(SOURCE_COMMENT);
            return match
                ? {text: line.replace(SOURCE_COMMENT, ''), source: {file: null, line: Number.parseInt(match[1], 10)}}
                : {text: line, source: null};
        });
        while (asm.length > 0 && asm[asm.length - 1].text === '') asm.pop();
        return {asm};
    }
}
