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
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
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

import {z} from 'zod';

import {splitArguments} from '../../shared/common-utils.js';
import type {ParsedAsmResult, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces.js';
import type {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import type {CompilationEnvironment} from '../compilation-env.js';
import {JavaCompiler} from './java.js';

const manifestSchema = z.object({
    version: z.literal(1),
    architecture: z.literal('amd64'),
    methods: z.array(
        z.object({
            name: z.string(),
            code: z.string().regex(/^(?:[0-9a-f]{2})+$/),
            patches: z.array(
                z.object({
                    offset: z.number().int().nonnegative(),
                    kind: z.enum(['call', 'data']),
                    target: z.string(),
                    direct: z.boolean().optional(),
                }),
            ),
            mappings: z.array(
                z.object({
                    start: z.number().int().nonnegative(),
                    end: z.number().int().nonnegative(),
                    line: z.number().int().positive(),
                }),
            ),
        }),
    ),
});

export type NativeImageMethod = z.infer<typeof manifestSchema>['methods'][number];

export function parseNativeImageDisassembly(method: NativeImageMethod, output: string): ParsedAsmResultLine[] {
    const lines: ParsedAsmResultLine[] = [{text: `${method.name}:`, source: null}];
    for (const line of output.split('\n')) {
        const match = line.match(/^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2}\s+)+)\s*(.+)$/);
        if (!match) continue;
        const offset = Number.parseInt(match[1], 16);
        const bytes = match[2].trim().split(/\s+/);
        let instruction = match[3].trim();
        const patches = method.patches.filter(p => p.offset >= offset && p.offset < offset + bytes.length);
        for (const patch of patches) {
            if (patch.kind === 'call' && patch.direct !== false) {
                instruction = `${instruction.split(/\s+/)[0]} <${patch.target}>`;
            } else {
                instruction =
                    instruction.replace(/\s+#.*$/, '') +
                    ` # ${patch.kind === 'call' ? 'indirect call' : 'unresolved'} ${patch.target}`;
            }
        }
        const source = method.mappings.find(m => offset >= m.start && offset < m.end);
        lines.push({
            text: `    ${instruction}`,
            opcodes: bytes,
            source: source ? {file: null, line: source.line} : null,
        });
    }
    const offsets = [...output.matchAll(/^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+)+\s*.+$/gm)].map(match =>
        Number.parseInt(match[1], 16),
    );
    const labelPrefix = `.L${createHash('sha256').update(method.name).digest('hex').slice(0, 12)}_`;
    const targets = new Set<number>();
    for (const line of lines) {
        const jump = line.text.match(/^(\s*j\w+\s+)(?:0x)?([0-9a-f]+)$/);
        if (!jump) continue;
        const target = Number.parseInt(jump[2], 16);
        if (!offsets.includes(target)) continue;
        targets.add(target);
        line.text = `${jump[1]}${labelPrefix}${target.toString(16)}`;
    }
    return lines
        .flatMap((line, index) =>
            targets.has(offsets[index - 1])
                ? [{text: `${labelPrefix}${offsets[index - 1].toString(16)}:`, source: null}, line]
                : [line],
        )
        .concat({text: '', source: null});
}

export class NativeImageCompiler extends BaseCompiler {
    static get key() {
        return 'native-image';
    }

    private readonly bytecodeParser: JavaCompiler;
    private readonly frontend: string;
    private readonly feature: string;
    private readonly javap: string;
    private readonly classpath: string;

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super({...info, disabledFilters: ['labels', 'directives', 'commentOnly', 'trim', 'debugCalls']}, env);
        this.compiler.supportsJvmBytecodeView = true;
        this.compiler.supportsExecute = false;
        this.compiler.supportsBinary = false;
        this.compiler.supportsBinaryObject = false;
        this.compiler.supportsIntel = true;
        this.frontend = this.compilerProps<string>(`compiler.${this.compiler.id}.frontend`);
        this.feature = this.compilerProps<string>(`compiler.${this.compiler.id}.nativeImageFeature`);
        this.javap = this.compilerProps<string>(
            `compiler.${this.compiler.id}.javap`,
            path.join(path.dirname(info.exe), 'javap'),
        );
        this.classpath = this.compilerProps<string>(`compiler.${this.compiler.id}.nativeImageClasspath`, '');
        this.bytecodeParser = new JavaCompiler({...info}, env);
    }

    override async getVersion() {
        const version = await super.getVersion();
        if (!version || !this.frontend || !this.feature) return version;
        const frontend = await this.exec(this.frontend, ['-version'], this.getDefaultExecOptions());
        if (frontend.code !== 0) return frontend;
        const hash = createHash('sha256');
        for (const file of [this.feature, ...this.classpath.split(path.delimiter).filter(Boolean)]) {
            hash.update(await fs.readFile(file));
        }
        return {
            ...version,
            stdout: `${version.stdout}\nFrontend: ${frontend.stdout}${frontend.stderr}\nExtraction adapter and classpath: ${hash.digest('hex')}\n`,
        };
    }

    override prepareArguments(userOptions: string[]) {
        return [...splitArguments(this.compiler.options), ...userOptions];
    }

    override getDefaultExecOptions(): ExecutionOptionsWithEnv {
        const options = super.getDefaultExecOptions();
        options.timeoutMs = this.compilerProps<number>(`compiler.${this.compiler.id}.nativeImageTimeoutMs`, 120000);
        options.env.JAVA_HOME = path.dirname(path.dirname(this.compiler.exe));
        return options;
    }

    override async postProcess(result: CompilationResult): ReturnType<BaseCompiler['postProcess']> {
        return [result, [], []];
    }

    override async processAsm(result: CompilationResult): Promise<ParsedAsmResult> {
        return {asm: Array.isArray(result.asm) ? result.asm : [{text: result.asm ?? '<No output>'}]};
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        const directory = path.dirname(inputFilename);
        const classes = path.join(directory, 'classes');
        const manifest = path.join(directory, 'native-image.json');
        const result: CompilationResult = {
            code: 0,
            timedOut: false,
            okToCache: true,
            stdout: [],
            stderr: [],
            inputFilename,
            languageId: 'asm',
            instructionSet: 'amd64',
        };
        const separator = options.indexOf('--');
        const nativeOptions = separator === -1 ? options : options.slice(0, separator);
        const frontendOptions = separator === -1 ? [] : options.slice(separator + 1);
        // Restrict builder options to code-generation choices; the adapter owns features and output paths.
        const invalid = nativeOptions.find(option => !/^-O[0-3bs]$/.test(option));
        if (invalid)
            return {
                ...result,
                code: -1,
                stderr: [
                    {
                        text: `Unsupported Native Image option: ${invalid}. Use -O0, -O1, -O2, -O3, -Ob or -Os; put frontend options after --.`,
                    },
                ],
            };
        if (!this.frontend || !this.feature)
            return {
                ...result,
                code: -1,
                stderr: [{text: 'Configure frontend and nativeImageFeature for this compiler.'}],
            };
        if (
            frontendOptions.some(
                option =>
                    !/^(?:-g(?::[a-z,]+)?|-parameters|-nowarn|-Werror|-Xlint(?::[a-z,-]+)?|-java-parameters|-Xno-param-assertions|-Xno-call-assertions)$/.test(
                        option,
                    ),
            )
        ) {
            return {
                ...result,
                code: -1,
                stderr: [
                    {text: 'Unsupported frontend option. See docs/native-image.md for the local prototype options.'},
                ],
            };
        }
        const deadline = Date.now() + (execOptions.timeoutMs ?? 120000);
        const run = async (exe: string, args: string[]) => {
            const remaining = deadline - Date.now();
            if (remaining <= 0) {
                result.timedOut = true;
                throw new Error('Compilation timed out');
            }
            const raw = await this.exec(exe, args, {...execOptions, timeoutMs: remaining, customCwd: directory});
            const step = this.transformToCompilationResult(raw, inputFilename);
            result.stdout.push(...step.stdout);
            result.stderr.push(...step.stderr);
            result.code = step.code;
            result.execTime = (result.execTime ?? 0) + raw.execTime;
            result.timedOut ||= step.timedOut;
            result.okToCache &&= step.okToCache;
            return raw;
        };
        try {
            await fs.mkdir(classes);
            const kotlin = this.lang.id === 'kotlin';
            const frontendArgs = [
                ...(kotlin ? [] : ['-g', '-proc:none']),
                ...frontendOptions,
                ...(this.classpath ? ['-classpath', this.classpath] : []),
                '-d',
                classes,
                inputFilename,
            ];
            if ((await run(this.frontend, frontendArgs)).code !== 0) return result;
            const classFiles = (await fs.readdir(classes, {recursive: true})).filter(f => f.endsWith('.class')).sort();
            const bytecode: {text: string}[] = [];
            for (const file of classFiles) {
                const stdoutLength = result.stdout.length;
                const output = await run(this.javap, ['-c', '-l', '-p', '-constants', path.join(classes, file)]);
                if (output.code !== 0) return result;
                // javap output belongs in its pane, not in compiler diagnostics.
                result.stdout.length = stdoutLength;
                bytecode.push({text: output.stdout});
            }
            result.jvmBytecodeOutput = (await this.bytecodeParser.processAsm({asm: bytecode})).asm;
            const exports: Record<string, string[]> = {
                'org.graalvm.nativeimage.builder': [
                    'com.oracle.svm.hosted',
                    'com.oracle.svm.hosted.code',
                    'com.oracle.svm.hosted.meta',
                    'com.oracle.svm.core.util',
                ],
                'org.graalvm.nativeimage.base': ['com.oracle.svm.common.meta'],
                'jdk.graal.compiler': [
                    'jdk.graal.compiler.code',
                    'jdk.graal.compiler.graph',
                    'jdk.graal.compiler.util.json',
                ],
                'jdk.internal.vm.ci': ['jdk.vm.ci.meta', 'jdk.vm.ci.code', 'jdk.vm.ci.code.site'],
            };
            const nativeArgs = [
                '--shared',
                '--no-fallback',
                '--parallelism=2',
                '-J-Xmx3g',
                '-march=x86-64',
                '-H:+UnlockExperimentalVMOptions',
                '-H:+TrackNodeSourcePosition',
                ...nativeOptions,
                '--features=ce.nativeimage.ExplorerFeature',
                `-Dce.nativeimage.classes=${classes}`,
                `-Dce.nativeimage.output=${manifest}`,
                ...Object.entries(exports).flatMap(([module, packages]) =>
                    packages.map(pkg => `-J--add-exports=${module}/${pkg}=ALL-UNNAMED`),
                ),
                '-cp',
                [classes, this.feature, this.classpath].filter(Boolean).join(path.delimiter),
                '-o',
                path.join(directory, 'unused-image'),
            ];
            if ((await run(compiler, nativeArgs)).code !== 0) return result;
            const stat = await fs.stat(manifest);
            if (stat.size > this.env.ceProps('max-asm-size', 64 * 1024 * 1024))
                throw new Error('Native Image output exceeds the size limit');
            const extracted = manifestSchema.parse(JSON.parse(await fs.readFile(manifest, 'utf8')));
            const asm: ParsedAsmResultLine[] = [];
            for (const [index, method] of extracted.methods.entries()) {
                const binary = path.join(directory, `method-${index}.bin`);
                await fs.writeFile(binary, Buffer.from(method.code, 'hex'));
                const stdoutLength = result.stdout.length;
                const output = await run(this.compiler.objdumper, [
                    '-D',
                    '-b',
                    'binary',
                    '-m',
                    'i386:x86-64',
                    '--insn-width=16',
                    ...(filters?.intel ? ['-M', 'intel'] : []),
                    binary,
                ]);
                result.stdout.length = stdoutLength;
                if (output.code !== 0) throw new Error(`Disassembly failed: ${output.stderr}`);
                asm.push(...parseNativeImageDisassembly(method, output.stdout));
            }
            result.stdout = result.stdout.filter(line => !line.text.startsWith('Failed generating '));
            result.stdout.push({
                text: 'Native Image methods extracted; image generation stopped before image creation.',
            });
            result.asm = asm.length ? asm : [{text: '<No compiled methods in submitted classes>'}];
            result.compilationOptions = nativeOptions;
            return result;
        } catch (error) {
            return {
                ...result,
                code: -1,
                okToCache: false,
                stderr: [
                    ...result.stderr,
                    {text: `Native Image extraction failed: ${error instanceof Error ? error.message : String(error)}`},
                ],
            };
        }
    }
}
