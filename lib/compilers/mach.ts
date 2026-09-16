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

import type {
    CacheKey,
    CompilationCacheKey,
    CompilationResult,
    ExecutionOptionsWithEnv,
    FiledataPair,
} from '../../types/compilation/compilation.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {BasicExecutionResult, UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import * as temp from '../temp.js';
import * as utils from '../utils.js';
import {MachParser} from './argument-parsers.js';

/** One platform tuple the compiler supports, keyed by the name a user passes to `--target`. */
export type MachTarget = {
    name: string;
    isa: string;
    os: string;
    abi: string;
    object: string;
};

/**
 * What the probe compiles. Every compilation realizes std into the project, and two of the three examples import it,
 * so a module that never reaches it would clear targets that no real program can use: std has no freestanding os
 * layer, and a `use std.print` there fails deep inside std rather than at the import.
 */
const probeSource = [
    'use print: std.print;',
    '',
    'pub fun probe() i64 {',
    '    print.println("probe");',
    '    ret 0;',
    '}',
    '',
].join('\n');

/**
 * The profile every compilation builds under. A target that cannot be resolved against it produces no view at all, so
 * the same lines drive the probe that decides which targets are worth offering.
 */
const profile = [
    '[profile.ce]',
    'default = true',
    'opt = 0',
    'debug = true',
    'simd = "scalarize"',
    'vectorize = true',
    'float_reassoc = false',
    '',
];

/**
 * Each tuple names its object format explicitly. Omitting it takes the os's default, which for freestanding is the
 * flat image: it carries neither a debug model nor linkable objects, so the ELF row of the same tuple never gets a
 * chance to be the one that builds.
 */
function targetSection(target: MachTarget): string[] {
    return [
        `[target.${target.name}]`,
        `isa = "${target.isa}"`,
        `os = "${target.os}"`,
        `abi = "${target.abi}"`,
        `of = "${target.object}"`,
        '',
    ];
}

/** The manifest key a tuple is probed under, positional so that tuples sharing a platform name stay distinct. */
function probeKey(index: number): string {
    return `t${index}`;
}

/** A platform name shared by several tuples is qualified by abi, then by object format, until every key is unique. */
function qualifyNames(rows: MachTarget[]): MachTarget[] {
    return rows.map(row => {
        const samePlatform = rows.filter(other => other.name === row.name);
        if (samePlatform.length === 1) return row;
        const name = `${row.name}-${row.abi}`;
        return samePlatform.filter(other => other.abi === row.abi).length === 1
            ? {...row, name}
            : {...row, name: `${name}-${row.object}`};
    });
}

/**
 * Mach has no single-file mode: every build is a project with a manifest and a realized dependency closure (std
 * included). Each compilation therefore lays out a project in the temp dir:
 *
 *   mach.toml               generated; one bin artifact, every buildable target, std as a path dependency
 *   src/example.mach        the user's source (and any extra files, rooted at src/)
 *   dep/std/                realized by `mach dep pull` from the compiler's std (see `stdPath` below)
 *   out/obj/example/*.o     per-module objects, disassembled with objdump against their DWARF line table
 *   out/bin/example         the linked executable
 *
 * Mach installs no std of its own: a project names std as a dependency. A compiler therefore needs a std to build
 * against, which is the std release it was released with. By default it lives beside the executable, in
 * `<dir of exe>/std`, which is the layout the release tarball plus a std checkout gives; `compiler.<id>.stdPath`
 * overrides it. docs/Mach.md describes the layout.
 *
 * The targets offered are the ones a program that uses std can build for, so no freestanding target is offered: std
 * has no freestanding os layer. See `targets()` and docs/Mach.md.
 */
export class MachCompiler extends BaseCompiler {
    static get key() {
        return 'mach';
    }

    private readonly stdPath: string;
    /** The probe runs once per compiler: every compilation needs the same answer to write its manifest. */
    private buildable?: Promise<MachTarget[]>;

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.stdPath =
            this.compilerProps<string>(`compiler.${this.compiler.id}.stdPath`) ??
            path.join(path.dirname(this.compiler.exe), 'std');
        this.compiler.supportsTarget = true;
    }

    override getArgumentParserClass() {
        return MachParser;
    }

    get projectId(): string {
        return path.parse(this.compileFilename).name;
    }

    projectRoot(inputFilename: string): string {
        return path.dirname(path.dirname(inputFilename));
    }

    /** Every (isa, os, abi, object) tuple `mach info targets` reports, named after mach's own platform name. */
    async supportedTuples(): Promise<MachTarget[]> {
        const result = await this.execCompilerCached(this.compiler.exe, ['info', 'targets']);
        if (result.code !== 0) throw new Error(`mach info targets failed: ${result.stderr}`);

        const rows: MachTarget[] = [];
        for (const line of utils.splitLines(result.stdout)) {
            const match = line.match(/^(\S+)\s+isa=(\S+)\s+os=(\S+)\s+abi=(\S+)\s+object=(\S+)/);
            if (!match) continue;
            const [, name, isa, os, abi, object] = match;
            rows.push({name, isa, os, abi, object});
        }
        return rows;
    }

    /**
     * The tuples this compiler can actually produce a view for, probed rather than listed. Two things sink a tuple: an
     * object format that registers no debug model, which the profile's debug information needs, and an os std has no
     * layer for. Neither is knowable from what `mach info targets` prints, and both move between releases, so the
     * answer comes from building rather than from a list here.
     */
    async targets(): Promise<MachTarget[]> {
        this.buildable ??= this.probeTargets();
        return await this.buildable;
    }

    private async probeTargets(): Promise<MachTarget[]> {
        if (!(await utils.fileExists(path.join(this.stdPath, 'mach.toml'))))
            throw new Error(
                `${this.compiler.id}: no mach std at ${this.stdPath}; install it there or set ` +
                    `compiler.${this.compiler.id}.stdPath (see docs/Mach.md)`,
            );
        const tuples = await this.supportedTuples();
        const dirPath = await this.newTempDir({hold: true});
        try {
            await this.layOutProbe(dirPath, tuples);
            const rows: MachTarget[] = [];
            for (const [index, target] of tuples.entries()) {
                if (await this.buildsUnderProfile(dirPath, probeKey(index))) rows.push(target);
            }
            return qualifyNames(rows);
        } finally {
            temp.release(dirPath);
            await fs.rm(dirPath, {recursive: true, force: true});
        }
    }

    /**
     * One project holding every tuple, which is the shape a compilation has: std is realized once, and each tuple is
     * then selected the way a user selects one. Tuples are keyed positionally because the names they will be offered
     * under are only settled once the unbuildable ones are gone.
     */
    private async layOutProbe(dirPath: string, tuples: MachTarget[]) {
        await fs.mkdir(path.join(dirPath, 'src'), {recursive: true});
        await fs.writeFile(path.join(dirPath, 'src', 'probe.mach'), probeSource);

        const lines = ['[project]', 'id = "probe"', 'version = "0.0.0"', 'src = "src"', 'out = "out"', '', ...profile];
        for (const [index, target] of tuples.entries())
            lines.push(...targetSection({...target, name: probeKey(index)}));
        lines.push(
            '[artifact.probe]',
            'kind = "bin"',
            'entry = "probe.mach"',
            'out = "bin/probe"',
            `targets = [${tuples.map((_, index) => `"${probeKey(index)}"`).join(', ')}]`,
            'link = []',
            'need = []',
            '',
            '[dep.std]',
            `path = ${JSON.stringify(this.stdPath)}`,
            '',
        );
        await fs.writeFile(path.join(dirPath, 'mach.toml'), lines.join('\n'));

        const pull = await this.exec(this.compiler.exe, ['dep', 'pull', dirPath], {
            ...this.getDefaultExecOptions(),
            customCwd: dirPath,
        });
        if (pull.code !== 0) throw new Error(`mach dep pull failed: ${pull.stderr || pull.stdout}`);
    }

    /** Build the probe for one tuple, the way a compilation does, to see whether the tuple survives it. */
    private async buildsUnderProfile(dirPath: string, key: string): Promise<boolean> {
        const result = await this.exec(this.compiler.exe, ['build', dirPath, '--emit', 'obj', '--target', key], {
            ...this.getDefaultExecOptions(),
            customCwd: dirPath,
        });
        return result.code === 0;
    }

    manifest(targets: MachTarget[]): string {
        const id = this.projectId;
        const lines = ['[project]', `id = "${id}"`, 'version = "0.0.0"', 'src = "src"', 'out = "out"', '', ...profile];
        for (const t of targets) lines.push(...targetSection(t));
        lines.push(
            `[artifact.${id}]`,
            'kind = "bin"',
            `entry = "${this.compileFilename}"`,
            `out = "bin/${id}"`,
            `targets = [${targets.map(t => `"${t.name}"`).join(', ')}]`,
            'link = []',
            'need = []',
            '',
            '[dep.std]',
            `path = ${JSON.stringify(this.stdPath)}`,
            '',
        );
        return lines.join('\n');
    }

    protected override async writeAllFiles(dirPath: string, source: string, files: FiledataPair[]) {
        if (!source) throw new Error(`File ${this.compileFilename} has no content or file is missing`);

        const srcDir = path.join(dirPath, 'src');
        await fs.mkdir(srcDir, {recursive: true});

        const inputFilename = path.join(srcDir, this.compileFilename);
        await fs.writeFile(inputFilename, source);
        if (files && files.length > 0) await this.writeMultipleFiles(files, srcDir);

        await fs.writeFile(path.join(dirPath, 'mach.toml'), this.manifest(await this.targets()));

        // copies std into the project; goes away once mach can use a path dependency in place (briar-systems/mach#3484)
        const pull = await this.exec(this.compiler.exe, ['dep', 'pull', dirPath], {
            ...this.getDefaultExecOptions(),
            customCwd: dirPath,
        });
        if (pull.code !== 0) throw new Error(`mach dep pull failed: ${pull.stderr || pull.stdout}`);

        return {inputFilename};
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: CacheKey | CompilationCacheKey) {
        if (this.isCacheKey(key) && key.filters?.binary)
            return this.getExecutableFilename(dirPath, outputFilebase, key);
        const id = this.projectId;
        return path.join(dirPath, 'out', 'obj', id, `${id}.o`);
    }

    override getExecutableFilename(dirPath: string, outputFilebase: string, key?: CacheKey | CompilationCacheKey) {
        return path.join(dirPath, 'out', 'bin', this.projectId);
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        // the object's disassembly is the asm view: mach's text listing is not an assembler file
        if (!filters.binary) filters.binaryObject = true;
        return ['--emit', filters.binary ? 'exe' : 'obj'];
    }

    override orderArguments(
        options: string[],
        inputFilename: string,
        libIncludes: string[],
        libOptions: string[],
        libPaths: string[],
        libLinks: string[],
        userOptions: string[],
        staticLibLinks: string[],
    ) {
        // an absolute project root makes the DWARF comp_dir absolute, which objdump needs to name source lines
        return ['build', this.projectRoot(inputFilename), ...options, ...userOptions];
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: ParseFiltersAndOutputOptions,
    ): Promise<CompilationResult> {
        execOptions.customCwd = this.projectRoot(inputFilename);
        return super.runCompiler(compiler, options, inputFilename, execOptions, filters);
    }

    override async buildExecutable(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
    ) {
        execOptions.customCwd = this.projectRoot(inputFilename);
        return super.buildExecutable(compiler, options, inputFilename, execOptions);
    }

    override processExecutionResult(input: UnprocessedExecResult, inputFilename?: string): BasicExecutionResult {
        return {
            ...input,
            stdout: utils.parseRustOutput(input.stdout, inputFilename),
            stderr: utils.parseRustOutput(input.stderr, inputFilename),
        };
    }
}
