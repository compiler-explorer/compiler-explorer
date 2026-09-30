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

import {readFileSync} from 'node:fs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi} from 'vitest';

import {MachCompiler} from '../lib/compilers/mach.js';
import {logger} from '../lib/logger.js';
import {AsmParser} from '../lib/parsers/asm-parser.js';
import {toEditorColumns} from '../lib/parsers/mach-diagnostics.js';
import {parseProperties} from '../lib/properties.js';
import * as utils from '../lib/utils.js';
import {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo, makeFakeParseFiltersAndOutputOptions} from './utils.js';

const languages = {
    mach: {id: 'mach' as LanguageKey, extensions: ['.mach']},
};

// `mach info targets` as mach 6.7.0 prints it.
const infoTargets = `linux-x86_64          isa=x86_64    os=linux         abi=sysv64   object=elf
linux-aarch64         isa=aarch64   os=linux         abi=aapcs64  object=elf
linux-riscv64         isa=rv64gc    os=linux         abi=lp64     object=elf
linux-riscv64         isa=rv64gc    os=linux         abi=lp64f    object=elf
linux-riscv64         isa=rv64gc    os=linux         abi=lp64d    object=elf
darwin-x86_64         isa=x86_64    os=darwin        abi=sysv64   object=macho
darwin-aarch64        isa=aarch64   os=darwin        abi=aapcs64  object=macho
windows-x86_64        isa=x86_64    os=windows       abi=win64    object=coff
freestanding-x86_64   isa=x86_64    os=freestanding  abi=sysv64   object=elf
freestanding-x86_64   isa=x86_64    os=freestanding  abi=sysv64   object=raw
freestanding-x86_64   isa=x86_64    os=freestanding  abi=win64    object=elf
freestanding-x86_64   isa=x86_64    os=freestanding  abi=win64    object=raw
freestanding-aarch64  isa=aarch64   os=freestanding  abi=aapcs64  object=elf
freestanding-aarch64  isa=aarch64   os=freestanding  abi=aapcs64  object=raw
freestanding-riscv64  isa=rv64gc    os=freestanding  abi=lp64     object=elf
freestanding-riscv64  isa=rv64gc    os=freestanding  abi=lp64     object=raw
freestanding-riscv64  isa=rv64gc    os=freestanding  abi=lp64f    object=elf
freestanding-riscv64  isa=rv64gc    os=freestanding  abi=lp64f    object=raw
freestanding-riscv64  isa=rv64gc    os=freestanding  abi=lp64d    object=elf
freestanding-riscv64  isa=rv64gc    os=freestanding  abi=lp64d    object=raw
freestanding-riscv32  isa=rv32imac  os=freestanding  abi=ilp32    object=elf
freestanding-riscv32  isa=rv32imac  os=freestanding  abi=ilp32    object=raw
freestanding-spirv    isa=spirv     os=freestanding  abi=spirv    object=spv
`;

// the formats that register a debug model in mach 6.7.0; the flat images refuse the adapter's debug profile, since they
// carry neither a debug model nor linkable objects.
const debugCapableFormats = new Set(['elf', 'macho', 'coff', 'spv']);

/**
 * Stands in for one probe build, reproducing what mach 6.7.0 with std 9.4.1 answers for the probe module: std has no layer for a
 * freestanding os, so a `use std.print` there fails inside std whatever the object format, and a format with no debug
 * model refuses the profile's debug information.
 */
async function fakeProbe(root: string, key: string) {
    const manifest = await fs.readFile(path.join(root, 'mach.toml'), 'utf8');
    const section = manifest.match(new RegExp(`^\\[target\\.${key}]\\n(?:\\w+ = "\\S+"\\n)+`, 'm'))![0];
    const os = section.match(/^os = "(\S+)"$/m)![1];
    const of = section.match(/^of = "(\S+)"$/m)![1];
    // std.types is header-only and resolves anywhere; anything above it reaches for an os
    const source = await fs.readFile(path.join(root, 'src', 'probe.mach'), 'utf8');
    const usesStdOs = /^use .*\bstd\.(?!types\b)/m.test(source);

    if (os === 'freestanding' && usesStdOs) {
        return {
            code: 1,
            stdout: '',
            stderr: 'error[comptime.user_error]: std.print needs the files capability (std.system.capability.HAS_FILES)',
        };
    }
    if (!debugCapableFormats.has(of)) {
        return {
            code: 2,
            stdout: '',
            stderr: 'error[compiler.internal]: debug info was requested, but this target registers no debug model',
        };
    }
    return {code: 0, stdout: '', stderr: ''};
}

/** A std checkout the probe can find, standing in for the one installed beside the compiler. */
async function makeStd() {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-mach-std'));
    await fs.writeFile(path.join(dir, 'mach.toml'), '[project]\nid = "std"\n');
    return dir;
}

/** Starts a compiler the way a prediscovered deploy does, which skips version and override discovery. */
function start(compiler: MachCompiler) {
    return compiler.initialise(new Date(), {libs: {}} as any, true);
}

/** Answers `mach info targets`, `mach dep pull` and each probe build the way mach 6.7.0 with std 9.4.1 does. */
function fakeMach(compiler: MachCompiler) {
    vi.spyOn(compiler, 'exec').mockImplementation(async (_exe, args) => {
        if (args[0] === 'info') return {code: 0, stdout: infoTargets, stderr: ''} as any;
        if (args[0] === 'dep') return {code: 0, stdout: '', stderr: ''} as any;
        return (await fakeProbe(args[1], args[5])) as any;
    });
}

function makeMach(stdPath: string) {
    return new MachCompiler(
        makeFakeCompilerInfo({id: 'mach', exe: '/opt/compiler-explorer/mach-6.7.0/mach', lang: 'mach', libsArr: []}),
        makeCompilationEnvironment({languages, props: {'compiler.mach.stdPath': stdPath}}),
    );
}

describe('Mach project layout', () => {
    let compiler: MachCompiler;
    let stdPath: string;

    beforeAll(async () => {
        stdPath = await makeStd();
        compiler = makeMach(stdPath);
        fakeMach(compiler);
        expect(await start(compiler)).toBe(compiler);
    });

    afterAll(async () => {
        await fs.rm(stdPath, {recursive: true, force: true});
    });

    it('offers only the tuples that build under the profile, named by the dimensions that disambiguate them', async () => {
        expect(compiler.targets().map(t => t.name)).toEqual([
            'linux-x86_64',
            'linux-aarch64',
            'linux-riscv64-lp64',
            'linux-riscv64-lp64f',
            'linux-riscv64-lp64d',
            'darwin-x86_64',
            'darwin-aarch64',
            'windows-x86_64',
        ]);
    });

    it('probes every tuple `mach info targets` reports at startup, against one project with std realized once', async () => {
        expect((await compiler.supportedTuples()).length).toEqual(23);
        const calls = (compiler.exec as any).mock.calls.map((call: any[]) => call[1]);
        expect(calls.filter((args: string[]) => args[0] === 'dep')).toHaveLength(1);
        expect(calls.filter((args: string[]) => args[0] === 'build')).toHaveLength(23);
    });

    it('declares every offered target with its object format, the source entry and the bundled std', async () => {
        const manifest = compiler.manifest(compiler.targets());
        expect(manifest).toContain(
            '[target.linux-riscv64-lp64d]\nisa = "rv64gc"\nos = "linux"\nabi = "lp64d"\nof = "elf"\n',
        );
        // the format is written out, not assumed: leaving it off takes the os default, which for freestanding is raw
        expect(manifest).toContain(
            '[target.darwin-aarch64]\nisa = "aarch64"\nos = "darwin"\nabi = "aapcs64"\nof = "macho"\n',
        );
        expect(manifest).toContain('[artifact.example]\nkind = "bin"\nentry = "example.mach"\n');
        expect(manifest).toContain(
            '[target.windows-x86_64]\nisa = "x86_64"\nos = "windows"\nabi = "win64"\nof = "coff"\n',
        );
        // std has no freestanding os layer, so no tuple on that os survives whatever its object format
        expect(manifest).not.toContain('freestanding');
        expect(manifest).toContain(`[dep.std]\npath = ${JSON.stringify(stdPath)}\n`);
    });

    it('builds the project root and disassembles the module object', () => {
        const root = path.join('/tmp', 'ce');
        const input = path.join(root, 'src', 'example.mach');
        expect(compiler.orderArguments(['--emit', 'obj'], input, [], [], [], [], ['-O2'], [])).toEqual([
            'build',
            root,
            '--diagnostics=json',
            '--emit',
            'obj',
            '-O2',
        ]);
        expect(compiler.getOutputFilename(root, 'output')).toEqual(
            path.join(root, 'out', 'obj', 'example', 'example.o'),
        );
        expect(compiler.getExecutableFilename(root, 'output')).toEqual(path.join(root, 'out', 'bin', 'example'));
    });

    it('states the compilers whose diagnostic records the adapter reads, whatever the compiler at hand', () => {
        const env = makeCompilationEnvironment({languages});
        const at = (semver?: string) =>
            new MachCompiler(makeFakeCompilerInfo({id: 'mach', exe: '/usr/bin/mach', lang: 'mach', semver}), env)
                .manifest([])
                .split('\n')
                .filter(line => line.startsWith('mach = '));
        expect(at('6.7.0')).toEqual(['mach = "^6.5"']);
        expect(at(undefined)).toEqual(['mach = "^6.5"']);
    });

    it('takes std from beside the executable by default', () => {
        const env = makeCompilationEnvironment({languages});
        const installed = new MachCompiler(
            makeFakeCompilerInfo({id: 'mach670', exe: '/opt/compiler-explorer/mach-6.7.0/mach', lang: 'mach'}),
            env,
        );
        expect(installed.manifest([])).toContain('[dep.std]\npath = "/opt/compiler-explorer/mach-6.7.0/std"\n');
    });

    it('refuses to offer any target when the compiler has no std', async () => {
        const env = makeCompilationEnvironment({languages});
        const bare = new MachCompiler(
            makeFakeCompilerInfo({
                id: 'machbare',
                exe: path.join(os.tmpdir(), 'ce-no-such-mach', 'mach'),
                lang: 'mach',
            }),
            env,
        );
        const error = vi.spyOn(logger, 'error').mockImplementation(() => logger);
        expect(await start(bare)).toBeNull();
        expect(String((error.mock.calls[0] as unknown[])[1])).toContain('set compiler.machbare.stdPath');
        error.mockRestore();
        expect(() => bare.targets()).toThrow('targets read before initialise() probed them');
    });

    it('takes std from stdPath when a compiler names one', () => {
        const env = makeCompilationEnvironment({languages, props: {'compiler.machdev.stdPath': '/src/mach/dep/std'}});
        const dev = new MachCompiler(makeFakeCompilerInfo({id: 'machdev', exe: '/usr/bin/mach', lang: 'mach'}), env);
        expect(dev.manifest([])).toContain('[dep.std]\npath = "/src/mach/dep/std"\n');
    });
});
/**
 * Every project the adapter lays out puts the user's sources under `src/`, so CE's file names become module paths
 * rooted at the project id: `src/util/fmt.mach` is the module `example.util.fmt`.
 */
describe('Mach multi-file projects', () => {
    let compiler: MachCompiler;
    let dirPath: string;
    let stdPath: string;

    beforeAll(async () => {
        stdPath = await makeStd();
        compiler = makeMach(stdPath);
        fakeMach(compiler);
        await start(compiler);
    });

    beforeEach(async () => {
        dirPath = await fs.mkdtemp(path.join(os.tmpdir(), 'ce-mach-layout'));
        vi.mocked(compiler.exec).mockClear();
    });

    afterEach(async () => {
        await fs.rm(dirPath, {recursive: true, force: true});
    });

    afterAll(async () => {
        await fs.rm(stdPath, {recursive: true, force: true});
    });

    it('roots every source at src/ and keeps the directories CE gave them', async () => {
        const pull = vi.spyOn(compiler, 'exec').mockResolvedValue({code: 0, stdout: '', stderr: ''} as any);

        const {inputFilename} = await (compiler as any).writeAllFiles(dirPath, 'entry', [
            {filename: 'util/fmt.mach', contents: 'fmt'},
            {filename: 'other.mach', contents: 'other'},
        ]);

        expect(inputFilename).toEqual(path.join(dirPath, 'src', 'example.mach'));
        expect(((await fs.readdir(dirPath, {recursive: true})) as string[]).sort()).toEqual([
            'mach.toml',
            'src',
            path.join('src', 'example.mach'),
            path.join('src', 'other.mach'),
            'src/util',
            path.join('src', 'util', 'fmt.mach'),
        ]);
        expect(await fs.readFile(path.join(dirPath, 'src', 'util', 'fmt.mach'), 'utf8')).toEqual('fmt');
        // the pull is the only thing a compilation runs here: the targets were probed at startup
        expect(pull.mock.calls).toEqual([
            [
                '/opt/compiler-explorer/mach-6.7.0/mach',
                ['dep', 'pull', dirPath],
                expect.objectContaining({customCwd: dirPath}),
            ],
        ]);
    });

    it('refuses an extra file that would land outside the project', async () => {
        vi.spyOn(compiler, 'exec').mockResolvedValue({code: 0, stdout: '', stderr: ''} as any);
        await expect(
            (compiler as any).writeAllFiles(dirPath, 'entry', [{filename: '../escape.mach', contents: 'no'}]),
        ).rejects.toThrow();
    });

    it('fails the compilation when the dependency pull fails', async () => {
        vi.spyOn(compiler, 'exec').mockResolvedValue({code: 1, stdout: '', stderr: 'no std'} as any);
        await expect((compiler as any).writeAllFiles(dirPath, 'entry', [])).rejects.toThrow(
            'mach dep pull failed: no std',
        );
    });
});
/**
 * Each capture is what `mach build --diagnostics=json` wrote to stderr at 6.7.0 with std 9.4.1, verbatim: the
 * diagnostic, failure and summary records of an error, a warning, both at once, help and a one-edit fix, a note and a
 * two-edit fix over a multi-line span, a related site, a second file, a std file, a link failure and two command-line
 * refusals with no location, and a line with non-ASCII text before the span.
 */
describe('Mach diagnostics', () => {
    const root = '/tmp/compiler-explorer-compiler-mach';
    const inputFilename = `${root}/src/example.mach`;
    let compiler: MachCompiler;

    beforeAll(() => {
        compiler = new MachCompiler(
            makeFakeCompilerInfo({id: 'mach', exe: '/opt/compiler-explorer/mach-6.7.0/mach', lang: 'mach'}),
            makeCompilationEnvironment({languages}),
        );
    });

    function parseText(stderr: string) {
        return compiler.processExecutionResult({code: 1, stdout: '', stderr} as any, inputFilename).stderr;
    }

    function capture(name: string) {
        return readFileSync(path.join(__dirname, 'mach', 'diagnostics', `${name}.ndjson`), 'utf8');
    }

    function parse(name: string) {
        return parseText(capture(name));
    }

    /** Every marker as [file, line, column, endline, endcolumn, severity, text]. */
    function marks(lines: ReturnType<typeof parse>) {
        return lines
            .filter(line => line.tag)
            .map(({tag}) => [
                tag!.file,
                tag!.line,
                tag!.column,
                tag!.endline,
                tag!.endcolumn,
                tag!.severity,
                tag!.text,
            ]);
    }

    function texts(lines: ReturnType<typeof parse>) {
        return lines.map(line => line.text);
    }

    it('renders an error as its headline and location, marks its span, and ends with the tally', () => {
        const lines = parse('error');
        expect(texts(lines)).toEqual([
            'error[name.unresolved]: unresolved identifier `bogus`',
            ' --> src/example.mach:2:9',
            '1 error / 0 warnings',
        ]);
        expect(marks(lines)).toEqual([
            ['example.mach', 2, 9, 2, 14, 3, 'error[name.unresolved]: unresolved identifier `bogus`'],
        ]);
    });

    it('marks a warning at warning severity, and keeps it apart from an error', () => {
        expect(marks(parse('warning'))).toEqual([
            ['example.mach', 1, 20, 1, 25, 2, 'warning[import.unused]: unused import `usize`'],
        ]);
        const both = parse('warning-and-error');
        expect(marks(both).map(([, line, , , , severity]) => [line, severity])).toEqual([
            [4, 3],
            [1, 2],
        ]);
        expect(texts(both).at(-1)).toEqual('1 error / 1 warning');
    });

    it('shows no tally for a build with nothing to report', () => {
        expect(
            parseText(
                '{"schema":1,"record":"summary","errors":0,"warnings":0,"notes":0,"outcome":"success","exit_code":0}',
            ),
        ).toEqual([]);
    });

    it('folds help into the marker and offers the fix as a quick fix over the span it replaces', () => {
        const lines = parse('help-and-fix');
        expect(texts(lines)).toEqual([
            'error[name.unresolved]: unresolved identifier `helpr`',
            ' --> src/example.mach:6:9',
            '  = help: did you mean `helper`?',
            '  = fix: replace with `helper`',
            '1 error / 0 warnings',
        ]);
        expect(lines[0].tag).toEqual({
            file: 'example.mach',
            line: 6,
            column: 9,
            endline: 6,
            endcolumn: 14,
            severity: 3,
            text: 'error[name.unresolved]: unresolved identifier `helpr`\nhelp: did you mean `helper`?',
            fixes: [
                {
                    title: 'replace with `helper`',
                    edits: [{file: 'example.mach', line: 6, column: 9, endline: 6, endcolumn: 14, text: 'helper'}],
                },
            ],
        });
    });

    it('folds a note into the marker, and offers every edit of a two-edit fix as one quick fix', () => {
        const [headline] = parse('two-edit-fix');
        expect(headline.tag?.text).toEqual(
            'error[type.mismatch]: type mismatch: expected i64, found i32\n' +
                'note: mach has no implicit widening; cast the value with `value::Type`',
        );
        // the primary span covers all three lines of the expression
        expect([headline.tag?.line, headline.tag?.column, headline.tag?.endline, headline.tag?.endcolumn]).toEqual([
            2, 5, 4, 11,
        ]);
        // an insertion is an empty edit where its text goes
        expect(headline.tag?.fixes).toEqual([
            {
                title: 'cast the value to `i64`',
                edits: [
                    {file: 'example.mach', line: 2, column: 9, endline: 2, endcolumn: 9, text: '('},
                    {file: 'example.mach', line: 4, column: 10, endline: 4, endcolumn: 10, text: ')::i64'},
                ],
            },
        ]);
    });

    it('marks a related site with its label', () => {
        const lines = parse('related');
        expect(texts(lines)).toContain(' --> src/example.mach:1:5: previous definition here');
        expect(marks(lines)).toEqual([
            [
                'example.mach',
                2,
                5,
                2,
                8,
                3,
                'error[name.duplicate]: duplicate definition: `dup` is already bound in this scope',
            ],
            ['example.mach', 1, 5, 1, 8, 1, 'previous definition here'],
        ]);
    });

    it('names a second file by the path the project tree knows it by', () => {
        expect(marks(parse('second-file'))).toEqual([
            ['util/fmt.mach', 2, 9, 2, 13, 3, 'error[name.unresolved]: unresolved identifier `nope`'],
        ]);
    });

    it('marks nothing in a std file, which belongs to no editor, but still shows where it is', () => {
        const lines = parse('std');
        expect(marks(lines)).toEqual([]);
        expect(texts(lines)).toContain(' --> dep/std/src/print.mach:24:5');
    });

    it.each(['link', 'unknown-target', 'flag-conflict'])('shows a %s failure with no location, unmarked', name => {
        const lines = parse(name);
        expect(marks(lines)).toEqual([]);
        expect(texts(lines)).toHaveLength(2);
        expect(texts(lines)[0]).toMatch(/^error\[[a-z_.]+]: /);
    });

    it('turns byte columns into the editor columns of a line with non-ASCII text before the span', () => {
        const source = utils.splitLines(readFileSync(path.join(__dirname, 'mach', 'diagnostics', 'utf8.mach'), 'utf8'));
        const [headline] = toEditorColumns(parse('utf8'), (file, line) =>
            file === 'example.mach' ? source[line - 1] : undefined,
        );
        // `é` and `ö` are two bytes each: byte column 39 is the 37th character
        expect([headline.tag?.column, headline.tag?.endcolumn]).toEqual([37, 41]);
        expect(source[1].slice(36, 40)).toEqual('nope');
    });

    it('remaps the edits of a quick fix with the marker', () => {
        const [headline] = toEditorColumns(parse('help-and-fix'), () => '    ret hélpr();');
        expect(headline.tag?.fixes?.[0].edits[0]).toMatchObject({column: 9, endcolumn: 13});
    });

    describe('shows as written', () => {
        it('a line that is not a record', () => {
            expect(parseText(`building in ${root}/out`)).toEqual([{text: 'building in out'}]);
        });

        it('a record of a schema this reader does not know', () => {
            const record = '{"schema":2,"record":"diagnostic"}';
            expect(parseText(record)).toEqual([{text: record}]);
        });
    });

    it('skips a record that is not a diagnostic', () => {
        expect(parseText('{"schema":1,"record":"test","name":"example#t","outcome":"pass"}')).toEqual([]);
    });

    it('offers no quick fix whose edits reach outside the marked editor', () => {
        const record = JSON.parse(capture('help-and-fix').split('\n')[0]);
        record.fixes[0].edits[0].file = 'src/other.mach';
        expect(parseText(JSON.stringify(record))[0].tag?.fixes).toBeUndefined();
    });
});

/**
 * Under the production sandbox the compile directory is bind-mounted at `/app`, so the project root the adapter
 * passes to `mach build` is `/app` and the DWARF comp_dir it records is `/app` too. The capture below is a real
 * objdump of an object built with the project root at `/app`.
 */
describe('Mach asm with an /app project root', () => {
    it('attributes an /app source line to the editor', () => {
        const objdump = readFileSync(path.join(__dirname, 'mach', 'app-objdump.asm'), 'utf8');
        const parsed = new AsmParser().process(
            objdump,
            makeFakeParseFiltersAndOutputOptions({binaryObject: true, directives: true}),
        );

        expect(parsed.asm.filter(line => line.source).map(line => line.source)).toContainEqual({
            file: null,
            line: 9,
            mainsource: true,
        });
        expect(parsed.asm.map(line => line.text)).toContain('main:');
    });

    it('strips the /app root when filenames are not masked', () => {
        const objdump = readFileSync(path.join(__dirname, 'mach', 'app-objdump.asm'), 'utf8');
        const parsed = new AsmParser().process(
            objdump,
            makeFakeParseFiltersAndOutputOptions({binaryObject: true, directives: true, dontMaskFilenames: true}),
        );

        expect(parsed.asm.filter(line => line.source).map(line => line.source)).toContainEqual({
            file: 'src/example.mach',
            line: 9,
            mainsource: true,
        });
    });
});

describe('Mach binary asm', () => {
    // a Hello World executable, trimmed to the head of each function: std and its runtime are linked in beside main
    const objdump = readFileSync(path.join(__dirname, 'mach', 'hello-binary.asm'), 'utf8');

    it.each(['amazon', 'defaults'])('shows only the user functions under the %s properties', env => {
        const file = path.join(__dirname, '..', 'etc', 'config', `mach.${env}.properties`);
        const props = parseProperties(readFileSync(file, 'utf8'), file);
        const parsed = new AsmParser((key: string) => props[key]).process(
            objdump,
            makeFakeParseFiltersAndOutputOptions({binary: true, directives: true}),
        );
        expect(parsed.asm.map(line => line.text).filter(text => text.endsWith(':'))).toEqual(['main:']);
    });
});
