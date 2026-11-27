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

import fs from 'node:fs';
import path from 'node:path';

import {afterEach, beforeAll, describe, expect, it, vi} from 'vitest';

import {CompilationEnvironment} from '../lib/compilation-env.js';
import {D8Compiler} from '../lib/compilers/d8.js';
import {Dex2OatCompiler} from '../lib/compilers/dex2oat.js';
import {JavaCompiler} from '../lib/compilers/java.js';
import {PascalWinCompiler} from '../lib/compilers/pascal-win.js';
import {SpiceCompiler} from '../lib/compilers/spice.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../types/features/filters.interfaces.js';
import {makeCompilationEnvironment} from './utils.js';

// A compiler object serves every request for its compiler id, and requests interleave across awaits: so anything
// worked out for one request has to stay with that request. Each test sets up two requests on one object, the second
// before the first is finished, and checks the first still gets its own.

const mapFilesRead = vi.hoisted(() => [] as string[]);
vi.mock('../lib/mapfiles/map-file-delphi.js', () => ({
    MapFileReaderDelphi: class {
        constructor(filename: string) {
            mapFilesRead.push(filename);
        }
        run() {}
    },
}));
vi.mock('../lib/pe32-support.js', () => ({
    PELabelReconstructor: class {
        constructor(
            public asmLines: string[],
            _dontLabel: boolean,
            _reader: unknown,
        ) {}
        run() {}
    },
}));

const languages = {
    spice: {id: 'spice'},
    pascal: {id: 'pascal'},
    java: {id: 'java'},
    androidJava: {id: 'android-java'},
};

function compilerInfo(lang: string, exe = 'compiler') {
    return {exe, remote: true, lang, libPath: []} as unknown as CompilerInfo;
}

function isolate(dir: string, name: string) {
    return path.join(path.sep, 'tmp', dir, name);
}

describe('Per-request state on shared compiler objects', () => {
    let env: CompilationEnvironment;

    beforeAll(() => {
        env = makeCompilationEnvironment({languages});
    });

    afterEach(() => {
        vi.restoreAllMocks();
        mapFilesRead.length = 0;
    });

    it('Spice names the IR file after the optimisation level of its own request', () => {
        const compiler = new SpiceCompiler(compilerInfo('spice'), env);
        const filters = {} as ParseFiltersAndOutputOptions;
        const first = isolate('first', 'example.spice');

        compiler.optionsForFilter(filters, isolate('first', 'out.s'), ['-O2']);
        compiler.optionsForFilter(filters, isolate('second', 'out.s'), ['-Os']);

        expect(compiler.getIrOutputFilename(first, filters, undefined, ['build', '-O2'])).toEqual(
            isolate('first', 'example-ir-code-O2.ll'),
        );
        expect(compiler.getIrOutputFilename(first, filters, undefined, ['-Os'])).toEqual(
            isolate('first', 'example-ir-code-O4.ll'),
        );
        expect(compiler.getIrOutputFilename(first, filters, undefined, ['-O3', '-O0'])).toEqual(
            isolate('first', 'example-ir-code.ll'),
        );
    });

    it('Pascal (Windows) reads the map file beside its own output', () => {
        const compiler = new PascalWinCompiler(compilerInfo('pascal', 'pascal.exe') as any, env);
        const firstFilters = {} as ParseFiltersAndOutputOptions;
        const secondFilters = {} as ParseFiltersAndOutputOptions;

        compiler.optionsForFilter(firstFilters, isolate('first', 'prog.exe'));
        compiler.optionsForFilter(secondFilters, isolate('second', 'prog.exe'));
        firstFilters.preProcessBinaryAsmLines!([]);
        secondFilters.preProcessBinaryAsmLines!([]);

        expect(mapFilesRead).toEqual([isolate('first', 'prog.map'), isolate('second', 'prog.map')]);
    });

    it('Java puts its own libraries on the classpath', async () => {
        const compiler = new JavaCompiler(compilerInfo('java'), env);
        vi.spyOn(compiler, 'findLibVersion').mockImplementation(
            lib =>
                ({path: [`/opt/libs/${lib.id}.jar`], liblink: [], staticliblink: [], libpath: [], options: []}) as any,
        );
        const exec = vi.spyOn(compiler, 'exec').mockResolvedValue({
            code: 0,
            stdout: '',
            stderr: '',
            filenameTransform: (f: string) => f,
        } as any);
        const firstInput = isolate('first', 'example.java');
        const firstOptions = compiler.prepareArguments(
            [],
            {},
            {},
            firstInput,
            isolate('first', 'example.class'),
            [{id: 'guava', version: '1'}],
            [],
        );
        compiler.prepareArguments([], {}, {}, isolate('second', 'example.java'), '', [], []);

        await compiler.runCompiler('javac', firstOptions, firstInput, {} as any);

        const args = exec.mock.calls[0][1];
        expect(args.slice(0, 2)).toEqual(['-cp', '/opt/libs/guava.jar']);
        expect(args.at(-1)).toEqual(firstInput);
    });

    it('D8 reads its own library paths back from its arguments', () => {
        const compiler = new D8Compiler(compilerInfo('android-java'), env);
        vi.spyOn(compiler, 'findLibVersion').mockImplementation(
            lib =>
                ({path: [`/opt/libs/${lib.id}.jar`], liblink: [], staticliblink: [], libpath: [], options: []}) as any,
        );
        const firstOptions = compiler.prepareArguments(
            ['--release'],
            {},
            {},
            isolate('first', 'example.java'),
            '',
            [{id: 'guava', version: '1'}],
            [],
        );
        const secondOptions = compiler.prepareArguments([], {}, {}, isolate('second', 'example.java'), '', [], []);

        expect(D8Compiler.getLibPathsFromOptions(firstOptions)).toEqual(['/opt/libs/guava.jar']);
        expect(D8Compiler.getLibPathsFromOptions(secondOptions)).toEqual([]);
        expect(D8Compiler.getClasspathArgument(['/a.jar', '/b.jar'])).toEqual(['-cp', '/a.jar:/b.jar']);
    });

    it('dex2oat decides full output from the options of each request', async () => {
        const compiler = new Dex2OatCompiler(compilerInfo('android-java', 'dex2oat'), env);
        const oatdump = fs.readFileSync('test/android/java/oatdump.asm', 'utf-8');
        const filters = compiler.getDefaultFilters();

        const [full, parsed] = await Promise.all([
            compiler.processAsm({asm: [{text: oatdump}]}, filters, [compiler.fullOutputFlag]),
            compiler.processAsm({asm: [{text: oatdump}]}, filters, []),
        ]);

        expect(full.asm).toEqual([{text: oatdump, source: null}]);
        expect(parsed.asm.length).toBeGreaterThan(1);
    });
});
