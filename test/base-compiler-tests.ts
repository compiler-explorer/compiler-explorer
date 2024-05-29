// Copyright (c) 2018, Compiler Explorer Authors
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

import {beforeAll, describe, expect, it} from 'vitest';

import {BaseCompiler} from '../lib/base-compiler.js';
import {BuildEnvSetupBase} from '../lib/buildenvsetup/index.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import {ClangCompiler} from '../lib/compilers/clang.js';
import {Win32Compiler} from '../lib/compilers/win32.js';
import {splitArguments} from '../lib/utils.js';
import {CompilerOverrideType, ConfiguredOverrides} from '../types/compilation/compiler-overrides.interfaces.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';

import {
    fs,
    makeCompilationEnvironment,
    makeFakeCompilerInfo,
    makeFakeParseFiltersAndOutputOptions,
    path,
    shouldExist,
} from './utils.js';

const languages = {
    'c++': {id: 'c++'},
} as const;

describe('Basic compiler invariants', () => {
    let ce: CompilationEnvironment;
    let compiler: BaseCompiler;

    const info: Partial<CompilerInfo> = {
        exe: '',
        remote: {
            target: 'foo',
            path: 'bar',
            cmakePath: 'cmake',
        },
        lang: 'c++',
        ldPath: [],
    };

    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
        compiler = new BaseCompiler(info as CompilerInfo, ce);
    });

    it('should recognize when optOutput has been request', () => {
        expect(compiler.optOutputRequested(['please', 'recognize', '-fsave-optimization-record'])).toBe(true);
        expect(compiler.optOutputRequested(['please', "don't", 'recognize'])).toBe(false);
    });
    it('should allow comments next to includes (Bug #874)', () => {
        expect(compiler.checkSource('#include <cmath> // std::(sin, cos, ...)')).toBeNull();
        const badSource = compiler.checkSource('#include </dev/null..> //Muehehehe');
        if (shouldExist(badSource)) {
            expect(badSource).toEqual('<stdin>:1:1: no absolute or relative includes please');
        }
    });
    it('should not warn of path-likes outside C++ includes (Bug #3045)', () => {
        function testIncludeG(text: string) {
            expect(compiler.checkSource(text)).toBeNull();
        }
        testIncludeG('#include <iostream>');
        testIncludeG('#include <iostream>  // <..>');
        testIncludeG('#include <type_traits> // for std::is_same_v<...>');
        testIncludeG('#include <ranges>      // for std::ranges::range<...> and std::ranges::range_type_v<...>');
        testIncludeG('#include <https://godbolt.com> // /home/');
    });
    it('should not allow path C++ includes', () => {
        function testIncludeNotG(text: string) {
            expect(compiler.checkSource(text)).toEqual('<stdin>:1:1: no absolute or relative includes please');
        }
        testIncludeNotG('#include <./.bashrc>');
        testIncludeNotG('#include </dev/null>  // <..>');
        testIncludeNotG('#include <../fish.config> // for std::is_same_v<...>');
        testIncludeNotG('#include <./>      // for std::ranges::range<...> and std::ranges::range_type_v<...>');
    });
    it('should skip version check if forced to', async () => {
        const newConfig: Partial<CompilerInfo> = {...info, explicitVersion: '123'};
        const forcedVersionCompiler = new BaseCompiler(newConfig as CompilerInfo, ce);
        const result = await forcedVersionCompiler.getVersion();
        expect(result.stdout).toEqual(['123']);
    });
});

describe('Compiler execution', () => {
    let ce: CompilationEnvironment;
    let compiler: BaseCompiler;
    // let compilerNoExec: BaseCompiler;
    let win32compiler: Win32Compiler;

    const executingCompilerInfo = makeFakeCompilerInfo({
        remote: {
            target: 'foo',
            path: 'bar',
            cmakePath: 'cmake',
        },
        lang: 'c++',
        ldPath: [],
        libPath: [],
        supportsExecute: true,
        supportsBinary: true,
        options: '--hello-abc -I"/opt/some thing 1.0/include" -march="magic 8bit"',
    });
    const win32CompilerInfo = makeFakeCompilerInfo({
        remote: {
            target: 'foo',
            path: 'bar',
            cmakePath: 'cmake',
        },
        lang: 'c++',
        ldPath: [],
        supportsExecute: true,
        supportsBinary: true,
        options: '/std=c++17 /I"C:/program files (x86)/Company name/Compiler 1.2.3/include" /D "MAGIC=magic 8bit"',
    });
    const noExecuteSupportCompilerInfo = makeFakeCompilerInfo({
        remote: {
            target: 'foo',
            path: 'bar',
            cmakePath: 'cmake',
        },
        lang: 'c++',
        ldPath: [],
        libPath: [],
    });
    const someOptionsCompilerInfo = makeFakeCompilerInfo({
        remote: {
            target: 'foo',
            path: 'bar',
            cmakePath: 'cmake',
        },
        lang: 'c++',
        ldPath: [],
        libPath: [],
        supportsExecute: true,
        supportsBinary: true,
        options: '--hello-abc -I"/opt/some thing 1.0/include"',
    });

    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
        compiler = new BaseCompiler(executingCompilerInfo, ce);
        win32compiler = new Win32Compiler(win32CompilerInfo, ce);
        // compilerNoExec = new BaseCompiler(noExecuteSupportCompilerInfo, ce);
    });

    // afterEach(() => restore());

    // function stubOutCallToExec(execStub, compiler, content, result, nthCall) {
    //     execStub.onCall(nthCall || 0).callsFake((compiler, args) => {
    //         const minusO = args.indexOf('-o');
    //         expect(minusO).toBeGreaterThanOrEqual(0);
    //         const output = args[minusO + 1];
    //         // Maybe we should mock out the FS too; but that requires a lot more work.
    //         fs.writeFileSync(output, content);
    //         result.filenameTransform = (x: string) => x;
    //         return Promise.resolve(result);
    //     });
    // }

    it('basecompiler should handle spaces in options correctly', () => {
        const userOptions = [];
        const filters = makeFakeParseFiltersAndOutputOptions({});
        const backendOptions = {};
        const inputFilename = 'example.cpp';
        const outputFilename = 'example.s';
        const libraries = [];

        const args = compiler.prepareArguments(
            userOptions,
            filters,
            backendOptions,
            inputFilename,
            outputFilename,
            libraries,
            [],
        );
        expect(args).toEqual([
            '-g',
            '-o',
            'example.s',
            '-S',
            '--hello-abc',
            '-I/opt/some thing 1.0/include',
            '-march=magic 8bit',
            'example.cpp',
        ]);
    });

    it('win32 compiler should handle spaces in options correctly', () => {
        const userOptions = [];
        const filters = makeFakeParseFiltersAndOutputOptions({});
        const backendOptions = {};
        const inputFilename = 'example.cpp';
        const outputFilename = 'example.s';
        const libraries = [];

        const win32args = win32compiler.prepareArguments(
            userOptions,
            filters,
            backendOptions,
            inputFilename,
            outputFilename,
            libraries,
            [],
        );
        expect(win32args).toEqual([
            '/nologo',
            '/FA',
            '/c',
            '/Faexample.s',
            '/Foexample.s.obj',
            '/std=c++17',
            '/IC:/program files (x86)/Company name/Compiler 1.2.3/include',
            '/D',
            'MAGIC=magic 8bit',
            'example.cpp',
        ]);
    });

    it('buildenv should handle spaces correctly', () => {
        const buildenv = new BuildEnvSetupBase(executingCompilerInfo, ce);
        expect(buildenv.getCompilerArch()).toEqual('magic 8bit');
    });

    it('buildenv compiler without target/march', () => {
        const buildenv = new BuildEnvSetupBase(noExecuteSupportCompilerInfo, ce);
        expect(buildenv.getCompilerArch()).toBe(false);
        expect(buildenv.compilerSupportsX86).toBe(true);
    });

    it('buildenv compiler without target/march but with options', () => {
        const buildenv = new BuildEnvSetupBase(someOptionsCompilerInfo, ce);
        expect(buildenv.getCompilerArch()).toBe(false);
        expect(buildenv.compilerSupportsX86).toBe(true);
    });

    it('compiler overrides should be sanitized', () => {
        const original_overrides: ConfiguredOverrides = [
            {
                name: CompilerOverrideType.env,
                values: [
                    {
                        name: 'somevar',
                        value: '123',
                    },
                    {
                        name: 'ABC$#%@6@5',
                        value: '456',
                    },
                    {
                        name: 'LD_PRELOAD',
                        value: '/path/to/my/malloc.so /bin/ls',
                    },
                ],
            },
        ];

        const sanitized = compiler.sanitizeCompilerOverrides(original_overrides);

        const execOptions = compiler.getDefaultExecOptions();

        compiler.applyOverridesToExecOptions(execOptions, sanitized);

        expect(execOptions.env).toHaveProperty('SOMEVAR');
        expect(execOptions.env['SOMEVAR']).toEqual('123');
        expect(execOptions.env).not.toHaveProperty('LD_PRELOAD');
        expect(execOptions.env).not.toHaveProperty('ABC$#%@6@5');
    });

    // it('should compile', async () => {
    //     const execStub = stub(compiler, 'exec');
    //     stubOutCallToExec(
    //         execStub,
    //         compiler,
    //         'This is the output file',
    //         {
    //             code: 0,
    //             okToCache: true,
    //             stdout: 'stdout',
    //             stderr: 'stderr',
    //         },
    //         undefined,
    //     );
    //     const result = await compiler.compile('source', 'options', {}, {}, false, [], {}, [], undefined);
    //     result.code.should.equal(0);
    //     result.compilationOptions.should.contain('options');
    //     result.compilationOptions.should.contain(result.inputFilename);
    //     result.okToCache.should.be.true;
    //     result.asm.should.deep.equal([{source: null, text: 'This is the output file', labels: []}]);
    //     result.stdout.should.deep.equal([{text: 'stdout'}]);
    //     result.stderr.should.deep.equal([{text: 'stderr'}]);
    //     result.popularArguments.should.deep.equal({});
    //     result.tools.should.deep.equal([]);
    //     execStub.called.should.be.true;
    // });
    //
    // it('should handle compilation failures', async () => {
    //     const execStub = stub(compiler, 'exec');
    //     stubOutCallToExec(
    //         execStub,
    //         compiler,
    //         'This is the output file',
    //         {
    //             code: 1,
    //             okToCache: true,
    //             stdout: '',
    //             stderr: 'oh noes',
    //         },
    //         undefined,
    //     );
    //     const result = await compiler.compile('source', 'options', {}, {}, false, [], {}, [], undefined);
    //     result.code.should.equal(1);
    //     result.asm.should.deep.equal([{labels: [], source: null, text: '<Compilation failed>'}]);
    // });
    //
    // it('should cache results (when asked)', async () => {
    //     const ceMock = mock(ce);
    //     const fakeExecResults = {
    //         code: 0,
    //         okToCache: true,
    //         stdout: 'stdout',
    //         stderr: 'stderr',
    //     };
    //     const execStub = stub(compiler, 'exec');
    //     stubOutCallToExec(execStub, compiler, 'This is the output file', fakeExecResults, undefined);
    //     const source = 'Some cacheable source';
    //     const options = 'Some cacheable options';
    //     ceMock
    //         .expects('cachePut')
    //         .withArgs(
    //             match({source, options}),
    //             match({
    //                 ...fakeExecResults,
    //                 stdout: [{text: 'stdout'}],
    //                 stderr: [{text: 'stderr'}],
    //             }),
    //         )
    //         .resolves();
    //     const uncachedResult = await compiler.compile(source, options, {}, {}, false, [], {}, [], undefined);
    //     uncachedResult.code.should.equal(0);
    //     ceMock.verify();
    // });
    //
    // it('should not cache results (when not asked)', async () => {
    //     const ceMock = mock(ce);
    //     const fakeExecResults = {
    //         code: 0,
    //         okToCache: false,
    //         stdout: 'stdout',
    //         stderr: 'stderr',
    //     };
    //     const execStub = stub(compiler, 'exec');
    //     stubOutCallToExec(execStub, compiler, 'This is the output file', fakeExecResults, undefined);
    //     ceMock.expects('cachePut').never();
    //     const source = 'Some cacheable source';
    //     const options = 'Some cacheable options';
    //     const uncachedResult = await compiler.compile(source, options, {}, {}, false, [], {}, [], undefined);
    //     uncachedResult.code.should.equal(0);
    //     ceMock.verify();
    // });
    //
    // it('should read from the cache (when asked)', async () => {
    //     const ceMock = mock(ce);
    //     const source = 'Some previously cached source';
    //     const options = 'Some previously cached options';
    //     ceMock.expects('cacheGet').withArgs(match({source, options})).resolves({code: 123});
    //     const cachedResult = await compiler.compile(source, options, {}, {}, false, [], {}, [], undefined);
    //     cachedResult.code.should.equal(123);
    //     ceMock.verify();
    // });
    //
    // it('should note read from the cache (when bypassed)', async () => {
    //     const ceMock = mock(ce);
    //     const fakeExecResults = {
    //         code: 0,
    //         okToCache: true,
    //         stdout: 'stdout',
    //         stderr: 'stderr',
    //     };
    //     const source = 'Some previously cached source';
    //     const options = 'Some previously cached options';
    //     ceMock.expects('cacheGet').never();
    //     const execStub = stub(compiler, 'exec');
    //     stubOutCallToExec(execStub, compiler, 'This is the output file', fakeExecResults, undefined);
    //     const uncachedResult = await compiler.compile(source, options, {}, {}, true, [], {}, [], undefined);
    //     uncachedResult.code.should.equal(0);
    //     ceMock.verify();
    // });
    //
    // it('should execute', async () => {
    //     const execMock = mock(exec);
    //     const execStub = stub(compiler, 'exec');
    //     stubOutCallToExec(
    //         execStub,
    //         compiler,
    //         'This is the output asm file',
    //         {
    //             code: 0,
    //             okToCache: true,
    //             stdout: 'asm stdout',
    //             stderr: 'asm stderr',
    //         },
    //         0,
    //     );
    //     stubOutCallToExec(
    //         execStub,
    //         compiler,
    //         'This is the output binary file',
    //         {
    //             code: 0,
    //             okToCache: true,
    //             stdout: 'binary stdout',
    //             stderr: 'binary stderr',
    //         },
    //         1,
    //     );
    //     execMock.expects('sandbox').withArgs(match.string, match.array, match.object).resolves({
    //         code: 0,
    //         stdout: 'exec stdout',
    //         stderr: 'exec stderr',
    //     });
    //     const result = await compiler.compile('source', 'options', {}, {execute: true}, false, [], {}, [], undefined);
    //     result.code.should.equal(0);
    //     result.execResult.didExecute.should.be.true;
    //     result.stdout.should.deep.equal([{text: 'asm stdout'}]);
    //     result.execResult.stdout.should.deep.equal([{text: 'exec stdout'}]);
    //     result.execResult.buildResult.stdout.should.deep.equal([{text: 'binary stdout'}]);
    //     result.stderr.should.deep.equal([{text: 'asm stderr'}]);
    //     result.execResult.stderr.should.deep.equal([{text: 'exec stderr'}]);
    //     result.execResult.buildResult.stderr.should.deep.equal([{text: 'binary stderr'}]);
    //     execMock.verify();
    // });
    //
    // it('should execute with an execution wrapper', async () => {
    //     const executionWrapper = '/some/wrapper/script.sh';
    //     (compiler as any).compiler.executionWrapper = executionWrapper;
    //     const execMock = mock(exec);
    //     const execStub = stub(compiler, 'exec');
    //     stubOutCallToExec(
    //         execStub,
    //         compiler,
    //         'This is the output asm file',
    //         {
    //             code: 0,
    //             okToCache: true,
    //             stdout: 'asm stdout',
    //             stderr: 'asm stderr',
    //         },
    //         0,
    //     );
    //     stubOutCallToExec(
    //         execStub,
    //         compiler,
    //         'This is the output binary file',
    //         {
    //             code: 0,
    //             okToCache: true,
    //             stdout: 'binary stdout',
    //             stderr: 'binary stderr',
    //         },
    //         1,
    //     );
    //     execMock.expects('sandbox').withArgs(executionWrapper, match.array, match.object).resolves({
    //         code: 0,
    //         stdout: 'exec stdout',
    //         stderr: 'exec stderr',
    //     });
    //     await compiler.compile('source', 'options', {}, {execute: true}, false, [], {}, [], undefined);
    //     execMock.verify();
    // });
    //
    // it('should not execute where not supported', async () => {
    //     const execMock = mock(exec);
    //     const execStub = stub(compilerNoExec, 'exec');
    //     stubOutCallToExec(
    //         execStub,
    //         compilerNoExec,
    //         'This is the output asm file',
    //         {
    //             code: 0,
    //             okToCache: true,
    //             stdout: 'asm stdout',
    //             stderr: 'asm stderr',
    //         },
    //         0,
    //     );
    //     stubOutCallToExec(
    //         execStub,
    //         compilerNoExec,
    //         'This is the output binary file',
    //         {
    //             code: 0,
    //             okToCache: true,
    //             stdout: 'binary stdout',
    //             stderr: 'binary stderr',
    //         },
    //         1,
    //     );
    //     const result = await compilerNoExec.compile(
    //         'source',
    //         'options',
    //         {},
    //         {execute: true},
    //         false,
    //         [],
    //         {},
    //         [],
    //         undefined,
    //     );
    //     result.code.should.equal(0);
    //     result.execResult.didExecute.should.be.false;
    //     result.stdout.should.deep.equal([{text: 'asm stdout'}]);
    //     result.execResult.stdout.should.deep.equal([]);
    //     result.execResult.buildResult.stdout.should.deep.equal([{text: 'binary stdout'}]);
    //     result.stderr.should.deep.equal([{text: 'asm stderr'}]);
    //     result.execResult.stderr.should.deep.equal([{text: 'Compiler does not support execution'}]);
    //     result.execResult.buildResult.stderr.should.deep.equal([{text: 'binary stderr'}]);
    //     execMock.verify();
    // });
    //
    // it('should demangle', async () => {
    //     const withDemangler = {...noExecuteSupportCompilerInfo, demangler: 'demangler-exe', demanglerType: 'cpp'};
    //     const compiler = new BaseCompiler(withDemangler, ce);
    //     const execStub = stub(compiler, 'exec');
    //     stubOutCallToExec(
    //         execStub,
    //         compiler,
    //         'someMangledSymbol:\n',
    //         {
    //             code: 0,
    //             okToCache: true,
    //             stdout: 'stdout',
    //             stderr: 'stderr',
    //         },
    //         undefined,
    //     );
    //     execStub.onCall(1).callsFake((demangler, args, options) => {
    //         demangler.should.equal('demangler-exe');
    //         options.input.should.equal('someMangledSymbol');
    //         return Promise.resolve({
    //             code: 0,
    //             filenameTransform: (x: any) => x,
    //             stdout: 'someDemangledSymbol\n',
    //             stderr: '',
    //         });
    //     });
    //
    //     const result = await compiler.compile('source', 'options', {}, {demangle: true}, false, [], {}, [], undefined);
    //     result.code.should.equal(0);
    //     result.asm.should.deep.equal([{source: null, labels: [], text: 'someDemangledSymbol:'}]);
    //     // TODO all with demangle: false
    // });
    //
    // async function objdumpTest(type, expectedArgs) {
    //     const withObjdumper = {
    //         ...noExecuteSupportCompilerInfo,
    //         objdumper: 'objdump-exe',
    //         objdumperType: type,
    //     };
    //     const compiler = new BaseCompiler(withObjdumper, ce);
    //     const execStub = stub(compiler, 'exec');
    //     execStub.onCall(0).callsFake((objdumper, args, options) => {
    //         objdumper.should.equal('objdump-exe');
    //         args.should.deep.equal(expectedArgs);
    //         options.maxOutput.should.equal(123456);
    //         return Promise.resolve({
    //             code: 0,
    //             filenameTransform: x => x,
    //             stdout: '<No output file output>',
    //             stderr: '',
    //         });
    //     });
    //     compiler.supportsObjdump().should.be.true;
    //     const result = await compiler.objdump(
    //         'output',
    //         {},
    //         123456,
    //         true,
    //         true,
    //         false,
    //         false,
    //         makeFakeParseFiltersAndOutputOptions({}),
    //     );
    //     result.asm.should.deep.equal('<No output file output>');
    // }

    // it('should run default objdump properly', async () => {
    //     return objdumpTest('default', ['-d', 'output', '-l', '--insn-width=16', '-C', '-M', 'intel']);
    // });
    //
    // it('should run binutils objdump properly', async () => {
    //     return objdumpTest('binutils', ['-d', 'output', '-l', '--insn-width=16', '-C', '-M', 'intel']);
    // });
    //
    // it('should run ELF Tool Chain objdump properly', async () => {
    //     return objdumpTest('elftoolchain', ['-d', 'output', '-l', '-C', '-M', 'intel']);
    // });
    //
    // it('should run LLVM objdump properly', async () => {
    //     return objdumpTest('llvm', ['-d', 'output', '-l', '-C', '--x86-asm-syntax=intel']);
    // });

    it('should run process opt output', async () => {
        const test = `--- !Missed
Pass: inline
Name: NeverInline
DebugLoc: { File: example.cpp, Line: 4, Column: 21 }
Function: main
Args: []
...
`;
        const dirPath = await compiler.newTempDir();
        const optPath = path.join(dirPath, 'temp.out');
        await fs.writeFile(optPath, test);
        expect(await compiler.processOptOutput(optPath)).toEqual([
            {
                Args: [],
                DebugLoc: {Column: 21, File: 'example.cpp', Line: 4},
                Function: 'main',
                Name: 'NeverInline',
                Pass: 'inline',
                displayString: '',
                optType: 'Missed',
            },
        ]);
    });

    it('should normalize extra file path', () => {
        const withDemangler = {...noExecuteSupportCompilerInfo, demangler: 'demangler-exe', demanglerType: 'cpp'};
        const compiler = new BaseCompiler(withDemangler, ce) as any; // to get to the protected...
        if (process.platform === 'win32') {
            expect(compiler.getExtraFilepath('c:/tmp/somefolder', 'test.h')).toEqual('c:\\tmp\\somefolder\\test.h');
        } else {
            expect(compiler.getExtraFilepath('/tmp/somefolder', 'test.h')).toEqual('/tmp/somefolder/test.h');
        }

        expect(() => compiler.getExtraFilepath('/tmp/somefolder', '../test.h')).toThrow(Error);
        expect(() => compiler.getExtraFilepath('/tmp/somefolder', './../test.h')).toThrow(Error);

        expect(compiler.getExtraFilepath('/tmp/somefolder', '/tmp/someotherfolder/test.h')).toEqual(
            path.normalize('/tmp/somefolder/tmp/someotherfolder/test.h'),
        );

        if (process.platform === 'win32') {
            expect(compiler.getExtraFilepath('/tmp/somefolder', '\\test.h')).toEqual('\\tmp\\somefolder\\test.h');
        }

        expect(() => compiler.getExtraFilepath('/tmp/somefolder', 'test_hello/../../etc/passwd')).toThrow(Error);

        if (process.platform === 'win32') {
            expect(compiler.getExtraFilepath('c:/tmp/somefolder', 'test.txt')).toEqual('c:\\tmp\\somefolder\\test.txt');
        } else {
            expect(compiler.getExtraFilepath('/tmp/somefolder', 'test.txt')).toEqual('/tmp/somefolder/test.txt');
        }

        expect(compiler.getExtraFilepath('/tmp/somefolder', 'subfolder/hello.h')).toEqual(
            path.normalize('/tmp/somefolder/subfolder/hello.h'),
        );
    });
});

describe('getDefaultExecOptions', () => {
    let ce: CompilationEnvironment;

    const noExecuteSupportCompilerInfo = makeFakeCompilerInfo({
        remote: {
            target: 'foo',
            path: 'bar',
            cmakePath: 'cmake',
        },
        lang: 'c++',
        ldPath: [],
        libPath: [],
        extraPath: ['/tmp/p1', '/tmp/p2'],
    });

    beforeAll(() => {
        ce = makeCompilationEnvironment({
            languages,
            props: {
                environmentPassThrough: '',
                ninjaPath: '/usr/local/ninja',
            },
        });
    });

    it('Have all the paths', () => {
        const compiler = new BaseCompiler(noExecuteSupportCompilerInfo, ce);
        const options = compiler.getDefaultExecOptions();
        expect(options.env).toHaveProperty('PATH');

        const paths = options.env.PATH.split(path.delimiter);
        expect(paths).toEqual(['/usr/local/ninja', '/tmp/p1', '/tmp/p2']);
    });
});

describe('Target hints', () => {
    let ce: CompilationEnvironment;

    const noExecuteSupportCompilerInfo = makeFakeCompilerInfo({
        exe: '/usr/bin/clang++',
        lang: 'c++',
        supportsTargetIs: true,
        supportsTarget: true,
        ldPath: [],
        libPath: [],
        extraPath: [],
    });

    beforeAll(() => {
        ce = makeCompilationEnvironment({
            languages,
            props: {
                environmentPassThrough: '',
                ninjaPath: '/usr/local/ninja',
            },
        });
    });

    it('Should determine the target for Clang', async () => {
        const compiler = new ClangCompiler(noExecuteSupportCompilerInfo, ce);

        const args =
            '-gdwarf-4 -g -o output.s -mllvm --x86-asm-syntax=intel -S --gcc-toolchain=/opt/compiler-explorer/gcc-13.2.0 -fcolor-diagnostics -fno-crash-diagnostics --target=riscv64 example.cpp -isystem/opt/compiler-explorer/libs/abseil';
        const argArray = splitArguments(args);
        const hint = compiler.getTargetHintFromCompilerArgs(argArray);
        expect(hint).toBe('riscv64');
        const iset = await compiler.getInstructionSetFromCompilerArgs(argArray);
        expect(iset).toBe('riscv64');
    });
});
