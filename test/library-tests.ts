// Copyright (c) 2024, Compiler Explorer Authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import path from 'path';

import fs from 'fs-extra';
import {beforeAll, describe, expect, it} from 'vitest';

import {BaseCompiler} from '../lib/base-compiler.js';
import {BuildEnvSetupBase} from '../lib/buildenvsetup/base.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import {FortranCompiler} from '../lib/compilers/fortran.js';
import {ClientOptionsType, OptionsHandlerLibrary} from '../lib/options-handler.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';

import {makeCompilationEnvironment} from './utils.js';

const languages = {
    'c++': {
        id: 'c++',
    },
    fortran: {
        id: 'fortran',
    },
} as const;

describe('Library directories (c++)', () => {
    let ce: CompilationEnvironment;
    let compiler: BaseCompiler;

    const info: Partial<CompilerInfo> = {
        exe: '',
        remote: {
            target: 'foo',
            path: 'bar',
            cmakePath: 'cmake',
            basePath: '/',
        },
        lang: 'c++',
        ldPath: [],
        libPath: [],
        libsArr: ['fmt.10', 'qt.660', 'cpptrace.030'],
    };

    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
        compiler = new BaseCompiler(info as CompilerInfo, ce);
        (compiler as any).buildenvsetup = new BuildEnvSetupBase(info as CompilerInfo, ce);
        compiler.initialiseLibraries({
            libs: {
                'c++': {
                    fmt: {
                        id: 'fmt',
                        name: '{fmt}',
                        versions: {
                            10: {
                                version: '1.0',
                                liblink: ['fmtd'],
                                libpath: [],
                                path: ['/opt/compiler-explorer/libs/fmt/1.0/include'],
                                packagedheaders: false,
                            },
                        },
                    } as unknown as OptionsHandlerLibrary,
                    qt: {
                        id: 'qt',
                        name: 'Qt',
                        versions: {
                            660: {
                                version: '6.6.0',
                                liblink: ['Qt6Core'],
                                libpath: [],
                                path: ['/opt/compiler-explorer/libs/qt/6.6.0/include'],
                                options: ['-DQT_NO_VERSION_TAGGING'],
                                packagedheaders: true,
                            },
                        },
                    } as unknown as OptionsHandlerLibrary,
                    cpptrace: {
                        id: 'cpptrace',
                        name: 'cpptrace',
                        versions: {
                            '030': {
                                version: '0.3.0',
                                staticliblink: ['cpptrace'],
                                dependencies: ['dwarf', 'dl', 'z'],
                                libpath: [],
                                path: ['/opt/compiler-explorer/libs/cpptrace/v0.3.0/include'],
                                packagedheaders: true,
                            },
                        },
                    } as unknown as OptionsHandlerLibrary,
                },
            },
        } as unknown as ClientOptionsType);
    });

    it('should add libpaths and link to libraries', () => {
        const links = compiler.getSharedLibraryLinks([{id: 'fmt', version: '10'}]);
        expect(links).toContain('-lfmtd');

        const fmtpaths = (compiler as any).getSharedLibraryPathsAsArguments(
            [{id: 'fmt', version: '10'}],
            undefined,
            undefined,
            '/tmp/compiler-explorer-compiler-123',
        );
        expect(fmtpaths).toContain('-L./lib');

        const qtpaths = (compiler as any).getSharedLibraryPathsAsArguments(
            [{id: 'qt', version: '660'}],
            undefined,
            undefined,
            '/tmp/compiler-explorer-compiler-123',
        );

        expect(qtpaths).toContain('-L' + path.normalize('/tmp/compiler-explorer-compiler-123/qt/lib'));
    });

    it('should add libpaths and link to libraries when using nsjail', () => {
        (compiler as any).executionType = 'nsjail';

        const fmtpaths = (compiler as any).getSharedLibraryPathsAsArguments(
            [{id: 'fmt', version: '10'}],
            undefined,
            undefined,
            '/tmp/compiler-explorer-compiler-123',
        );

        expect(fmtpaths).toContain('-L' + path.normalize('/tmp/compiler-explorer-compiler-123/fmt/lib'));

        const qtpaths = (compiler as any).getSharedLibraryPathsAsArguments(
            [{id: 'qt', version: '660'}],
            undefined,
            undefined,
            '/tmp/compiler-explorer-compiler-123',
        );

        expect(qtpaths).toContain('-L' + path.normalize('/tmp/compiler-explorer-compiler-123/qt/lib'));
    });

    it('should add extra include paths when using packagedheaders', () => {
        (compiler as any).executionType = 'nsjail';

        const fmtpaths = (compiler as any).getIncludeArguments(
            [{id: 'fmt', version: '10'}],
            '/tmp/compiler-explorer-compiler-123',
        );
        expect(fmtpaths).not.toContain('-I/tmp/compiler-explorer-compiler-123/fmt/include');
        expect(fmtpaths).toContain('-I/opt/compiler-explorer/libs/fmt/1.0/include');

        const qtpaths = (compiler as BaseCompiler).getIncludeArguments(
            [{id: 'qt', version: '660'}],
            '/tmp/compiler-explorer-compiler-123',
        );

        expect(qtpaths).toContain('-I/opt/compiler-explorer/libs/qt/6.6.0/include');
        // paths in options are passed through, but this is a composited path and so is windows formatted
        expect(qtpaths).toContain('-I' + path.normalize('/tmp/compiler-explorer-compiler-123/qt/include'));
    });

    it('should set LD_LIBRARY_PATH when executing', () => {
        (compiler as any).sandboxType = 'nsjail';

        const qtpaths = (compiler as BaseCompiler).getSharedLibraryPathsAsLdLibraryPathsForExecution(
            {
                libraries: [{id: 'qt', version: '660'}],
                compiler: undefined,
                source: '',
                options: [],
                backendOptions: undefined,
                tools: [],
                files: [],
            },
            '/tmp/compiler-explorer-compiler-123',
        );

        expect(qtpaths).toContain(path.normalize('/tmp/compiler-explorer-compiler-123/qt/lib'));
    });

    it('should add libpaths and link when statically linking', () => {
        (compiler as any).executionType = 'nsjail';

        const staticlinks = compiler.getStaticLibraryLinks([{id: 'cpptrace', version: '030'}], []);
        expect(staticlinks).toContain('-lcpptrace');
        expect(staticlinks).toContain('-ldwarf');
        expect(staticlinks).toContain('-ldl');
        expect(staticlinks).toContain('-lz');

        const libpaths = (compiler as any).getSharedLibraryPathsAsArguments(
            [{id: 'cpptrace', version: '030'}],
            undefined,
            undefined,
            '/tmp/compiler-explorer-compiler-123',
        );

        expect(libpaths).toContain('-L' + path.normalize('/tmp/compiler-explorer-compiler-123/cpptrace/lib'));
    });
});

describe('Library directories (fortran)', () => {
    let ce: CompilationEnvironment;
    let compiler: BaseCompiler;

    const info: Partial<CompilerInfo> = {
        exe: '',
        remote: {
            target: 'foo',
            path: 'bar',
            cmakePath: 'cmake',
            basePath: '/',
        },
        lang: 'fortran',
        ldPath: [],
        libPath: [],
        libsArr: ['json_fortran.830', 'curl.7831'],
    };

    beforeAll(() => {
        ce = makeCompilationEnvironment({languages});
        compiler = new FortranCompiler(info as CompilerInfo, ce);
        (compiler as any).buildenvsetup = new BuildEnvSetupBase(info as CompilerInfo, ce);
        compiler.initialiseLibraries({
            libs: {
                fortran: {
                    json_fortran: {
                        id: 'json_fortran',
                        name: 'json-fortran',
                        versions: {
                            830: {
                                version: '8.3.0',
                                liblink: [],
                                staticliblink: ['json-fortran'],
                                libpath: [],
                                path: [],
                                packagedheaders: true,
                            },
                        },
                    } as unknown as OptionsHandlerLibrary,
                    curl: {
                        id: 'curl',
                        name: 'curl',
                        versions: {
                            7831: {
                                version: '7.83.1',
                                liblink: ['curl-d'],
                                staticliblink: [],
                                libpath: [],
                                path: ['/opt/compiler-explorer/libs/curl/7.83.1/include'],
                            },
                        },
                    } as unknown as OptionsHandlerLibrary,
                },
            },
        } as unknown as ClientOptionsType);
    });

    it('should not add libpaths and link to libraries when they dont exist', async () => {
        (compiler as any).executionType = 'nsjail';

        const dirPath = await compiler.newTempDir();

        const libPath = path.join(dirPath, 'json_fortran/lib');
        await fs.mkdir(libPath, {recursive: true});

        const libPaths = compiler.getSharedLibraryPaths([{id: 'json_fortran', version: '830'}], dirPath);
        expect(libPaths).toContain(libPath);

        const libJsonFilepath = path.join(libPath, 'libjson-fortran.a');

        const failedLinks = compiler.getStaticLibraryLinks([{id: 'json_fortran', version: '830'}], libPaths);
        expect(failedLinks).not.toContain(libJsonFilepath);
    });

    it('should add libpaths and link to libraries', async () => {
        (compiler as any).executionType = 'nsjail';

        const dirPath = await compiler.newTempDir();
        const libPath = path.join(dirPath, 'json_fortran/lib');
        await fs.mkdir(libPath, {recursive: true});
        const libJsonFilepath = path.join(libPath, 'libjson-fortran.a');

        const libPaths = compiler.getSharedLibraryPaths([{id: 'json_fortran', version: '830'}], dirPath);
        expect(libPaths).toContain(libPath);

        await fs.writeFile(libJsonFilepath, 'hello, world!');

        // the file is now here and Should be linked to
        const links = compiler.getStaticLibraryLinks([{id: 'json_fortran', version: '830'}], libPaths);
        expect(links).toContain(libJsonFilepath);

        const paths = (compiler as any).getSharedLibraryPathsAsArguments(
            [{id: 'json_fortran', version: '830'}],
            undefined,
            undefined,
            dirPath,
        );
        expect(paths).toContain('-L' + libPath);
    });

    it('should add includes for packaged libraries', async () => {
        (compiler as any).executionType = 'nsjail';
        (compiler as any).compiler.includeFlag = '-isystem';

        const dirPath = await compiler.newTempDir();
        const fortranInclude = path.join(dirPath, 'json_fortran/mod');
        const cInclude = path.join(dirPath, 'json_fortran/include');

        const paths = (compiler as any).getIncludeArguments([{id: 'json_fortran', version: '830'}], dirPath);
        expect(paths).toContain('-I' + fortranInclude);
        expect(paths).toContain('-isystem' + cInclude);
    });

    it('should add includes for non-packaged C libraries', async () => {
        (compiler as any).executionType = 'nsjail';
        (compiler as any).compiler.includeFlag = '-isystem';

        const dirPath = await compiler.newTempDir();

        const paths = (compiler as any).getIncludeArguments([{id: 'curl', version: '7831'}], dirPath);
        expect(paths).toContain('-isystem/opt/compiler-explorer/libs/curl/7.83.1/include');
    });
});
