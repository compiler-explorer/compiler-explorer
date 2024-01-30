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

import {BaseCompiler} from '../lib/base-compiler.js';
import {BuildEnvSetupBase} from '../lib/buildenvsetup/base.js';
import {CompilationEnvironment} from '../lib/compilation-env.js';
import {ClientOptionsType, OptionsHandlerLibrary} from '../lib/options-handler.js';
import {CompilerInfo} from '../types/compiler.interfaces.js';

import {makeCompilationEnvironment} from './utils.js';

const languages = {
    'c++': {
        id: 'c++',
    },
} as const;

describe('Library directories', () => {
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
        libPath: [],
        libsArr: ['fmt.10', 'qt.660', 'cpptrace.030'],
    };

    before(() => {
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
        links.should.include('-lfmtd');

        const fmtpaths = (compiler as any).getSharedLibraryPathsAsArguments(
            [{id: 'fmt', version: '10'}],
            undefined,
            undefined,
            '/tmp/compiler-explorer-compiler-123',
        );
        fmtpaths.should.include('-L./lib');

        const qtpaths = (compiler as any).getSharedLibraryPathsAsArguments(
            [{id: 'qt', version: '660'}],
            undefined,
            undefined,
            '/tmp/compiler-explorer-compiler-123',
        );
        qtpaths.should.include('-L/tmp/compiler-explorer-compiler-123/qt/lib');
    });

    it('should add libpaths and link to libraries when using nsjail', () => {
        (compiler as any).executionType = 'nsjail';

        const fmtpaths = (compiler as any).getSharedLibraryPathsAsArguments(
            [{id: 'fmt', version: '10'}],
            undefined,
            undefined,
            '/tmp/compiler-explorer-compiler-123',
        );
        fmtpaths.should.include('-L/app/fmt/lib');

        const qtpaths = (compiler as any).getSharedLibraryPathsAsArguments(
            [{id: 'qt', version: '660'}],
            undefined,
            undefined,
            '/tmp/compiler-explorer-compiler-123',
        );
        qtpaths.should.include('-L/app/qt/lib');
    });

    it('should add extra include paths when using packagedheaders', () => {
        (compiler as any).executionType = 'nsjail';

        const fmtpaths = (compiler as any).getIncludeArguments(
            [{id: 'fmt', version: '10'}],
            '/tmp/compiler-explorer-compiler-123',
        );
        fmtpaths.should.not.include('-I/app/fmt/include');
        fmtpaths.should.include('-I/opt/compiler-explorer/libs/fmt/1.0/include');

        const qtpaths = (compiler as any).getIncludeArguments(
            [{id: 'qt', version: '660'}],
            '/tmp/compiler-explorer-compiler-123',
        );
        qtpaths.should.include('-I/opt/compiler-explorer/libs/qt/6.6.0/include');
        qtpaths.should.include('-I/app/qt/include');
    });

    it('should set LD_LIBRARY_PATH when executing', () => {
        (compiler as any).sandboxType = 'nsjail';

        const qtpaths = (compiler as any).getSharedLibraryPathsAsLdLibraryPathsForExecution(
            [{id: 'qt', version: '660'}],
            '/tmp/compiler-explorer-compiler-123',
        );
        qtpaths.should.include('/app/qt/lib');
    });

    it('should add libpaths and link when statically linking', () => {
        (compiler as any).executionType = 'nsjail';

        const staticlinks = compiler.getStaticLibraryLinks([{id: 'cpptrace', version: '030'}], []);
        staticlinks.should.include('-lcpptrace');
        staticlinks.should.include('-ldwarf');
        staticlinks.should.include('-ldl');
        staticlinks.should.include('-lz');

        const libpaths = (compiler as any).getSharedLibraryPathsAsArguments(
            [{id: 'cpptrace', version: '030'}],
            undefined,
            undefined,
            '/tmp/compiler-explorer-compiler-123',
        );
        libpaths.should.include('-L/app/cpptrace/lib');
    });
});
