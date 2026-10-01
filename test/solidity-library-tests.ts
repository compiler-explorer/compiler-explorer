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

import {beforeAll, describe, expect, it} from 'vitest';

import {BaseCompiler} from '../lib/base-compiler.js';
import type {CompilationEnvironment} from '../lib/compilation-env.js';
import {ResolcCompiler, SolidityZKsyncCompiler, SolxCompiler} from '../lib/compilers/index.js';
import {ClientOptionsType, OptionsHandlerLibrary} from '../lib/options-handler.js';
import type {CompilerInfo} from '../types/compiler.interfaces.js';
import type {LanguageKey} from '../types/languages.interfaces.js';
import {makeCompilationEnvironment, makeFakeCompilerInfo} from './utils.js';

const libraryPath = '/opt/compiler-explorer/libs/openzeppelin/v5.4.0';
const openzeppelin = [{id: 'openzeppelin', version: '540'}];
const languages = {solidity: {id: 'solidity' as LanguageKey}};

describe('Solidity library imports', () => {
    let env: CompilationEnvironment;

    beforeAll(() => {
        env = makeCompilationEnvironment({languages});
    });

    function initialiseOpenZeppelin(compiler: SolidityZKsyncCompiler | SolxCompiler | ResolcCompiler): void {
        compiler.initialiseLibraries({
            libs: {
                solidity: {
                    openzeppelin: {
                        id: 'openzeppelin',
                        versions: {
                            540: {
                                dependencies: [],
                                liblink: [],
                                libpath: [],
                                options: [`@openzeppelin/=${libraryPath}/`],
                                path: [libraryPath],
                                staticliblink: [],
                            },
                        },
                    } as unknown as OptionsHandlerLibrary,
                },
            },
        } as unknown as ClientOptionsType);
    }

    function makeCompiler<T extends SolidityZKsyncCompiler | SolxCompiler | ResolcCompiler>(
        Compiler: new (info: CompilerInfo, env: CompilationEnvironment) => T,
    ): T {
        const compiler = new Compiler(makeFakeCompilerInfo({lang: 'solidity', libsArr: ['openzeppelin.540']}), env);
        initialiseOpenZeppelin(compiler);
        return compiler;
    }

    it('does not pass -I to zksolc', () => {
        expect(
            (makeCompiler(SolidityZKsyncCompiler) as BaseCompiler).getIncludeArguments(openzeppelin, '/tmp'),
        ).toEqual([]);
    });

    it('allows OpenZeppelin paths when invoking solx', () => {
        expect(makeCompiler(SolxCompiler).getIncludeArguments(openzeppelin)).toEqual(['--allow-paths', libraryPath]);
    });

    it('does not pass -I to resolc', () => {
        expect((makeCompiler(ResolcCompiler) as BaseCompiler).getIncludeArguments(openzeppelin, '/tmp')).toEqual([]);
    });
});
