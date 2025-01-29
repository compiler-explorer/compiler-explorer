// Copyright (c) 2018, Microsoft Corporation
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

import path from 'path';

import _ from 'underscore';

import {splitArguments} from '../../shared/common-utils.js';
import type {CacheKey, ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {ConfiguredOverrides} from '../../types/compilation/compiler-overrides.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {MapFileReaderVS} from '../mapfiles/map-file-vs.js';
import {AsmParser} from '../parsers/asm-parser.js';
import {PELabelReconstructor} from '../pe32-support.js';

export class Win32Compiler extends BaseCompiler {
    static get key() {
        return 'win32';
    }

    binaryAsmParser: AsmParser;

    constructor(compilerInfo: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);

        this.binaryAsmParser = new AsmParser(this.compilerProps);
    }

    override getStdverFlags(): string[] {
        return ['/std:<value>'];
    }

    override getExecutableFilename(dirPath: string, outputFilebase: string, key?: CacheKey) {
        return this.getOutputFilename(dirPath, outputFilebase, key) + '.exe';
    }

    override getObjdumpOutputFilename(defaultOutputFilename: string) {
        return this.getExecutableFilename(path.dirname(defaultOutputFilename), 'output');
    }

    override getSharedLibraryPathsAsArguments(libraries: SelectedLibraryVersion[]) {
        const libPathFlag = this.compiler.libpathFlag || '/LIBPATH:';

        return this.getSharedLibraryPaths(libraries).map(path => libPathFlag + path);
    }

    // Ofek: foundVersion having 'liblink' makes me suspicious of the decision to annotate everywhere
    // with `SelectedLibraryVersion`, but can't test at this time
    override getSharedLibraryLinks(libraries: any[]): string[] {
        return _.flatten(
            libraries
                .map(selectedLib => [selectedLib, this.findLibVersion(selectedLib)])
                .filter(([selectedLib, foundVersion]) => !!foundVersion)
                .map(([selectedLib, foundVersion]) => {
                    return foundVersion.liblink.filter(Boolean).map((lib: string) => `"${lib}.lib"`);
                })
                .map(([selectedLib, foundVersion]) => selectedLib),
        );
    }

    override getStaticLibraryLinks(libraries: SelectedLibraryVersion[]) {
        return super.getSortedStaticLibraries(libraries).map(lib => {
            return '"' + lib + '.lib"';
        });
    }

    override prepareArguments(
        userOptions: string[],
        filters: ParseFiltersAndOutputOptions,
        backendOptions: Record<string, any>,
        inputFilename: string,
        outputFilename: string,
        libraries: SelectedLibraryVersion[],
        overrides: ConfiguredOverrides,
    ) {
        let options = this.optionsForFilter(filters, outputFilename, userOptions);
        backendOptions = backendOptions || {};

        if (this.compiler.options) {
            options = options.concat(splitArguments(this.compiler.options));
        }

        if (this.compiler.supportsOptOutput && backendOptions.produceOptInfo) {
            options = options.concat(unwrap(this.compiler.optArg));
        }

        const libIncludes = this.getIncludeArguments(libraries, path.dirname(inputFilename));
        const libOptions = this.getLibraryOptions(libraries);
        let libLinks: any[] = [];
        let libPaths: string[] = [];
        let preLink: string[] = [];
        let staticlibLinks: string[] = [];

        if (filters.binary) {
            preLink = ['/link'];
            libLinks = this.getSharedLibraryLinks(libraries);
            libPaths = this.getSharedLibraryPathsAsArguments(libraries);
            staticlibLinks = this.getStaticLibraryLinks(libraries);
        }

        userOptions = this.filterUserOptions(userOptions) || [];
        [options, overrides] = this.fixIncompatibleOptions(options, userOptions, overrides);
        this.changeOptionsBasedOnOverrides(options, overrides);

        // `/link` and all that follows must come after the filename
        const linkIndex = userOptions.indexOf('/link');
        let linkUserOptions: string[] = [];
        let compileUserOptions = userOptions;
        if (linkIndex !== -1) {
            linkUserOptions = userOptions.slice(linkIndex + 1);
            compileUserOptions = userOptions.slice(0, linkIndex);
            preLink = ['/link'];
        }

        return options.concat(
            libIncludes,
            libOptions,
            compileUserOptions,
            [this.filename(inputFilename)],
            preLink,
            linkUserOptions,
            libPaths,
            libLinks,
            staticlibLinks,
        );
    }

    override fixIncompatibleOptions(
        options: string[],
        userOptions: string[],
        overrides: ConfiguredOverrides,
    ): [string[], ConfiguredOverrides] {
        // If userOptions contains anything starting with /source-charset or /execution-charset, remove /utf-8 from options
        if (
            userOptions.some(option => option.startsWith('/source-charset') || option.startsWith('/execution-charset'))
        ) {
            options = options.filter(option => option !== '/utf-8');
        }
        return [options, overrides];
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        if (filters.binary) {
            const mapFilename = outputFilename + '.map';
            const mapFileReader = new MapFileReaderVS(mapFilename);

            filters.preProcessBinaryAsmLines = asmLines => {
                const reconstructor = new PELabelReconstructor(asmLines, false, mapFileReader);
                reconstructor.run('output.s.obj');

                return reconstructor.asmLines;
            };

            return [
                '/nologo',
                '/FA',
                '/Fa' + this.filename(outputFilename.replace(/\.exe$/, '')),
                '/Fo' + this.filename(outputFilename.replace(/\.exe$/, '') + '.obj'),
                '/Fm' + this.filename(mapFilename),
                '/Fe' + this.filename(this.getExecutableFilename(path.dirname(outputFilename), 'output')),
            ];
        } else {
            return [
                '/nologo',
                '/FA',
                '/c',
                '/Fa' + this.filename(outputFilename),
                '/Fo' + this.filename(outputFilename + '.obj'),
            ];
        }
    }

    override async processAsm(result, filters: ParseFiltersAndOutputOptions) {
        if (filters.binary) {
            filters.dontMaskFilenames = true;
            return this.binaryAsmParser.process(result.asm, filters);
        } else {
            return this.asm.process(result.asm, filters);
        }
    }

    override exec(compiler: string, args: string[], options_: ExecutionOptions) {
        const options = Object.assign({}, options_);
        options.env = Object.assign({}, options.env);

        if (this.compiler.includePath) {
            options.env['INCLUDE'] = this.compiler.includePath;
        }
        if (this.compiler.libPath) {
            options.env['LIB'] = this.compiler.libPath.join(';');
        }
        for (const [env, to] of this.compiler.envVars) {
            options.env[env] = to;
        }

        return super.exec(compiler, args, options);
    }
}
