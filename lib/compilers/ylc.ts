// Copyright (c) 2024, Compiler Explorer Authors
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

import path from 'node:path';

import {splitArguments} from '../../shared/common-utils.js';
import {ConfiguredOverrides} from '../../types/compilation/compiler-overrides.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import {BaseParser} from './argument-parsers.js';

export class YLCCompiler extends BaseCompiler {
    static get key() {
        return 'ylc';
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions: string[]) {
        return ['-o=' + this.filename(outputFilename)];
    }

    override getArgumentParserClass() {
        return BaseParser;
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

        options = options.concat(this.optionsForBackend(backendOptions, outputFilename));

        if (this.compiler.options) {
            options = options.concat(splitArguments(this.compiler.options));
        }

        if (this.compiler.supportsOptOutput && backendOptions.produceOptInfo) {
            options = options.concat(unwrap(this.compiler.optArg));
        }
        if (this.compiler.supportsStackUsageOutput && backendOptions.produceStackUsageInfo) {
            options = options.concat(unwrap(this.compiler.stackUsageArg));
        }

        const toolchainPath = this.getDefaultOrOverridenToolchainPath(backendOptions.overrides || []);

        const dirPath = path.dirname(inputFilename);

        const libIncludes = this.getIncludeArguments(libraries, dirPath);
        const libOptions = this.getLibraryOptions(libraries);
        let libLinks: string[] = [];
        let libPaths: string[] = [];
        let libPathsAsFlags: string[] = [];
        let staticLibLinks: string[] = [];

        if (filters.binary) {
            libLinks = (this.getSharedLibraryLinks(libraries).filter(Boolean) as string[]) || [];
            libPathsAsFlags = this.getSharedLibraryPathsAsArguments(libraries, undefined, toolchainPath, dirPath);
            libPaths = this.getSharedLibraryPaths(libraries, dirPath);
            staticLibLinks = (this.getStaticLibraryLinks(libraries, libPaths).filter(Boolean) as string[]) || [];
        }

        userOptions = this.filterUserOptions(userOptions) || [];
        [options, overrides] = this.fixIncompatibleOptions(options, userOptions, overrides);
        options = this.changeOptionsBasedOnOverrides(options, overrides);

        return this.orderArguments(
            options,
            inputFilename,
            libIncludes,
            libOptions,
            libPathsAsFlags,
            libLinks,
            userOptions,
            staticLibLinks,
        );
    }
}
