import path from 'path';

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
