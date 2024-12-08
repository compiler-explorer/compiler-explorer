import path from 'path';

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';

export class OdinCompiler extends BaseCompiler {
    private clangPath?: string;

    static get key() {
        return 'odin';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compiler.supportsIrView = true;
        this.compiler.irArg = [];
        this.compiler.supportsIntel = false;
        this.clangPath = this.compilerProps<string>('clangPath', undefined);
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        if (filters.execute || filters.binary) {
            return ['-debug', '-keep-temp-files', `-out:${this.filename(outputFilename)}`];
        }
        return ['-build-mode:asm', '-debug', '-keep-temp-files', `-out:${this.filename(outputFilename)}`];
    }

    override orderArguments(
        options,
        inputFilename,
        libIncludes,
        libOptions,
        libPaths,
        libLinks,
        userOptions,
        staticLibLinks,
    ) {
        return ['build', this.filename(inputFilename), '-file'].concat(options, userOptions);
    }

    override getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        if (this.clangPath) {
            execOptions.env.ODIN_CLANG_PATH = this.clangPath;
        }

        return execOptions;
    }

    override async checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters) {
        let newOutputFilename = outputFilename;
        if (!filters.binary && !filters.execute) newOutputFilename = outputFilename.replace(/.s$/, '.S');
        return super.checkOutputFileAndDoPostProcess(asmResult, newOutputFilename, filters);
    }

    override getIrOutputFilename(inputFilename: string): string {
        return this.filename(path.dirname(inputFilename) + '/output.ll');
    }

    override async postProcessAsm(result, filters?: ParseFiltersAndOutputOptions) {
        // we dont need demangling
        return result;
    }
}
