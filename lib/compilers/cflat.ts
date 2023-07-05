import path from 'path';
import {BaseCompiler} from '../base-compiler.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {ConfiguredOverrides} from '../../types/compilation/compiler-overrides.interfaces.js';

export class CflatCompiler extends BaseCompiler {
    static get key() {
        return 'cflat';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super(compilerInfo, env);
    }

    override getOutputFilename(dirPath: string, outputFilebase: string) {
        return path.join(dirPath, outputFilebase + '.asm');
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions,
    ) {
        const compilerExecResult = await this.exec(compiler, options, execOptions);
        return this.transformToCompilationResult(compilerExecResult, inputFilename);
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        return [outputFilename];
    }

    override orderArguments(options: string[], inputFilename: string, userOptions: string[]) {
        return options.concat(this.filename(inputFilename), userOptions);
    }

    override prepareArguments(
        userOptions: string[],
        filters: ParseFiltersAndOutputOptions,
        backendOptions: Record<string, any>,
        inputFilename: string,
        outputFilename: string,
        libraries,
        overrides: ConfiguredOverrides,
    ) {
        const options = this.optionsForFilter(filters, outputFilename);
        return [this.filename(inputFilename)].concat(options);
    }
}
