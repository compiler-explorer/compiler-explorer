import path from 'node:path';
import Semver from 'semver';
import {splitArguments} from '../../shared/common-utils.js';
import type {
    ActiveTool,
    BypassCache,
    ExecutionParams,
    FiledataPair,
} from '../../types/compilation/compilation.interfaces.js';
import type {ConfiguredOverrides} from '../../types/compilation/compiler-overrides.interfaces.js';
import type {BasicExecutionResult, ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import type {IExecutionEnvironment} from '../execution/execution-env.interfaces.js';

export class MicroPythonCompiler extends BaseCompiler {
    static get key() {
        return 'micropython';
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, `${path.basename(this.compileFilename, this.lang.extensions[0])}.mpy`);
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
        if (Semver.eq(this.compiler.semver, '1.20.0')) {
            return [
                ...['-o', outputFilename],
                ...(filters.dontMaskFilenames ? [] : ['-s', this.compileFilename]),
                ...(this.compiler.options ? splitArguments(this.compiler.options) : []),
                ...userOptions,
                ...[inputFilename],
            ];
        } else {
            return [
                ...['-o', outputFilename],
                ...(filters.dontMaskFilenames ? [] : ['-s', this.compileFilename]),
                ...(this.compiler.options ? splitArguments(this.compiler.options) : []),
                ...userOptions,
                ...['--', inputFilename],
            ];
        }
    }

    override async compile(
        source: string,
        options: string[],
        backendOptions: Record<string, any>,
        filters: ParseFiltersAndOutputOptions,
        bypassCache: BypassCache,
        tools: ActiveTool[],
        executeParameters: ExecutionParams,
        libraries: SelectedLibraryVersion[],
        files: FiledataPair[],
    ) {
        filters.binary = true;
        return super.compile(
            source,
            options,
            backendOptions,
            filters,
            bypassCache,
            tools,
            executeParameters,
            libraries,
            files,
        );
    }

    override async runExecutable(
        executable: string,
        executeParameters: ExecutableExecutionOptions,
        homeDir: string,
    ): Promise<BasicExecutionResult> {
        const execOptionsCopy: ExecutableExecutionOptions = JSON.parse(
            JSON.stringify(executeParameters),
        ) as ExecutableExecutionOptions;

        execOptionsCopy.args = [
            ...this.compiler.executionWrapperArgs,
            ...['-m', path.basename(executable, '.mpy')],
            ...execOptionsCopy.args,
        ];
        executable = this.compiler.executionWrapper;

        const execEnv: IExecutionEnvironment = new this.executionEnvironmentClass(this.env);
        return execEnv.execBinary(executable, execOptionsCopy, homeDir);
    }
}
