import path from 'path';

import {BasicExecutionResult, ExecutableExecutionOptions} from '../../types/execution/execution.interfaces.js';
import * as utils from '../utils.js';

import {LocalExecutionEnvironment} from './base-execution-env.js';

export const AssemblyName = 'CompilerExplorer';

export type DotnetExtraConfiguration = {
    buildConfig: string;
    clrBuildDir: string;
    langVersion: string;
    targetFramework: string;
    corerunPath: string;
};

export class DotnetExecutionEnvironment extends LocalExecutionEnvironment {
    static override get key() {
        return 'local-dotnet';
    }

    override async execBinary(
        executable: string,
        executeParameters: ExecutableExecutionOptions,
        homeDir: string,
        extraConfiguration: any,
    ): Promise<BasicExecutionResult> {
        const programDir = path.dirname(executable);
        const programOutputPath = path.join(
            programDir,
            'bin',
            extraConfiguration.buildConfig,
            extraConfiguration.targetFramework,
        );
        const programDllPath = path.join(programOutputPath, `${AssemblyName}.dll`);
        const execOptions = this.getDefaultExecOptions(executeParameters);
        execOptions.maxOutput = this.maxExecOutputSize;
        execOptions.timeoutMs = this.timeoutMs;
        execOptions.ldPath = executeParameters.ldPath;
        execOptions.customCwd = homeDir;
        execOptions.appHome = homeDir;
        execOptions.env = executeParameters.env;
        execOptions.env.DOTNET_EnableWriteXorExecute = '0';
        execOptions.env.DOTNET_CLI_HOME = programDir;
        execOptions.env.CORE_ROOT = extraConfiguration.clrBuildDir;
        execOptions.input = executeParameters.stdin;
        const execArgs = ['-p', 'System.Runtime.TieredCompilation=false', programDllPath, ...executeParameters.args];
        try {
            return this.execBinaryMaybeWrapped(
                extraConfiguration.corerunPath,
                execArgs,
                execOptions,
                executeParameters,
                homeDir,
            );
        } catch (err: any) {
            if (err.code && err.stderr) {
                return utils.processExecutionResult(err);
            } else {
                return {
                    ...utils.getEmptyExecutionResult(),
                    stdout: err.stdout ? utils.parseOutput(err.stdout) : [],
                    stderr: err.stderr ? utils.parseOutput(err.stderr) : [],
                    code: err.code === undefined ? -1 : err.code,
                };
            }
        }
    }
}
