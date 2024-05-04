import os from 'os';
import path from 'path';

import fs from 'fs-extra';
import temp from 'temp';

import {BuildResult, ExecutionOptions, ExecutionParams} from '../types/compilation/compilation.interfaces.js';
import {RuntimeToolType, UnprocessedExecResult} from '../types/execution/execution.interfaces.js';

import {assert, unwrap} from './assert.js';
import {CompilationEnvironment} from './compilation-env.js';
import * as exec from './exec.js';
import {logger} from './logger.js';
import {Packager} from './packager.js';
import * as utils from './utils.js';

export interface IExecutionEnvironment {
    downloadExecutablePackage(hash: string): Promise<void>;
    execute(params: ExecutionParams): Promise<UnprocessedExecResult>;
}

export class LocalExecutionEnvironment implements IExecutionEnvironment {
    protected packager: Packager;
    protected dirPath: string;
    protected buildResult: BuildResult | undefined;
    protected environment: CompilationEnvironment;

    constructor(environment: CompilationEnvironment) {
        this.environment = environment;
        this.packager = new Packager();
        this.dirPath = 'not initialized';
    }

    protected async executableGet(hash: string, destinationFolder: string) {
        const result = await this.environment.executableCache.get(hash);
        if (!result.hit) return null;
        const filepath = destinationFolder + '/' + hash;
        await fs.writeFile(filepath, unwrap(result.data));
        return filepath;
    }

    protected async loadPackageWithExecutable(hash: string, dirPath: string): Promise<BuildResult> {
        const compilationResultFilename = 'compilation-result.json';
        try {
            const startTime = process.hrtime.bigint();
            const outputFilename = await this.executableGet(hash, dirPath);
            if (outputFilename) {
                logger.debug(`Using cached package ${outputFilename}`);
                await this.packager.unpack(outputFilename, dirPath);
                const buildResultsBuf = await fs.readFile(path.join(dirPath, compilationResultFilename));
                const buildResults = JSON.parse(buildResultsBuf.toString('utf8'));
                // logger.info(hash + ' => ' + JSON.stringify(buildResults));
                const endTime = process.hrtime.bigint();

                let inputFilename = '';
                if (buildResults.inputFilename) {
                    inputFilename = path.join(dirPath, path.basename(buildResults.inputFilename));
                }

                let executableFilename = '';
                if (buildResults.executableFilename) {
                    const execPath = utils.maskRootdir(buildResults.executableFilename);
                    executableFilename = path.join(dirPath, execPath);
                }

                return Object.assign({}, buildResults, {
                    code: 0,
                    inputFilename: inputFilename,
                    dirPath: dirPath,
                    executableFilename: executableFilename,
                    packageDownloadAndUnzipTime: ((endTime - startTime) / BigInt(1000000)).toString(),
                });
            } else {
                throw new Error('Tried to get executable from cache, but got a cache miss');
            }
        } catch (err) {
            throw new Error('Tried to get executable from cache, but got an error: ' + JSON.stringify(err));
        }
    }

    async downloadExecutablePackage(hash: string): Promise<void> {
        this.dirPath = await temp.mkdir({prefix: utils.ce_temp_prefix, dir: os.tmpdir()});

        this.buildResult = await this.loadPackageWithExecutable(hash, this.dirPath);
    }

    private getDefaultExecOptions(params: ExecutionParams): ExecutionOptions & {env: Record<string, string>} {
        const env: Record<string, string> = {};
        env.PATH = '';

        if (params.runtimeTools) {
            const runtimeEnv = params.runtimeTools.find(tool => tool.name === RuntimeToolType.env);
            if (runtimeEnv) {
                for (const opt of runtimeEnv.options) {
                    env[(opt.name = opt.value)];
                }
            }
        }

        // todo: what to do about the rest of the runtimeTools?

        if (
            this.buildResult &&
            this.buildResult.defaultExecOptions &&
            this.buildResult.defaultExecOptions.env &&
            this.buildResult.defaultExecOptions.env.PATH
        ) {
            if (env.PATH.length > 0)
                env.PATH = env.PATH + path.delimiter + this.buildResult.defaultExecOptions.env.PATH;
            else env.PATH = this.buildResult.defaultExecOptions.env.PATH;
        }

        const execOptions: ExecutionOptions & {env: Record<string, string>} = {
            env,
        };

        if (this.buildResult && this.buildResult.preparedLdPaths) {
            execOptions.ldPath = this.buildResult.preparedLdPaths;
        }

        return execOptions;
    }

    async execute(params: ExecutionParams): Promise<UnprocessedExecResult> {
        assert(this.buildResult);

        return await exec.sandbox(
            this.buildResult.executableFilename,
            typeof params.args === 'string' ? utils.splitArguments(params.args) : params.args || [],
            this.getDefaultExecOptions(params),
        );
    }
}
