import os from 'os';
import path from 'path';

import fs from 'fs-extra';
import temp from 'temp';

import {ExecutionOptions, ExecutionParams} from '../types/compilation/compilation.interfaces.js';
import {UnprocessedExecResult} from '../types/execution/execution.interfaces.js';

import {unwrap} from './assert.js';
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
    protected executableCache: any;
    protected dirPath: string;
    protected buildResult: any;

    constructor() {
        this.packager = new Packager();
        this.dirPath = 'not initialized';
    }

    protected async executableGet(key: string, destinationFolder: string) {
        const result = await this.executableCache.get(key);
        if (!result.hit) return null;
        const filepath = destinationFolder + '/' + key;
        await fs.writeFile(filepath, unwrap(result.data));
        return filepath;
    }

    protected async loadPackageWithExecutable(hash: string, dirPath: string) {
        const compilationResultFilename = 'compilation-result.json';
        try {
            const startTime = process.hrtime.bigint();
            const outputFilename = await this.executableGet(hash, dirPath);
            if (outputFilename) {
                logger.debug(`Using cached package ${outputFilename}`);
                await this.packager.unpack(outputFilename, dirPath);
                const buildResultsBuf = await fs.readFile(path.join(dirPath, compilationResultFilename));
                const buildResults = JSON.parse(buildResultsBuf.toString('utf8'));
                const endTime = process.hrtime.bigint();

                // todo: get inputFilename and executableFilename from somewhere else?

                // let inputFilename = path.join(dirPath, this.compileFilename);
                // if (buildResults.inputFilename) {
                //     inputFilename = path.join(dirPath, path.basename(buildResults.inputFilename));
                // }

                return Object.assign({}, buildResults, {
                    code: 0,
                    // inputFilename: inputFilename,
                    dirPath: dirPath,
                    // executableFilename: this.getExecutableFilename(dirPath, this.outputFilebase),
                    packageDownloadAndUnzipTime: ((endTime - startTime) / BigInt(1000000)).toString(),
                });
            }
            logger.debug('Tried to get executable from cache, but got a cache miss');
        } catch (err) {
            logger.error('Tried to get executable from cache, but got an error:', {err});
        }
        return false;
    }

    async downloadExecutablePackage(hash: string): Promise<void> {
        this.dirPath = await temp.mkdir({prefix: utils.ce_temp_prefix, dir: os.tmpdir()});

        this.buildResult = await this.loadPackageWithExecutable(hash, this.dirPath);
    }

    private getDefaultExecOptions(): ExecutionOptions & {env: Record<string, string>} {
        // const env = this.env.getEnv(this.compiler.needsMulti);
        const env: Record<string, string> = {};
        if (!env.PATH) env.PATH = '';
        // env.PATH = [...this.getExtraPaths(), env.PATH].filter(Boolean).join(path.delimiter);

        return {
            // timeoutMs: this.env.ceProps('compileTimeoutMs', 7500),
            // maxErrorOutput: this.env.ceProps('max-error-output', 5000),
            env,
            // wrapper: this.compilerWrapper,
        };
    }

    async execute(params: ExecutionParams): Promise<UnprocessedExecResult> {
        return await exec.execute(
            this.buildResult.executableFilename,
            typeof params.args === 'string' ? utils.splitArguments(params.args) : params.args || [],
            this.getDefaultExecOptions(),
        );
    }
}
