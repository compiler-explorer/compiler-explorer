import fs from 'fs';
import path from 'path';

import {ParsedAsmResult} from '../../types/asmresult/asmresult.interfaces';
import {TypicalExecutionFunc, UnprocessedExecResult} from '../../types/execution/execution.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {logger} from '../logger';
import {maskRootdir} from '../utils';

import {IExternalParser} from './external-parser.interface';

const starterScriptName = 'dump-and-parse.sh';

export class ExternalParserBase implements IExternalParser {
    private readonly objdumperPath: string;
    private readonly parserPath: string;
    private readonly execFunc: TypicalExecutionFunc;
    private compilerInfo;
    private envInfo;

    constructor(compilerInfo, envInfo, execFunc: TypicalExecutionFunc) {
        this.compilerInfo = compilerInfo;
        this.envInfo = envInfo;
        this.objdumperPath = compilerInfo.objdumper;
        this.parserPath = compilerInfo.externalparser.props('exe', '');
        if (!fs.existsSync(this.parserPath)) {
            logger.error(`External parser ${this.parserPath} does not exist`);
            process.exit(1);
        }
        this.execFunc = execFunc;
    }

    private getParserArguments(filters: ParseFiltersAndOutputOptions, fromStdin: boolean): string[] {
        const parameters = ['-plt'];

        if (fromStdin) parameters.push('-stdin');
        if (filters.binary) parameters.push('-binary');
        if (filters.binaryObject) parameters.push('-binary');
        if (filters.labels) parameters.push('-unused_labels');
        if (filters.directives) parameters.push('-directives');
        if (filters.commentOnly) parameters.push('-comment_only');
        if (filters.trim) parameters.push('-whitespace');
        if (filters.libraryCode) parameters.push('-library_functions');
        if (filters.dontMaskFilenames) parameters.push('-dont_mask_filenames');

        return parameters;
    }

    private getObjdumpStarterScriptContent(filters: ParseFiltersAndOutputOptions) {
        const parserArgs = this.getParserArguments(filters, true);

        return (
            '#!/bin/bash\n' +
            `OBJDUMP=${this.objdumperPath}\n` +
            `ASMPARSER=${this.parserPath}\n` +
            `$OBJDUMP "$@" | $ASMPARSER ${parserArgs.join(' ')}\n`
        );
    }

    private async writeStarterScriptObjdump(
        buildfolder: string,
        filters: ParseFiltersAndOutputOptions
    ): Promise<string> {
        const scriptFilepath = path.join(buildfolder, starterScriptName);

        return new Promise(resolve => {
            fs.writeFile(
                scriptFilepath,
                this.getObjdumpStarterScriptContent(filters),
                {
                    encoding: 'utf8',
                    mode: 0o777,
                },
                () => {
                    resolve(maskRootdir(scriptFilepath));
                }
            );
        });
    }

    private parseAsmExecResult(execResult: UnprocessedExecResult): ParsedAsmResult {
        if (execResult.code !== 0) {
            throw new Error(`Internal error running asm parser: ${execResult.stdout}\n${execResult.stderr}`);
        }
        const result = Object.assign({}, execResult, JSON.parse(execResult.stdout));
        delete result.stdout;
        delete result.stderr;
        result.externalParserUsed = true;
        return result;
    }

    public async objdumpAndParseAssembly(
        buildfolder: string,
        objdumpArgs: string[],
        filters: ParseFiltersAndOutputOptions
    ): Promise<ParsedAsmResult> {
        objdumpArgs = objdumpArgs.map(v => {
            return maskRootdir(v);
        });
        await this.writeStarterScriptObjdump(buildfolder, filters);
        const execOptions = {
            env: this.envInfo.getEnv(this.compilerInfo.needsMulti),
            customCwd: buildfolder,
            maxOutput: 1024 * 1024 * 1024,
        };
        const execResult = await this.execFunc(`./${starterScriptName}`, objdumpArgs, execOptions);
        return this.parseAsmExecResult(execResult);
    }

    public async parseAssembly(filepath: string, filters: ParseFiltersAndOutputOptions): Promise<ParsedAsmResult> {
        const execOptions = {
            env: this.envInfo.getEnv(this.compilerInfo.needsMulti),
        };

        const parserArgs = this.getParserArguments(filters, false);
        parserArgs.push(filepath);

        const execResult = await this.execFunc(this.parserPath, parserArgs, execOptions);
        return this.parseAsmExecResult(execResult);
    }
}
