import fs from 'fs';
import path from 'path';

import { TypicalExecutionFunc } from '../../types/execution/execution.interfaces';
import { IParseFilters } from '../../types/features/filters.interfaces';

import { IParsedAsmResult } from './external-parser.interface';

export class ExternalParserBase {
    private objdumperPath: string;
    private parserPath: string;
    private execFunc: TypicalExecutionFunc;

    constructor(objdumperPath: string, parserPath: string, execFunc: TypicalExecutionFunc) {
        this.objdumperPath = objdumperPath;
        this.parserPath = parserPath;
        this.execFunc = execFunc;
    }

    private getParserArguments(filters: IParseFilters, fromStdin: boolean): string[] {
        const parameters = [];

        if (fromStdin) parameters.push('-stdin');
        if (filters.binary) parameters.push('-binary');
        if (filters.labels) parameters.push('-unused_labels');
        if (filters.directives) parameters.push('-directives');
        if (filters.commentOnly) parameters.push('-comment_only');
        if (filters.trim) parameters.push('-whitespace');
        if (filters.libraryCode) parameters.push('-library_functions');

        return parameters;
    }

    private getObjdumpStarterScriptContent(filters: IParseFilters) {
        const parserArgs = this.getParserArguments(filters, true);

        return '#!/bin/sh\n' +
            `OBJDUMP=${this.objdumperPath}\n` +
            `ASMPARSER=${this.parserPath}\n` +
            `$OBJDUMP "$@" | $ASMPARSER ${parserArgs.join(' ')}\n`;
    }

    private async writeStarterScriptObjdump(buildfolder: string, filters: IParseFilters): Promise<string> {
        const scriptFilepath = path.join(buildfolder, 'dump-and-parse.sh');

        return new Promise((resolve) => {
            fs.writeFile(scriptFilepath,
                this.getObjdumpStarterScriptContent(filters), {
                encoding: 'utf8',
                mode: 0o777,
            }, () => {
                resolve(scriptFilepath);
            });
        });
    }

    private parseAsmExecResult(execResult): IParsedAsmResult {
        const result = Object.assign({}, JSON.parse(execResult.stdout));
        if (result.stderr) {
            throw result.stderr;
        }
        return result;
    }

    public async objdumpAndParseAssembly(buildfolder: string, objdumpArgs: string[],
        filters: IParseFilters): Promise<IParsedAsmResult> {
        const scriptFilepath = await this.writeStarterScriptObjdump(buildfolder, filters);
        const execOptions = {
            customCwd: buildfolder,
        };
        const execResult = await this.execFunc(scriptFilepath, objdumpArgs, execOptions);
        return this.parseAsmExecResult(execResult);
    }

    public async parseAssembly(filepath: string, filters: IParseFilters): Promise<IParsedAsmResult>  {
        const execOptions = {
            customCwd: path.dirname(filepath),
        };

        const parserArgs = this.getParserArguments(filters, false);
        parserArgs.push(filepath);

        const execResult = await this.execFunc(this.parserPath, parserArgs, execOptions);
        return this.parseAsmExecResult(execResult);
    }
}
