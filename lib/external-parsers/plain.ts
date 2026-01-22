import fs from 'node:fs';
import path from 'node:path';
import type {ParsedAsmResult} from '../../types/asmresult/asmresult.interfaces.js';
import {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import type {TypicalExecutionFunc, UnprocessedExecResult} from '../../types/execution/execution.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';
import {IExternalParser} from './external-parser.interface.js';

export class PlainParser implements IExternalParser {
    protected readonly parserPath: string;
    protected readonly parserArgs: string[];
    protected readonly execFunc: TypicalExecutionFunc;
    protected compilerInfo: CompilerInfo;
    protected envInfo;

    constructor(compilerInfo: CompilerInfo, envInfo: CompilationEnvironment, execFunc: TypicalExecutionFunc) {
        this.compilerInfo = compilerInfo;
        this.envInfo = envInfo;
        this.parserPath = compilerInfo.externalparser.exe;
        this.parserArgs = (compilerInfo.externalparser.args || '').split('|');
        if (!fs.existsSync(this.parserPath)) {
            const msg = `External parser for compiler ${compilerInfo.id} does not exist: "${this.parserPath}"`;
            logger.error(msg);
            // Delay exit to allow async log transports (e.g., Loki) to flush
            setTimeout(() => process.exit(1), 5000);
            throw new Error(msg);
        }
        this.execFunc = execFunc;
    }

    protected parseAsmExecResult(execResult: UnprocessedExecResult): ParsedAsmResult {
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
        filters: ParseFiltersAndOutputOptions,
    ): Promise<ParsedAsmResult> {
        return this.parseAssembly(objdumpArgs[objdumpArgs.length - 1], filters);
    }

    public async parseAssembly(filepath: string, filters: ParseFiltersAndOutputOptions): Promise<ParsedAsmResult> {
        const execOptions: ExecutionOptions = {
            env: this.envInfo.getEnv(this.compilerInfo.needsMulti),
            customCwd: path.dirname(filepath),
        };

        const args = [...this.parserArgs, filepath];

        const execResult = await this.execFunc(this.parserPath, args, execOptions);
        return this.parseAsmExecResult(execResult);
    }

    static get key() {
        return 'plain';
    }
}
