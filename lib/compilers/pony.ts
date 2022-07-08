import path from 'path';

import {ParseFilters} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';
import {CompilationResult, ExecutionOptions} from '../../types/compilation/compilation.interfaces';

export class PonyCompiler extends BaseCompiler {
    static get key() {
        return 'pony';
    }

    /* 
  constructor(info: any, env: any) {
    super(info, env);

    this.compiler.supportsIrView = true;
    this.compiler.irArg = ['--pass', 'ir'];
  } 
  */

    override optionsForFilter(filters: ParseFilters, outputFilename: any, userOptions?: any): string[] {
        let options = ['-d', '-b', path.parse(outputFilename).name];

        if (!filters.binary) {
            options = options.concat(['--pass', 'asm']);
        }

        return options;
    }

    override preProcess(source: string, filters: any) {
        // I do not think you can make a Pony "library", so you must always have a main.
        // Looking at the stdlib, the main is used as a test harness.
        if (!this.stubRe.test(source)) {
            source += `\n${this.stubText}\n`;
        }
        return source;
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions,
    ): Promise<CompilationResult> {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        if (!execOptions.customCwd) {
            execOptions.customCwd = path.dirname(inputFilename);
        }

        // Pony operates upon the directory as a whole, not files it seems
        // So we must set the input to the directory rather than a file.
        const input_i = options.indexOf(inputFilename);
        options[input_i] = path.dirname(options[input_i]);

        const result = await this.exec(compiler, options, execOptions);
        result.inputFilename = inputFilename;
        const transformedInput = result.filenameTransform(inputFilename);
        this.parseCompilationOutput(result, transformedInput);
        return result;
    }
}
