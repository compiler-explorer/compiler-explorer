import path from 'path';
import {fileURLToPath} from 'url';

import _ from 'underscore';

import {CompilerOutputOptions, ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';
import {TaskingHlObjdumper} from '../objdumper';
import {AsmParserTasking} from '../parsers/asm-parser-tasking';

export class TaskingCompiler extends BaseCompiler {
    protected override objdumperClass = TaskingHlObjdumper;
    protected override asm = new AsmParserTasking();
    protected srcpath: string;
    protected objpath: string;

    static get key() {
        return 'tasking';
    }

    protected override optionsForFilter(
        filters: ParseFiltersAndOutputOptions,
        outputFilename: string,
        userOptions?: string[] | undefined,
    ): string[] {
        const options: string[] = ['-g', '--core=tc1.8', '-O0'];
        if (this.lang.id === 'c++') {
            options.push('--force-c++');
        }
        if (filters.binaryObject) {
            const cfd = path.dirname(fileURLToPath(import.meta.url));
            const script_path = path.resolve(cfd, '..\\..\\etc\\link-scripts\\tc49x.lsl');
            options.push('--lsl-core=tc0', '--lsl-file=' + script_path);
        } else {
            options.push('-co');
        }
        options.push('-o', outputFilename);
        return options;
    }

    override preProcess(source: string, filters: CompilerOutputOptions): string {
        if (filters.binaryObject && !this.stubRe.test(source)) {
            source += `\n${this.stubText}\n`;
        }
        return source;
    }

    override async doCompilation(
        inputFilename: any,
        dirPath: any,
        key: any,
        options: any,
        filters: any,
        backendOptions: any,
        libraries: any,
        tools: any,
    ): Promise<any> {
        filters.binary = false;
        const inputFilenameSafe = this.filename(inputFilename);
        const outputFilename = this.getOutputFilename(dirPath, this.outputFilebase, key);
        options = _.compact(
            this.prepareArguments(options, filters, backendOptions, inputFilename, outputFilename, libraries),
        );
        this.srcpath = inputFilenameSafe;
        this.objpath = outputFilename;

        const execOptions = this.getDefaultExecOptions();
        execOptions.ldPath = this.getSharedLibraryPathsAsLdLibraryPaths([]);
        const [output] = await Promise.all([
            this.runCompiler(this.compiler.exe, options, inputFilenameSafe, execOptions),
        ]);
        filters.binary = true;
        return this.checkOutputFileAndDoPostProcess(output, outputFilename, filters);
    }

    override processAsm(result: any, filters: any, options: any) {
        this.asm.objpath = this.objpath;
        this.asm.setSrcPath(this.srcpath);
        return this.asm.process(result.asm, filters);
    }
}
