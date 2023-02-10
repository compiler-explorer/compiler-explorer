import path from 'path';
import {fileURLToPath} from 'url';

import {CompilationResult, ExecutionOptions} from '../../types/compilation/compilation.interfaces';
import {CompilerInfo} from '../../types/compiler.interfaces';
import {CompilerOutputOptions, ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';
import {CompilationEnvironment} from '../compilation-env';
import {AsmParserTasking} from '../parsers/asm-parser-tasking';

export class TaskingCompiler extends BaseCompiler {
    override asm: AsmParserTasking;
    filtersBinary: boolean;

    static get key() {
        return 'tasking';
    }

    constructor(info: CompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.asm = new AsmParserTasking(this.compilerProps);
        this.compiler.exe = this.compiler.exe.replace('cctc.exe', 'hldumptc.exe');
    }

    override optionsForFilter(filters, outputFilename) {
        //hldumptc -cc -FCdFHMNSY -is $(OUTPATH)/$< -o $(OUTPATH)/cppdemo.asm
        if (filters.binaryObject) this.filtersBinary = true;
        else this.filtersBinary = false;

        return ['-cc', '-FCdFHMNSY', '-o', this.filename(outputFilename)];
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptions,
    ) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        execOptions.customCwd = path.dirname(inputFilename);

        const index = compiler.indexOf('hldumptc.exe');
        const compilertasking = compiler.slice(0, index) + 'cctc.exe';
        const length = 0;
        let optionstasking: string[] = Array.from({length});
        if (options[4].endsWith('.cpp')) {
            const cpppath: string = options[4].replace('.cpp', '.o');
            optionstasking = [
                '-g',
                '--core=tc1.8',
                '--force-c++',
                '--pending-instantiations=200',
                '-c',
                '-O0',
                '-o',
                cpppath,
                options[4],
            ];
            options[4] = optionstasking[7];
        } else {
            const cpath = options[4].replace('.c', '.o');
            optionstasking = ['--core=tc1.8', '-c', '-g', '-o', cpath, options[4]];
            options[4] = optionstasking[4];
        }
        const result1 = await this.exec(compilertasking, optionstasking, execOptions);

        if (this.filtersBinary && inputFilename.endsWith('.cpp')) {
            const file = path.dirname(fileURLToPath(import.meta.url)) + '\\..\\..\\tc49x.lsl';
            const cpppath: string = options[4].replace('.cpp', '.o');
            optionstasking = [
                '--force-c++',
                '--pending-instantiations=200',
                file,
                '--lsl-core=tc0',
                '-o',
                cpppath,
                options[4],
            ];
            const result2 = await this.exec(compilertasking, optionstasking, execOptions);
            if (result2.code !== 0) options[4] = "Linkerror: can't generate .exe file.";
        } else if (this.filtersBinary && inputFilename.endsWith('.c')) {
            const file = path.dirname(fileURLToPath(import.meta.url)) + '\\..\\..\\tc49x.lsl';
            const cpath = options[4].replace('.c', '.o');
            optionstasking = ['-d', file, '--lsl-core=tc0', '-o', cpath, options[4]];
            const result2 = await this.exec(compilertasking, optionstasking, execOptions);
            if (result2.code !== 0) options[4] = "Linkerror: can't generate .exe file.";
        }

        this.asm._elffilepath = options[4];

        return super.runCompiler(compiler, options, inputFilename, execOptions);
    }

    override preProcess(source: string, filters: CompilerOutputOptions): string {
        if (filters.binaryObject && !this.stubRe.test(source)) {
            source += `\n${this.stubText}\n`;
        }
        return source;
    }

    protected override getSharedLibraryPathsAsArguments(libraries: any): string[] {
        return [];
    }

    override supportsObjdump(): boolean {
        return false;
    }
}
