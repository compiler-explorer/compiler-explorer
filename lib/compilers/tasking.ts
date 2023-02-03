import {fileURLToPath} from 'url';
import path from 'path';


import {CompilationResult, ExecutionOptions} from '../../types/compilation/compilation.interfaces';
import {CompilerOutputOptions, ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {BaseCompiler} from '../base-compiler';
import {AsmParserTasking} from '../parsers/asm-parser-tasking';

export class TaskingCompiler extends BaseCompiler {
    override asm: AsmParserTasking;
    filtersBinary: boolean;

    static get key() {
        return 'tasking';
    }

    constructor(info, env) {
        super(info, env);
        this.asm = new AsmParserTasking(this.compilerProps);
        const index = this.compiler.exe.indexOf('cctc.exe');
        this.compiler.exe = this.compiler.exe.slice(0, index) + 'hldumptc.exe';
    }

    override optionsForFilter(filters, outputFilename) {
        //hldumptc -cc -FCdFHMNSY -is $(OUTPATH)/$< -o $(OUTPATH)/cppdemo.asm
        if (filters.binary) this.filtersBinary = true;
        else this.filtersBinary = false;

        return ['-cc', '-FCdFHMNSY', '-o', this.filename(outputFilename)];
    }

    override async runCompiler(compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = this.getDefaultExecOptions();
        }

        execOptions.customCwd = path.dirname(inputFilename);

        const index = compiler.indexOf('hldumptc.exe');
        const compilertasking = compiler.slice(0, index) + 'cctc.exe';
        const optionstasking = ['1', '2', '3', '4'];
        if (options[4].endsWith('.cpp')) {
            //cctc -g --core=tc1.8 --force-c++ --pending-instantiations=200 -c -O0 -o example.o  example.cpp
            optionstasking[0] = '-g';
            optionstasking[1] = '--core=tc1.8';
            optionstasking[2] = '--force-c++';
            optionstasking[3] = '--pending-instantiations=200';
            optionstasking[4] = '-c';
            optionstasking[5] = '-O0';
            optionstasking[6] = '-o';
            optionstasking[7] = options[4];
            // optionstasking[7] = optionstasking[7].substring(0, optionstasking[7].length - 3);
            const end = optionstasking[7].indexOf('.c');
            optionstasking[7] = optionstasking[7].slice(0, end);
            optionstasking[7] += '.o';
            optionstasking[8] = options[4];
            optionstasking.length = 9;
            options[4] = optionstasking[7];
        } else {
            //cctc  --core=tc1.8 -c -o $(OUTPATH)/$@  $<
            optionstasking[0] = '--core=tc1.8';
            optionstasking[1] = '-c';
            optionstasking[2] = '-g';
            optionstasking[3] = '-o';
            const end = options[4].indexOf('.c');
            optionstasking[4] = options[4].slice(0, end);
            optionstasking[4] += '.o';
            optionstasking[5] = options[4];
            options[4] = optionstasking[4];
        }
        const result1 = await this.exec(compilertasking, optionstasking, execOptions);

        if (this.filtersBinary && inputFilename.endsWith('.cpp')) {
            //cctc
            optionstasking[0] = '--force-c++';
            optionstasking[1] = '--pending-instantiations=200';
            optionstasking[2] = '-d';
            const file = path.dirname(fileURLToPath(import.meta.url))+"\\..\\..\\tc49x.lsl";
            console.log(file);
            optionstasking[3] = file;
            optionstasking[4] = '--lsl-core=tc0';
            optionstasking[5] = '-o';
            const end = options[4].indexOf('.o');
            optionstasking[6] = options[4].slice(0, end) + '.o';
            optionstasking[7] = options[4];

            optionstasking.length = 8;
            const result2 = await this.exec(compilertasking, optionstasking, execOptions);
            if (result2.code !== 0) options[4] = "Linkerror: can't generate .exe file.";
        } else if(this.filtersBinary && inputFilename.endsWith('.c')){
            optionstasking[0] = '-d';
            const file = path.dirname(fileURLToPath(import.meta.url))+"\\..\\..\\tc49x.lsl";
            optionstasking[1] = file;
            optionstasking[2] = '--lsl-core=tc0';
            optionstasking[3] = '-o';
            const end = options[4].indexOf('.o');
            optionstasking[4] = options[4].slice(0, end) + '.o';
            optionstasking[5] = options[4];

            optionstasking.length = 6;

            const result2 = await this.exec(compilertasking, optionstasking, execOptions);
            if (result2.code !== 0) options[4] = "Linkerror: can't generate .exe file.";
        }

        this.asm._elffilepath = options[4];

        return super.runCompiler(compiler, options, inputFilename, execOptions);
    }

    override preProcess(source: string, filters: CompilerOutputOptions): string {
        if (filters.binary && !this.stubRe.test(source)) {
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
