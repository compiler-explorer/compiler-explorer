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
        let optionstasking = new Array<string>();
        const sourceFile = options[options.length - 1];
        const objectFile = sourceFile.replace(/\.cpp|\.c/g, '.o');
        if (sourceFile.endsWith('.cpp')) {
            optionstasking = [
                '-g',
                '--core=tc1.8',
                '--force-c++',
                '--pending-instantiations=200',
                '-c',
                '-O0',
                '-o',
                objectFile,
                sourceFile,
            ];
        } else {
            optionstasking = ['--core=tc1.8', '-c', '-g', '-o', objectFile, sourceFile];
        }
        if (options.length > 5) {
            optionstasking = optionstasking.concat(options.slice(4, -1));
        }

        const result1 = await this.exec(compilertasking, optionstasking, execOptions);

        if (this.filtersBinary && inputFilename.endsWith('.cpp')) {
            const file = path.dirname(fileURLToPath(import.meta.url)) + '\\..\\..\\tc49x.lsl';
            optionstasking = [
                '--force-c++',
                '--pending-instantiations=200',
                file,
                '--lsl-core=tc0',
                '-o',
                objectFile,
                objectFile,
            ];
            const result2 = await this.exec(compilertasking, optionstasking, execOptions);
            if (result2.code !== 0) options[4] = "Linkerror: can't generate .exe file.";
        } else if (this.filtersBinary && inputFilename.endsWith('.c')) {
            const file = path.dirname(fileURLToPath(import.meta.url)) + '\\..\\..\\tc49x.lsl';
            optionstasking = ['-d', file, '--lsl-core=tc0', '-o', objectFile, objectFile];
            const result2 = await this.exec(compilertasking, optionstasking, execOptions);
            if (result2.code !== 0) options[4] = "Linkerror: can't generate .exe file.";
        }
        options.length = 4;
        options.push(objectFile);
        this.asm.objpath = objectFile;
        this.asm.setSrcPath(inputFilename);
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
