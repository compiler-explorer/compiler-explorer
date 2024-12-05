import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';

export class OdinCompiler extends BaseCompiler {
    static get key() {
        return 'odin';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compiler.supportsIrView = false;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        if (filters.execute || filters.binary) {
            return [`-out:${this.filename(outputFilename)}`];
        }
        return ['-build-mode:asm', '-debug', `-out:${this.filename(outputFilename)}`];
    }

    override orderArguments(
        options,
        inputFilename,
        libIncludes,
        libOptions,
        libPaths,
        libLinks,
        userOptions,
        staticLibLinks,
    ) {
        return ['build', this.filename(inputFilename), '-file'].concat(options, userOptions);
    }

    override async checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters) {
        let newOutputFilename = outputFilename;
        if (!filters.binary && !filters.execute) newOutputFilename = outputFilename.replace(/.s$/, '.S');
        return super.checkOutputFileAndDoPostProcess(asmResult, newOutputFilename, filters);
    }
}
