import path from 'node:path';

import Semver from 'semver';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {asSafeVer} from '../utils.js';

export class C3Compiler extends BaseCompiler {
    static get key() {
        return 'c3c';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['--emit-llvm'];
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        const options = ['compile-only', '-g', '-l', 'pthread', '--no-obj', '--emit-asm'];
        if (Semver.gte(asSafeVer(this.compiler.semver), '0.6.8', true)) {
            options.push('--llvm-out', '.', '--asm-out', '.');
        }
        return options;
    }

    override getIrOutputFilename(inputFilename: string): string {
        return this.filename(path.dirname(inputFilename) + '/output.ll');
    }
}
