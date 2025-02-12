// Customization point for Speedata MFCC compiler

import path from 'node:path';

import fs from 'fs-extra';

import type {CacheKey, CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {ConfiguredOverrides} from '../../types/compilation/compiler-overrides.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {logger} from '../logger.js';

import {splitArguments} from '../../shared/common-utils.js';
import {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {ClangParser} from './argument-parsers.js';

export class MfccCompiler extends BaseCompiler {
    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);

        // Mfcc is really a python script - it does not accept clang switches so supportsIrView
        // by default is false. However invoking it does generate IR with no extra switches.
        // This hack needlessly invoke the compilation twice but it seems good enough for now.

        this.compiler.supportsIrView = true;
        this.compiler.irArg = [''];
        this.compiler.minIrArgs = [''];
        this.outputFilebase = 'example';
    }

    static get key() {
        return 'mfcc';
    }

    override prepareArguments(
        userOptions: string[],
        filters: ParseFiltersAndOutputOptions,
        backendOptions: Record<string, any>,
        inputFilename: string,
        outputFilename: string,
        libraries: SelectedLibraryVersion[],
        overrides: ConfiguredOverrides,
    ) {
        let options = [''];
        if (this.compiler.options) {
            options = options.concat(splitArguments(this.compiler.options));
        }

        return [this.filename(inputFilename)].concat(options, userOptions);
    }

    override getOutputFilename(dirPath: string, outputFilebase: string, key?: CacheKey): string {
        return path.join(dirPath, `${outputFilebase}.s`);
    }

    override getArgumentParserClass() {
        return ClangParser;
    }

    override async checkOutputFileAndDoPostProcess(
        asmResult: CompilationResult,
        outputFilename: string,
        filters: ParseFiltersAndOutputOptions,
    ) {
        logger.info(`*** stating ${outputFilename}`);
        asmResult.asmSize = 1;
        if (fs.existsSync(outputFilename)) {
            logger.info(`*** Found file: ${outputFilename}`);
            const stat = fs.statSync(outputFilename);
            asmResult.asmSize = stat.size;
        } else {
            asmResult.asm = [{text: '<No output: missing output assembly file'}];
        }
        logger.info(`*** calling postProcess with ${outputFilename}, ${asmResult.asmSize}`);
        return await this.postProcess(asmResult, outputFilename, filters);
    }
}
