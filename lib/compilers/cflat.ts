import fs from "fs";
import path from 'path';
import {BaseCompiler} from '../base-compiler.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ExecutionOptions} from '../../types/compilation/compilation.interfaces.js';

export class CflatCompiler extends BaseCompiler {
  static get key() {
    return 'cflat';
  }

  constructor(compilerInfo: PreliminaryCompilerInfo, env) {
    super(compilerInfo, env);
    console.log("Cflat compiler called!");
    // this.toolchainPath = path.resolve(path.dirname(compilerInfo.exe), '..');
  }

  override getOutputFilename(dirPath: string, outputFilebase: string) {
    // console.log("local get output file: " + dirPath);
    // console.log("local get output file name: " + outputFilebase);
    return path.join(dirPath, outputFilebase + ".asm");
  }

  override async runCompiler(
      compiler: string,
      options: string[],
      inputFilename: string,
      execOptions: ExecutionOptions,
  ) {
    console.log("exepOptions:" + execOptions);
    console.log("Run compiler");

    const compilerExecResult = await this.exec(compiler, options, execOptions);
    const result = this.transformToCompilationResult(compilerExecResult, inputFilename);
    return result;
  }

  override optionsForBackend(backendOptions: Record<string, any>, outputFilename: string): string[] {
    return [];
  }
}

