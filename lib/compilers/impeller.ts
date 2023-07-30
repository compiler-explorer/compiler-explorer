import { ExecutionOptions, CompilationResult } from "../../types/compilation/compilation.interfaces.js";
import { PreliminaryCompilerInfo } from "../../types/compiler.interfaces.js";
import { ParseFiltersAndOutputOptions, preProcessLinesFunc } from "../../types/features/filters.interfaces.js";
import { BaseCompiler } from "../base-compiler.js";
import * as utils from '../utils.js';

export class ImpellerCompiler extends BaseCompiler {
  disassemblerPath: any;

  static get key() {
      return 'impeller';
  }

  constructor(compilerInfo: PreliminaryCompilerInfo, env) {
      super(compilerInfo, env);

      this.disassemblerPath = this.compilerProps('disassemblerPath');
  }


  override optionsForFilter(
    filters: ParseFiltersAndOutputOptions,
    outputFilename: string,
    userOptions?: string[]): string[] {
      var options = [
        `--sl=${outputFilename}`,
        `--include=/usr/local/include/shader_lib`,
        `--include=/src/examples/impeller`,
      ];

      // Pick a default platform if the user has not specified one of
      // the known platforms.
      const knownPlatforms = new Set([
        '--metal-ios',
        '--metal-desktop',
        '--opengl-desktop',
        '--opengl-es',
        '--runtime-stage-gles',
        '--runtime-stage-metal',
        '--runtime-stage-vulkan',
        '--sksl',
        '--vulkan',
      ]);
      var userSpecifiesPlatform = false;
      userOptions?.forEach((option) => {
        userSpecifiesPlatform ||= knownPlatforms.has(option);
      });
      if (!userSpecifiesPlatform) {
        options.push('--metal-ios');
      }
      return options;
  }

  async disassembleSPIRVIfNecessary(
    promise_result: Promise<CompilationResult>,
    disassemble: boolean,
    outputFileName: string,
    execOptions: ExecutionOptions & { env: Record<string, string>; }
    ) : Promise<CompilationResult> {
    if (!disassemble) {
      return promise_result;
    }

    var result : CompilationResult = await promise_result;

    if (result.code != 0 || !(await utils.fileExists(outputFileName))) {
      return result;
    }

    // Transform the binary SPIRV into text form.
    var spv_result = await this.exec(this.disassemblerPath, [outputFileName, '-o', outputFileName], execOptions);
    if (spv_result.code != 0) {
      return result;
    }

    result.stdout = result.stdout.concat(utils.parseOutput(spv_result.stdout));
    return result;
  }

  async clangFormatIfNecessary(
    promise_result: Promise<CompilationResult>,
    format: boolean,
    outputFileName: string,
    execOptions: ExecutionOptions & { env: Record<string, string>; }
    ) : Promise<CompilationResult> {
    if (!format) {
      return promise_result;
    }

    var result : CompilationResult = await promise_result;

    if (result.code != 0 || !(await utils.fileExists(outputFileName))) {
      return result;
    }

    var spv_result = await this.exec('clang-format', ['--style=Chromium', '-i', outputFileName], execOptions);
    if (spv_result.code != 0) {
      return result;
    }

    result.stdout = result.stdout.concat(utils.parseOutput(spv_result.stdout));
    return result;
  }

  override async runCompiler(
    compiler: string,
    options: string[],
    inputFilename: string,
    execOptions: ExecutionOptions & { env: Record<string, string>; })
      : Promise<CompilationResult> {
      // Impeller expects the input as a flag while CE expects it a positional.
      // Apply those fixups here.
      options.pop();
      options.push(`--input=${inputFilename}`);

      // If the user has requested reflection JSON, header, or CC file,
      // ask ImpellerC to redirect that to the output location and move
      // the SL stuff to a different file file.
      var outputFileName = "";
      options.forEach((option) => {
        if (option.startsWith('--sl=')) {
          outputFileName = option.substring('--sl='.length);
        }
      });

      var outputIsBinarySPIRV = false;

      options.forEach((option) => {
        outputIsBinarySPIRV ||= option.startsWith('--vulkan');
      });

      var rewriteSL = false;
      var hasSPIRVFlag = false;
      for (var i = 0; i < options.length; i++) {
        if (options[i].startsWith('--reflection-json')) {
          rewriteSL = true;
          options[i] = `--reflection-json=${outputFileName}`;
        } else if (options[i].startsWith('--reflection-header')) {
          rewriteSL = true;
          options[i] = `--reflection-header=${outputFileName}`;
        } else if (options[i].startsWith('--reflection-cc')) {
          rewriteSL = true;
          options[i] = `--reflection-cc=${outputFileName}`;
        } else if (options[i].startsWith('--spirv')) {
          rewriteSL = true;
          hasSPIRVFlag = true;
          outputIsBinarySPIRV = true;
          options[i] = `--spirv=${outputFileName}`;
        }
      }

      var applyClangFormat = false;
      options.forEach((option) => {
        if (option.startsWith('--reflection-header')
         || option.startsWith('--reflection-cc')
         || option.startsWith('--metal-ios')
         || option.startsWith('--metal-desktop')
         || option.startsWith('--opengl-es')
         || option.startsWith('--opengl-desktop')
         || option.startsWith('--sksl')
          ) {
            applyClangFormat = true;
          }
      });

      options.forEach((option) => {
        if (option.startsWith('--reflection-json')) {
            applyClangFormat = false;
          }
      });

      if (hasSPIRVFlag) {
        applyClangFormat = false;
      }

      if (rewriteSL) {
        for (var i = 0; i < options.length; i++) {
          if (options[i].startsWith('--sl=')) {
            options[i] = options[i] + '.movsl';
          }
        }
      }

      if (!hasSPIRVFlag) {
        options.push(`--spirv=${inputFilename}.spv`);
      }

      var result = super.runCompiler(compiler, options, inputFilename, execOptions);
      result = this.disassembleSPIRVIfNecessary(result, outputIsBinarySPIRV, outputFileName, execOptions);
      result = this.clangFormatIfNecessary(result, applyClangFormat, outputFileName, execOptions);
      return result;
  }
}
