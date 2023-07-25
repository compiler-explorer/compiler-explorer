import { ExecutionOptions, CompilationResult } from "../../types/compilation/compilation.interfaces.js";
import { ParseFiltersAndOutputOptions, preProcessLinesFunc } from "../../types/features/filters.interfaces.js";
import { BaseCompiler } from "../base-compiler.js";

export class ImpellerCompiler extends BaseCompiler {
  static get key() {
      return 'impeller';
  }

  override optionsForFilter(
    filters: ParseFiltersAndOutputOptions,
    outputFilename: string,
    userOptions?: string[]): string[] {
      var options = [
        `--sl=${outputFilename}`,
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
          options[i] = `--spirv=${outputFileName}`;
        }
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

      return super.runCompiler(compiler, options, inputFilename, execOptions);
  }
}
