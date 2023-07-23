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
      let spirvFileName = outputFilename + ".spv";
      var options = [
        `--sl=${outputFilename}`,
        `--spirv=${spirvFileName}`,
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
      var fixed_options = options;
      fixed_options.pop();
      fixed_options.push(`--input=${inputFilename}`);
      return super.runCompiler(compiler, fixed_options, inputFilename, execOptions);
  }
}
