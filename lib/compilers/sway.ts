import path from 'path';

import fsExtra from 'fs-extra';

import {CompilationResult, ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import {LLVMIrBackendOptions} from '../../types/compilation/ir.interfaces.js';
import {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';

interface SymbolMap {
    paths: string[];
    map: {
        [key: string]: {
            path: number;
            range: {
                start: {line: number; col: number};
                end: {line: number; col: number};
            };
        };
    };
}

export class SwayCompiler extends BaseCompiler {
    static get key() {
        return 'sway-compiler';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.compiler.supportsIrView = true;
        this.compiler.irArg = ['build', '--ir', 'final'];
        this.compiler.supportsIntel = true;
    }

    override async checkOutputFileAndDoPostProcess(asmResult: CompilationResult): Promise<[any, any[], any[]]> {
        return [asmResult, [], []];
    }

    override async processAsm(result: any) {
        // If compilation failed or we have no assembly, return as is
        if (result.code !== 0 || !result.asm || result.asm.length === 0) {
            result.asm = '<Compilation failed>';
            return result;
        }
        // The asm array should already be properly formatted from runCompiler
        return {
            asm: result.asm,
            labelDefinitions: {},
        };
    }

    override async generateIR(
        inputFilename: string,
        options: string[],
        irOptions: LLVMIrBackendOptions,
        produceCfg: boolean,
        filters: ParseFiltersAndOutputOptions,
    ) {
        // We can use runCompiler since it already handles all the project setup
        const result = await this.runCompiler(
            this.compiler.exe,
            ['build', '--ir', 'final'],
            inputFilename,
            this.getDefaultExecOptions(),
            filters,
        );

        return {
            code: result.code,
            stdout: [],
            stderr: result.stderr,
            asm: result.irOutput?.asm || [],
            timedOut: result.timedOut,
            execTime: result.execTime,
            okToCache: true,
            inputFilename: result.inputFilename,
            dirPath: result.dirPath,
        };
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        // return an array of command line options for the compiler
        return ['-o', outputFilename];
    }

    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: Partial<ParseFiltersAndOutputOptions>,
    ): Promise<CompilationResult> {
        // Make a temp directory for a forc project
        const projectDir = await this.newTempDir();
        const {symbolsPath} = await setupForcProject(projectDir, inputFilename);

        // Run `forc build`
        // "compiler" is the path to the forc binary from .properties
        const buildResult = await this.exec(compiler, ['build', '-g', symbolsPath], {
            ...execOptions,
            customCwd: projectDir,
        });

        // If build succeeded, parse the bytecode
        let asm: ResultLine[] = [];
        if (buildResult.code === 0) {
            const artifactPath = path.join(projectDir, 'out', 'debug', 'godbolt.bin');

            if (filters?.intel) {
                const asmResult = await this.exec(compiler, ['build', '--asm', 'all'], {
                    ...execOptions,
                    customCwd: projectDir,
                });
                const lines = splitLines(asmResult.stdout);
                const startIndex = lines.findIndex(line => line.includes(';; ASM: Virtual abstract program'));
                const endIndex = lines.findIndex(line => line.includes('[1;32mFinished'));
                if (startIndex === -1 || endIndex === -1 || startIndex >= endIndex) {
                    asm = [{text: '<Error extracting ASM output>'}];
                } else {
                    asm = lines
                        .slice(startIndex, endIndex)
                        .filter(line => line.trim() !== '')
                        .map(line => ({text: line}));
                }
            } else {
                const parseResult = await this.exec(compiler, ['parse-bytecode', artifactPath], {
                    ...execOptions,
                    customCwd: projectDir,
                });
                let symbols: SymbolMap | undefined;
                if (await fsExtra.pathExists(symbolsPath)) {
                    const symbolsContent = await fsExtra.readFile(symbolsPath, 'utf8');
                    symbols = JSON.parse(symbolsContent);
                }

                // Map the bytecode lines
                const contentLines = splitLines(parseResult.stdout)
                    .filter(line => line.trim() !== '')
                    .map(line => {
                        const match = line.match(/^\s*(\d+)\s+(\d+)\s+/);
                        if (match && symbols) {
                            const opcodeIndex = match[1];
                            const symbolInfo = symbols.map[opcodeIndex];
                            if (symbolInfo && symbolInfo.path === 1) {
                                return {
                                    text: line,
                                    source: {
                                        file: symbols.paths[symbolInfo.path],
                                        line: symbolInfo.range.start.line,
                                        column: symbolInfo.range.start.col,
                                        mainsource: true,
                                    },
                                };
                            }
                        }
                        return {text: line};
                    });
                asm.push(...contentLines);
            }
        }

        // Run `forc build --ir final` to gather IR output and store it in `result.irOutput`.
        let irLines: ResultLine[] = [];
        if (buildResult.code === 0) {
            const irResult = await this.exec(compiler, ['build', '--ir', 'final'], {
                ...execOptions,
                customCwd: projectDir,
            });
            const lastIrMarkerIndex = irResult.stdout.lastIndexOf('// IR: Final');
            if (lastIrMarkerIndex >= 0) {
                const relevantIr =
                    irResult.stdout
                        .slice(lastIrMarkerIndex)
                        .split('\n')
                        .slice(1)
                        .join('\n')
                        .match(/(script|library|contract|predicate)\s*{[^]*?^}/m)?.[0] || '';
                irLines = relevantIr.split('\n').map(line => ({text: line}));
            }
        }

        // Construct and return a CompilationResult
        const result: CompilationResult = {
            code: buildResult.code,
            timedOut: buildResult.timedOut ?? false,
            stdout: splitLines(buildResult.stdout).map(line => ({text: line})),
            stderr: splitLines(buildResult.stderr).map(line => ({text: line})),
            asm,
            inputFilename,
            execTime: buildResult.execTime,
            okToCache: true,
            dirPath: projectDir,
            irOutput:
                irLines.length > 0
                    ? {
                          asm: irLines.map(line => ({
                              text: line.text,
                          })),
                      }
                    : undefined,
        };

        return result;
    }
}

const FORC_TOML_CONTENT = `[project]
entry = "main.sw"
license = "Apache-2.0"
name = "godbolt"

[dependencies]
std = { git = "https://github.com/FuelLabs/sway", tag = "v0.66.7" }
`;

async function setupForcProject(
    projectDir: string,
    inputFilename: string,
): Promise<{mainSw: string; symbolsPath: string}> {
    const outDebugDir = path.join(projectDir, 'out', 'debug');
    const symbolsPath = path.join(outDebugDir, 'symbols.json');
    await fsExtra.mkdirp(outDebugDir);

    // Write Forc.toml file
    const forcTomlPath = path.join(projectDir, 'Forc.toml');
    await fsExtra.writeFile(forcTomlPath, FORC_TOML_CONTENT);

    // Copy input file to src/main.sw
    const srcDir = path.join(projectDir, 'src');
    await fsExtra.mkdirp(srcDir);
    const mainSw = path.join(srcDir, 'main.sw');
    await fsExtra.copyFile(inputFilename, mainSw);

    return {mainSw, symbolsPath};
}

/**
 * Splits a multi-line string into an array of lines, omitting the trailing newline if present.
 */
function splitLines(str: string): string[] {
    return str.split(/\r?\n/);
}
