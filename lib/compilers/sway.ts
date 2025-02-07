import path from 'path';
import fsExtra from 'fs-extra';

import {BaseCompiler} from '../base-compiler.js';
import {CompilationResult} from '../../types/compilation/compilation.interfaces.js';
import type {ExecutionOptionsWithEnv} from '../../types/compilation/compilation.interfaces.js';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import { ResultLine } from '../../types/resultline/resultline.interfaces.js';

interface SymbolMap {
    paths: string[];
    map: {
        [key: string]: {
            path: number;
            range: {
                start: { line: number, col: number };
                end: { line: number, col: number };
            }
        }
    }
}

export class SwayCompiler extends BaseCompiler {
    static get key() { return 'sway-compiler'; }

    override async checkOutputFileAndDoPostProcess(
        asmResult: CompilationResult,
        outputFilename: string,
        filters: ParseFiltersAndOutputOptions
    ): Promise<[any, any[], any[]]> {
        // No need to check for files since we already have our ASM
        return [asmResult, [], []];
    }

    override async processAsm(result: any) {
        //console.log("processAsm got:", result);
        //console.log("processAsm asm type:", typeof result.asm);
        //console.log("processAsm asm is array:", Array.isArray(result.asm));

        // If compilation failed or we have no assembly, return as is
        if (result.code !== 0 || !result.asm || result.asm.length === 0) {
            result.asm = '<Compilation failed>';
            return result;
        }
    
        // The asm array should already be properly formatted from runCompiler
        return {
            asm: result.asm,
            labelDefinitions: {}
        };
    }

    // override optionsForFilter(filters, outputFilename) {
    //     // Normally BaseCompiler might add '-g' if filters.debug is true.
    //     // Return your own argument list instead:
    //     return [];
    // }
    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string): string[] {
        // We need to return an array of command line options for the compiler
        let options = ['-o', outputFilename];
        
        // Only show asm output if we're not building an executable
        if (!filters.binary && !filters.binaryObject) {
            // Add any specific options needed to generate assembly output
            // options.push('--emit', 'asm');  // uncomment if sway has an option for this
        }
    
        // You might want to add additional options based on filters
        if (filters.intel) {
            // Add options for Intel syntax if supported
        }
    
        return options;
    }

    // Overriding runCompiler with the correct signature:
    override async runCompiler(
        compiler: string,
        options: string[],
        inputFilename: string,
        execOptions: ExecutionOptionsWithEnv,
        filters?: Partial<ParseFiltersAndOutputOptions>,
    ): Promise<CompilationResult> {
        // 1) Make a temp directory for a forc project
        const projectDir = await this.newTempDir();

        // 2) Create out/debug dir for forc to put its output there
        const outDebugDir = path.join(projectDir, 'out', 'debug');
        const symbolsPath = path.join(projectDir, 'out', 'debug', 'symbols.json');
        await fsExtra.mkdirp(outDebugDir);

        // 2) Write a Forc.toml
        const forcTomlPath = path.join(projectDir, 'Forc.toml');
        await fsExtra.writeFile(
            forcTomlPath,
            `[project]
entry = "main.sw"
license = "Apache-2.0"
name = "godbolt"

[dependencies]
std = { git = "https://github.com/FuelLabs/sway", tag = "v0.66.6" }
`
        );

        // 3) Copy input file to src/main.sw
        const srcDir = path.join(projectDir, 'src');
        await fsExtra.mkdirp(srcDir);
        const mainSw = path.join(srcDir, 'main.sw');
        await fsExtra.copyFile(inputFilename, mainSw);

        // 4) Actually run `forc build` in that folder
        //    "compiler" is the path to your forc binary from .properties
        const buildResult = await this.exec(compiler, ['build', '-g', symbolsPath], {
            ...execOptions,
            customCwd: projectDir,
        });
        

        // 5) Convert stdout & stderr to arrays of lines CE expects
        const stdout = splitLines(buildResult.stdout).map(line => ({ text: line }));
        const stderr = splitLines(buildResult.stderr).map(line => ({ text: line }));

        // 6) If build succeeded, parse the bytecode
        let asm: ResultLine[] = [];  // Explicitly type this
        if (buildResult.code === 0) {
            const artifactPath = path.join(projectDir, 'out', 'debug', 'godbolt.bin');
            
            if (await fsExtra.pathExists(artifactPath)) {
                const parseResult = await this.exec(compiler, ['parse-bytecode', artifactPath], {
                    ...execOptions,
                    customCwd: projectDir,
                });

// After your build command when checking for the symbols file:
const symbolsPath = path.join(projectDir, 'out', 'debug', 'symbols.json');
console.log("Looking for symbols at:", symbolsPath);

if (await fsExtra.pathExists(symbolsPath)) {
    console.log("Found symbols file!");
    const symbolsContent = await fsExtra.readFile(symbolsPath, 'utf8');
    const symbols: SymbolMap = JSON.parse(symbolsContent);
    console.log("Loaded symbols:", symbols);

    // When mapping each line:
    // When mapping lines:
const lines = splitLines(parseResult.stdout)
.filter(line => line.trim() !== '')
.map(line => {
    const match = line.match(/^\s*(\d+)\s+(\d+)\s+/);
    if (match) {
        const opcodeIndex = match[1]; // The half-word index
        const symbolInfo = symbols.map[opcodeIndex];
        
        // Only map if it's from our source file (path 1) not the standard library
        if (symbolInfo && symbolInfo.path === 1) {
            console.log(`Found source mapping for instruction ${opcodeIndex}:`, {
                sourceLine: symbolInfo.range.start.line,
                sourceCol: symbolInfo.range.start.col
            });
            
            return {
                text: line,
                source: {
                    file: symbols.paths[symbolInfo.path],
                    line: symbolInfo.range.start.line,
                    column: symbolInfo.range.start.col
                }
            };
        }
    }
    return { text: line };
});

// Log final assembly
console.log("Final assembly lines:", lines.map(l => ({
text: l.text,
source: l.source
})));
}           
        
                // Load symbols file
                const symbolsContent = await fsExtra.readFile(symbolsPath, 'utf8');
                const symbols: SymbolMap = JSON.parse(symbolsContent);
        
                asm = [{ text: "  half-word   byte   op                                                 raw           notes" }];
        
                // Add each line with source mapping if available
                const lines = splitLines(parseResult.stdout)
                    .filter(line => line.trim() !== '')
                    .map(line => {
                        const match = line.match(/^\s*(\d+)\s+(\d+)\s+/);
                        if (match) {
                            const opcodeIndex = match[1]; // The half-word index
                            const symbolInfo = symbols.map[opcodeIndex];
                            
                            if (symbolInfo) {
                                return {
                                    text: line,
                                    source: {
                                        file: symbols.paths[symbolInfo.path],
                                        line: symbolInfo.range.start.line,
                                        column: symbolInfo.range.start.col
                                    }
                                };
                            }
                        }
                        return { text: line };
                    });
        
                asm.push(...lines);
            }
        }

        // 7) Construct and return a CompilationResult
        const result: CompilationResult = {
            code: buildResult.code,
            timedOut: buildResult.timedOut ?? false,
            stdout,
            stderr,
            asm,
            inputFilename,
            execTime: buildResult.execTime,
            okToCache: true,
            // executableFilename: buildResult.code === 0 ? path.join(outDebugDir, 'godbolt.bin') : undefined,
            dirPath: projectDir
        };
        // console.log("ASM output:", asm);

        //console.log("SWAY RESULT:", result);  // Let's see what the result looks like
        //console.log("SWAY FILTERS:", filters);  // And what filters we're getting

        return result;
    }

    // If you need custom behavior, override methods like:
    //  - optionsForFilter()
    //  - runCompiler()
    //  - etc.
}

/**
 * Splits a multi-line string into an array of lines, omitting the trailing newline if present.
 */
function splitLines(str: string): string[] {
    return str.split(/\r?\n/);
}