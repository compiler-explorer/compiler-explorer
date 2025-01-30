import path from 'path';

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {CompilerOutputOptions, ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import {BaseCompiler} from '../base-compiler.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {OdinAsmParser} from '../parsers/asm-parser-odin.js';
import * as utils from '../utils.js';

export class OdinCompiler extends BaseCompiler {
    private clangPath?: string;

    static get key() {
        return 'odin';
    }

    constructor(info: PreliminaryCompilerInfo, env: CompilationEnvironment) {
        super(info, env);
        this.asm = new OdinAsmParser(this.compilerProps);
        this.compiler.supportsIrView = true;
        this.compiler.irArg = [];
        this.compiler.supportsIntel = false;
        this.clangPath = this.compilerProps<string>('clangPath', undefined);
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string) {
        if (filters.execute || filters.binary) {
            return ['-debug', '-keep-temp-files', `-out:${this.filename(outputFilename)}`];
        }

        filters.preProcessLines = this.preProcessLines.bind(this);
        return ['-build-mode:asm', '-debug', '-keep-temp-files', `-out:${this.filename(outputFilename)}`];
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

    override getDefaultExecOptions() {
        const execOptions = super.getDefaultExecOptions();
        if (this.clangPath) {
            execOptions.env.ODIN_CLANG_PATH = this.clangPath;
        }

        return execOptions;
    }

    override async checkOutputFileAndDoPostProcess(asmResult, outputFilename, filters) {
        let newOutputFilename = outputFilename;
        if (!filters.binary && !filters.execute) newOutputFilename = outputFilename.replace(/.s$/, '.S');
        return super.checkOutputFileAndDoPostProcess(asmResult, newOutputFilename, filters);
    }

    override getIrOutputFilename(inputFilename: string): string {
        return this.filename(path.dirname(inputFilename) + '/output.ll');
    }

    override async postProcessAsm(result, filters?: ParseFiltersAndOutputOptions) {
        // we dont need demangling
        return result;
    }

    /**
     * Preprocess the source code to '@require' all the functions
     * so that their asm is emitted
     */
    override preProcess(source: string, filters: CompilerOutputOptions): string {
        if (filters.binary && !this.stubRe.test(source)) {
            source += `\n${this.stubText}\n`;
        }

        if (!filters.binary && !filters.execute) {
            const sourceLines = utils.splitLines(source);
            const outputLines: string[] = [];
            const procRE = /^\s*([A-Za-z]\w+)\s*:\s*:\s*proc\s*("\w+")?\s*\(/;
            let lastLine = '';
            for (const line of sourceLines) {
                // skip if the line doesn't contain proc keyword
                if (!line.includes('proc')) {
                    outputLines.push(line);
                    lastLine = line;
                    continue;
                }

                const match = line.match(procRE);
                if (!match) {
                    outputLines.push(line);
                    lastLine = line;
                    continue;
                }

                // last line already has require?
                if (lastLine.includes('@require') || lastLine.includes('@(require)')) {
                    outputLines.push(line);
                    continue;
                }

                // @require the function so they dont get inlined
                // put @require on the same line so that output source
                // has same number of lines so that asm<->source mapping
                // in ui works correctly.
                outputLines.push('@(require) ' + line);
                lastLine = line;
            }

            let text = outputLines.join('\n');
            // append a final trailing newline
            if (!text.endsWith('\n')) {
                text += '\n';
            }
            return text;
        }

        return source;
    }

    preProcessLines(asmLines: string[]) {
        let i = 0;
        let funcStart = -1;
        while (i < asmLines.length) {
            const line = asmLines[i];
            // filter out __$ builtin functions
            if (funcStart === -1 && line.startsWith('__$') && line.endsWith(':')) {
                // ensure there is cfi_startproc
                for (let j = i; j < asmLines.length && j < i + 5; j++) {
                    if (asmLines[j].includes('.cfi_startproc')) {
                        funcStart = i;
                        break;
                    }
                }

                // make sure this is a globl
                if (funcStart !== -1) {
                    for (let j = i; j >= 0 && j >= i - 3; --j) {
                        if (asmLines[j].includes('.globl')) {
                            funcStart = j;
                            break;
                        }
                    }
                }
            }

            if (funcStart !== -1 && line.includes('.cfi_endproc')) {
                const len = i - funcStart;
                asmLines.splice(funcStart, len + 1);
                i = funcStart - 1;
                funcStart = -1;
            }
            i++;
        }

        return this.removeNonMainSourceFunctions(asmLines);
    }

    parseFiles(asmLines: string[]) {
        const files: Record<number, string> = {};
        const fileFind = /^\s*\.(?:cv_)?file\s+(\d+)\s+"([^"]+)"(\s+"([^"]+)")?.*/;
        for (const line of asmLines) {
            const match = line.match(fileFind);
            if (match) {
                const lineNum = parseInt(match[1]);
                if (match[4] && !line.includes('.cv_file')) {
                    // Clang-style file directive '.file X "dir" "filename"'
                    if (match[4].startsWith('/')) {
                        files[lineNum] = match[4];
                    } else {
                        files[lineNum] = match[2] + '/' + match[4];
                    }
                } else {
                    files[lineNum] = match[2];
                }
            }
        }
        return files;
    }

    removeNonMainSourceFunctions(asmLines: string[]): string[] {
        const files = this.parseFiles(asmLines);
        const sourceTag = /^\s*\.loc\s+(\d+)\s+(\d+)\s+(.*)/;
        const stdInLooking = /<stdin>|^-$|example\.[^/]+$|<source>/;
        let i = 0;
        let funcStart = -1;
        while (i < asmLines.length) {
            const line = asmLines[i];
            // find a label
            if (funcStart === -1 && line.endsWith(':')) {
                // scan the next 5 lines for a source loc label
                for (let j = i; j < i + 5 && j < asmLines.length; j++) {
                    const fline = asmLines[j];
                    const match = fline.match(sourceTag);
                    if (!match) continue;
                    const file = files[match[1]];
                    if (!file) continue;
                    // if the file is not current file we remove this function
                    if (!stdInLooking.test(file)) {
                        funcStart = i;
                        break;
                    }
                }

                // ensure there is cfi_startproc
                if (funcStart !== -1) {
                    funcStart = -1;
                    for (let j = i; j < asmLines.length && j < i + 5; j++) {
                        if (asmLines[j].includes('.cfi_startproc')) {
                            funcStart = i;
                            break;
                        }
                    }
                }

                if (funcStart !== -1) {
                    for (let j = i; j >= 0 && j >= i - 3; --j) {
                        if (asmLines[j].includes('.type') && asmLines[j].endsWith('@function')) {
                            funcStart = j;
                            break;
                        }
                    }
                }
            }

            if (funcStart !== -1 && line.includes('.cfi_endproc')) {
                const len = i - funcStart;
                asmLines.splice(funcStart, len + 1);
                i = funcStart - 1;
                funcStart = -1;
            }
            i++;
        }

        return asmLines;
    }
}
