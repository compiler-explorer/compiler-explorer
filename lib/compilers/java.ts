// Copyright (c) 2019, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import path from 'path';

import fs from 'fs-extra';

import {ParsedAsmResult, ParsedAsmResultLine} from '../../types/asmresult/asmresult.interfaces';
import {PreliminaryCompilerInfo} from '../../types/compiler.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {unwrap} from '../assert';
import {BaseCompiler} from '../base-compiler';
import {logger} from '../logger';
import * as utils from '../utils';

import {JavaParser} from './argument-parsers';

export class JavaCompiler extends BaseCompiler {
    static get key() {
        return 'java';
    }

    javaRuntime: string;
    mainRegex: RegExp;

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super(
            {
                // Default is to disable all "cosmetic" filters
                disabledFilters: ['labels', 'directives', 'commentOnly', 'trim'],
                ...compilerInfo,
            },
            env
        );
        this.javaRuntime = this.compilerProps<string>(`compiler.${this.compiler.id}.runtime`);
        this.mainRegex = /public static ?(.*?) void main\(java\.lang\.String\[]\)/;
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override async objdump(outputFilename, result: any, maxSize: number) {
        const dirPath = path.dirname(outputFilename);
        const files = await fs.readdir(dirPath);
        logger.verbose('Class files: ', files);
        const results = await Promise.all(
            files
                .filter(f => f.endsWith('.class'))
                .map(async classFile => {
                    const args = [
                        // Prints out disassembled code, i.e., the instructions that comprise the Java bytecodes,
                        // for each of the methods in the class.
                        '-c',
                        // Prints out line and local variable tables.
                        '-l',
                        // Private things
                        '-p',
                        // Final constants
                        '-constants',
                        // Verbose - ideally we'd enable this and then we get constant pools too. Needs work to parse.
                        //'-v',
                        classFile,
                    ];
                    const objResult = await this.exec(this.compiler.objdumper, args, {
                        maxOutput: maxSize,
                        customCwd: dirPath,
                    });
                    const oneResult: ParsedAsmResult = {
                        asm: [
                            {
                                text: objResult.stdout,
                            },
                        ],
                    };

                    if (objResult.code === 0) {
                        oneResult.objdumpTime = objResult.execTime;
                    } else {
                        oneResult.asm = [
                            {
                                text: `<No output: javap returned ${objResult.code}>`,
                            },
                        ];
                    }
                    return oneResult;
                })
        );

        const merged: ParsedAsmResult = {asm: []};
        for (const result of results) {
            const asmBackup = merged.asm;
            Object.assign(merged, result);
            merged.asm = [...asmBackup, ...result.asm];
        }

        result.asm = merged.asm;
        return result;
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions) {
        // Forcibly enable javap
        filters.binary = true;

        return ['-Xlint:all', '-encoding', 'utf8'];
    }

    override async handleInterpreting(key, executeParameters) {
        const compileResult = await this.getOrBuildExecutable(key);
        if (compileResult.code === 0) {
            executeParameters.args = [
                '-Xss136K', // Reduce thread stack size
                '-XX:CICompilerCount=2', // Reduce JIT compilation threads. 2 is minimum
                '-XX:-UseDynamicNumberOfCompilerThreads',
                '-XX:-UseDynamicNumberOfGCThreads',
                '-XX:+UseSerialGC', // Disable parallell/concurrent garbage collector
                await this.getMainClassName(compileResult.dirPath),
                '-cp',
                compileResult.dirPath,
                ...executeParameters.args,
            ];
            const result = await this.runExecutable(this.javaRuntime, executeParameters, compileResult.dirPath);
            return {
                ...result,
                didExecute: true,
                buildResult: compileResult,
            };
        } else {
            return {
                stdout: compileResult.stdout,
                stderr: compileResult.stderr,
                code: compileResult.code,
                didExecute: false,
                buildResult: compileResult,
                timedOut: false,
            };
        }
    }

    async getMainClassName(dirPath: string) {
        const maxSize = this.env.ceProps('max-asm-size', 64 * 1024 * 1024);
        const files = await fs.readdir(dirPath);
        const results = await Promise.all(
            files
                .filter(f => f.endsWith('.class'))
                .map(async classFile => {
                    const options = {
                        maxOutput: maxSize,
                        customCwd: dirPath,
                    };
                    const objResult = await this.exec(this.compiler.objdumper, [classFile], options);
                    if (objResult.code !== 0) {
                        return null;
                    }

                    if (this.mainRegex.test(objResult.stdout)) {
                        return classFile;
                    }
                    return null;
                })
        );

        const candidates = results.filter(file => file !== null);
        if (candidates.length > 0) {
            // In case of multiple candidates, we'll just take the first one.
            const fileName = unwrap(candidates[0]);
            return fileName.substring(0, fileName.lastIndexOf('.'));
        }
        // We were unable to find a main method, let's error out assuming "Main"
        return 'Main';
    }

    override getArgumentParser() {
        return JavaParser;
    }

    override getOutputFilename(dirPath: string) {
        return path.join(dirPath, `${path.basename(this.compileFilename, this.lang.extensions[0])}.class`);
    }

    filterUserOptionsWithArg(userOptions: string[], oneArgForbiddenList: Set<string>) {
        const filteredOptions: string[] = [];
        let toSkip = 0;

        for (const userOption of userOptions) {
            if (toSkip > 0) {
                toSkip--;
                continue;
            }
            if (oneArgForbiddenList.has(userOption)) {
                toSkip = 1;
                continue;
            }

            filteredOptions.push(userOption);
        }

        return filteredOptions;
    }

    override filterUserOptions(userOptions: string[]) {
        const oneArgForbiddenList = new Set([
            // -d directory
            // Sets the destination directory for class files.
            '-d',
            // -s directory
            // Specifies the directory used to place the generated source files.
            '-s',
            // --source-path path or -sourcepath path
            // Specifies where to find input source files.
            '--source-path',
            '-sourcepath',
        ]);

        return this.filterUserOptionsWithArg(userOptions, oneArgForbiddenList);
    }

    override processAsm(result) {
        // Handle "error" documents.
        if (!result.asm.includes('\n') && result.asm[0] === '<') {
            return [{text: result.asm, source: null}];
        }

        // result.asm is an array of javap stdouts
        const parseds = result.asm.map(asm => this.parseAsmForClass(asm.text));
        // Sort class file outputs according to first source line they reference
        parseds.sort((o1, o2) => o1.firstSourceLine - o2.firstSourceLine);

        const segments: ParsedAsmResultLine[] = [];
        for (const [classNumber, parsed] of parseds.entries()) {
            if (classNumber > 0) {
                // Separate classes with two line breaks
                segments.push({text: '', source: null}, {text: '', source: null});
            }
            for (let i = 0; i < parsed.textsBeforeMethod.length; i++) {
                // Line-based highlighting doesn't work if some segments span multiple lines,
                // even if they don't have a source line number
                // -> split the lines and create segment for each separately
                for (const line of utils.splitLines(parsed.textsBeforeMethod[i])) {
                    // javap output always starts with "Compiled from" on first line, discard these lines.
                    if (line.startsWith('Compiled from')) {
                        continue;
                    }
                    segments.push({
                        text: line,
                        source: null,
                    });
                }

                // textsBeforeMethod[last] is actually *after* the last method.
                // Check whether there is a method following the text block
                if (i < parsed.methods.length) {
                    for (const {text, sourceLine} of parsed.methods[i].instructions) {
                        segments.push({text: text, source: {file: null, line: sourceLine}});
                    }
                }
            }
        }
        return {asm: segments};
    }

    parseAsmForClass(javapOut) {
        const textsBeforeMethod: string[] = [];
        const methods: {instructions: any[]; startLine?: number}[] = [];
        // javap output puts `    Code:` after every signature. (Line will not be shown to user)
        // We use this to find the individual methods.
        // Before the first `Code:` occurrence, there is the method signature as well as the name of the class.
        // Subsequent matches are always followed by lists of assembly instructions as well as line info mappings

        // Regex idea: make sure `Code:` is the only thing on the line. Also consume trailing line ending!
        const [classNameAndFirstMethodSignature, ...codeAndLineNumberTables] = javapOut.split(/^\s+Code:\s*$\r?\n/m);
        textsBeforeMethod.push(classNameAndFirstMethodSignature.trimEnd()); // possible trailing \r on windows

        for (const codeAndLineNumberTable of codeAndLineNumberTables) {
            const method = {
                instructions: [],
            } as (typeof methods)[0];
            methods.push(method);

            for (const codeLineCandidate of utils.splitLines(codeAndLineNumberTable)) {
                // Match
                //       1: invokespecial #1                  // Method java/lang/Object."<init>":()V
                // Or match the "default: <code>" block inside a lookupswitch instruction
                const match = codeLineCandidate.match(/\s+(\d+|default): (.*)/);
                if (match) {
                    const instrOffset = Number.parseInt(match[1]);
                    method.instructions.push({
                        instrOffset: instrOffset,
                        // Should an instruction ever not be followed by a line number table,
                        // it might contain a trailing \r on Windows -> trim it, otherwise this would not be necessary
                        text: codeLineCandidate.trimEnd(),
                    });
                } else {
                    // Attempt to match the closing } of a lookupswitch. If we don't include the closing bracket, then
                    // the brackets will be misaligned, and it may be confusing to read.
                    const isClosingCurlyBrace = codeLineCandidate.match(/\s+}/);
                    if (isClosingCurlyBrace) {
                        // Put closing curly brace in asm output
                        method.instructions.push({
                            text: codeLineCandidate.trimEnd(),
                        });
                        continue;
                    }
                    break;
                }
            }

            const lineRegex = /line\s*(\d+):\s*(\d+)/g;
            let m;
            let currentInstr = 0;
            let currentSourceLine = -1;
            let lastIndex = -1;
            do {
                m = lineRegex.exec(codeAndLineNumberTable);
                if (m) {
                    // If exec doesn't find a match anymore, lineRegex.lastIndex will be reset to 0
                    // therefore, cache value here on match
                    lastIndex = lineRegex.lastIndex;
                    const [, sourceLineS, instructionS] = m;
                    logger.verbose('Found source mapping: ', sourceLineS, 'to instruction', instructionS);
                    const instrOffset = Number.parseInt(instructionS);

                    // Some instructions don't receive an explicit line number.
                    // They are all assigned to the previous explicit line number,
                    // because the line consists of multiple instructions.
                    while (
                        currentInstr < method.instructions.length &&
                        method.instructions[currentInstr].instrOffset !== instrOffset
                    ) {
                        if (currentSourceLine === -1) {
                            logger.error('Skipping over instruction even though currentSourceLine == -1');
                        } else {
                            // instructions without explicit line number get assigned the last explicit/same line number
                            method.instructions[currentInstr].sourceLine = currentSourceLine;
                        }
                        currentInstr++;
                    }

                    const sourceLine = Number.parseInt(sourceLineS);
                    currentSourceLine = sourceLine;
                    if (method.instructions[currentInstr]) {
                        method.instructions[currentInstr].sourceLine = currentSourceLine;
                    }

                    if (method.startLine === undefined) {
                        method.startLine = sourceLine;
                    }
                    // method.instructions.push({sourceLine: instrOffset});
                }
            } while (m);
            if (lastIndex !== -1) {
                // Get "interesting" text after the LineNumbers table (header of next method/tail of file)
                // trimRight() because of trailing \r on Windows
                textsBeforeMethod.push(codeAndLineNumberTable.substr(lastIndex).trimEnd());
            }

            if (currentSourceLine !== -1) {
                // Assign remaining instructions to the same source line
                while (currentInstr + 1 < method.instructions.length) {
                    currentInstr++;
                    method.instructions[currentInstr].sourceLine = currentSourceLine;
                }
            }
        }
        return {
            // Used for sorting
            firstSourceLine: methods.reduce((prev, method) => {
                if (method.startLine) {
                    return prev === -1 ? method.startLine : Math.min(prev, method.startLine);
                } else {
                    return prev;
                }
            }, -1),
            methods: methods,
            textsBeforeMethod,
        };
    }
}
