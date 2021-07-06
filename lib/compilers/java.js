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

import { BaseCompiler } from '../base-compiler';
import { logger } from '../logger';
import * as utils from '../utils';

import { JavaParser } from './argument-parsers';

export class JavaCompiler extends BaseCompiler {
    static get key() {
        return 'java';
    }

    constructor(compilerInfo, env) {
        // Default is to disable all "cosmetic" filters
        if (!compilerInfo.disabledFilters) {
            compilerInfo.disabledFilters = ['labels', 'directives', 'commentOnly', 'trim'];
        }
        super(compilerInfo, env);
        this.javaRuntime = this.compilerProps(`compiler.${this.compiler.id}.runtime`);
    }

    getSharedLibraryPathsAsArguments() {
        return [];
    }

    async objdump(outputFilename, result, maxSize) {
        const dirPath = path.dirname(outputFilename);
        const files = await fs.readdir(dirPath);
        logger.verbose('Class files: ', files);
        const results = await Promise.all(files.filter(f => f.endsWith('.class')).map(async classFile => {
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
            const objResult = await this.exec(this.compiler.objdumper, args, {maxOutput: maxSize, customCwd: dirPath});
            const oneResult = {
                asm: objResult.stdout,
            };

            if (objResult.code !== 0) {
                oneResult.asm = '<No output: javap returned ' + objResult.code + '>';
            } else {
                oneResult.objdumpTime = objResult.execTime;
            }
            return oneResult;
        }));

        const merged = {asm: []};
        for (const result of results) {
            const asmBackup = merged.asm;
            Object.assign(merged, result);
            merged.asm = asmBackup;
            merged.asm.push(result.asm);
        }

        result.asm = merged.asm;
        return result;
    }

    optionsForFilter(filters) {
        // Forcibly enable javap
        filters.binary = true;

        return [
            '-Xlint:all',
        ];
    }

    async handleInterpreting(source, executeParameters) {
        const dirPath = await this.newTempDir();
        const inputFile = path.join(dirPath, this.compileFilename);
        await fs.writeFile(inputFile, source);
        const execOpts = this.getDefaultExecOptions();
        const compileResult = await this.runCompiler(this.compiler.exe, [inputFile], inputFile, execOpts);

        executeParameters.args = [
            '-Xss136K', // Reduce thread stack size
            '-XX:CICompilerCount=2', // Reduce JIT compilation threads. 2 is minimum
            '-XX:-UseDynamicNumberOfCompilerThreads',
            '-XX:-UseDynamicNumberOfGCThreads',
            '-XX:+UseSerialGC', // Disable parallell/concurrent garbage collector
            this.getMainClassName(),
            ...executeParameters.args,
        ];
        const result = await this.runExecutable(this.javaRuntime, executeParameters, dirPath);
        result.didExecute = true;
        result.buildResult = compileResult;
        return result;
    }

    getMainClassName() {
        // TODO(supergrecko): The main class name is currently hardcoded
        return 'Main';
    }

    getArgumentParser() {
        return JavaParser;
    }

    getOutputFilename(dirPath) {
        return path.join(dirPath, `${path.basename(this.compileFilename, this.lang.extensions[0])}.class`);
    }

    filterUserOptionsWithArg(userOptions, oneArgForbiddenList) {
        const filteredOptions = [];
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

    filterUserOptions(userOptions) {
        const oneArgForbiddenList = new Set([
            // -d directory
            // Sets the destination directory for class files.
            '-d',
            // -s directory
            // Specifies the directory used to place the generated source files.
            '-s',
            // --source-path path or -sourcepath path
            // Specifies where to find input source files.
            '--source-path', '-sourcepath',
        ]);

        return this.filterUserOptionsWithArg(userOptions, oneArgForbiddenList);
    }

    processAsm(result) {
        // Handle "error" documents.
        if (!result.asm.includes('\n') && result.asm[0] === '<') {
            return [{text: result.asm, source: null}];
        }

        // result.asm is an array of javap stdouts
        const parseds = result.asm.map(asm => this.parseAsmForClass(asm));
        // Sort class file outputs according to first source line they reference
        parseds.sort((o1, o2) => o1.firstSourceLine - o2.firstSourceLine);

        const segments = [];
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
        const textsBeforeMethod = [];
        const methods = [];
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
            };
            methods.push(method);

            for (const codeLineCandidate of utils.splitLines(codeAndLineNumberTable)) {
                // Match
                //       1: invokespecial #1                  // Method java/lang/Object."<init>":()V
                const match = codeLineCandidate.match(/\s+(\d+): (.*)/);
                if (match) {
                    const instrOffset = Number.parseInt(match[1]);
                    method.instructions.push({
                        instrOffset: instrOffset,
                        // Should an instruction ever not be followed by a line number table,
                        // it might contain a trailing \r on Windows -> trim it, otherwise this would not be necessary
                        text: codeLineCandidate.trimEnd(),
                    });
                } else {
                    break;
                }
            }

            let lineRegex = /line\s*(\d+):\s*(\d+)/g;
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
                    while (currentInstr < method.instructions.length
                    && method.instructions[currentInstr].instrOffset !== instrOffset) {
                        if (currentSourceLine !== -1) {
                            // instructions without explicit line number get assigned the last explicit/same line number
                            method.instructions[currentInstr].sourceLine = currentSourceLine;
                        } else {
                            logger.error('Skipping over instruction even though currentSourceLine == -1');
                        }
                        currentInstr++;
                    }

                    const sourceLine = Number.parseInt(sourceLineS);
                    currentSourceLine = sourceLine;
                    if (method.instructions[currentInstr]) {
                        method.instructions[currentInstr].sourceLine = currentSourceLine;
                    }

                    if (typeof method.startLine === 'undefined') {
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
            firstSourceLine: methods.reduce((p, m) => p === -1 ? m.startLine : Math.min(p, m.startLine), -1),
            methods: methods,
            textsBeforeMethod,
        };
    }
}
