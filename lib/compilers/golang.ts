// Copyright (c) 2022, Compiler Explorer Authors
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

import _ from 'underscore';

import type {PreliminaryCompilerInfo} from '../../types/compiler.interfaces.js';
import type {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces.js';
import type {ResultLine} from '../../types/resultline/resultline.interfaces.js';
import {unwrap} from '../assert.js';
import {BaseCompiler} from '../base-compiler.js';
import * as utils from '../utils.js';

import {GolangParser} from './argument-parsers.js';

// Each arch has a list of jump instructions in
// Go source src/cmd/asm/internal/arch.
// x86 -> j, b
// arm -> cb, tb
// s390x -> cmpb, cmpub
const JUMP_RE = /^(j|b|cb|tb|cmpb|cmpub).*/i;
const LINE_RE = /^\s+(0[Xx]?[\dA-Za-z]+)?\s?(\d+)\s*\(([^:]+):(\d+)\)\s*([A-Z]+)(.*)/;
const UNKNOWN_RE = /^\s+(0[Xx]?[\dA-Za-z]+)?\s?(\d+)\s*\(<unknown line number>\)\s*([A-Z]+)(.*)/;
const FUNC_RE = /TEXT\s+[".]*(\S+)\(SB\)/;
const LOGGING_RE = /^[^:]+:\d+:(\d+:)?\s.*/;
const DECIMAL_RE = /(\s+)(\d+)(\s?)$/;

type GoEnv = {
    GOROOT?: string;
    GOARCH?: string;
    GOOS?: string;
};

export class GolangCompiler extends BaseCompiler {
    private readonly GOENV: GoEnv;

    static get key() {
        return 'golang';
    }

    constructor(compilerInfo: PreliminaryCompilerInfo, env) {
        super(compilerInfo, env);
        const group = this.compiler.group;

        const goroot = this.compilerProps<string | undefined>(
            'goroot',
            this.compilerProps<string | undefined>(`group.${group}.goroot`),
        );
        const goarch = this.compilerProps<string | undefined>(
            'goarch',
            this.compilerProps<string | undefined>(`group.${group}.goarch`),
        );
        const goos = this.compilerProps<string | undefined>(
            'goos',
            this.compilerProps<string | undefined>(`group.${group}.goos`),
        );

        this.GOENV = {};
        if (goroot) {
            this.GOENV.GOROOT = goroot;
        }
        if (goarch) {
            this.GOENV.GOARCH = goarch;
        }
        if (goos) {
            this.GOENV.GOOS = goos;
        }
    }

    convertNewGoL(code: ResultLine[]): string {
        let prevLine: string | null = null;
        let file: string | null = null;
        let fileCount = 0;
        let func: string | null = null;
        const funcCollisions: Record<string, number> = {};
        const labels: Record<string, boolean> = {};
        const usedLabels: Record<string, boolean> = {};
        const lines = code.map(obj => {
            let pcMatch: string | null = null;
            let fileMatch: string | null = null;
            let lineMatch: string | null = null;
            let ins: string | null = null;
            let args: string | null = null;

            const line = obj.text;
            let match = line.match(LINE_RE);
            if (match) {
                pcMatch = match[2];
                fileMatch = match[3];
                lineMatch = match[4];
                ins = match[5];
                args = match[6];
            } else {
                match = line.match(UNKNOWN_RE);
                if (match) {
                    pcMatch = match[2];
                    ins = match[3];
                    args = match[4];
                } else {
                    return [];
                }
            }

            match = line.match(FUNC_RE);
            if (match) {
                // Normalize function name.
                func = match[1].replace(/[()*.]+/g, '_');

                // It's possible for normalized function names to collide.
                // Keep a count of collisions per function name. Labels get
                // suffixed with _[collisions] when collisions > 0.
                let collisions = funcCollisions[func];
                if (collisions == null) {
                    collisions = 0;
                } else {
                    collisions++;
                }

                funcCollisions[func] = collisions;
            }

            const res: string[] = [];
            if (pcMatch && !labels[pcMatch]) {
                // Create pseudo-label.
                let label = pcMatch.replace(/^0{0,4}/, '');
                let suffix = '';
                if (func && funcCollisions[func] > 0) {
                    suffix = `_${funcCollisions[func]}`;
                }

                label = `${func}_pc${label}${suffix}:`;
                if (!labels[label]) {
                    res.push(label);
                    labels[label] = true;
                }
            }

            if (fileMatch && file !== fileMatch) {
                fileCount++;
                res.push(`\t.file ${fileCount} "${fileMatch}"`);
                file = fileMatch;
            }

            if (lineMatch && prevLine !== lineMatch) {
                res.push(`\t.loc ${fileCount} ${lineMatch} 0`);
                prevLine = lineMatch;
            }

            if (func) {
                args = this.replaceJump(func, funcCollisions[func], ins, args, usedLabels);
                res.push(`\t${ins}${args}`);
            }
            return res;
        });

        // Find unused pseudo-labels so they can be filtered out.
        const unusedLabels = _.mapObject(labels, (val, key) => !usedLabels[key]);

        return lines
            .flat()
            .filter(line => !unusedLabels[line])
            .join('\n');
    }

    replaceJump(
        func: string,
        collisions: number,
        ins: string,
        args: string,
        usedLabels: Record<string, boolean>,
    ): string {
        // Check if last argument is a decimal number.
        const match = args.match(DECIMAL_RE);
        if (!match) {
            return args;
        }

        // Check instruction has a jump prefix
        if (JUMP_RE.test(ins)) {
            let label = `${func}_pc${match[2]}`;
            if (collisions > 0) {
                label += `_${collisions}`;
            }
            usedLabels[label + ':'] = true; // record label use for later filtering
            return `${match[1]}${label}${match[3]}`;
        }

        return args;
    }

    extractLogging(stdout: ResultLine[]): string {
        const filepath = `./${this.compileFilename}`;

        return stdout
            .filter(obj => obj.text.match(LOGGING_RE))
            .map(obj => obj.text.replace(filepath, '<source>'))
            .join('\n');
    }

    override async postProcess(result) {
        let out = result.stderr;
        if (this.compiler.id === '6g141') {
            out = result.stdout;
        }
        const logging = this.extractLogging(out);
        result.asm = this.convertNewGoL(out);
        result.stderr = [];
        result.stdout = utils.parseOutput(logging, result.inputFilename);
        return Promise.all([result, '', '']);
    }

    override getSharedLibraryPathsAsArguments() {
        return [];
    }

    override optionsForFilter(filters: ParseFiltersAndOutputOptions, outputFilename: string, userOptions?: string[]) {
        // If we're dealing with an older version...
        if (this.compiler.id === '6g141') {
            return ['tool', '6g', '-g', '-o', outputFilename, '-S'];
        }

        if (filters.binary) {
            return ['build', '-o', outputFilename, '-gcflags=' + unwrap(userOptions).join(' ')];
        } else {
            // Add userOptions to -gcflags to preserve previous behavior.
            return ['build', '-o', outputFilename, '-gcflags=-S ' + unwrap(userOptions).join(' ')];
        }
    }

    override filterUserOptions(userOptions: string[]) {
        if (this.compiler.id === '6g141') {
            return userOptions;
        }
        // userOptions are added to -gcflags in optionsForFilter
        return [];
    }

    override getDefaultExecOptions() {
        const options = {
            ...super.getDefaultExecOptions(),
        };

        options.env = {
            ...options.env,
            ...this.GOENV,
        };

        return options;
    }

    override getArgumentParser(): any {
        return GolangParser;
    }
}
