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

import path from 'path';

import type {
    Fix,
    Link,
    MessageWithLocation,
    ResultLine,
    ResultLineTag,
} from '../../types/resultline/resultline.interfaces.js';
import type {Artifact, ToolResult} from '../../types/tool.interfaces.js';
import * as utils from '../utils.js';

import {BaseTool} from './base-tool.js';
import {ToolEnv} from './base-tool.interface.js';
import {ToolInfo} from '../../types/tool.interfaces.js';

export class SonarTool extends BaseTool {
    reproducer?: Artifact;

    static get key() {
        return 'sonar-tool';
    }

    constructor(toolInfo: ToolInfo, env: ToolEnv) {
        super(toolInfo, env);

        this.addOptionsToToolArgs = false;
        this.parseOutput = this.parseOutputAndFixes;
    }

    makeURL(rule: string): string {
        return `https://rules.sonarsource.com/cpp/RSPEC-${rule.substring(1)}`;
    }

    readTag(tag: any, output: ResultLine[], severity: number) {
        const text = `${tag.text} (cpp:${tag.ruleKey})`;
        const fixes: Fix[] = tag.fixes.map(f => ({
            title: f.message,
            edits: f.edits.map(e => ({
                line: e.startLine,
                column: e.startColumn,
                text: e.text,
                endline: e.endLine,
                endcolumn: e.endColumn,
            })),
        }));
        const flow: MessageWithLocation[] = tag.parts.map(p => ({
            line: p.line,
            column: p.column,
            file: p.filename,
            text: p.text,
            endline: p.endLine,
            endcolumn: p.endColumn,
        }));
        const link: Link = {
            text: 'More...',
            url: this.makeURL(tag.ruleKey),
        };
        const cetag: ResultLineTag = {
            severity,
            text,
            line: tag.line,
            column: tag.column,
            endline: tag.endLine,
            endcolumn: tag.endColumn,
            link,
            fixes,
            flow,
        };
        output.push({
            text: `${(tag.line + ':' + tag.column).padEnd(6)} ${flow.length > 0 ? '\u2795 ' : ''}${text}`,
            tag: cetag,
        });
        if (fixes.length > 0) {
            output.push({text: '\t\u21B3 \u001B[3m\uD83D\uDCA1 A Quick Fix is available for this issue\u001B[0m'});
        }
        if (flow.length > 0) {
            output.push(
                ...flow
                    .filter(f => f.text)
                    .map((f, i) => ({
                        text: `\t\u21B3 ${flow.length > 1 ? ' ' + (i + 1) : ''} ${f.text}`,
                    })),
            );
        }
        output.push({text: `\t\u001B[2m\u21B3 ${this.makeURL(tag.ruleKey)}\u001B[0m`});
    }

    simplifyPathes(lines: string, inputFilepath?: string, pathPrefix?: string): string {
        if (inputFilepath) {
            lines = lines.replaceAll(inputFilepath, '<source>').replaceAll('<stdin>', '<source>');
        }
        if (pathPrefix) {
            lines = lines.replaceAll(pathPrefix, '');
        }
        return lines;
    }

    parseOutputAndFixes(lines: string, inputFilepath?: string, pathPrefix?: string): ResultLine[] {
        if (!lines) return [];
        let output: ResultLine[] = [];
        try {
            const results = JSON.parse(this.simplifyPathes(lines, inputFilepath, pathPrefix));
            if (results.header) {
                output.push(
                    ...utils.splitLines('\u001B[3m\u001B[32m' + results.header + '\u001B[0m').map(s => ({text: s})),
                );
            }
            if (results.parsingErrors && results.parsingErrors.length > 0) {
                output.push(
                    {text: ''},
                    {
                        text:
                            '\u001B[1m\u26A0\uFE0F ' +
                            'Parsing errors\u001B[0m (these can affect the quality of the analysis):',
                    },
                );
                for (const e of results.parsingErrors) {
                    this.readTag(e, output, 8);
                }
            }
            if (results.issues && results.issues.length > 0) {
                output.push({text: ''}, {text: '\u001B[1m\uD83D\uDC1E Issues:\u001B[0m'});
                for (const e of results.issues) {
                    this.readTag(e, output, 4);
                }
                output.push({text: ''});
            } else if (!results.parsingErrors || results.parsingErrors.length === 0) {
                output.push({text: ''}, {text: '\u001B[1m\u2728 No issue found\u001B[0m'});
            }
            if (results.logs && results.logs.length > 0) {
                output.push(
                    {text: ''},
                    {text: '\u001B[1m\uD83D\uDCDC Logs:\u001B[0m'},
                    ...results.logs.map(l => ({text: `[${l.level}] ${l.message}`})),
                );
            }
            if (results.reproducer) {
                this.reproducer = {
                    content: results.reproducer.content,
                    type: 'application/octet-stream',
                    name: 'sonar-reproducer.zip',
                    title: 'reproducer',
                };
            }
        } catch (err) {
            output = utils.splitLines(lines).map(l => ({text: l}));
        }
        return output;
    }

    buildCompilationCMD(compilationInfo: Record<any, any>, inputFilePath: string) {
        const cmd: any[] = [];
        cmd.push(compilationInfo.compiler.exe);

        // Collecting the flags of compilation

        let compileFlags: string[] = utils.splitArguments(compilationInfo.compiler.options);
        const includeflags = super.getIncludeArguments(compilationInfo.libraries, compilationInfo.compiler);
        if (typeof includeflags === 'string') {
            compileFlags = compileFlags.concat(includeflags);
        }
        const libOptions = super.getLibraryOptions(compilationInfo.libraries, compilationInfo.compiler);
        compileFlags = compileFlags.concat(libOptions);
        const manualCompileFlags = compilationInfo.options.filter(option => option !== inputFilePath);
        compileFlags = compileFlags.concat(manualCompileFlags);
        compileFlags = compileFlags.filter(f => f !== '');

        return cmd.concat(compileFlags);
    }

    override async runTool(
        compilationInfo: Record<any, any>,
        inputFilePath?: string,
        args?: string[],
    ): Promise<ToolResult> {
        if (inputFilePath == null) {
            return new Promise(resolve => {
                resolve(this.createErrorResponse('Unable to run tool'));
            });
        }
        this.reproducer = undefined;
        let sonarArgs: string[] = (args ?? [])
            .filter(a => !a.includes('subprocess'))
            .concat(['--directory', path.dirname(inputFilePath), '--']);
        sonarArgs = sonarArgs.concat(this.buildCompilationCMD(compilationInfo, inputFilePath));

        const res: ToolResult = await super.runTool(compilationInfo, inputFilePath, sonarArgs);
        if (this.reproducer) {
            res.artifact = this.reproducer;
        }
        return res;
    }
}
