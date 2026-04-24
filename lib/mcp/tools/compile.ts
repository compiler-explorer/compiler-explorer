// Copyright (C) 2026 Hudson River Trading LLC <opensource@hudson-trading.com>
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

import type {McpServer} from '@modelcontextprotocol/sdk/server/mcp.js';
import {z} from 'zod';

import {BypassCache} from '../../../types/compilation/compilation.interfaces.js';
import type {LanguageKey} from '../../../types/languages.interfaces.js';
import {CompileHandler} from '../../handlers/compile.js';
import {truncateLines} from '../utils.js';

const DEFAULT_MAX_ASM_LINES = 500;
const DEFAULT_MAX_STDOUT_LINES = 100;
const DEFAULT_MAX_STDERR_LINES = 100;

export function registerCompileTool(server: McpServer, compileHandler: CompileHandler): void {
    server.tool(
        'compile',
        'Compile source code and return assembly output, stdout, and stderr',
        {
            source: z.string().describe('Source code to compile'),
            language: z.string().describe('Language ID (e.g. "c++", "c", "rust", "python")'),
            compiler: z.string().describe('Compiler ID (e.g. "g142", "clang_trunk", "rustc")'),
            options: z.string().optional().describe('Compiler flags (e.g. "-O2 -std=c++20 -Wall")'),
            execute: z.boolean().optional().describe('Whether to also execute the compiled program'),
            stdin: z.string().optional().describe('Standard input for execution (requires execute=true)'),
            filters: z
                .object({
                    intel: z.boolean().optional().describe('Use Intel assembly syntax (default: true)'),
                    demangle: z.boolean().optional().describe('Demangle symbol names (default: true)'),
                    directives: z.boolean().optional().describe('Filter assembler directives (default: true)'),
                    commentOnly: z.boolean().optional().describe('Filter comment-only lines (default: true)'),
                    labels: z.boolean().optional().describe('Filter unused labels (default: true)'),
                    libraryCode: z.boolean().optional().describe('Filter library code (default: false)'),
                    trim: z.boolean().optional().describe('Trim whitespace (default: false)'),
                })
                .optional()
                .describe('Output filters'),
            libraries: z
                .array(
                    z.object({
                        id: z.string().describe('Library ID'),
                        version: z.string().describe('Library version'),
                    }),
                )
                .optional()
                .describe('Libraries to link'),
            maxAsmLines: z
                .number()
                .int()
                .positive()
                .optional()
                .describe(`Cap assembly output to this many lines (default ${DEFAULT_MAX_ASM_LINES})`),
            maxStdoutLines: z
                .number()
                .int()
                .positive()
                .optional()
                .describe(`Cap stdout (compile + execute) to this many lines (default ${DEFAULT_MAX_STDOUT_LINES})`),
            maxStderrLines: z
                .number()
                .int()
                .positive()
                .optional()
                .describe(`Cap stderr (compile + execute) to this many lines (default ${DEFAULT_MAX_STDERR_LINES})`),
        },
        async ({
            source,
            language,
            compiler: compilerId,
            options,
            execute,
            stdin,
            filters,
            libraries,
            maxAsmLines,
            maxStdoutLines,
            maxStderrLines,
        }) => {
            const baseCompiler = compileHandler.findCompiler(language as LanguageKey, compilerId);
            if (!baseCompiler) {
                return {
                    content: [
                        {
                            type: 'text',
                            text: `Compiler "${compilerId}" not found for language "${language}"`,
                        },
                    ],
                    isError: true,
                };
            }

            try {
                const body = {
                    source,
                    options: {
                        userArguments: options || '',
                        compilerOptions: {executorRequest: !!execute},
                        filters: filters || {},
                        tools: [],
                        libraries: libraries || [],
                        executeParameters: {
                            args: [],
                            stdin: stdin || '',
                            runtimeTools: [],
                        },
                    },
                };

                const parsed = CompileHandler.parseRequestReusable(true, {}, body, baseCompiler);
                const result = await baseCompiler.compile(
                    parsed.source,
                    parsed.options,
                    parsed.backendOptions,
                    parsed.filters,
                    BypassCache.None,
                    parsed.tools,
                    parsed.executeParameters,
                    parsed.libraries,
                    [],
                );

                const asmCap = maxAsmLines ?? DEFAULT_MAX_ASM_LINES;
                const stdoutCap = maxStdoutLines ?? DEFAULT_MAX_STDOUT_LINES;
                const stderrCap = maxStderrLines ?? DEFAULT_MAX_STDERR_LINES;

                const asm = truncateLines(result.asm, asmCap);
                const stdout = truncateLines(result.stdout, stdoutCap);
                const stderr = truncateLines(result.stderr, stderrCap);

                const output: Record<string, unknown> = {
                    code: result.code,
                    asm: asm.text,
                    stdout: stdout.text,
                    stderr: stderr.text,
                    ...(asm.truncated && {asmTruncated: true, asmTotalLines: asm.totalLines}),
                    ...(stdout.truncated && {stdoutTruncated: true, stdoutTotalLines: stdout.totalLines}),
                    ...(stderr.truncated && {stderrTruncated: true, stderrTotalLines: stderr.totalLines}),
                };

                if (result.execResult) {
                    const execStdout = truncateLines(result.execResult.stdout, stdoutCap);
                    const execStderr = truncateLines(result.execResult.stderr, stderrCap);
                    output.execResult = {
                        code: result.execResult.code,
                        stdout: execStdout.text,
                        stderr: execStderr.text,
                        didExecute: result.execResult.didExecute,
                        ...(execStdout.truncated && {
                            stdoutTruncated: true,
                            stdoutTotalLines: execStdout.totalLines,
                        }),
                        ...(execStderr.truncated && {
                            stderrTruncated: true,
                            stderrTotalLines: execStderr.totalLines,
                        }),
                    };
                }

                if (asm.truncated || stdout.truncated || stderr.truncated) {
                    output.hint =
                        'Some output was capped. Raise maxAsmLines / maxStdoutLines / maxStderrLines to retrieve more.';
                }

                return {content: [{type: 'text', text: JSON.stringify(output, null, 2)}]};
            } catch (e) {
                return {
                    content: [{type: 'text', text: `Compilation error: ${(e as Error).message}`}],
                    isError: true,
                };
            }
        },
    );
}
