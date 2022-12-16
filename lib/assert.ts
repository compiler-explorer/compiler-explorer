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

import * as fs from 'fs';
import path from 'path';

// Based on stack-trace https://github.com/felixge/node-stack-trace
// eslint-disable-next-line @typescript-eslint/no-namespace
export namespace stacktrace {
    type StackFrame = {
        fileName: string | undefined;
        lineNumber?: number;
        functionName?: string;
        typeName?: string;
        methodName?: string;
        columnNumber?: number;
        native?: boolean;
    };

    export function parse(err: Error) {
        if (!err.stack) {
            return [];
        }

        return err.stack
            .split('\n')
            .slice(1)
            .map((line): StackFrame | undefined => {
                if (/^\s*-{4,}$/.test(line)) {
                    return {
                        fileName: line,
                    };
                }

                const lineMatch = line.match(/at (?:(.+?)\s+\()?(?:(.+?):(\d+)(?::(\d+))?|([^)]+))\)?/);
                if (!lineMatch) {
                    return;
                }

                let object: string | undefined;
                let method: string | undefined;
                let functionName: string | undefined;
                let typeName: string | undefined;
                let methodName: string | undefined;
                const isNative = lineMatch[5] === 'native';

                if (lineMatch[1]) {
                    functionName = lineMatch[1];
                    let methodStart = functionName.lastIndexOf('.');
                    if (functionName[methodStart - 1] === '.') methodStart--;
                    if (methodStart > 0) {
                        object = functionName.substring(0, methodStart);
                        method = functionName.substring(methodStart + 1);
                        const objectEnd = object.indexOf('.Module');
                        if (objectEnd > 0) {
                            functionName = functionName.substring(objectEnd + 1);
                            object = object.substring(0, objectEnd);
                        }
                    }
                }

                if (method) {
                    typeName = object;
                    methodName = method;
                }

                if (method === '<anonymous>') {
                    methodName = undefined;
                    functionName = undefined;
                }

                return {
                    fileName: lineMatch[2] || undefined,
                    lineNumber: parseInt(lineMatch[3], 10) || undefined,
                    functionName: functionName,
                    typeName: typeName,
                    methodName: methodName,
                    columnNumber: parseInt(lineMatch[4], 10) || undefined,
                    native: isNative,
                };
            })
            .filter(frame => frame !== undefined) as StackFrame[];
    }
}

function check_path(parent: string, directory: string) {
    // https://stackoverflow.com/a/45242825/15675011
    const relative = path.relative(parent, directory);
    if (relative && !relative.startsWith('..') && !path.isAbsolute(relative)) {
        return relative;
    } else {
        return false;
    }
}

function get_diagnostic() {
    const e = new Error(); // eslint-disable-line unicorn/error-message
    const trace = stacktrace.parse(e);
    const invoker_frame = trace[3];
    if (invoker_frame.fileName && invoker_frame.lineNumber) {
        // Just out of an abundance of caution...
        const relative = check_path(global.ce_base_directory, invoker_frame.fileName);
        if (relative) {
            try {
                const file = fs.readFileSync(invoker_frame.fileName, 'utf-8');
                const lines = file.split('\n');
                return {
                    file: relative,
                    line: invoker_frame.lineNumber,
                    src: lines[invoker_frame.lineNumber - 1].trim(),
                };
            } catch (e: any) {}
        }
    }
}

function fail(fail_message: string, user_message: string | undefined, args: any[]): never {
    // Assertions will look like:
    // Assertion failed
    // Assertion failed: Foobar
    // Assertion failed: Foobar, [{"foo": "bar"}]
    // Assertion failed: Foobar, [{"foo": "bar"}], at `assert(x.foo.length < 2, "Foobar", x)`
    let assert_line = fail_message;
    if (user_message) {
        assert_line += `: ${user_message}`;
    }
    if (args.length > 0) {
        try {
            assert_line += ', ' + JSON.stringify(args);
        } catch (e) {}
    }

    const diagnostic = get_diagnostic();
    if (diagnostic) {
        throw new Error(assert_line + `, at ${diagnostic.file}:${diagnostic.line} \`${diagnostic.src}\``);
    } else {
        throw new Error(assert_line);
    }
}

export function assert<C>(c: C, message?: string, ...args: any[]): asserts c {
    if (!c) {
        fail('Assertion failed', message, args);
    }
}

export function unwrap<T>(x: T | undefined | null, message?: string, ...args: any[]): T {
    if (x === undefined || x === null) {
        fail('Unwrap failed', message, args);
    }
    return x;
}
