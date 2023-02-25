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

// Based on stack-trace https://github.com/felixge/node-stack-trace

type StackFrame = {
    fileName: string | undefined;
    lineNumber?: number;
    functionName?: string;
    typeName?: string;
    methodName?: string;
    columnNumber?: number;
    native?: boolean;
};

enum TraceFormat {
    V8,
    Firefox,
}

export function parse(err: Error) {
    if (!err.stack) {
        return [];
    }

    let format: TraceFormat;

    if (typeof window === 'undefined') {
        // node
        format = TraceFormat.V8;
    } else {
        if (navigator.userAgent.includes('AppleWebKit')) {
            // Just going with V8 for now...
            format = TraceFormat.V8;
        } else if ((window as any).chrome) {
            format = TraceFormat.V8;
        } else if (navigator.userAgent.toLowerCase().includes('firefox')) {
            format = TraceFormat.Firefox;
        } else {
            // We'll just default to V8 for now...
            format = TraceFormat.V8;
        }
    }
    if (format === TraceFormat.V8) {
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
    } else {
        return err.stack
            .split('\n')
            .map((line): StackFrame | undefined => {
                const lineMatch = line.match(/(.*)@(.*):(\d+):(\d+)/);
                if (!lineMatch) {
                    return;
                }

                let object: string | undefined;
                let method: string | undefined;
                let functionName: string | undefined;
                let typeName: string | undefined;
                let methodName: string | undefined;

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
                };
            })
            .filter(frame => frame !== undefined) as StackFrame[];
    }
}
