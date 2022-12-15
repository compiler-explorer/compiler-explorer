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

import * as fs from 'fs-extra';

/* eslint-disable unicorn/no-abusive-eslint-disable */
/* eslint-disable */
namespace stacktrace {
    export function get(belowFn) {
        const oldLimit = Error.stackTraceLimit;
        Error.stackTraceLimit = Infinity;

        const dummyObject: any = {};

        const v8Handler = Error.prepareStackTrace;
        Error.prepareStackTrace = function (dummyObject, v8StackTrace) {
            return v8StackTrace;
        };
        Error.captureStackTrace(dummyObject, belowFn || get);

        const v8StackTrace = dummyObject.stack;
        Error.prepareStackTrace = v8Handler;
        Error.stackTraceLimit = oldLimit;

        return v8StackTrace;
    }

    export function parse(err) {
        if (!err.stack) {
            return [];
        }

        const lines = err.stack.split('\n').slice(1);
        return lines
            .map(function (line) {
                if (line.match(/^\s*[-]{4,}$/)) {
                    return createParsedCallSite({
                        fileName: line,
                        lineNumber: null,
                        functionName: null,
                        typeName: null,
                        methodName: null,
                        columnNumber: null,
                        native: null,
                    });
                }

                const lineMatch = line.match(/at (?:(.+?)\s+\()?(?:(.+?):(\d+)(?::(\d+))?|([^)]+))\)?/);
                if (!lineMatch) {
                    return;
                }

                let object: string | null = null;
                let method: string | null = null;
                let functionName: string | null = null;
                let typeName: string | null = null;
                let methodName: string | null = null;
                let isNative = lineMatch[5] === 'native';

                if (lineMatch[1]) {
                    functionName = lineMatch[1];
                    let methodStart = functionName!.lastIndexOf('.');
                    if (functionName![methodStart - 1] == '.') methodStart--;
                    if (methodStart > 0) {
                        object = functionName!.substr(0, methodStart);
                        method = functionName!.substr(methodStart + 1);
                        const objectEnd = object.indexOf('.Module');
                        if (objectEnd > 0) {
                            functionName = functionName!.substr(objectEnd + 1);
                            object = object.substr(0, objectEnd);
                        }
                    }
                }

                if (method) {
                    typeName = object;
                    methodName = method;
                }

                if (method === '<anonymous>') {
                    methodName = null;
                    functionName = null;
                }

                const properties = {
                    fileName: lineMatch[2] || null,
                    lineNumber: parseInt(lineMatch[3], 10) || null,
                    functionName: functionName,
                    typeName: typeName,
                    methodName: methodName,
                    columnNumber: parseInt(lineMatch[4], 10) || null,
                    native: isNative,
                };

                return createParsedCallSite(properties);
            })
            .filter(function (callSite) {
                return !!callSite;
            });
    }

    function CallSite(this: any, properties) {
        for (const property in properties) {
            this[property] = properties[property];
        }
    }

    const strProperties = [
        'this',
        'typeName',
        'functionName',
        'methodName',
        'fileName',
        'lineNumber',
        'columnNumber',
        'function',
        'evalOrigin',
    ];

    const boolProperties = ['topLevel', 'eval', 'native', 'constructor'];

    strProperties.forEach(function (property) {
        CallSite.prototype[property] = null;
        CallSite.prototype['get' + property[0].toUpperCase() + property.substr(1)] = function () {
            return this[property];
        };
    });

    boolProperties.forEach(function (property) {
        CallSite.prototype[property] = false;
        CallSite.prototype['is' + property[0].toUpperCase() + property.substr(1)] = function () {
            return this[property];
        };
    });

    function createParsedCallSite(properties) {
        return new CallSite(properties);
    }
}
/* eslint-enable */
/* eslint-enable unicorn/no-abusive-eslint-disable */

export function assert<C>(c: C, message?: string): asserts c {
    if (!c) {
        const e = new Error(); // eslint-disable-line unicorn/error-message
        const trace = stacktrace.parse(e);
        // eslint-disable-next-line import/namespace
        const file = fs.readFileSync(trace[1].fileName, 'utf-8');
        const lines = file.split('\n');
        throw new Error('Assertion failed' + (message ? ': ' + message : '') + '\n\n' + lines[trace[1].lineNumber - 1]);
    }
}

export function unwrap<T>(x: T | undefined | null, message?: string): T {
    assert(x, message);
    return x;
}
