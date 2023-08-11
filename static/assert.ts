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

import {isString} from '../shared/common-utils.js';
import {parse} from '../shared/stacktrace.js';

// This file defines three assert utilities:
// assert(condition, message?, extra_info...?): asserts condition
// unwrap(x: T | undefined | null, message?, extra_info...?): T
// assert_type(x, class, message?, extra_info...?)

function get_diagnostic() {
    const e = new Error(); // eslint-disable-line unicorn/error-message
    const trace = parse(e);
    if (trace.length >= 4) {
        const invoker_frame = trace[3];
        if (invoker_frame.fileName && invoker_frame.lineNumber) {
            return {
                file: invoker_frame.fileName,
                line: invoker_frame.lineNumber,
            };
        }
    }
}

function fail(fail_message: string, user_message: string | undefined, args: any[]): never {
    // Assertions will look like:
    // Assertion failed
    // Assertion failed: Foobar
    // Assertion failed: Foobar, [{"foo": "bar"}]
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
        throw new Error(assert_line + `, at ${diagnostic.file}:${diagnostic.line}`);
    } else {
        throw new Error(assert_line);
    }
}

export function assert<C>(c: C, message?: string, ...extra_info: any[]): asserts c {
    if (!c) {
        fail('Assertion failed', message, extra_info);
    }
}

export function unwrap<T>(x: T | undefined | null, message?: string, ...extra_info: any[]): T {
    if (x === undefined || x === null) {
        fail('Unwrap failed', message, extra_info);
    }
    return x;
}

// This mainly a utility for JQuery.val(): string | number | string[] | undefined, in our code we typically want a
// single string.
// T is syntax sugar for unwrapping to a string union
export function unwrapString<T extends string>(x: any, message?: string, ...extra_info: any[]): T {
    if (!isString(x)) {
        fail('String unwrap failed', message, extra_info);
    }
    return x as T;
}
