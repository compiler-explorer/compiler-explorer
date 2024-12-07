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

export function isString(x: any): x is string {
    return typeof x === 'string' || x instanceof String;
}

// Object.keys is typed as returning :string[] for some reason
// This util is for cases where the key is a union of a few possible keys and we
// want the resulting array properly typed.
export function keys<K extends string | number | symbol>(o: Partial<Record<K, any>>): K[] {
    return Object.keys(o) as K[];
}

export function unique<V>(arr: V[]): V[] {
    return [...new Set(arr)];
}

export function intersection<V>(a: V[], b: V[]): V[] {
    const B = new Set(b);
    return [...a].filter(item => B.has(item));
}

// arr.filter(x => x !== null) returns a (T | null)[] even though it is a T[]
// Apparently the idiomatic solution is arr.filter((x): x is T => x !== null), but this is shorter (and the type
// predicate also isn't type checked so it doesn't seem safe to me)
export function remove<U, V extends U>(arr: U[], v: V) {
    return arr.filter(item => item !== v) as Exclude<U, V extends null | undefined ? V : never>[];
}

// https://www.typescriptlang.org/play?#code/KYDwDg9gTgLgBAMwK4DsDGMCWEVysAWwgDdgAeAVQBo4A1OUGYFAEwGc4KA+ACgEMoUAFycA2gF0axEbQCUcAN4AoOKrzAYSKLgFQAdAkwAbJlB6YmBOAF4ucC4TgBCa9bjF5fDgFEQaI0gs5NR0DCBMrBwoSEZGcAA+cKhBhijALHAA-KEiaaRQXBIA3EoAvkpKQf4CwHBoOGzwuiI8jVCYKADmCXDRsbLFFfUojXAgNupEpPyCNH1GsiUA9EtqcAB6mUMN8ACeE-hTwDNQNABECBAQZ4tKK2ubFUA

// For Array.prototype.sort
export function basic_comparator<T>(a: T, b: T) {
    if (a < b) {
        return -1;
    } else if (a > b) {
        return 1;
    } else {
        return 0;
    }
}

// https://stackoverflow.com/questions/41253310/typescript-retrieve-element-type-information-from-array-type
export type ElementType<ArrayType extends readonly unknown[]> = ArrayType extends readonly (infer T)[] ? T : never;

const EscapeMap = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#x27;',
    '`': '&#x60;',
};
const EscapeRE = new RegExp(`(?:${Object.keys(EscapeMap).join('|')})`, 'g');

export function escapeHTML(text: string) {
    return text.replace(EscapeRE, str => EscapeMap[str as keyof typeof EscapeMap]);
}

function splitIntoChunks(s: string, chunkSize: number): string[] {
    const chunks: string[] = [];
    const isNegative = s.slice(0, 1) === '-';
    if (isNegative) {
        s = s.slice(1);
    }
    const firstChunkLength = s.length % chunkSize;
    if (firstChunkLength !== 0) {
        chunks.push(s.slice(0, firstChunkLength));
    }
    for (let i = firstChunkLength; i < s.length; i += chunkSize) {
        chunks.push(s.slice(i, i + chunkSize));
    }
    if (isNegative) {
        chunks[0] = '-' + (chunks[0] ?? '');
    }
    return chunks;
}

export function addDigitSeparator(n: string, digitSeparator: string, chunkSize: number): string {
    return splitIntoChunks(n, chunkSize).join(digitSeparator);
}

// Initially based on shell-split (whose source has gone missing?) and then heavily modified to fit CE's use cases.
export function splitArguments(str = ''): string[] {
    let rest = str.replace(/^\s+/, '');

    if (!rest) return [];

    const results: string[] = [];
    const addResult = (result: string) => {
        if (results[position] === undefined) results[position] = result;
        else results[position] += result;
    };
    let position = 0;
    let inDoubleQuotes = false;

    while (rest) {
        if (inDoubleQuotes) {
            const match = /["\\]/.exec(rest);
            const matchEnd = match ? match.index : rest.length;

            addResult(rest.slice(0, matchEnd));

            // append backslash escapes
            if (match && match[0] === '\\') {
                if (rest.length - matchEnd < 2) {
                    addResult('\\');
                    rest = '';
                } else {
                    // don't append escaped newlines
                    if (rest.charAt(1) !== '\n') addResult(rest.charAt(matchEnd + 1));

                    rest = rest.slice(matchEnd + 2);
                }
                // for the other recognized decision point ("), exit double-quoting
            } else {
                rest = rest.slice(matchEnd + 1);
                inDoubleQuotes = false;
            }

            // out of double-quoted context
        } else {
            // split at any whitespace out of quoted context
            const wsMatch = /^\s+/.exec(rest);

            if (wsMatch) {
                rest = rest.slice(wsMatch[0].length);
                ++position;
            }

            // find the next decision point
            const nextMatch = /[\s'"\\]/.exec(rest);

            // if there is an upcoming split or context switch
            if (nextMatch) {
                addResult(rest.slice(0, nextMatch.index));
                rest = rest.slice(nextMatch.index);

                // append backslash escapes
                if (nextMatch[0] === '\\') {
                    if (rest.length < 2) {
                        // Special case for an unterminated `\` at the end of the string.
                        addResult('\\');
                        rest = '';
                    } else {
                        // don't append escaped newlines
                        if (rest.charAt(1) !== '\n') addResult(rest.charAt(1));

                        rest = rest.slice(2);
                    }

                    // append single-quoted strings
                } else if (nextMatch[0] === "'") {
                    let quotePos = rest.indexOf("'", 1);
                    if (quotePos === -1) quotePos = rest.length;
                    addResult(rest.slice(1, quotePos));
                    rest = rest.slice(quotePos + 1);

                    // enter double-quoted context
                } else if (nextMatch[0] === '"') {
                    // move into double-quoted string
                    rest = rest.slice(1);
                    inDoubleQuotes = true;
                } // otherwise, we assume it's whitespace and will reiterate

                // if we won't be entering any more contexts
            } else {
                // finish results
                addResult(rest);
                rest = '';
            }
        }
    }
    return results;
}
