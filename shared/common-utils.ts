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
    return text.replace(EscapeRE, str => EscapeMap[str]);
}
