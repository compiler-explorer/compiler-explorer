// Copyright (c) 2021, Compiler Explorer Authors
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

import {assert} from '../assert.js';

// A prefix tree, really a trie, but I find the name annoyingly pompous, and
// as it's pronounced the same way as "tree", super confusing.
// Essentially we have a N-way tree, for N possible ASCII characters. Each
// mapping is added to the tree, and the terminal nodes (that code for an actual
// match) have an addition 'result' entry for their result.
// * It's linear in the number of entries to build (though it's a super high
//   fan out tree, so RAM usage is pretty bad, and cache locality poor).
// * It's linear in the length of a match to find the longest prefix, or a match.
// It's the "find longest prefix" performance characteristic that we want for the
// demangler.

type Node = Node[] & {result?: string};

type charRange = {
    startCol: number;
    endCol: number;
};

export type replaceAction = {
    newText: string;
    mapRanges: Record<number, Record<number, charRange>>;
    mapNames: Record<string, string>;
};

export class PrefixTree {
    root: Node = [];

    constructor(mappings: [string, string][]) {
        if (mappings) {
            for (const [from, to] of mappings) this.add(from, to);
        }
    }

    add(from: string, to: string) {
        let node = this.root;
        for (let i = 0; i < from.length; ++i) {
            const character = from.codePointAt(i);
            assert(character !== undefined, 'Undefined code point encountered in PrefixTree');
            if (!node[character]) node[character] = [];
            node = node[character];
        }
        node.result = to;
    }

    // Finds the longest possible match by walking along the N-way tree until we
    // mismatch or reach the end of the input string. Along the way, we note the
    // most recent match (if any), which will be our return value.
    findLongestMatch(needle: string) {
        let node = this.root;
        let match: [string, string] | [null, null] = [null, null];
        for (let i = 0; i < needle.length; ++i) {
            const character = needle.codePointAt(i);
            assert(character !== undefined, 'Undefined code point encountered in PrefixTree');
            node = node[character];
            if (!node) break;
            if (node.result) match = [needle.substring(0, i + 1), node.result];
        }
        return match;
    }

    findExact(needle: string) {
        let node = this.root;
        for (let i = 0; i < needle.length; ++i) {
            const character = needle.codePointAt(i);
            assert(character !== undefined, 'Undefined code point encountered in PrefixTree');
            node = node[character];
            if (!node) break;
        }
        if (node && node['result']) return node['result'];
        return null;
    }

    // Replace all matches (longest match first) in a line.
    replaceAll(line: string): replaceAction {
        let newText = '';
        let idxInOld = 0;
        let idxInNew = 0;
        const mapRanges = {};
        const mapNames = {};
        // Loop over each possible replacement point in the line.
        // Use a binary search to find the replacements (allowing a prefix match). If we couldn't find a match, skip
        // on, else use the replacement, and skip by that amount.
        while (idxInOld < line.length) {
            const lineBit = line.substring(idxInOld);
            const [oldValue, newValue] = this.findLongestMatch(lineBit);
            if (oldValue) {
                // We found a replacement.
                newText += newValue;
                mapNames[oldValue] = newValue;

                idxInOld += oldValue.length;
                idxInNew += newValue.length;

                // The annoying +1 ultimately comes from monaco.Position being 1-based.
                const oldStart = idxInOld - oldValue.length + 1;
                const oldEnd = idxInOld + 1;
                const newStart = idxInNew - newValue.length + 1;
                const newEnd = idxInNew + 1;

                // JS/TS don't allow using an object such as {oldStart, oldEnd} as a key in a dictionary/Map,
                // we use a nested dictionary instead.
                if (!mapRanges[oldStart]) mapRanges[oldStart] = {};
                mapRanges[oldStart][oldEnd] = {startCol: newStart, endCol: newEnd};
            } else {
                // No match; output the unmatched character, and keep looking.
                newText += line[idxInOld];
                idxInOld++;
                idxInNew++;
            }
        }
        return {
            newText: newText,
            mapRanges: mapRanges,
            mapNames: mapNames,
        };
    }
}
