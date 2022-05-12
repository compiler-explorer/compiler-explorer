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
export class PrefixTree {
    constructor(mappings) {
        this.root = [];
        if (mappings) {
            for (const [from, to] of mappings) this.add(from, to);
        }
    }

    add(from, to) {
        let node = this.root;
        for (let i = 0; i < from.length; ++i) {
            const character = from.codePointAt(i);
            if (!node[character]) node[character] = [];
            node = node[character];
        }
        node.result = to;
    }

    // Finds the longest possible match by walking along the N-way tree until we
    // mismatch or reach the end of the input string. Along the way, we note the
    // most recent match (if any), which will be our return value.
    findLongestMatch(needle) {
        let node = this.root;
        let match = [null, null];
        for (let i = 0; i < needle.length; ++i) {
            const character = needle.codePointAt(i);
            node = node[character];
            if (!node) break;
            if (node.result) match = [needle.substr(0, i + 1), node.result];
        }
        return match;
    }

    findExact(needle) {
        let node = this.root;
        for (let i = 0; i < needle.length; ++i) {
            const character = needle.codePointAt(i);
            node = node[character];
            if (!node) break;
        }
        if (node && node['result']) return node['result'];
        return null;
    }

    // Replace all matches (longest match first) in a line.
    replaceAll(line) {
        let result = '';
        let index = 0;
        // Loop over each possible replacement point in the line.
        // Use a binary search to find the replacements (allowing a prefix match). If we couldn't find a match, skip
        // on, else use the replacement, and skip by that amount.
        while (index < line.length) {
            const lineBit = line.substr(index);
            const [oldValue, newValue] = this.findLongestMatch(lineBit);
            if (oldValue) {
                // We found a replacement.
                result += newValue;
                index += oldValue.length;
            } else {
                // No match; output the unmatched character, and keep looking.
                result += line[index];
                index++;
            }
        }
        return result;
    }
}
