// Copyright (c) 2023, Compiler Explorer Authors
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

function regexExecAll(base_re: RegExp, s: string) {
    const re = new RegExp(base_re.source, base_re.flags + 'gd');
    let m: any;
    const matches: [number, number][] = [];
    while ((m = re.exec(s)) != null) {
        // TODO(jeremy-rifkin): Find a way to get TS to understand that m.indices is real
        matches.push([m.indices[0][0], m.indices[0][1] - m.indices[0][0]]);
    }
    return matches;
}

export function highlight(str: string, regexes: RegExp[]) {
    // At the moment, because compiler names are short, I think this solution is best. It's easiest to just
    // track intervals with an array. The ideal solution in the general case is probably something along the
    // lines of an interval tree.
    // [start, length]
    const intervals: [number, number][] = [];
    for (const regex of regexes) {
        intervals.push(...regexExecAll(regex, str));
    }
    // sort
    intervals.sort((a, b) => a[0] - b[0]);
    // combine intervals
    let i = 0;
    while (i < intervals.length - 1) {
        const intervalA = intervals[i];
        const intervalB = intervals[i + 1];
        if (intervalA[0] + intervalA[1] >= intervalB[0]) {
            intervals.splice(i, 2, [intervalA[0], intervalB[0] + intervalB[1] - intervalA[0]]);
        } else {
            i++;
        }
    }
    // for each interval, highlight
    let offset = 0;
    for (const [start, length] of intervals) {
        const tagStart = '<span class="highlight">';
        const tagEnd = '</span>';
        const intervalStart = offset + start;
        const intervalEnd = offset + start + length;
        str =
            str.slice(0, intervalStart) +
            tagStart +
            str.slice(intervalStart, intervalEnd) +
            tagEnd +
            str.slice(intervalEnd);
        offset += tagStart.length + tagEnd.length;
    }
    return str;
}
