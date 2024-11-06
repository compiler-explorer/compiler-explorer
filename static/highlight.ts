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

type Interval = {start: number; length: number};

function regexExecAll(base_re: RegExp, s: string) {
    const re = new RegExp(base_re.source, base_re.flags + 'gd');
    let m: any;
    const matches: Interval[] = [];
    while ((m = re.exec(s)) != null) {
        // TODO(jeremy-rifkin): Find a way to get TS to understand that m.indices is real
        matches.push({
            start: m.indices[0][0],
            length: m.indices[0][1] - m.indices[0][0],
        });
    }
    return matches;
}

export function highlight(str: string, regexes: RegExp[]) {
    // At the moment, because compiler names are short, I think this solution is best. It's easiest to just
    // track intervals with an array. The ideal solution in the general case is probably something along the
    // lines of an interval tree.
    const intervals: Interval[] = [];
    for (const regex of regexes) {
        intervals.push(...regexExecAll(regex, str));
    }
    // sort by interval start
    intervals.sort((a, b) => a.start - b.start);
    // combine intervals
    let i = 0;
    while (i < intervals.length - 1) {
        const {start: AStart, length: ALength} = intervals[i];
        const {start: BStart, length: BLength} = intervals[i + 1];
        if (AStart + ALength >= BStart) {
            intervals.splice(i, 2, {
                start: AStart,
                length: BStart + BLength - AStart,
            });
        } else {
            i++;
        }
    }
    // for each interval, highlight
    let offset = 0;
    for (const {start, length} of intervals) {
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
