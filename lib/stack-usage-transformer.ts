// Copyright (c) 2023, Compiler Explorer Authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

type Path = string;

type StackInfo = {
    displayString: string;
};

export type StackUsageInfo = StackInfo & {
    DebugLoc: DebugLoc;
    Function: string;
    BytesUsed: number;
    Qualifier: 'static' | 'dynamic' | 'dynamic,bounded';
};

type DebugLoc = {
    File: Path;
    Line: number;
    Column: number;
};

export function parse(suText: string) {
    const output: StackUsageInfo[] = [];
    suText
        .split('\n')
        .filter(e => e)
        .forEach(line => {
            const c = line.split('\t');
            const pathLocName = c[0].split(':');
            const lineNumber = +pathLocName[1];
            const qualifier = c.at(-1);
            const su = {
                DebugLoc: {File: pathLocName[0], Line: lineNumber, Column: 0},
                Function: pathLocName.at(-1),
                Qualifier: qualifier,
                BytesUsed: parseInt(c[1]),
                displayString: c[1] + ' bytes, ' + qualifier,
            };
            output.push(su as StackUsageInfo);
        });
    return output;
}
