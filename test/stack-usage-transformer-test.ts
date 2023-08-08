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

import {parse} from '../lib/stack-usage-transformer.js';

describe('stack usage transformer', () => {
    it('should work', async () => {
        const doc = `example.cpp:2:5:int square(int)\t16\tstatic
example.cpp:6:5:int f()\t32\tdynamic
example.cpp:7:5:int h()\t64\tdynamic,bounded
`;
        const output = parse(doc);
        output.should.deep.equal([
            {
                BytesUsed: 16,
                DebugLoc: {
                    File: 'example.cpp',
                    Line: 2,
                    Column: 0,
                },
                Function: 'int square(int)',
                Qualifier: 'static',
                displayString: '16 bytes, static',
            },
            {
                BytesUsed: 32,
                DebugLoc: {
                    File: 'example.cpp',
                    Line: 6,
                    Column: 0,
                },
                Function: 'int f()',
                Qualifier: 'dynamic',
                displayString: '32 bytes, dynamic',
            },
            {
                BytesUsed: 64,
                DebugLoc: {
                    File: 'example.cpp',
                    Line: 7,
                    Column: 0,
                },
                Function: 'int h()',
                Qualifier: 'dynamic,bounded',
                displayString: '64 bytes, dynamic,bounded',
            },
        ]);
    });
});
