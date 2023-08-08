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

import * as stream from 'stream';

import {LLVMOptTransformer} from '../lib/llvm-opt-transformer.js';

describe('LLVM opt transformer', () => {
    it('should work', async () => {
        const doc = `--- !Analysis
Pass:            prologepilog
Name:            StackSize
DebugLoc:        { File: example.cpp, Line: 2, Column: 0 }
Function:        _Z6squarei
Args:
  - NumStackBytes:   '8'
  - String:          ' stack bytes in function'
...
--- !Analysis
Pass:            asm-printer
Name:            InstructionCount
DebugLoc:        { File: example.cpp, Line: 2, Column: 0 }
Function:        _Z6squarei
Args:
  - NumInstructions: '7'
  - String:          ' instructions in function'
...
`;
        const readString = new stream.PassThrough();
        readString.push(doc);
        readString.end();
        const optStream = readString.pipe(new LLVMOptTransformer());

        const output: object[] = [];
        for await (const opt of optStream) {
            output.push(opt);
        }
        output.should.deep.equal([
            {
                Args: [
                    {
                        NumStackBytes: '8',
                    },
                    {
                        String: ' stack bytes in function',
                    },
                ],
                DebugLoc: {
                    Column: 0,
                    File: 'example.cpp',
                    Line: 2,
                },
                Function: '_Z6squarei',
                Name: 'StackSize',
                Pass: 'prologepilog',
                displayString: '8 stack bytes in function',
                optType: 'Analysis',
            },
            {
                Args: [
                    {
                        NumInstructions: '7',
                    },
                    {
                        String: ' instructions in function',
                    },
                ],
                DebugLoc: {
                    Column: 0,
                    File: 'example.cpp',
                    Line: 2,
                },
                Function: '_Z6squarei',
                Name: 'InstructionCount',
                Pass: 'asm-printer',
                displayString: '7 instructions in function',
                optType: 'Analysis',
            },
        ]);
    });
});
