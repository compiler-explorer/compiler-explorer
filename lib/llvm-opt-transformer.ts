// Copyright (c) 2017 Jared Wyles, fixes by Aviv Polak and Ofek Shilon
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

import {Transform, TransformCallback} from 'stream';

import * as R from 'ramda';
import * as YAML from 'yamljs';

import {logger} from './logger';

type Path = string;
type OptType = 'Missed' | 'Passed' | 'Analysis';

interface OptInfo {
    optType: OptType;
    displayString: string;
}

interface LLVMOptInfo extends OptInfo {
    Pass: string;
    Name: string;
    DebugLoc: DebugLoc;
    Function: string;
    Args: Array<object>;
}

interface DebugLoc {
    File: Path;
    Line: number;
    Column: number;
}

function DisplayOptInfo(optInfo: LLVMOptInfo) {
    return optInfo.Args.reduce((acc, x) => {
        let inc = '';
        for (const [key, value] of Object.entries(x)) {
            if (key === 'DebugLoc') {
                if (value['Line'] !== 0) {
                    inc += ' (' + value['Line'] + ':' + value['Column'] + ')';
                }
            } else {
                inc += value;
            }
        }
        return acc + inc;
    }, '');
}

const optTypeMatcher = /---\s(.*)\r?\n/;
const docStart = '---';
const docEnd = '\n...';
const IsDocumentStart = (x: string) => x.substring(0, 3) === docStart;
const FindDocumentEnd = (x: string) => {
    const index = x.indexOf(docEnd);
    return {found: index > -1, endpos: index + docEnd.length};
};

export class LLVMOptTransformer extends Transform {
    _buffer: string;
    _prevOpts: Set<string>; // Avoid duplicate display of remarks
    constructor(options?: object) {
        super(R.merge(options || {}, {objectMode: true}));
        this._buffer = '';
        this._prevOpts = new Set<string>();
    }
    override _flush(done: TransformCallback) {
        this.processBuffer();
        done();
    }
    override _transform(chunk: any, encoding: string, done: TransformCallback) {
        this._buffer += chunk.toString();
        //buffer until we have a start and and end
        //if at any time i care about improving performance stash the offset
        this.processBuffer();
        done();
    }
    processBuffer() {
        while (IsDocumentStart(this._buffer)) {
            const {found, endpos} = FindDocumentEnd(this._buffer);
            if (found) {
                const [head, tail] = R.splitAt(endpos, this._buffer);
                const optTypeMatch = head.match(optTypeMatcher);
                const opt = YAML.parse(head);
                const strOpt = JSON.stringify(opt);
                if (!this._prevOpts.has(strOpt)) {
                    this._prevOpts.add(strOpt);

                    if (optTypeMatch) {
                        opt.optType = optTypeMatch[1].replace('!', '');
                    } else {
                        logger.warn('missing optimization type');
                    }
                    opt.displayString = DisplayOptInfo(opt);
                    this.push(opt as LLVMOptInfo);
                }
                this._buffer = tail.replace(/^\n/, '');
            } else {
                break;
            }
        }
    }
}
