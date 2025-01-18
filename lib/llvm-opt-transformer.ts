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

import {parseAllDocuments} from 'yaml';

import {OptRemark} from '../static/panes/opt-view.interfaces.js';

import {logger} from './logger.js';

function DisplayOptInfo(optInfo: OptRemark) {
    let displayString = optInfo.Args.reduce((acc, x) => {
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
    displayString = displayString.replaceAll('\n', ' ').replaceAll('\r', ' ');
    return displayString;
}

export function processRawOptRemarks(buffer: string, compileFileName: string = ''): OptRemark[] {
    const output: OptRemark[] = [];
    const remarksSet: Set<string> = new Set<string>();
    const remarks: any = parseAllDocuments(buffer);
    for (const doc of remarks) {
        if (doc.errors !== undefined && doc.errors.length > 0) {
            logger.warn('YAMLParseError: ' + JSON.stringify(doc.errors[0]));
            continue;
        }

        const opt = doc.toJS();
        if (!opt.DebugLoc || !opt.DebugLoc.File || !opt.DebugLoc.File.includes(compileFileName)) continue;

        const strOpt = JSON.stringify(opt);
        if (!remarksSet.has(strOpt)) {
            remarksSet.add(strOpt);
            opt.optType = doc.contents.tag.substring(1); // remove leading '!'
            opt.displayString = DisplayOptInfo(opt);
            output.push(opt as OptRemark);
        }
    }

    return output;
}
