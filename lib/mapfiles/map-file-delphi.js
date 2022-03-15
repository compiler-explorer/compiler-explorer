// Copyright (c) 2017, Compiler Explorer Authors
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

import { MapFileReader } from './map-file';

export class MapFileReaderDelphi extends MapFileReader {
    /**
     * constructor
     *
     * @param {string} mapFilename
     */
    constructor(mapFilename) {
        super(mapFilename);

        this.regexDelphiCodeSegmentOffset = /^\s([\da-f]*):([\da-f]*)\s*([\da-f]*)h\s*(\.[$a-z]*)\s*([a-z]*)$/i;
        this.regexDelphiCodeSegment = /^\s([\da-f]*):([\da-f]*)\s*([\da-f]*)\s*c=code\s*s=.text\s*g=.*m=([\w.]*)\s.*/i;
        this.regexDelphiICodeSegment = /^\s([\da-f]*):([\da-f]*)\s*([\da-f]*)\s*c=icode\s*s=.itext\s*g=.*m=([\w.]*)\s.*/i;
        this.regexDelphiNames = /^\s([\da-f]*):([\da-f]*)\s*([\w$.<>@{}]*)$/i;
        this.regexDelphiLineNumbersStart = /line numbers for (.*)\(.*\) segment \.text/i;
        this.regexDelphiLineNumber = /^(\d*)\s([\da-f]*):([\da-f]*)/i;
        this.regexDelphiLineNumbersStartIText = /line numbers for (.*)\(.*\) segment \.itext/i;
    }

    /**
     * Tries to match the given line to code segment information
     *  Matches in order:
     *   1. segment offset info
     *   2. code segment delphi map
     *   3. icode segment delphi map
     *   4. code segment vs map
     *
     * @param {string} line
     */
    tryReadingCodeSegmentInfo(line) {
        let codesegmentObject = false;

        let matches = line.match(this.regexDelphiCodeSegmentOffset);
        if (matches && !matches[4].includes('$') && (parseInt(matches[2], 16) >= this.preferredLoadAddress)) {
            const addressWithOffset = parseInt(matches[2], 16);
            this.segmentOffsets.push({
                segment: matches[1],
                addressInt: addressWithOffset,
                address: addressWithOffset.toString(16),
                segmentLength: parseInt(matches[3], 16),
            });
        } else {
            matches = line.match(this.regexDelphiCodeSegment);
            if (matches) {
                codesegmentObject = this.addressToObject(matches[1], matches[2]);
                codesegmentObject.id = this.segments.length + 1;
                codesegmentObject.segmentLength = parseInt(matches[3], 16);
                codesegmentObject.unitName = matches[4];

                if (codesegmentObject.unitName === 'prog') {
                    codesegmentObject.unitName = 'prog.dpr';
                } else {
                    codesegmentObject.unitName = codesegmentObject.unitName + '.pas';
                }

                this.segments.push(codesegmentObject);
            } else {
                matches = line.match(this.regexDelphiICodeSegment);
                if (matches) {
                    codesegmentObject = this.addressToObject(matches[1], matches[2]);
                    codesegmentObject.id = this.isegments.length + 1;
                    codesegmentObject.segmentLength = parseInt(matches[3], 16);
                    codesegmentObject.unitName = matches[4];

                    if (codesegmentObject.unitName === 'prog') {
                        codesegmentObject.unitName = 'prog.dpr';
                    } else {
                        codesegmentObject.unitName = codesegmentObject.unitName + '.pas';
                    }
    
                    this.isegments.push(codesegmentObject);
                }
            }
        }
    }

    /**
     * Try to match information about the address where a symbol is
     *
     * @param {string} line
     */
    tryReadingNamedAddress(line) {
        let symbolObject = false;

        const matches = line.match(this.regexDelphiNames);
        if (matches) {
            if (!this.getSymbolInfoByName(matches[3])) {
                symbolObject = this.addressToObject(matches[1], matches[2]);
                symbolObject.displayName = matches[3];

                this.namedAddresses.push(symbolObject);
            }
        }
    }

    /**
     *
     * @param {string} line
     */
    isStartOfLineNumbers(line) {
        const matches = line.match(this.regexDelphiLineNumbersStart);
        return !!matches;
    }

    /**
     * Retreives line number references from supplied Map line
     *
     * @param {string} line
     * @returns {boolean}
     */
    tryReadingLineNumbers(line) {
        let hasLineNumbers = false;

        const references = line.split('    ');    // 4 spaces
        for (const reference of references) {
            const matches = reference.match(this.regexDelphiLineNumber);
            if (matches) {
                const lineObject = this.addressToObject(matches[2], matches[3]);
                lineObject.lineNumber = parseInt(matches[1], 10);

                this.lineNumbers.push(lineObject);

                hasLineNumbers = true;
            }
        }

        return hasLineNumbers;
    }
}
