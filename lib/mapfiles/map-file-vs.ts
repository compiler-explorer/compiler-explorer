// Copyright (c) 2018, Compiler Explorer Authors
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

import {MapFileReader} from './map-file';

export class MapFileReaderVS extends MapFileReader {
    regexVsNames = /^\s([\da-f]*):([\da-f]*)\s*([\w$.?@]*)\s*([\da-f]*)(\sf\si\s*|\sf\s*|\s*)([\w.:<>-]*)$/i;
    regexVSLoadAddress = /\spreferred load address is ([\da-f]*)/i;
    regexVsCodeSegment = /^\s([\da-f]*):([\da-f]*)\s*([\da-f]*)h\s*\.text\$mn\s*code.*/i;

    override tryReadingPreferredAddress(line: string) {
        const matches = line.match(this.regexVSLoadAddress);
        if (matches) {
            this.preferredLoadAddress = parseInt(matches[1], 16);
        }
    }

    /**
     * Tries to match the given line to code segment information
     *  Matches in order:
     *   1. segment offset info
     *   2. code segment delphi map
     *   3. icode segment delphi map
     *   4. code segment vs map
     */
    override tryReadingCodeSegmentInfo(line: string) {
        const matches = line.match(this.regexVsCodeSegment);
        if (matches) {
            this.segments.push({
                ...this.addressToObject(matches[1], matches[2]),
                id: this.segments.length + 1,
                segmentLength: parseInt(matches[3], 16),
                unitName: false,
            });
        }
    }

    /**
     * Try to match information about the address where a symbol is
     */
    override tryReadingNamedAddress(line: string) {
        const matches = line.match(this.regexVsNames);
        if (matches && matches.length >= 7 && matches[4] !== '') {
            const addressWithOffset = parseInt(matches[4], 16);
            const symbolObject = {
                segment: matches[1],
                addressWithoutOffset: parseInt(matches[2], 16),
                addressInt: addressWithOffset,
                address: addressWithOffset.toString(16),
                displayName: matches[3],
                unitName: matches[6],
                isStaticSymbol: false,
                segmentLength: 0,
            };
            this.namedAddresses.push(symbolObject);

            this.setSegmentOffset(symbolObject.segment, symbolObject.addressInt - symbolObject.addressWithoutOffset);

            const segment = this.getSegmentInfoAddressWithoutOffsetIsIn(
                symbolObject.segment,
                symbolObject.addressWithoutOffset,
            );
            if (segment && !segment.unitName) {
                segment.unitName = matches[6];
            }
        }
    }
}
