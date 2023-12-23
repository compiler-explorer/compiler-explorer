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

import {LineNumber, MapFileReader} from './map-file.js';

export class MapFileReaderMinGW extends MapFileReader {
    regexLineWithAddressAndSymbol = /^\s+(0x[0-9a-f]*)\s+([\w().]*)$/i;
    regexObjectOffset = /^\s(.[\w]*)\s+(0x[0-9a-f]*)\s+(0x[0-9a-f]*)\s(.*)$/i;
    regexStartOfSegment = /^(.[\w]*)\s+(0x[0-9a-f]*)\s+(0x[0-9a-f]*)$/i;

    private currentSegment = '';

    constructor(mapFilename: string) {
        super(mapFilename);

        this.regexEntryPoint = /^\s+(0x[0-9a-f]*)\s+__image_base__\s=\s(0x[0-9a-f]*)$/i;
    }

    override tryReadingEntryPoint(line: string) {
        const matches = line.match(this.regexEntryPoint);
        if (matches) {
            this.entryPoint = {
                segment: '.text',
                addressWithoutOffset: matches[1],
                addressWithoutOffsetInt: parseInt(matches[1]),
            };
        }
    }

    override addressToObject(segment: string, address: string): LineNumber {
        // mingw map works with full addresses everywhere, so we'll have to do this differently than usual
        const fullAddress = parseInt(address, 16);
        const segOffset = this.getSegmentOffset(segment);
        const addressWithoutOffset = fullAddress - segOffset;
        const addressWithOffset = fullAddress;

        return {
            segment: segment,
            addressWithoutOffset: addressWithoutOffset,
            addressInt: addressWithOffset,
            address: addressWithOffset.toString(16),
        };
    }

    override tryReadingCodeSegmentInfo(line: string) {
        const matchStart = line.match(this.regexStartOfSegment);
        if (matchStart && matchStart[1].trim() !== '') {
            const address = parseInt(matchStart[2], 16);
            this.segmentOffsets.push({
                segment: matchStart[1],
                addressInt: address,
                address: address.toString(16),
                segmentLength: parseInt(matchStart[3], 16),
            });
        } else {
            const match = line.match(this.regexObjectOffset);
            if (match && match[1].trim() !== '') {
                this.currentSegment = match[1];

                this.segments.push({
                    ...this.addressToObject(match[1], match[2]),
                    id: this.segments.length + 1,
                    segmentLength: parseInt(match[3], 16),
                    unitName: match[4],
                });
            }
        }
    }

    override tryReadingNamedAddress(line: string) {
        const matches = line.match(this.regexLineWithAddressAndSymbol);
        if (matches) {
            if (!this.getSymbolInfoByName(matches[2])) {
                this.namedAddresses.push({
                    ...this.addressToObject(this.currentSegment, matches[1]),
                    displayName: matches[2],
                    segmentLength: 0,
                });
            }
        }
    }
}
