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

import fs from 'fs';

import * as utils from '../utils';

type SegmentOffset = {
    segment: string;
    addressInt: number;
    address: string;
    segmentLength: number;
};

type Segment = {
    segment: string;
    addressInt: number;
    address: string;
    addressWithoutOffset: number;
    segmentLength: number;
    unitName?: string | false;
    id?: number;
    displayName?: string;
};

type ReconstructedSegment = {
    addressInt: number;
    address: string;
    endAddress?: string;
    segmentLength: number;
    unitName?: string | false;
};

type EntryPoint = {
    segment: string;
    addressWithoutOffset: string;
};

type LineNumber = {
    addressInt: number;
    address: string;
    segment: string;
    addressWithoutOffset: number;
    lineNumber?: number;
};

type AddressRangeInformation = {
    startAddress: number;
    startAddressHex: string;
    endAddress: number;
    endAddressHex: string;
};

export class MapFileReader {
    preferredLoadAddress = 0x400000;
    segmentMultiplier = 0x1000;
    segmentOffsets: SegmentOffset[] = [];
    segments: Segment[] = [];
    isegments: Segment[] = [];
    namedAddresses: Segment[] = [];
    entryPoint: '' | EntryPoint = '';

    lineNumbers: LineNumber[] = [];
    reconstructedSegments: ReconstructedSegment[] = [];

    regexEntryPoint = /^\sentry point at\s*([\da-f]*):([\da-f]*)$/i;

    /**
     * constructor of MapFileReader
     *  Note that this is a base class and should be overriden. (see for example map-file-vs.js)
     *  Note that this base class retains and uses state,
     *   so when you want to read a new file you need to instantiate a new object.
     */
    constructor(protected readonly mapFilename: string) {}

    /**
     * The function to call to load a map file (not async)
     */
    run() {
        if (this.mapFilename) {
            this.loadMap();
        }
    }

    getLineInfoByAddress(segment: string | boolean, address: number) {
        for (let idx = 0; idx < this.lineNumbers.length; idx++) {
            const lineInfo = this.lineNumbers[idx];
            if (!segment && lineInfo.addressInt === address) {
                return lineInfo;
            } else if (segment === lineInfo.segment && lineInfo.addressWithoutOffset === address) {
                return lineInfo;
            }
        }

        return false;
    }

    getSegmentOffset(segment: string): number {
        if (this.segmentOffsets.length > 0) {
            for (let idx = 0; idx < this.segmentOffsets.length; idx++) {
                const info = this.segmentOffsets[idx];
                if (info.segment === segment) {
                    return info.addressInt;
                }
            }
        }

        return this.preferredLoadAddress + parseInt(segment, 16) * this.segmentMultiplier;
    }

    setSegmentOffset(segment: string, address: number) {
        for (let idx = 0; idx < this.segments.length; idx++) {
            const info = this.segments[idx];
            if (info.segment === segment) {
                this.segments[idx].addressInt = address;
                this.segments[idx].address = address.toString(16);
            }
        }

        if (this.segmentOffsets.length > 0) {
            for (let idx = 0; idx < this.segmentOffsets.length; idx++) {
                const info = this.segmentOffsets[idx];
                if (info.segment === segment) {
                    this.segmentOffsets[idx].addressInt = address;
                    this.segmentOffsets[idx].address = address.toString(16);
                    return;
                }
            }
        }

        this.segmentOffsets.push({
            segment: segment,
            addressInt: address,
            address: address.toString(16),
            segmentLength: 0,
        });
    }

    getSegmentInfoByUnitName(unitName: string) {
        for (let idx = 0; idx < this.segments.length; idx++) {
            const info = this.segments[idx];
            if (info.unitName === unitName) {
                return info;
            }
        }

        return false;
    }

    getICodeSegmentInfoByUnitName(unitName: string) {
        for (let idx = 0; idx < this.isegments.length; idx++) {
            const info = this.isegments[idx];
            if (info.unitName === unitName) {
                return info;
            }
        }

        return false;
    }

    getSegmentIdByUnitName(unitName: string) {
        const info = this.getSegmentInfoByUnitName(unitName);
        if (info) {
            return info.id;
        }

        return 0;
    }

    /**
     * Get Segment info for exact address
     */
    getSegmentInfoByStartingAddress(segment: string | boolean, address: number) {
        for (let idx = 0; idx < this.segments.length; idx++) {
            const info = this.segments[idx];
            if (!segment && info.addressInt === address) {
                return info;
            } else if (info.segment === segment && info.addressWithoutOffset === address) {
                return info;
            }
        }

        return false;
    }

    /**
     * Get Segment info for the segment where the given address is in
     */
    getSegmentInfoAddressIsIn(segment: string | boolean, address: number) {
        for (let idx = 0; idx < this.segments.length; idx++) {
            const info = this.segments[idx];
            if (!segment && address >= info.addressInt && address < info.addressInt + info.segmentLength) {
                return info;
            } else if (
                segment === info.segment &&
                address >= info.addressWithoutOffset &&
                address < info.addressWithoutOffset + info.segmentLength
            ) {
                return info;
            }
        }

        return false;
    }

    /**
     * Get Segment info for the segment where the given address is in
     */
    getSegmentInfoAddressWithoutOffsetIsIn(segment: string, address: number) {
        for (let idx = 0; idx < this.segments.length; idx++) {
            const info = this.segments[idx];
            if (
                segment === info.segment &&
                address >= info.addressWithoutOffset &&
                address < info.addressWithoutOffset + info.segmentLength
            ) {
                return info;
            }
        }

        return false;
    }

    getSymbolAt(segment: string, address: number) {
        for (let idx = 0; idx < this.namedAddresses.length; idx++) {
            const info = this.namedAddresses[idx];
            if (!segment && info.addressInt === address) {
                return info;
            } else if (segment === info.segment && info.addressWithoutOffset === address) {
                return info;
            }
        }

        return false;
    }

    getSymbolBefore(segment: string, address: number) {
        let maxNamed: false | Segment = false;

        for (let idx = 0; idx < this.namedAddresses.length; idx++) {
            const info = this.namedAddresses[idx];
            if (!segment && info.addressInt <= address) {
                if (!maxNamed || info.addressInt > maxNamed.addressInt) {
                    maxNamed = info;
                }
            } else if (segment === info.segment && info.addressWithoutOffset <= address) {
                if (!maxNamed || info.addressInt > maxNamed.addressInt) {
                    maxNamed = info;
                }
            }
        }

        return maxNamed;
    }

    getSymbolInfoByName(name: string) {
        for (let idx = 0; idx < this.namedAddresses.length; idx++) {
            const info = this.namedAddresses[idx];
            if (info.displayName === name) {
                return info;
            }
        }

        return false;
    }

    addressToObject(segment: string, address: string): LineNumber {
        const addressWithoutOffset = parseInt(address, 16);
        const addressWithOffset = this.getSegmentOffset(segment) + addressWithoutOffset;

        return {
            segment: segment,
            addressWithoutOffset: addressWithoutOffset,
            addressInt: addressWithOffset,
            address: addressWithOffset.toString(16),
        };
    }

    /**
     * Try to match information about the address where a symbol is
     */
    // eslint-disable-next-line no-unused-vars
    tryReadingNamedAddress(line: string) {}

    /**
     * Tries to match the given line to code segment information.
     *  Implementation specific, so this base function is empty
     */
    // eslint-disable-next-line no-unused-vars
    tryReadingCodeSegmentInfo(line: string) {}

    tryReadingEntryPoint(line: string) {
        const matches = line.match(this.regexEntryPoint);
        if (matches) {
            this.entryPoint = {
                segment: matches[1],
                addressWithoutOffset: matches[2],
            };
        }
    }

    // eslint-disable-next-line no-unused-vars
    tryReadingPreferredAddress(line: string) {}

    // eslint-disable-next-line no-unused-vars
    tryReadingLineNumbers(line: string): boolean {
        return false;
    }

    // eslint-disable-next-line no-unused-vars
    isStartOfLineNumbers(line: string) {
        return false;
    }

    /**
     * Tries to reconstruct segments information from contiguous named addresses
     */
    reconstructSegmentsFromNamedAddresses() {
        let currentUnit: false | string | undefined = false;
        let addressStart = 0;
        for (let idxSymbol = 0; idxSymbol < this.namedAddresses.length; ++idxSymbol) {
            const symbolObject = this.namedAddresses[idxSymbol];

            if (symbolObject.addressInt < this.preferredLoadAddress) continue;

            if (!currentUnit) {
                addressStart = symbolObject.addressInt;
                currentUnit = symbolObject.unitName;
            } else if (symbolObject.unitName !== currentUnit) {
                const segmentLen = symbolObject.addressInt - addressStart;

                this.reconstructedSegments.push({
                    addressInt: addressStart,
                    address: addressStart.toString(16),
                    endAddress: (addressStart + segmentLen).toString(16),
                    segmentLength: segmentLen,
                    unitName: currentUnit,
                });

                addressStart = symbolObject.addressInt;
                currentUnit = symbolObject.unitName;
            }

            if (idxSymbol === this.namedAddresses.length - 1) {
                this.reconstructedSegments.push({
                    addressInt: addressStart,
                    address: addressStart.toString(16),
                    segmentLength: -1,
                    unitName: symbolObject.unitName,
                });
            }
        }
    }

    /**
     * Returns an array of objects with address range information for a given unit (filename)
     */
    getReconstructedUnitAddressSpace(unitName: string): AddressRangeInformation[] {
        const addressSpace: AddressRangeInformation[] = [];

        for (let idxSegment = 0; idxSegment < this.reconstructedSegments.length; ++idxSegment) {
            const segment = this.reconstructedSegments[idxSegment];
            if (segment.unitName === unitName) {
                addressSpace.push({
                    startAddress: segment.addressInt,
                    startAddressHex: segment.addressInt.toString(16),
                    endAddress: segment.addressInt + segment.segmentLength,
                    endAddressHex: (segment.addressInt + segment.segmentLength).toString(16),
                });
            }
        }

        return addressSpace;
    }

    /**
     * Returns true if the address (and every address until address+size) is within
     *  one of the given spaces. Spaces should be of format returned by getReconstructedUnitAddressSpace()
     */
    isWithinAddressSpace(spaces: AddressRangeInformation[], address: number, size: number) {
        for (const spaceAddress of spaces) {
            if (
                (address >= spaceAddress.startAddress && address < spaceAddress.endAddress) ||
                (address < spaceAddress.startAddress && address + size >= spaceAddress.startAddress)
            ) {
                return true;
            }
        }

        return false;
    }

    /**
     * Retreives information from Map file contents
     */
    loadMapFromString(contents: string) {
        const mapLines = utils.splitLines(contents);

        let readLineNumbersMode = false;

        let lineIdx = 0;
        while (lineIdx < mapLines.length) {
            const line = mapLines[lineIdx];

            if (readLineNumbersMode) {
                if (!this.tryReadingLineNumbers(line)) {
                    readLineNumbersMode = false;
                }
            } else {
                this.tryReadingPreferredAddress(line);
                this.tryReadingEntryPoint(line);
                this.tryReadingCodeSegmentInfo(line);
                this.tryReadingNamedAddress(line);

                if (this.isStartOfLineNumbers(line)) {
                    readLineNumbersMode = true;
                    lineIdx++;
                }
            }

            lineIdx++;
        }

        this.reconstructSegmentsFromNamedAddresses();
    }

    /**
     * Reads the actual mapfile from disk synchronously and load it into this class
     */
    loadMap() {
        const data = fs.readFileSync(this.mapFilename);

        this.loadMapFromString(data.toString());
    }
}
