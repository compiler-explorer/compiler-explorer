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
"use strict";

const fs = require("fs"),
    utils = require("./utils");

class MapFileReader {
    /**
     * constructor of MapFileReader
     *  Note that this is a base class and should be overriden. (see for example map-file-vs.js)
     *  Note that this base class retains and uses state,
     *   so when you want to read a new file you need to instantiate a new object.
     * @param {string} mapFilename
     */
    constructor(mapFilename) {
        this.mapFilename = mapFilename;
        this.preferredLoadAddress = 0x400000;
        this.segmentMultiplier = 0x1000;
        this.segmentOffsets = [];
        this.segments = [];
        this.isegments = [];
        this.namedAddresses = [];
        this.entryPoint = "";

        this.lineNumbers = [];
        this.reconstructedSegments = [];

        this.regexEntryPoint = /^\sentry point at\s*([0-9a-f]*):([0-9a-f]*)$/i;
    }

    /**
     * The function to call to load a map file (not async)
     */
    run() {
        if (this.mapFilename) {
            this.loadMap();
        }
    }

    /**
     *
     * @param {(string|boolean)} segment
     * @param {number} address
     * @returns {(Object|boolean)}
     */
    getLineInfoByAddress(segment, address) {
        for (let idx = 0; idx < this.lineNumbers.length; idx++) {
            const lineInfo = this.lineNumbers[idx];
            if (!segment && (lineInfo.addressInt === address)) {
                return lineInfo;
            } else if ((segment === lineInfo.segment) && (lineInfo.addressWithoutOffset === address)) {
                return lineInfo;
            }
        }

        return false;
    }

    /**
     *
     * @param {string} segment
     * @returns {number}
     */
    getSegmentOffset(segment) {
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

    /**
     *
     * @param {string} segment
     * @param {number} address
     */
    setSegmentOffset(segment, address) {
        let info = false;
        let idx = 0;

        for (idx = 0; idx < this.segments.length; idx++) {
            info = this.segments[idx];
            if (info.segment === segment) {
                this.segments[idx].addressInt = address;
                this.segments[idx].address = address.toString(16);
            }
        }

        if (this.segmentOffsets.length > 0) {
            for (idx = 0; idx < this.segmentOffsets.length; idx++) {
                info = this.segmentOffsets[idx];
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
            segmentLength: 0
        });
    }

    /**
     *
     * @param {string} unitName
     * @returns {(Object|boolean)}
     */
    getSegmentInfoByUnitName(unitName) {
        for (let idx = 0; idx < this.segments.length; idx++) {
            const info = this.segments[idx];
            if (info.unitName === unitName) {
                return info;
            }
        }

        return false;
    }

    /**
     *
     * @param {string} unitName
     * @returns {(Object|boolean)}
     */
    getICodeSegmentInfoByUnitName(unitName) {
        for (let idx = 0; idx < this.isegments.length; idx++) {
            const info = this.isegments[idx];
            if (info.unitName === unitName) {
                return info;
            }
        }

        return false;
    }

    /**
     *
     * @param {string} unitName
     * @returns {number}
     */
    getSegmentIdByUnitName(unitName) {
        const info = this.getSegmentInfoByUnitName(unitName);
        if (info) {
            return info.id;
        }

        return 0;
    }

    /**
     * Get Segment info for exact address
     * @param {(string|boolean)} segment
     * @param {number} address
     * @returns {(Object|boolean)}
     */
    getSegmentInfoByStartingAddress(segment, address) {
        for (let idx = 0; idx < this.segments.length; idx++) {
            const info = this.segments[idx];
            if (!segment && (info.addressInt === address)) {
                return info;
            } else if ((info.segment === segment) && (info.addressWithoutOffset === address)) {
                return info;
            }
        }

        return false;
    }

    /**
     * Get Segment info for the segment where the given address is in
     * @param {(string|boolean)} segment
     * @param {number} address
     * @returns {(Object|boolean)}
     */
    getSegmentInfoAddressIsIn(segment, address) {
        for (let idx = 0; idx < this.segments.length; idx++) {
            const info = this.segments[idx];
            if (!segment && (address >= info.addressInt) && (address < (info.addressInt + info.segmentLength))) {
                return info;
            } else if ((segment === info.segment) &&
                (address >= info.addressWithoutOffset) &&
                (address < (info.addressWithoutOffset + info.segmentLength))) {
                return info;
            }
        }

        return false;
    }

    /**
     * Get Segment info for the segment where the given address is in
     * @param {string} segment
     * @param {number} address
     * @returns {(Object|boolean)}
     */
    getSegmentInfoAddressWithoutOffsetIsIn(segment, address) {
        for (let idx = 0; idx < this.segments.length; idx++) {
            const info = this.segments[idx];
            if ((segment === info.segment) &&
                (address >= info.addressWithoutOffset) &&
                (address < (info.addressWithoutOffset + info.segmentLength))) {
                return info;
            }
        }

        return false;
    }

    /**
     *
     * @param {string} segment
     * @param {number} address
     * @returns {(Object|boolean)}
     */
    getSymbolAt(segment, address) {
        for (let idx = 0; idx < this.namedAddresses.length; idx++) {
            const info = this.namedAddresses[idx];
            if (!segment && (info.addressInt === address)) {
                return info;
            } else if ((segment === info.segment) && (info.addressWithoutOffset === address)) {
                return info;
            }
        }

        return false;
    }

    /**
     *
     * @param {string} segment
     * @param {number} address
     * @returns {(Object|boolean)}
     */
    getSymbolBefore(segment, address) {
        let maxNamed = false;

        for (let idx = 0; idx < this.namedAddresses.length; idx++) {
            const info = this.namedAddresses[idx];
            if (!segment && (info.addressInt <= address)) {
                if (!maxNamed || (info.addressInt > maxNamed.addressInt)) {
                    maxNamed = info;
                }
            } else if ((segment === info.segment) && (info.addressWithoutOffset <= address)) {
                if (!maxNamed || (info.addressInt > maxNamed.addressInt)) {
                    maxNamed = info;
                }
            }
        }

        return maxNamed;
    }

    /**
     *
     * @param {string} name
     * @returns {(Object|boolean)}
     */
    getSymbolInfoByName(name) {
        for (let idx = 0; idx < this.namedAddresses.length; idx++) {
            const info = this.namedAddresses[idx];
            if (info.displayName === name) {
                return info;
            }
        }

        return false;
    }

    /**
     *
     * @param {string} segment
     * @param {string} address
     */
    addressToObject(segment, address) {
        const addressWithoutOffset = parseInt(address, 16);
        const addressWithOffset = this.getSegmentOffset(segment) + addressWithoutOffset;

        return {
            segment: segment,
            addressWithoutOffset: addressWithoutOffset,
            addressInt: addressWithOffset,
            address: addressWithOffset.toString(16)
        };
    }

    /**
     * Try to match information about the address where a symbol is
     * @param {string} line
     */
    tryReadingNamedAddress() {
    }

    /**
     * Tries to match the given line to code segment information.
     *  Implementation specific, so this base function is empty
     * @param {string} line
     */
    tryReadingCodeSegmentInfo() {
    }

    /**
     *
     * @param {string} line
     */
    tryReadingEntryPoint(line) {
        const matches = line.match(this.regexEntryPoint);
        if (matches) {
            this.entryPoint = {
                segment: matches[1],
                addressWithoutOffset: matches[2]
            };
        }
    }

    /**
     *
     * @param {string} line
     */
    // eslint-disable-next-line no-unused-vars
    tryReadingPreferredAddress(line) {
    }

    /**
     * Retreives line number references from supplied Map line
     * @param {string} line
     * @returns {boolean}
     */
    // eslint-disable-next-line no-unused-vars
    tryReadingLineNumbers(line) {
        return false;
    }

    /**
     *
     * @param {string} line
     */
    // eslint-disable-next-line no-unused-vars
    isStartOfLineNumbers(line) {
        return false;
    }

    /**
     * Tries to reconstruct segments information from contiguous named addresses
     */
    reconstructSegmentsFromNamedAddresses() {
        let currentUnit = false;
        let addressStart = 0;
        for (let idxSymbol = 0; idxSymbol < this.namedAddresses.length; ++idxSymbol) {
            const symbolObject = this.namedAddresses[idxSymbol];

            if (symbolObject.addressInt < this.preferredLoadAddress) continue;

            if (!currentUnit) {
                addressStart = symbolObject.addressInt;
                currentUnit = symbolObject.unitName;
            } else if ((symbolObject.unitName !== currentUnit)) {
                const segmentLen = symbolObject.addressInt - addressStart;

                this.reconstructedSegments.push({
                    addressInt: addressStart,
                    address: addressStart.toString(16),
                    endAddress: (addressStart + segmentLen).toString(16),
                    segmentLength: segmentLen,
                    unitName: currentUnit
                });

                addressStart = symbolObject.addressInt;
                currentUnit = symbolObject.unitName;
            }

            if (idxSymbol === this.namedAddresses.length - 1) {
                this.reconstructedSegments.push({
                    addressInt: addressStart,
                    address: addressStart.toString(16),
                    segmentLength: -1,
                    unitName: symbolObject.unitName
                });
            }
        }
    }

    /**
     * Returns an array of objects with address range information for a given unit (filename)
     *  [{startAddress: int, startAddressHex: string, endAddress: int, endAddressHex: string}]
     * @param {string} unitName
     */
    getReconstructedUnitAddressSpace(unitName) {
        let addressSpace = [];

        for (let idxSegment = 0; idxSegment < this.reconstructedSegments.length; ++idxSegment) {
            const segment = this.reconstructedSegments[idxSegment];
            if (segment.unitName === unitName) {
                addressSpace.push({
                    startAddress: segment.addressInt,
                    startAddressHex: segment.addressInt.toString(16),
                    endAddress: segment.addressInt + segment.segmentLength,
                    endAddressHex: (segment.addressInt + segment.segmentLength).toString(16)
                });
            }
        }

        return addressSpace;
    }

    /**
     * Returns true if the address (and every address until address+size) is within
     *  one of the given spaces. Spaces should be of format returned by getReconstructedUnitAddressSpace()
     * @param {array of objects} spaces
     * @param {int} address
     * @param {int} size
     */
    isWithinAddressSpace(spaces, address, size) {
        for (let idxSpace = 0; idxSpace < spaces.length; ++idxSpace) {
            const spaceAddress = spaces[idxSpace];

            if (((address >= spaceAddress.startAddress) &&
                (address < spaceAddress.endAddress)) ||
                ((address < spaceAddress.startAddress) &&
                    (address + size >= spaceAddress.startAddress))) {
                return true;
            }
        }

        return false;
    }

    /**
     * Retreives information from Map file contents
     * @param {string} contents
     */
    loadMapFromString(contents) {
        const mapLines = utils.splitLines(contents);

        let readLineNumbersMode = false;

        let lineIdx = 0;
        while (lineIdx < mapLines.length) {
            const line = mapLines[lineIdx];

            if (!readLineNumbersMode) {
                this.tryReadingPreferredAddress(line);
                this.tryReadingEntryPoint(line);
                this.tryReadingCodeSegmentInfo(line);
                this.tryReadingNamedAddress(line);

                if (this.isStartOfLineNumbers(line)) {
                    readLineNumbersMode = true;
                    lineIdx++;
                }
            } else {
                if (!this.tryReadingLineNumbers(line)) {
                    readLineNumbersMode = false;
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

exports.MapFileReader = MapFileReader;
