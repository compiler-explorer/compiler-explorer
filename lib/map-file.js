// Copyright (c) 2017, Patrick Quist
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

var fs = require("fs"),
    utils = require("./utils"),
    logger = require("./logger").logger;

class MapFileReader {
    /**
     * constructor
     * @param {string} mapFilename 
     * @param {function} callback 
     */
    constructor(mapFilename) {
        this.mapFilename = mapFilename;
        this.preferredLoadAddress = 0x400000;
        this.segmentMultiplier = 0x1000;
        this.segmentOffsets = [];
        this.segments = [];
        this.isegments = [];
        this.namedAddresses = [];
        this.lineNumbers = [];
        this.entryPoint = "";
        
        this.regexVSLoadAddress = /\sPreferred load address is ([0-9a-f]*)/i;

        this.regexDelphiCodeSegmentOffset = /^\s([0-9a-f]*):([0-9a-f]*)\s*([0-9a-f]*)H\s*(\.[a-z\$]*)\s*([A-Z]*)$/i;

        this.regexDelphiCodeSegment = /^\s([0-9a-f]*):([0-9a-f]*)\s*([0-9a-f]*)\s*C=CODE\s*S=.text\s*G=.*M=([\w\d]*)\s.*/i;
        this.regexVsCodeSegment = /^\s([0-9a-f]*):([0-9a-f]*)\s*([0-9a-f]*)H\s*\.text\$mn\s*CODE.*/i;

        this.regexDelphiICodeSegment = /^\s([0-9a-f]*):([0-9a-f]*)\s*([0-9a-f]*)\s*C=ICODE\s*S=.itext\s*G=.*M=([\w\d]*)\s.*/i;

        this.regexDelphiNames = /^\s([0-9a-f]*):([0-9a-f]*)\s*([a-z0-9_@\.]*)$/i;
        this.regexVsNames = /^\s([0-9a-f]*):([0-9a-f]*)\s*([a-z0-9\?\$_@\.]*)\s*([0-9a-f]*)(\sf\si\s*|\sf\s*|\s*)([a-z0-9\-\._<>:]*)$/i;

        this.regexDelphiLineNumbersStart = /Line numbers for (.*)\(.*\) segment \.text/i;
        this.regexDelphiLineNumber = /^([0-9]*)\s([0-9a-f]*):([0-9a-f]*)/i;

        this.regexDelphiLineNumbersStartIText = /Line numbers for (.*)\(.*\) segment \.itext/i;
    }

    Run() {
        if (this.mapFilename) {
            this.LoadMap();
        }
    }

    /**
     * 
     * @param {string} segment 
     * @returns {number} 
     */
    GetSegmentOffset(segment) {
        if (this.segmentOffsets.length > 0) {
            for (var idx = 0; idx < this.segmentOffsets.length; idx++) {
                var info = this.segmentOffsets[idx];
                if (info.segment === segment) {
                    return info.addressInt;
                }
            }
        }

        // default
        return this.preferredLoadAddress + parseInt(segment, 16) * this.segmentMultiplier;
    }

    /**
     * 
     * @param {string} segment 
     * @param {number} address 
     */
    SetSegmentOffset(segment, address) {
        var info = false;
        var idx = 0;

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

        // else, add it
        this.segmentOffsets.push({
            "segment": segment,
            "addressInt": address,
            "address": address.toString(16),
            "segmentLength": 0
        });
    }

    /**
     * 
     * @param {string} unitName 
     * @returns {(Object|boolean)} 
     */
    GetSegmentInfoByUnitName(unitName) {
        for (var idx = 0; idx < this.segments.length; idx++) {
            var info = this.segments[idx];
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
    GetICodeSegmentInfoByUnitName(unitName) {
        for (var idx = 0; idx < this.isegments.length; idx++) {
            var info = this.isegments[idx];
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
    GetSegmentIdByUnitName(unitName) {
        var info = this.GetSegmentInfoByUnitName(unitName);
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
    GetSegmentInfoByStartingAddress(segment, address) {
        for (var idx = 0; idx < this.segments.length; idx++) {
            var info = this.segments[idx];
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
    GetSegmentInfoAddressIsIn(segment, address) {
        for (var idx = 0; idx < this.segments.length; idx++) {
            var info = this.segments[idx];
            if (!segment && (address >= info.addressInt) && (address < (info.addressInt + info.segmentLength))) {
                return info;
            } else if ((segment === info.segment) && (address >= info.addressWithoutOffset) && (address < (info.addressWithoutOffset + info.segmentLength))) {
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
    GetSegmentInfoAddressWithoutOffsetIsIn(segment, address) {
        for (var idx = 0; idx < this.segments.length; idx++) {
            var info = this.segments[idx];
            if ((segment === info.segment) && (address >= info.addressWithoutOffset) && (address < (info.addressWithoutOffset + info.segmentLength))) {
                return info;
            }
        }

        return false;
    }

    /**
     * 
     * @param {(string|boolean)} segment 
     * @param {number} address 
     * @returns {(Object|boolean)} 
     */
    GetLineInfoByAddress(segment, address) {
        for (var idx = 0; idx < this.lineNumbers.length; idx++) {
            var lineInfo = this.lineNumbers[idx];
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
     * @param {number} address 
     * @returns {(Object|boolean)} 
     */
    GetSymbolAt(segment, address) {
        for (var idx = 0; idx < this.namedAddresses.length; idx++) {
            var info = this.namedAddresses[idx];
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
     * @param {string} name 
     * @returns {(Object|boolean)} 
     */
    GetSymbolInfoByName(name) {
        for (var idx = 0; idx < this.namedAddresses.length; idx++) {
            var info = this.namedAddresses[idx];
            if (info.displayName === name) {
                return info;
            }
        }

        return false;
    }

    /**
     * 
     * @param {string} line 
     */
    TryReadingPreferredAddress(line) {
        var matches = line.match(this.regexVSLoadAddress);
        if (matches) {
            this.preferredLoadAddress = parseInt(matches[1], 16);
        }
    }

    AddressToObject(segment, address) {
        var addressWithoutOffset = parseInt(address, 16);
        var addressWithOffset = this.GetSegmentOffset(segment) + addressWithoutOffset;

        return {
            "segment": segment,
            "addressWithoutOffset": addressWithoutOffset,
            "addressInt": addressWithOffset,
            "address": addressWithOffset.toString(16)
        };
    }

    /**
     * Tries to match the given line to code segment information
     *  Matches in order:
     *   1. segment offset info
     *   2. code segment delphi map
     *   3. icode segment delphi map
     *   4. code segment vs map
     * @param {string} line 
     */
    TryReadingCodeSegmentInfo(line) {
        var codesegmentObject = false;

        var matches = line.match(this.regexDelphiCodeSegmentOffset);
        if (matches && !matches[4].includes('$') && (parseInt(matches[2], 16) >= this.preferredLoadAddress)) {
            var addressWithOffset = parseInt(matches[2], 16);
            this.segmentOffsets.push({
                "segment": matches[1],
                "addressInt": addressWithOffset,
                "address": addressWithOffset.toString(16),
                "segmentLength": parseInt(matches[3], 16)
            });
        } else {
            matches = line.match(this.regexDelphiCodeSegment);
            if (matches) {
                codesegmentObject = this.AddressToObject(matches[1], matches[2]);
                codesegmentObject.id = this.segments.length + 1;
                codesegmentObject.segmentLength = parseInt(matches[3], 16);
                codesegmentObject.unitName = matches[4];

                this.segments.push(codesegmentObject);
            } else {
                matches = line.match(this.regexDelphiICodeSegment);
                if (matches) {
                    codesegmentObject = this.AddressToObject(matches[1], matches[2]);
                    codesegmentObject.id = this.isegments.length + 1;
                    codesegmentObject.segmentLength = parseInt(matches[3], 16);
                    codesegmentObject.unitName = matches[4];

                    this.isegments.push(codesegmentObject);
                } else {
                    matches = line.match(this.regexVsCodeSegment);
                    if (matches) {
                        codesegmentObject = this.AddressToObject(matches[1], matches[2]);
                        codesegmentObject.id = this.segments.length + 1;
                        codesegmentObject.segmentLength = parseInt(matches[3], 16);
                        codesegmentObject.unitName = false;

                        this.segments.push(codesegmentObject);
                    }
                }
            }
        }
    }

    /**
     * Try to match information about the address where a symbol is
     * @param {string} line 
     */
    TryReadingNamedAddress(line) {
        var symbolObject = false;

        var matches = line.match(this.regexDelphiNames);
        if (matches) {
            if (!this.GetSymbolInfoByName(matches[3])) {
                symbolObject = this.AddressToObject(matches[1], matches[2]);
                symbolObject.displayName = matches[3];

                this.namedAddresses.push(symbolObject);
            }
        }

        matches = line.match(this.regexVsNames);
        if (matches && (matches.length >= 7) && (matches[4] !== "")) {
            var addressWithOffset = parseInt(matches[4], 16);
            symbolObject = {
                "segment": matches[1],
                "addressWithoutOffset": parseInt(matches[2], 16),
                "addressInt": addressWithOffset,
                "address": addressWithOffset.toString(16),
                "displayName": matches[3]
            };
            this.namedAddresses.push(symbolObject);

            this.SetSegmentOffset(symbolObject.segment, symbolObject.addressInt - symbolObject.addressWithoutOffset);

            var segment = this.GetSegmentInfoAddressWithoutOffsetIsIn(symbolObject.segment, symbolObject.addressWithoutOffset);
            if (segment && !segment.unitName) {
                segment.unitName = matches[6];
            }
        }
    }

    /**
     * Retreives line number references from supplied Map line
     * @param {string} line 
     * @returns {boolean}
     */
    TryReadingLineNumbers(line) {
        var hasLineNumbers = false;

        var references = line.split("    ");    // 4 spaces
        for (var refIdx = 0; refIdx < references.length; refIdx++) {
            var matches = references[refIdx].match(this.regexDelphiLineNumber);
            if (matches) {
                var lineObject = this.AddressToObject(matches[2], matches[3]);
                lineObject.lineNumber = parseInt(matches[1], 10);

                this.lineNumbers.push(lineObject);

                hasLineNumbers = true;
            }
        }

        return hasLineNumbers;
    }

    /**
     * Retreives information from Map file contents
     * @param {string} contents 
     */
    LoadMapFromString(contents) {
        var mapLines = utils.splitLines(contents);
        
        var readLineNumbersMode = false;
        var lineNumbersForUnit = "";

        var lineIdx = 0;
        while (lineIdx < mapLines.length) {
            var line = mapLines[lineIdx];

            if (!readLineNumbersMode) {
                this.TryReadingPreferredAddress(line);
                this.TryReadingCodeSegmentInfo(line);
                this.TryReadingNamedAddress(line);

                var matches = line.match(this.regexDelphiLineNumbersStart);
                if (matches) {
                    readLineNumbersMode = true;
                    lineNumbersForUnit = matches[1];
                    lineIdx++;
                }
            } else {
                if (!this.TryReadingLineNumbers(line)) {
                    readLineNumbersMode = false;
                }
            }

            lineIdx++;
        }
    }

    LoadMap() {
        var data = fs.readFileSync(this.mapFilename);

        this.LoadMapFromString(data.toString());
    }
}

exports.MapFileReader = MapFileReader;
