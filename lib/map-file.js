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

var fs = require("fs");

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
        this.segments = [];
        this.namedAddresses = [];
        this.lineNumbers = [];
        this.entryPoint = "";
        
        this.regexVSLoadAddress = /\sPreferred load address is ([0-9a-f]*)/i;

        this.regexDelphiCodeSegment = /^\s([0-9a-f]*):([0-9a-f]*)\s*([0-9a-f]*)\s*C=CODE\s*S=.text\s*G=.*M=([\w\d]*)\s.*/i;
        this.regexVsCodeSegment = /^\s([0-9a-f]*):([0-9a-f]*)\s*([0-9a-f]*)H\s*\.text\$mn\s*CODE.*/i;

        this.regexDelphiNames = /^\s([0-9a-f]*):([0-9a-f]*)\s*([a-z0-9_@\.]*)$/i;
        this.regexVsNames = /^\s([0-9a-f]*):([0-9a-f]*)\s*([a-z0-9\?\$_@\.]*)\s*([0-9a-f]*)(\sf\si\s*|\sf\s*|\s*)([a-z0-9\-\._<>:]*)$/i;

        this.regexDelphiLineNumbersStart = /Line numbers for (.*)\(.*\) segment \.text/i;
        this.regexDelphiLineNumber = /^([0-9]*)\s([0-9a-f]*):([0-9a-f]*)/i;
    }

    Run() {
        if (this.mapFilename) {
            this.LoadMap();
        }
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
            } else if ((info.segment === segment) && (info.addressInt === address)) {
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
    GetSegmentInfoAddressIsIn(segment, address) {
        for (var idx = 0; idx < this.segments.length; idx++) {
            var info = this.segments[idx];
            if ((segment === info.segment) && (address >= info.addressInt) && (address < (info.addressInt + info.segmentLength))) {
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
     * @param {string} segment 
     * @param {number} address 
     * @returns {(Object|boolean)} 
     */
    GetSegmentInfoForAddressWithoutOffset(segment, address) {
        for (var idx = 0; idx < this.segments.length; idx++) {
            var info = this.segments[idx];
            if ((segment === info.segment) && (address === info.addressWithoutOffset)) {
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
            } else if ((segment === lineInfo.segment) && (lineInfo.addressInt === address)) {
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
            } else if ((segment === info.segment) && (info.addressInt === address)) {
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

    /**
     * Tries to match the given line to code segment information
     * @param {string} line 
     */
    TryReadingCodeSegmentInfo(line) {
        var addressWithoutOffset = false;
        var addressWithOffset = false;
        var segmentStr = false;

        var matches = line.match(this.regexDelphiCodeSegment);
        if (matches) {
            segmentStr = matches[1];
            addressWithoutOffset = parseInt(matches[2], 16);
            addressWithOffset = (parseInt(segmentStr, 16) * this.segmentMultiplier) + addressWithoutOffset + this.preferredLoadAddress;
            
            this.segments.push({
                "id": this.segments.length + 1,
                "segment": segmentStr,
                "addressWithoutOffset": addressWithoutOffset,
                "addressInt": addressWithOffset,
                "address": addressWithOffset.toString(16),
                "segmentLength": parseInt(matches[3], 16),
                "unitName": matches[4]
            });
        }

        matches = line.match(this.regexVsCodeSegment);
        if (matches) {
            addressWithoutOffset = parseInt(matches[2], 16);
            addressWithOffset = addressWithoutOffset + this.preferredLoadAddress;
            
            this.segments.push({
                "id": this.segments.length + 1,
                "segment": matches[1],
                "addressWithoutOffset": addressWithoutOffset,
                "addressInt": addressWithOffset,
                "address": addressWithOffset.toString(16),
                "segmentLength": parseInt(matches[3], 16),
                "unitName": false
            });
        }
    }

    /**
     * Try to match information about the address where a symbol is
     * @param {string} line 
     */
    TryReadingNamedAddress(line) {
        var addressWithoutOffset = false;
        var addressWithOffset = false;
        var segmentStr = false;

        var matches = line.match(this.regexDelphiNames);
        if (matches) {
            addressWithoutOffset = parseInt(matches[2], 16);
            segmentStr = matches[1];
            addressWithOffset = (parseInt(segmentStr, 16) * this.segmentMultiplier) + addressWithoutOffset + this.preferredLoadAddress;

            if (!this.GetSymbolInfoByName(matches[3])) {
                this.namedAddresses.push({
                    "segment": segmentStr,
                    "addressWithoutOffset": addressWithoutOffset,
                    "addressInt": addressWithOffset,
                    "address": addressWithOffset.toString(16),
                    "displayName": matches[3]
                });
            }
        }

        matches = line.match(this.regexVsNames);
        if (matches && (matches.length >= 7) && (matches[4] !== "")) {
            segmentStr = matches[1];
            addressWithoutOffset = parseInt(matches[2], 16);
            addressWithOffset = parseInt(matches[4], 16);

            this.namedAddresses.push({
                "segment": segmentStr,
                "addressWithoutOffset": addressWithoutOffset,
                "addressInt": addressWithOffset,
                "address": addressWithOffset.toString(16),
                "displayName": matches[3]
            });

            var segment = this.GetSegmentInfoAddressWithoutOffsetIsIn(segmentStr, addressWithoutOffset);
            if (segment && !segment.unitName) {
                segment.unitName = matches[6];
            }
        }
    }

    /**
     * 
     * @param {string} line 
     * @returns {boolean}
     */
    TryReadingLineNumbers(line) {
        var hasLineNumbers = false;

        var references = line.split("    ");    // 4 spaces
        for (var refIdx = 0; refIdx < references.length; refIdx++) {
            var matches = references[refIdx].match(this.regexDelphiLineNumber);
            if (matches) {
                var segmentStr = matches[2];
                var addressWithoutOffset = parseInt(matches[3], 16);
                var addressWithOffset = (parseInt(segmentStr, 16) * this.segmentMultiplier) + addressWithoutOffset + this.preferredLoadAddress;
                
                this.lineNumbers.push({
                    "segment": segmentStr,
                    "addressWithoutOffset": addressWithoutOffset,
                    "addressInt": addressWithOffset,
                    "address": addressWithOffset.toString(16),
                    "lineNumber": parseInt(matches[1], 10)
                });

                hasLineNumbers = true;
            }
        }

        return hasLineNumbers;
    }

    LoadMap() {
        var self = this;
        var data = fs.readFileSync(this.mapFilename);
        var mapLines = data.toString().split("\r\n");
        
        var readLineNumbersMode = false;
        var lineNumbersForUnit = "";

        var lineIdx = 0;
        while (lineIdx < mapLines.length) {
            var line = mapLines[lineIdx];

            if (!readLineNumbersMode) {
                self.TryReadingPreferredAddress(line);
                self.TryReadingCodeSegmentInfo(line);
                self.TryReadingNamedAddress(line);

                var matches = line.match(self.regexDelphiLineNumbersStart);
                if (matches) {
                    readLineNumbersMode = true;
                    lineNumbersForUnit = matches[1];
                    lineIdx++;
                }
            } else {
                if (!self.TryReadingLineNumbers(line)) {
                    readLineNumbersMode = false;
                }
            }

            lineIdx++;
        }
    }
}

exports.MapFileReader = MapFileReader;
