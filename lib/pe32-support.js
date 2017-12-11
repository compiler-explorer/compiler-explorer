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

var MapFileReader = require('./map-file').MapFileReader,
    logger = require('./logger').logger;

class PELabelReconstructor {
    /**
     * 
     * @param {Array} asmLines 
     * @param {string} mapFilename 
     * @param {boolean} dontLabelUnmappedAddresses 
     * @param {function} callback 
     */
    constructor(asmLines, mapFilename, dontLabelUnmappedAddresses) {
        this.asmLines = asmLines;
        this.addressesToLabel = [];
        this.dontLabelUnmappedAddresses = dontLabelUnmappedAddresses;

        this.addressRegex = /^\s*([0-9a-z]*):/i;
        this.mapFileReader = null;
        this.mapFilename = mapFilename;
    }

    Run() {
        this.mapFileReader = new MapFileReader(this.mapFilename);
        this.mapFileReader.Run();

        var info = this.mapFileReader.GetSegmentInfoByUnitName("output");
        if (info) {
            this.DeleteLinesBeforeAddress(info.addressInt, info.segmentLength);
        }
    
        this.CollectJumpsAndCalls();
        this.InsertLabels();
    }

    /**
     * 
     * @param {number} address 
     * @param {number} segmentLength 
     */
    DeleteLinesBeforeAddress(address, segmentLength) {
        var startIdx = 0;

        var matches = false;
        var line = false;
        var lineAddr = false;

        var lineIdx = 0;
        while (lineIdx < this.asmLines.length) {
            line = this.asmLines[lineIdx];

            if (line.endsWith("<CODE>:")) {
                startIdx = lineIdx;
            } else if (line.endsWith("<.text>:")) {
                startIdx = lineIdx;
            }

            matches = line.match(this.addressRegex);
            if (matches) {
                lineAddr = parseInt(matches[1], 16);
                if (lineAddr >= address) {
                    this.asmLines.splice(startIdx, lineIdx - startIdx);
                    break;
                }
            }

            lineIdx++;
        }

        if (segmentLength) {
            var endAddress = address + segmentLength;

            lineIdx = 0;
            while (lineIdx < this.asmLines.length) {
                line = this.asmLines[lineIdx];
    
                matches = line.match(this.addressRegex);
                if (matches) {
                    lineAddr = parseInt(matches[1], 16);
                    if (lineAddr >= endAddress) {
                        this.asmLines = this.asmLines.splice(0, lineIdx);
                        break;
                    }
                }
    
                lineIdx++;
            }
        }
    }

    CollectJumpsAndCalls() {
        var jumpRegex = /(\sj[a-z]*)(\s*)0x([0-9a-f]*)/i;
        var callRegex = /(\scall)(\s*)0x([0-9a-f]*)/i;

        for (var lineIdx = 0; lineIdx < this.asmLines.length; lineIdx++) {
            var line = this.asmLines[lineIdx];

            var namedAddr = false;
            var labelName = false;
            var address = false;

            var matches = line.match(jumpRegex);
            if (matches) {
                address = matches[3];
                if (!address.includes('+') && !address.includes('-')) {
                    labelName = "L" + address;
                    namedAddr = this.mapFileReader.GetSymbolAt(false, parseInt(address, 16));
                    if (namedAddr) {
                        labelName = namedAddr.displayName;
                    }

                    if (!this.dontLabelUnmappedAddresses || namedAddr) {
                        this.addressesToLabel.push(address);
                        this.asmLines[lineIdx] = line.replace(jumpRegex, " " + matches[1] + matches[2] + labelName);
                    }
                }
            }

            matches = line.match(callRegex);
            if (matches && !matches[3].includes('+') && !matches[3].includes('-')) {
                address = matches[3];
                if (!address.includes('+') && !address.includes('-')) {
                    labelName = "L" + address;
                    namedAddr = this.mapFileReader.GetSymbolAt(false, parseInt(address, 16));
                    if (namedAddr) {
                        labelName = namedAddr.displayName;
                    }

                    if (!this.dontLabelUnmappedAddresses || namedAddr) {
                        this.addressesToLabel.push(address);
                        this.asmLines[lineIdx] = line.replace(callRegex, " " + matches[1] + matches[2] + labelName);
                    }
                }
            }
        }
    }

    InsertLabels() {
        var sourceFileId = this.mapFileReader.GetSegmentIdByUnitName("output");

        var currentSegment = false;
        var currentSymbol = false;

        var lineIdx = 0;
        while (lineIdx < this.asmLines.length) {
            var line = this.asmLines[lineIdx];

            var matches = line.match(this.addressRegex);
            if (matches) {
                var addressStr = matches[1];
                var address = parseInt(addressStr, 16);

                var segmentInfo = this.mapFileReader.GetSegmentInfoByStartingAddress(false, address);
                if (segmentInfo) {
                    currentSegment = segmentInfo;
                }

                var namedAddr = false;
                var labelLine = false;

                var isReferenced = this.addressesToLabel.indexOf(addressStr);
                if (isReferenced !== -1) {
                    labelLine = matches[1] + " <L" + addressStr + ">:";

                    namedAddr = this.mapFileReader.GetSymbolAt(false, address);
                    if (namedAddr) {
                        if (currentSymbol) {
                            // note: this might be wrong, we simply don't know
                            // this.asmLines.splice(lineIdx, 0, "  " + addressStr + " .cfi_endproc");
                            // lineIdx++;
                        }
                        
                        currentSymbol = namedAddr.displayName;
                        labelLine = matches[1] + " <" + namedAddr.displayName + ">:";
                    }
                    
                    if (!this.dontLabelUnmappedAddresses || namedAddr) {
                        this.asmLines.splice(lineIdx, 0, labelLine);
                        lineIdx++;
                    }
                } else {
                    // we might have missed the reference to this address, but if it's listed as a symbol, we should still label it
                    // todo: the call might be in <.itext>, should we include that part of the assembly?
                    namedAddr = this.mapFileReader.GetSymbolAt(false, address);
                    if (namedAddr) {
                        currentSymbol = namedAddr.displayName;
                        labelLine = matches[1] + " <" + namedAddr.displayName + ">:";

                        this.asmLines.splice(lineIdx, 0, labelLine);
                        lineIdx++;
                    }
                }

                var lineInfo = this.mapFileReader.GetLineInfoByAddress(false, address);
                if (lineInfo && currentSegment.unitName.startsWith("output")) {
                    this.asmLines.splice(lineIdx, 0, "/" + sourceFileId + ":" + lineInfo.lineNumber);
                    lineIdx++;
                }
            }
            
            lineIdx++;
        }
    }
}

exports.labelReconstructor = PELabelReconstructor;
