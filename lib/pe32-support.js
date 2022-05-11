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

export class PELabelReconstructor {
    /**
     *
     * @param {Array} asmLines
     * @param {boolean} dontLabelUnmappedAddresses
     * @param {MapFileReader} mapFileReader
     */
    constructor(asmLines, dontLabelUnmappedAddresses, mapFileReader, needsReconstruction = true) {
        this.asmLines = asmLines;
        this.addressesToLabel = [];
        this.dontLabelUnmappedAddresses = dontLabelUnmappedAddresses;

        this.addressRegex = /^\s*([\da-f]*):/i;
        this.jumpRegex = /(\sj[a-z]*)(\s*)0x([\da-f]*)/i;
        this.callRegex = /(\scall)(\s*)0x([\da-f]*)/i;
        this.int3Regex = /\tcc\s*\tint3\s*$/i;

        this.mapFileReader = mapFileReader;
        this.needsReconstruction = needsReconstruction;
    }

    /**
     * Start reconstructing labels using the mapfile and remove unneccessary assembly
     *
     */
    run(/*unitName*/) {
        this.mapFileReader.run();

        //this.deleteEverythingBut(unitName);
        this.deleteSystemUnits();
        this.shortenInt3s();

        this.collectJumpsAndCalls();
        this.insertLabels();
    }

    /**
     * Remove any alignment NOP/int3 opcodes and replace them by a single line "..."
     */
    shortenInt3s() {
        let lineIdx = 0;
        let inInt3 = false;

        while (lineIdx < this.asmLines.length) {
            const line = this.asmLines[lineIdx];

            if (this.int3Regex.test(line)) {
                if (inInt3) {
                    this.asmLines.splice(lineIdx, 1);
                    lineIdx--;
                } else {
                    inInt3 = true;
                    this.asmLines[lineIdx] = '...';
                }
            } else {
                inInt3 = false;
            }

            lineIdx++;
        }
    }

    /**
     * Remove any assembly or data that isn't part of the given unit
     *
     * @param {string} unitName
     */
    deleteEverythingBut(unitName) {
        if (this.needsReconstruction) {
            let unitAddressSpaces = this.mapFileReader.getReconstructedUnitAddressSpace(unitName);

            for (let idx = 0; idx < this.mapFileReader.reconstructedSegments.length; idx++) {
                const info = this.mapFileReader.reconstructedSegments[idx];
                if (info.unitName !== unitName) {
                    if (info.segmentLength > 0) {
                        if (
                            !this.mapFileReader.isWithinAddressSpace(
                                unitAddressSpaces,
                                info.addressInt,
                                info.segmentLength,
                            )
                        ) {
                            this.deleteLinesBetweenAddresses(info.addressInt, info.addressInt + info.segmentLength);
                        }
                    }
                }
            }
        } else {
            let idx, info;
            for (idx = 0; idx < this.mapFileReader.segments.length; idx++) {
                info = this.mapFileReader.segments[idx];
                if (info.unitName !== unitName) {
                    this.deleteLinesBetweenAddresses(info.addressInt, info.addressInt + info.segmentLength);
                }
            }

            for (idx = 0; idx < this.mapFileReader.isegments.length; idx++) {
                info = this.mapFileReader.isegments[idx];
                if (info.unitName !== unitName) {
                    this.deleteLinesBetweenAddresses(info.addressInt, info.addressInt + info.segmentLength);
                }
            }
        }
    }

    deleteSystemUnits() {
        const systemUnits = new Set(['SysInit.pas', 'System.pas', 'SysUtils.pas', 'Classes.pas']);

        let idx, info;
        for (idx = 0; idx < this.mapFileReader.segments.length; idx++) {
            info = this.mapFileReader.segments[idx];
            if (systemUnits.has(info.unitName)) {
                this.deleteLinesBetweenAddresses(info.addressInt, info.addressInt + info.segmentLength);
            }
        }

        for (idx = 0; idx < this.mapFileReader.isegments.length; idx++) {
            info = this.mapFileReader.isegments[idx];
            if (systemUnits.has(info.unitName)) {
                this.deleteLinesBetweenAddresses(info.addressInt, info.addressInt + info.segmentLength);
            }
        }
    }

    /**
     *
     * @param {number} beginAddress
     * @param {number} endAddress
     */
    deleteLinesBetweenAddresses(beginAddress, endAddress) {
        let startIdx = -1;
        let linesRemoved = false;
        let lineIdx = 0;

        while (lineIdx < this.asmLines.length) {
            const line = this.asmLines[lineIdx];

            const matches = line.match(this.addressRegex);
            if (matches) {
                const lineAddr = parseInt(matches[1], 16);
                if (startIdx === -1 && lineAddr >= beginAddress) {
                    startIdx = lineIdx;
                    if (line.endsWith('<CODE>:') || line.endsWith('<.text>:') || line.endsWith('<.itext>:')) {
                        startIdx++;
                    }
                } else if (endAddress && lineAddr >= endAddress) {
                    this.asmLines.splice(startIdx, lineIdx - startIdx - 1);
                    linesRemoved = true;
                    break;
                }
            }

            lineIdx++;
        }

        if (!linesRemoved && startIdx !== -1) {
            this.asmLines.splice(startIdx, this.asmLines.length - startIdx);
        }
    }

    /**
     * Replaces an address used in a jmp or call instruction by its label.
     *  Does not replace an address if it has an offset.
     *
     * @param {int} lineIdx
     * @param {regex} regex
     */
    addAddressAsLabelAndReplaceLine(lineIdx, regex) {
        const line = this.asmLines[lineIdx];
        let matches = line.match(regex);
        if (matches) {
            const address = matches[3];
            if (!address.includes('+') && !address.includes('-')) {
                let labelName = 'L' + address;
                const namedAddr = this.mapFileReader.getSymbolAt(false, parseInt(address, 16));
                if (namedAddr) {
                    labelName = namedAddr.displayName;
                }

                if (!this.dontLabelUnmappedAddresses || namedAddr) {
                    this.addressesToLabel.push(address);

                    this.asmLines[lineIdx] = line.replace(regex, ' ' + matches[1] + matches[2] + labelName);
                }
            }
        }
    }

    /**
     * Collects addresses that are referred to by the assembly, through jump and call instructions
     */
    collectJumpsAndCalls() {
        for (let lineIdx = 0; lineIdx < this.asmLines.length; lineIdx++) {
            this.addAddressAsLabelAndReplaceLine(lineIdx, this.jumpRegex);
            this.addAddressAsLabelAndReplaceLine(lineIdx, this.callRegex);
        }
    }

    /**
     * Injects labels into the assembly where addresses are referred to
     *  if an address doesn't have a mapped name, it is called <Laddress>
     */
    insertLabels() {
        let currentSegment = false;
        let segmentChanged = false;

        let lineIdx = 0;
        while (lineIdx < this.asmLines.length) {
            const line = this.asmLines[lineIdx];

            const matches = line.match(this.addressRegex);
            if (matches) {
                const addressStr = matches[1];
                const address = parseInt(addressStr, 16);

                const segmentInfo = this.mapFileReader.getSegmentInfoByStartingAddress(false, address);
                if (segmentInfo) {
                    currentSegment = segmentInfo;
                    segmentChanged = true;
                }

                let namedAddr = false;
                let labelLine = false;

                const isReferenced = this.addressesToLabel.indexOf(addressStr);
                if (isReferenced !== -1) {
                    labelLine = matches[1] + ' <L' + addressStr + '>:';

                    namedAddr = this.mapFileReader.getSymbolAt(false, address);
                    if (namedAddr) {
                        labelLine = matches[1] + ' <' + namedAddr.displayName + '>:';
                    }

                    if (!this.dontLabelUnmappedAddresses || namedAddr) {
                        this.asmLines.splice(lineIdx, 0, labelLine);
                        lineIdx++;
                    }
                } else {
                    // we might have missed the reference to this address,
                    //  but if it's listed as a symbol, we should still label it.
                    // todo: the call might be in <.itext>, should we include that part of the assembly?
                    namedAddr = this.mapFileReader.getSymbolAt(false, address);
                    if (namedAddr) {
                        labelLine = matches[1] + ' <' + namedAddr.displayName + '>:';

                        this.asmLines.splice(lineIdx, 0, labelLine);
                        lineIdx++;
                    }
                }

                const lineInfo = this.mapFileReader.getLineInfoByAddress(false, address);
                if (lineInfo && currentSegment.unitName) {
                    this.asmLines.splice(lineIdx, 0, '/app/' + currentSegment.unitName + ':' + lineInfo.lineNumber);
                    lineIdx++;
                } else if (segmentChanged) {
                    this.asmLines.splice(lineIdx, 0, '/app/' + currentSegment.unitName + ':0');
                    lineIdx++;
                }
            }

            lineIdx++;
        }
    }
}
