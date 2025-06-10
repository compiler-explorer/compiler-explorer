// Copyright (c) 2025, Compiler Explorer Authors
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
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import {AsmResultSource} from '../../types/asmresult/asmresult.interfaces.js';

export class ParsingState {
    public mayRemovePreviousLabel = true;
    public keepInlineCode = false;
    public lastOwnSource: AsmResultSource | undefined | null = null;
    public inNvccDef = false;
    public inNvccCode = false;
    public inCustomAssembly = 0;
    public inVLIWpacket = false;
    public idxLine = 0;

    constructor(
        public files: Record<number, string>,
        public source: AsmResultSource | undefined | null,
        public prevLabel: string,
        public prevLabelIsUserFunction: boolean,
        public dontMaskFilenames: boolean,
    ) {}

    reset() {
        this.mayRemovePreviousLabel = true;
        this.keepInlineCode = false;
        this.lastOwnSource = null;
        this.inNvccDef = false;
        this.inNvccCode = false;
        this.inCustomAssembly = 0;
        this.inVLIWpacket = false;
        this.idxLine = 0;
        this.source = null;
        this.prevLabel = '';
        this.prevLabelIsUserFunction = false;
    }

    updateSource(newSource: AsmResultSource | null | undefined) {
        this.source = newSource;
        if (newSource?.file === null || newSource?.mainsource) {
            this.lastOwnSource = newSource;
        }
    }

    resetToBlockEnd() {
        this.source = null;
        this.prevLabel = '';
        this.lastOwnSource = null;
    }

    enterCustomAssembly() {
        this.inCustomAssembly++;
    }

    exitCustomAssembly() {
        this.inCustomAssembly--;
    }

    isInCustomAssembly(): boolean {
        return this.inCustomAssembly > 0;
    }

    setVLIWPacket(inVLIWpacket: boolean) {
        this.inVLIWpacket = inVLIWpacket;
    }

    enterNvccDef() {
        this.inNvccDef = true;
        this.inNvccCode = true;
    }

    exitNvccDef() {
        this.inNvccDef = false;
    }

    shouldFilterLibraryCode(filters: {libraryCode?: boolean}): boolean {
        return Boolean(
            filters.libraryCode &&
                !this.prevLabelIsUserFunction &&
                !this.lastOwnSource &&
                this.source?.file !== null &&
                !this.source?.mainsource,
        );
    }

    shouldRemovePreviousLabel(): boolean {
        return this.mayRemovePreviousLabel;
    }

    setMayRemovePreviousLabel(value: boolean) {
        this.mayRemovePreviousLabel = value;
    }

    setKeepInlineCode(value: boolean) {
        this.keepInlineCode = value;
    }

    shouldKeepInlineCode(): boolean {
        return this.keepInlineCode;
    }

    updatePrevLabel(label: string, isUserFunction = false) {
        this.prevLabel = label;
        this.prevLabelIsUserFunction = isUserFunction;
    }

    clearPrevLabel() {
        this.prevLabel = '';
        this.prevLabelIsUserFunction = false;
    }

    nextLine() {
        this.idxLine++;
    }

    getCurrentLineIndex(): number {
        return this.idxLine;
    }
}
