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
    private currentIndex = 0;

    constructor(
        public files: Record<number, string>,
        public source: AsmResultSource | undefined | null,
        public prevLabel: string,
        public prevLabelIsUserFunction: boolean,
        public dontMaskFilenames: boolean,
        private asmLines: string[],
    ) {}

    getCurrentLineIndex(): number {
        return this.currentIndex;
    }

    *[Symbol.iterator](): Generator<string, void, unknown> {
        while (this.currentIndex < this.asmLines.length) {
            const line = this.asmLines[this.currentIndex];
            this.currentIndex++;
            yield line;
        }
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
        const isLibraryCodeFilterEnabled = Boolean(filters.libraryCode);
        const isNotUserFunction = !this.prevLabelIsUserFunction;
        const hasNoLastOwnSource = !this.lastOwnSource;
        const hasSourceFile = Boolean(this.source?.file);
        const isNotMainSource = !this.source?.mainsource;

        return (
            isLibraryCodeFilterEnabled && isNotUserFunction && hasNoLastOwnSource && hasSourceFile && isNotMainSource
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
}
