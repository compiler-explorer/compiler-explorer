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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import {AsmResultSource} from '../../types/asmresult/asmresult.interfaces.js';
import * as utils from '../utils.js';

export type SourceHandlerContext = {
    files: Record<number, string>;
    dontMaskFilenames: boolean;
};

// STAB debugging format constants
// See: http://www.math.utah.edu/docs/info/stabs_11.html#SEC48
const STAB_N_SLINE = 68; // Source line: maps line numbers to addresses
const STAB_N_SO = 100; // Source file: marks beginning of source file debugging info
const STAB_N_SOL = 132; // Included file: tracks #included files

export class SourceLineHandler {
    private sourceTag: RegExp;
    private sourceD2Tag: RegExp;
    private sourceCVTag: RegExp;
    private source6502Dbg: RegExp;
    private source6502DbgEnd: RegExp;
    private sourceStab: RegExp;
    private stdInLooking: RegExp;

    constructor() {
        this.sourceTag = /^\s*\.loc\s+(\d+)\s+(\d+)\s+(.*)/;
        this.sourceD2Tag = /^\s*\.d2line\s+(\d+),?\s*(\d*).*/;
        this.sourceCVTag = /^\s*\.cv_loc\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+).*/;
        this.source6502Dbg = /^\s*\.dbg\s+line,\s*"([^"]+)",\s*(\d+)/;
        this.source6502DbgEnd = /^\s*\.dbg\s+line[^,]/;
        this.sourceStab = /^\s*\.stabn\s+(\d+),0,(\d+),.*/;
        this.stdInLooking = /<stdin>|^-$|example\.[^/]+$|<source>/;
    }

    private createSource(file: string, line: number, context: SourceHandlerContext, column?: number): AsmResultSource {
        const isMainSource = this.stdInLooking.test(file);
        const source: AsmResultSource = context.dontMaskFilenames
            ? {
                  file,
                  line,
                  mainsource: isMainSource,
              }
            : {
                  file: isMainSource ? null : file,
                  line,
              };

        if (column !== undefined && !Number.isNaN(column) && column !== 0) {
            source.column = column;
        }

        return source;
    }

    handleSourceTag(line: string, context: SourceHandlerContext): AsmResultSource | null {
        const match = line.match(this.sourceTag);
        if (!match) return null;

        const file = utils.maskRootdir(context.files[Number.parseInt(match[1], 10)]);
        const sourceLine = Number.parseInt(match[2], 10);

        if (!file) return null;

        return this.createSource(file, sourceLine, context, Number.parseInt(match[3], 10));
    }

    handleD2Tag(line: string): AsmResultSource | null {
        const match = line.match(this.sourceD2Tag);
        if (!match) return null;

        return {
            file: null,
            line: Number.parseInt(match[1], 10),
        };
    }

    handleCVTag(line: string, context: SourceHandlerContext): AsmResultSource | null {
        const match = line.match(this.sourceCVTag);
        if (!match) return null;

        const sourceLine = Number.parseInt(match[3], 10);
        const file = utils.maskRootdir(context.files[Number.parseInt(match[2], 10)]);

        return this.createSource(file, sourceLine, context, Number.parseInt(match[4], 10));
    }

    handle6502Debug(line: string, context: SourceHandlerContext): AsmResultSource | null {
        if (this.source6502DbgEnd.test(line)) {
            return null;
        }

        const match = line.match(this.source6502Dbg);
        if (!match) return null;

        const file = utils.maskRootdir(match[1]);
        const sourceLine = Number.parseInt(match[2], 10);

        return this.createSource(file, sourceLine, context);
    }

    handleStabs(line: string): AsmResultSource | null | undefined {
        const match = line.match(this.sourceStab);
        if (!match) return undefined;

        // cf http://www.math.utah.edu/docs/info/stabs_11.html#SEC48
        switch (Number.parseInt(match[1], 10)) {
            case STAB_N_SLINE:
                return {file: null, line: Number.parseInt(match[2], 10)};
            case STAB_N_SO:
            case STAB_N_SOL:
                return null;
            default:
                return undefined;
        }
    }

    processSourceLine(
        line: string,
        context: SourceHandlerContext,
    ): {
        source: AsmResultSource | null | undefined;
        resetPrevLabel: boolean;
    } {
        // Try each source handler in order
        const handlers: Array<() => AsmResultSource | null> = [
            () => this.handleSourceTag(line, context),
            () => this.handleD2Tag(line),
            () => this.handleCVTag(line, context),
            () => this.handle6502Debug(line, context),
        ];

        for (const handler of handlers) {
            const source = handler();
            if (source) {
                return {source, resetPrevLabel: false};
            }
        }

        // Special handling for stabs
        const stabResult = this.handleStabs(line);
        if (stabResult !== undefined) {
            const stabMatch = line.match(this.sourceStab);
            const resetPrevLabel =
                stabResult === null && (stabMatch?.[1] === String(STAB_N_SOL) || stabMatch?.[1] === String(STAB_N_SO));
            return {source: stabResult, resetPrevLabel};
        }

        return {source: undefined, resetPrevLabel: false};
    }
}
