// Copyright (c) 2021, Compiler Explorer Authors
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

import _ from 'underscore';
import {MultifileService} from './multifile-service';

interface ColouredSourcelineInfo {
    sourceLine: number;
    compilerId: number;
    compilerLine: number;
    colourIdx: number;
}

export class LineColouring {
    private colouredSourceLinesByEditor: ColouredSourcelineInfo[][];
    private multifileService: MultifileService;
    private linesAndColourByCompiler: Record<number, Record<number, number>>;
    private linesAndColourByEditor: Record<number, Record<number, number>>;

    constructor(multifileService: MultifileService) {
        this.multifileService = multifileService;

        this.clear();
    }

    public clear() {
        this.colouredSourceLinesByEditor = [];
        this.linesAndColourByCompiler = {};
        this.linesAndColourByEditor = {};
    }

    public addFromAssembly(compilerId, asm) {
        let asmLineIdx = 0;
        for (const asmLine of asm) {
            if (asmLine.source && asmLine.source.line > 0) {
                const editorId = this.multifileService.getEditorIdByFilename(asmLine.source.file);
                if (editorId != null && editorId > 0) {
                    if (!this.colouredSourceLinesByEditor[editorId]) {
                        this.colouredSourceLinesByEditor[editorId] = [];
                    }

                    if (!(compilerId in this.linesAndColourByCompiler)) {
                        this.linesAndColourByCompiler[compilerId] = {};
                    }

                    if (!(editorId in this.linesAndColourByEditor)) {
                        this.linesAndColourByEditor[editorId] = {};
                    }

                    this.colouredSourceLinesByEditor[editorId].push({
                        sourceLine: asmLine.source.line - 1,
                        compilerId: compilerId,
                        compilerLine: asmLineIdx,
                        colourIdx: -1,
                    });
                }
            }
            asmLineIdx++;
        }
    }

    private getUniqueLinesForEditor(editorId: number): number[] {
        const lines: number[] = [];

        for (const info of this.colouredSourceLinesByEditor[editorId]) {
            if (!lines.includes(info.sourceLine)) lines.push(info.sourceLine);
        }

        return lines;
    }

    private setColourBySourceline(editorId: number, line: number, colourIdx: number) {
        for (const info of this.colouredSourceLinesByEditor[editorId]) {
            if (info.sourceLine === line) {
                info.colourIdx = colourIdx;
            }
        }
    }

    public calculate() {
        let colourIdx = 0;

        for (const editorIdStr of _.keys(this.colouredSourceLinesByEditor)) {
            const editorId = parseInt(editorIdStr);

            const lines = this.getUniqueLinesForEditor(editorId);
            for (const line of lines) {
                this.setColourBySourceline(editorId, line, colourIdx);
                colourIdx++;
            }
        }

        const compilerIds = _.keys(this.linesAndColourByCompiler);
        const editorIds = _.keys(this.linesAndColourByEditor);

        for (const compilerIdStr of compilerIds) {
            const compilerId = parseInt(compilerIdStr);
            for (const editorId of _.keys(this.colouredSourceLinesByEditor)) {
                for (const info of this.colouredSourceLinesByEditor[editorId]) {
                    if (info.compilerId === compilerId && info.colourIdx >= 0) {
                        this.linesAndColourByCompiler[compilerId][info.compilerLine] = info.colourIdx;
                    }
                }
            }
        }

        for (const editorId of editorIds) {
            for (const info of this.colouredSourceLinesByEditor[editorId]) {
                if (info.colourIdx >= 0) {
                    this.linesAndColourByEditor[editorId][info.sourceLine] = info.colourIdx;
                }
            }
        }
    }

    public getColoursForCompiler(compilerId: number): Record<number, number> {
        if (compilerId in this.linesAndColourByCompiler) {
            return this.linesAndColourByCompiler[compilerId];
        } else {
            return {};
        }
    }

    public getColoursForEditor(editorId: number): Record<number, number> {
        if (editorId in this.linesAndColourByEditor) {
            return this.linesAndColourByEditor[editorId];
        } else {
            return {};
        }
    }
}
