import _ from 'underscore';
import { MultifileService } from './multifile-service';

class ColouredSourcelineInfo {
    sourceLine: number;
    compilerId: number;
    compilerLine: number;
    colourIdx: number;
};

export class LineColouring {
    private colouredSourceLinesByEditor: Object;
    private multifileService: MultifileService;
    private linesAndColourByCompiler: Object;
    private linesAndColourByEditor: Object;

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
        for (const asmLine of asm ) {
            if (asmLine.source && asmLine.source.line > 0) {
                const editorId = this.multifileService.getEditorIdByFilename(asmLine.source.file);
                if (editorId > 0) {
                    if (!this.colouredSourceLinesByEditor[editorId]) {
                        this.colouredSourceLinesByEditor[editorId] = new Array<ColouredSourcelineInfo>();
                    }

                    if (!this.linesAndColourByCompiler[compilerId]) {
                        this.linesAndColourByCompiler[compilerId] = {};
                    }

                    if (!this.linesAndColourByEditor[editorId]) {
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

    private getUniqueLinesForEditor(editorId: number) {
        const lines = [];

        for (const info of this.colouredSourceLinesByEditor[editorId]) {
            if (!lines.includes(info.sourceLine))
                lines.push(info.sourceLine);
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

    public getColoursForCompiler(compilerId: number): Object {
        return this.linesAndColourByCompiler[compilerId];
    }

    public getColoursForEditor(editorId: number): Object {
        return this.linesAndColourByEditor[editorId];
    }
};
