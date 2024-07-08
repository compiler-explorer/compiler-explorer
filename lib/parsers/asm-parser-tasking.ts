import {
    AsmResultLabel,
    AsmResultSource,
    ParsedAsmResult,
    ParsedAsmResultLine,
} from '../../types/asmresult/asmresult.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {PropertyGetter} from '../properties.interfaces';
import {ElfParserTool} from '../tooling/tasking-elfparse-tool';
import * as utils from '../utils';

import {AsmParser} from './asm-parser';
import {IAsmParser} from './asm-parser.interfaces';
import {AsmRegex} from './asmregex';

export class AsmParserTasking extends AsmParser implements IAsmParser {
    taskingText: RegExp;
    taskingMachineCode: RegExp;
    objpath: string;
    srcpath: string;

    constructor(compilerProps?: PropertyGetter) {
        super(compilerProps);
        this.taskingText = /^\s+(.sect|.sdecl)\s+'(.*)'.*/;
        this.taskingMachineCode = /(^\w+)((?:\s*(?:\d|[a-f]){2}){2,4})\s+(.*:)?(\s*(.*)\s+(.*))/;
    }

    public setSrcPath(path: string) {
        this.srcpath = path;
    }

    override processAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        if (filters.binaryObject) return this.processBinaryAsm(asmResult, filters);
        const startTime = process.hrtime.bigint();
        const asm: ParsedAsmResultLine[] = [];
        const labelDefinitions: Record<string, number> = {};

        const asmLines = utils.splitLines(asmResult);
        const startingLineCount = asmLines.length;

        let source: AsmResultSource | undefined | null;

        function maybeAddBlank() {
            const lastBlank = asm.length === 0 || asm[asm.length - 1].text === '';
            if (!lastBlank) asm.push({text: '', source: null, labels: []});
        }

        let filetext = '';
        let address = '';

        const elfParseTool = new ElfParserTool(this.objpath, this.srcpath, filters.binaryObject, filters.libraryCode);
        const elf = elfParseTool.start();

        for (let line of asmLines) {
            if (line.trim() === '') {
                maybeAddBlank();
                continue;
            }

            const match = null;
            line = utils.expandTabs(line);

            const matchtext = line.match(this.taskingText);
            if (matchtext) {
                if (matchtext[1] === '.sect') {
                    filetext = matchtext[2];
                    if (elf.lineSet.has(filetext) || filters.libraryCode) {
                        asm.push({
                            text: filetext,
                        });
                    }
                }
                continue;
            }

            const matchaddr = line.match(this.taskingMachineCode);
            if (matchaddr) {
                address = matchaddr[1];
                if (matchaddr[5].includes('call') || matchaddr[5].trim() === 'j') {
                    if (matchaddr[1] === '00000000') {
                        const map = elf.relaMap.get(filetext);
                        if (map) {
                            const value = map.get(0);
                            if (value) {
                                matchaddr[4] = ' ' + matchaddr[5] + value;
                            }
                        }
                    } else {
                        const key = parseInt(matchaddr[6]);
                        const map = elf.relaMap.get(filetext);
                        if (map) {
                            const value = map.get(key);
                            if (value) {
                                matchaddr[4] = ' ' + matchaddr[5] + value;
                            }
                        }
                    }
                }
                if (filters.directives) {
                    line = '  ' + matchaddr[4].trim();
                }
            }

            const text = AsmRegex.filterAsmLine(line, filters);

            const labelsInLine = match ? [] : this.getUsedLabelsInLine(text);

            const map = elf.lineMap.get(filetext);
            if (map) {
                const _linenumber: number | undefined = map.get(address);
                if (_linenumber) {
                    source = {
                        column: 1,
                        file: null,
                        line: _linenumber,
                    };
                }
            }

            if (elf.lineSet.has(filetext) || filters.libraryCode) {
                if (elf.lineSet.has(filetext)) {
                    asm.push({
                        text: text,
                        source: source,
                        labels: labelsInLine,
                    });
                } else {
                    asm.push({
                        text: text,
                        source: null,
                        labels: labelsInLine,
                    });
                }
            }
        }

        const endTime = process.hrtime.bigint();
        return {
            asm: asm,
            labelDefinitions: labelDefinitions,
            parsingTime: ((endTime - startTime) / BigInt(1000000)).toString(),
            filteredCount: startingLineCount - asm.length,
        };
    }

    override processBinaryAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        const startTime = process.hrtime.bigint();
        const asm: ParsedAsmResultLine[] = [];
        const labelDefinitions: Record<string, number> = {};

        const asmLines = utils.splitLines(asmResult);
        const startingLineCount = asmLines.length;

        let source: AsmResultSource | undefined | null;

        function maybeAddBlank() {
            const lastBlank = asm.length === 0 || asm[asm.length - 1].text === '';
            if (!lastBlank) asm.push({text: '', source: null, labels: []});
        }

        const elfParseTool = new ElfParserTool(this.objpath, this.srcpath, filters.binaryObject, filters.libraryCode);
        const elf = elfParseTool.start();
        const map = elf.lineMap.get(this.srcpath);

        let minaddress = 'ffffffff';
        let maxaddress = '00000000';

        if (map) {
            for (const key of map.keys()) {
                if (key < minaddress && key !== '00000000') {
                    minaddress = key;
                }
                if (key > maxaddress) {
                    maxaddress = key;
                }
            }
        }

        let completeTag1 = false;
        let completeTag2 = false;

        for (let line of asmLines) {
            let address = '';
            if (line.trim() === '') {
                maybeAddBlank();
                continue;
            }

            const match = null;
            line = utils.expandTabs(line);

            const matchaddr = line.match(this.taskingMachineCode);
            if (matchaddr) {
                completeTag1 = true;
                address = matchaddr[1];
                if (filters.directives) {
                    line = '  ' + matchaddr[4].trim();
                }
            } else {
                completeTag1 = false;
                completeTag2 = false;
            }

            const text = AsmRegex.filterAsmLine(line, filters);

            const labelsInLine = match ? [] : this.getUsedLabelsInLine(text);

            if (address >= minaddress && address <= maxaddress) {
                if (map) {
                    completeTag2 = true;
                    const _linenumber: number | undefined = map.get(address);
                    if (_linenumber) {
                        source = {
                            column: 1,
                            file: null,
                            line: _linenumber,
                        };
                    }
                    asm.push({
                        text: text,
                        source: source,
                        labels: labelsInLine,
                    });
                }
                continue;
            } else {
                if (filters.libraryCode) {
                    asm.push({
                        text: text,
                        source: null,
                        labels: labelsInLine,
                    });
                }
                if (completeTag1 && completeTag2) {
                    asm.push({
                        text: text,
                        source: source,
                        labels: labelsInLine,
                    });
                }
            }
        }

        const endTime = process.hrtime.bigint();
        return {
            asm: asm,
            labelDefinitions: labelDefinitions,
            parsingTime: ((endTime - startTime) / BigInt(1000000)).toString(),
            filteredCount: startingLineCount - asm.length,
        };
    }
}
