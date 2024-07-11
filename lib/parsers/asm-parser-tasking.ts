import {
    AsmResultLabel,
    AsmResultSource,
    ParsedAsmResult,
    ParsedAsmResultLine,
} from '../../types/asmresult/asmresult.interfaces';
import {ParseFiltersAndOutputOptions} from '../../types/features/filters.interfaces';
import {assert} from '../assert';
import {PropertyGetter} from '../properties.interfaces';
import {ElfParserTool} from '../tooling/tasking-elfparse-tool';
import * as utils from '../utils';

import {AsmParser} from './asm-parser';
import {IAsmParser} from './asm-parser.interfaces';

interface Isntruction {
    addr: number;
    mcode: string[];
    operator: string;
    operands: string[];
    labels: AsmResultLabel[];
}

export class AsmParserTasking extends AsmParser implements IAsmParser {
    sdeclRe: RegExp;
    sectRe: RegExp;
    instRe: RegExp;
    brInstRe: RegExp;
    brAddrRe: RegExp;
    callInstRe: RegExp;
    filters: ParseFiltersAndOutputOptions;
    objpath: string;
    srcpath: string;
    elfParseTool: ElfParserTool;

    constructor(compilerProps?: PropertyGetter) {
        super(compilerProps);
        this.sdeclRe = /^\s*\.sdecl\s*'\.text\.(\b[\w.]+\b)',\s*CODE\s+AT\s0x([\da-f]{0,8})\s*$/;
        this.sectRe = /^\s*\.sect\s*'\.text\.\b[\w.]+\b'\s*$/;
        this.instRe = /([\da-f]{8})\s+((?:\s*[\da-f]{2})+)\s+([a-z]+(?:\.[a-z]+)?)(?:\s+(.+))?\s*$/;
        this.brInstRe = /^j.*$/;
        this.callInstRe = /^.*call?$/;
        this.brAddrRe = /^0x([\da-f]{0,8})$/;
    }

    public setSrcPath(path: string) {
        this.srcpath = path;
    }

    parseLines(lines: string[]) {
        const sec_insts = new Map<string, Isntruction[]>();
        const links = new Map<number, string>();
        let sec_name = '__NONE_SECTION__';
        let cur_insts: Isntruction[] = [];
        for (const line of lines) {
            let ma_res = line.match(this.sdeclRe);
            if (ma_res) {
                assert(ma_res !== undefined && ma_res !== null);
                sec_name = ma_res[1];
                cur_insts = [];
                sec_insts.set(sec_name, cur_insts);
                const sec_addr = ma_res[2];
                const addr = BigInt.asUintN(32, BigInt('0x' + sec_addr));
                links.set(addr === 0n ? links.size : Number(addr), sec_name);
                continue;
            }
            ma_res = line.match(this.instRe);
            if (ma_res) {
                assert(ma_res !== undefined && ma_res !== null);
                const operands = ma_res[4] ? ma_res[4].split(',') : [];
                const addr = BigInt.asUintN(32, BigInt('0x' + ma_res[1]));
                cur_insts.push({
                    addr: Number(addr),
                    mcode: ma_res[2].split(' '),
                    operator: ma_res[3],
                    operands: operands,
                    labels: [],
                });
            }
        }
        return {links: links, insts: sec_insts};
    }

    processRelas(relas: Map<number, string>, insts: Isntruction[]) {
        const labels = new Map<number, string>();
        for (const inst of insts) {
            const ma_res = inst.operator.match(this.callInstRe);
            if (ma_res) {
                for (const [i, op] of inst.operands.entries()) {
                    const _ma_res = op.match(this.brAddrRe);
                    if (_ma_res === null) {
                        continue;
                    }
                    const offset = inst.addr;
                    const value = Number(BigInt.asUintN(32, BigInt('0x' + _ma_res[1])));
                    const addend = value - offset;
                    const _target = relas.get(offset);
                    if (_target) {
                        const ndx =
                            _target.lastIndexOf('..') === -1 ? _target.lastIndexOf('.') : _target.lastIndexOf('..');
                        const target = _target.substring(ndx + 1);
                        inst.operands[i] = addend === 0 ? `${target}` : `${target}+${addend}`;
                        inst.labels.push({
                            name: inst.operands[i],
                            range: {startCol: 1, endCol: target.length},
                        });
                    }
                }
            }
        }
    }

    processLinks(link: Map<number, string>, insts: Isntruction[]) {
        const labels = new Map<number, string>();
        for (const inst of insts) {
            const ma_res = inst.operator.match(this.callInstRe);
            if (ma_res) {
                for (const [i, op] of inst.operands.entries()) {
                    const _ma_res = op.match(this.brAddrRe);
                    if (_ma_res === null) {
                        continue;
                    }
                    const addr = Number(BigInt.asUintN(32, BigInt('0x' + _ma_res[1])));
                    const _target = link.get(addr);
                    if (_target) {
                        const ndx =
                            _target.lastIndexOf('..') === -1 ? _target.lastIndexOf('.') : _target.lastIndexOf('..');
                        const target = _target.substring(ndx + 1);
                        inst.operands[i] = `${target}`;
                        inst.labels.push({
                            name: inst.operands[i],
                            range: {startCol: 1, endCol: target.length},
                        });
                    }
                }
            }
        }
    }

    findAddrNest(link: Map<number, string>, addr: number) {
        const sec_addrs = [...link.keys()].sort();
        let left = 0;
        let right = sec_addrs.length;
        let ptr = Math.floor((left + right) / 2);
        while (left < right) {
            if (sec_addrs[ptr] < addr) {
                left = ptr + 1;
            } else if (sec_addrs[ptr] > addr) {
                right = ptr - 1;
            } else {
                break;
            }
            ptr = Math.floor((left + right) / 2);
        }
        return sec_addrs[ptr];
    }

    composeAsmText(inst: Isntruction) {
        let text = '';
        if (!this.filters.directives) {
            text += this.elfParseTool.toAddrStr(inst.addr) + ' ';
            const mcode = inst.mcode.join(' ');
            text += mcode + ' '.repeat(20).substring(mcode.length, 20);
        }
        text += inst.operator;
        if (inst.operands.length > 0) {
            text += ' '.repeat(10 - inst.operator.length) + inst.operands.join(',');
        }
        return text;
    }

    isSrcSection(sec: string) {
        const srcname = this.elfParseTool.getSrcname();
        return sec.startsWith(srcname) && !sec.substring(srcname.length + 1).startsWith('.');
    }

    stripHeader(sec: string) {
        return sec.substring(sec.lastIndexOf('.') + 1);
    }

    override processAsm(asmResult: string, filters: ParseFiltersAndOutputOptions): ParsedAsmResult {
        this.filters = filters;
        if (filters.binaryObject) return this.processBinaryAsm(asmResult, filters);
        const startTime = process.hrtime.bigint();
        this.elfParseTool = new ElfParserTool(this.objpath, this.srcpath, filters.binaryObject, filters.libraryCode);
        const elf = this.elfParseTool.start();
        const asm: ParsedAsmResultLine[] = [];
        const labelDefinitions: Record<string, number> = {};

        const asmLines = utils.splitLines(asmResult);
        const startingLineCount = asmLines.length;

        const parseResult = this.parseLines(asmLines);
        const links = parseResult.links;
        const sec_insts = parseResult.insts;
        for (const sec of sec_insts.keys()) {
            if (!filters.libraryCode && !this.isSrcSection(sec)) {
                continue;
            }
            const src_map = elf.lineMap.get('.text.' + sec);
            const rel_map = elf.relaMap.get('.text.' + sec);
            const insts = sec_insts.get(sec);
            assert(insts !== undefined && insts !== null);
            if (rel_map) {
                this.processRelas(rel_map, insts);
            }
            asm.push({
                text: this.stripHeader(sec) + ':',
                source: null,
                labels: [],
            });
            labelDefinitions[this.stripHeader(sec)] = asm.length;
            let last_line = -1;
            for (const inst of insts) {
                const addr = this.elfParseTool.toAddrStr(inst.addr);
                const line = src_map ? src_map.get(addr) : -1;
                if (line) {
                    last_line = line;
                }
                const src: AsmResultSource = {
                    file: null,
                    line: last_line,
                };
                const text = this.composeAsmText(inst);
                if (inst.labels.length > 0) {
                    inst.labels[0].range.startCol += text.lastIndexOf(' ') + 1;
                    inst.labels[0].range.endCol += inst.labels[0].range.startCol;
                }
                asm.push({
                    text: text,
                    // opcodes: inst.mcode,
                    // address: inst.addr,
                    source: src,
                    labels: inst.labels,
                });
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
        this.elfParseTool = new ElfParserTool(this.objpath, this.srcpath, filters.binaryObject, filters.libraryCode);
        const elf = this.elfParseTool.start();
        const asm: ParsedAsmResultLine[] = [];
        const labelDefinitions: Record<string, number> = {};

        const asmLines = utils.splitLines(asmResult);
        const startingLineCount = asmLines.length;

        const parseResult = this.parseLines(asmLines);
        const links = parseResult.links;
        const sec_insts = parseResult.insts;
        const src_map = elf.lineMap.get(this.srcpath);
        for (const sec of sec_insts.keys()) {
            if (!filters.libraryCode && !this.isSrcSection(sec)) {
                continue;
            }
            const insts = sec_insts.get(sec);
            assert(insts !== undefined && insts !== null);
            this.processLinks(links, insts);
            asm.push({
                text: this.stripHeader(sec) + ':',
                source: null,
                labels: [],
            });
            labelDefinitions[this.stripHeader(sec)] = asm.length;
            let last_line = -1;
            for (const inst of insts) {
                const addr = this.elfParseTool.toAddrStr(inst.addr);
                const line = src_map ? src_map.get(addr) : -1;
                if (line) {
                    last_line = line;
                }
                const src: AsmResultSource = {
                    file: null,
                    line: last_line,
                };
                const text = this.composeAsmText(inst);
                if (inst.labels.length > 0) {
                    inst.labels[0].range.startCol += text.lastIndexOf(' ') + 1;
                    inst.labels[0].range.endCol += inst.labels[0].range.startCol;
                }
                asm.push({
                    text: text,
                    // opcodes: inst.mcode,
                    // address: inst.addr,
                    source: src,
                    labels: inst.labels,
                });
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
