import * as fs from 'fs';

import {assert} from '../../assert';

import {LineInfoItem as _LineInfoItem, DwarfLineReader} from './dwarf-line-reader';
import {ElfReader, SecHeader, SH_TYPE} from './elf-reader';
import {Addr, uWord} from './elf-types';

export interface LineInfoItem extends _LineInfoItem {
    objpath: string;
    secname: string;
}

function extendItemFrom(_base: _LineInfoItem, objpath: string, sec_name: string) {
    const item: LineInfoItem = {
        address_start: _base.address_start,
        address_end: _base.address_end,
        inc_dir: _base.inc_dir,
        srcpath: _base.srcpath,
        line: _base.line,
        column: _base.column,
        objpath: objpath,
        secname: sec_name,
    };
    return item;
}

function pad(num_str: string) {
    const s = '00000000' + num_str;
    return s.substring(num_str.length);
}

export class ElfParser {
    protected elfPath: string;
    protected elfContent: Uint8Array;
    protected elfReader: ElfReader;
    protected lineReader: DwarfLineReader;

    bindFile(filepath: string) {
        this.elfPath = filepath;
        this.elfReader = new ElfReader();
        this.lineReader = new DwarfLineReader();
        const file_content = fs.readFileSync(filepath);
        this.elfReader.readElf(file_content);
        this.elfContent = file_content;
    }

    toAddrStr(addr: number) {
        return pad(addr.toString(16));
    }

    getSecName(sec: SecHeader) {
        return this.elfReader.readSecName(sec);
    }

    getSrcPaths() {
        const paths: string[] = [];
        for (const file_info of this.lineReader.fileInfo()) {
            paths.push(file_info.filename);
        }
        return paths;
    }

    getLinkedSymTable(): Map<number, string> {
        const table = new Map<number, string>();

        const sh_syms = this.elfReader.findSecsBy((sec: SecHeader) => {
            return sec.sh_type === SH_TYPE.SHT_SYMTAB;
        });
        for (const sh_sym of sh_syms) {
            const entries = this.elfReader.readSymEntries(sh_sym);
            for (const entry of entries) {
                const name = this.elfReader.readSymName(entry);
                const addr = Number(BigInt.asUintN(32, entry.st_value));
                table.set(addr, name);
            }
        }
        return table;
    }

    getDislinkedSymTable(): Map<string, Map<number, string>> {
        return this.getRelaMap();
    }

    getRelaMap() {
        const relaMap = new Map<string, Map<number, string>>();
        const rels = this.elfReader.getRelaocations();
        for (const [key, array] of Object.entries(rels)) {
            const record = new Map<number, string>();
            for (const rela of array) {
                const addr = Number(BigInt.asUintN(32, rela.r_offset));
                record.set(addr, this.elfReader.readRelTargetName(rela));
            }
            relaMap.set(key, record);
        }
        return relaMap;
    }

    getSecHeaders(filter: (sec: SecHeader) => boolean) {
        return this.elfReader.getSecHeaders(filter);
    }

    getRangesOf(secs: SecHeader[]) {
        const ranges: {begin: Addr; end: Addr}[] = [];
        for (const sec of secs) {
            ranges.push({
                begin: sec.sh_addr,
                end: sec.sh_addr + BigInt(sec.sh_size),
            });
        }
        return ranges;
    }

    protected processIntegrated(filter: (item: LineInfoItem) => boolean) {
        const lineInfoItems: LineInfoItem[] = [];
        const debug_lines = this.elfReader.findSecsBy((sec: SecHeader) => {
            return this.elfReader.readSecName(sec).startsWith('.debug_line');
        });
        this.lineReader.resetAll();
        for (const dbg of debug_lines) {
            const content = this.elfReader.getContentsOf([dbg])[0];
            this.lineReader.readEntries(content);
            const linInfos = this.lineReader.lineInfo();
            for (const l of linInfos) {
                const lineInfo = extendItemFrom(l, this.elfPath, '');
                if (filter(lineInfo)) {
                    lineInfoItems.push(lineInfo);
                }
            }
        }
        return lineInfoItems;
    }

    protected processGrouped(groups: any[], filter: (item: LineInfoItem) => boolean) {
        this.lineReader.resetAll();
        const lineInfoItems: LineInfoItem[] = [];
        for (const group of groups) {
            const debug_lines = this.elfReader.getDbgLineSecsOf(group);
            const texts = this.elfReader.getTextSecsOf(group);
            if (texts.length === 0 || debug_lines.length === 0) {
                continue;
            }
            for (const sec of texts) {
                const sec_name = this.elfReader.readSecName(sec);
                const contents = this.elfReader.getContentsOf(debug_lines);
                for (const content of contents) {
                    this.lineReader.readEntries(content);
                }
                for (const l of this.lineReader.lineInfo()) {
                    const lineInfo = extendItemFrom(l, this.elfPath, sec_name);
                    if (filter(lineInfo)) {
                        lineInfoItems.push(lineInfo);
                    }
                }
                this.lineReader.clearItems();
            }
        }
        return lineInfoItems;
    }

    getLineInfoItems(filter: (item: LineInfoItem) => boolean) {
        const groups = this.elfReader.getGroups();
        if (groups.length === 0) {
            return this.processIntegrated(filter);
        } else {
            return this.processGrouped(groups, filter);
        }
    }

    getLineMap(filter: (item: LineInfoItem) => boolean) {
        const lineMap = new Map<string, Map<string, number>>();
        const linInfoItems = this.getLineInfoItems(filter);
        for (const lineInfo of linInfoItems) {
            if (!lineMap.has(lineInfo.srcpath)) {
                lineMap.set(lineInfo.srcpath, new Map<string, number>());
            }
            if (!lineMap.has(lineInfo.secname)) {
                lineMap.set(lineInfo.secname, new Map<string, number>());
            }
            const src_map = lineMap.get(lineInfo.srcpath);
            const sec_map = lineMap.get(lineInfo.secname);
            assert(src_map !== undefined && src_map !== null);
            assert(sec_map !== undefined && sec_map !== null);
            const start = lineInfo.address_start;
            const end = lineInfo.address_end;
            for (let addr = start; addr < end; addr++) {
                const formated = pad(addr.toString(16));
                src_map.set(formated, lineInfo.line);
                sec_map.set(formated, lineInfo.line);
            }
        }
        return lineMap;
    }
}
