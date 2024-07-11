import {assert} from '../assert';

import {ElfParser, LineInfoItem} from './readers/elf-parser';
import {SecHeader} from './readers/elf-reader';
import {Addr} from './readers/elf-types';

export class ElfParserTool {
    protected elfParser: ElfParser;
    protected libCode: boolean;
    protected srcPath: string;
    protected linked: boolean;
    protected srcname: string;

    constructor(objpath: string, srcpath: string, linked: boolean, libCode: boolean) {
        this.elfParser = new ElfParser();
        this.elfParser.bindFile(objpath);
        this.setSrcPath(srcpath);
        this.linked = linked;
        this.libCode = libCode;
    }

    setSrcPath(path: string) {
        this.srcPath = path;
    }

    getSrcname() {
        return this.srcname;
    }

    toAddrStr(addr: number) {
        return this.elfParser.toAddrStr(addr);
    }

    start() {
        const srcPath = this.srcPath;
        const basename = srcPath.substring(srcPath.lastIndexOf('\\') + 1, srcPath.length);
        this.srcname = basename.substring(0, basename.search(/(\.c|\.cpp|\.cxx|)$/g));
        const ranges: {begin: Addr; end: Addr}[] = [];
        if (this.linked) {
            const texts = this.elfParser.getSecHeaders((sec: SecHeader) => {
                return this.elfParser.getSecName(sec).startsWith('.text.' + this.srcname);
            });
            ranges.push(...this.elfParser.getRangesOf(texts));
        }
        const maps = this.elfParser.getLineMap((item: LineInfoItem) => {
            if (item.srcpath !== srcPath) {
                return false;
            }
            if (ranges.length === 0) {
                return true;
            }
            for (const range of ranges) {
                if (
                    range.begin <= item.address_start &&
                    item.address_start <= item.address_end &&
                    item.address_end <= range.end
                ) {
                    return true;
                }
            }
            return false;
        });
        const lineMap = new Map<string, Map<string, number>>();
        for (const text of maps.keys()) {
            if (text.startsWith('.text.' + this.srcname) || text === srcPath) {
                const map = maps.get(text);
                assert(map !== undefined && map !== null);
                lineMap.set(text, map);
            }
        }
        return {
            lineMap: lineMap,
            lineSet: new Set<string>(lineMap.keys()),
            relaMap: this.elfParser.getRelaMap(),
            srcPath: srcPath,
        };
    }
}
