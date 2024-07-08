import {assert, warn} from 'console';

import {BytesReader} from './byte-reader';
import {Addr, ByteArray, sByte, signedLeb128, sWord, uByte, uHalf, uLeb128, uWord} from './elf-types';

type Elf32_Byte = uByte;
type Elf32_Half = uHalf;
type Elf32_Word = uWord;
type Elf32_Sword = sWord;
type Elf32_Addr = Addr;
type Elf32_Off = Addr;
type Elf32_ByteArray<l> = ByteArray;

interface ElfHeader {
    e_ident: Elf32_ByteArray<16>;
    e_type: Elf32_Half;
    e_machine: Elf32_Half;
    e_version: Elf32_Word;
    e_entry: Elf32_Addr;
    e_phoff: Elf32_Off;
    e_shoff: Elf32_Off;
    e_flags: Elf32_Word;
    e_ehsize: Elf32_Half;
    e_phentsize: Elf32_Half;
    e_phnum: Elf32_Half;
    e_shentsize: Elf32_Half;
    e_shnum: Elf32_Half;
    e_shstrndx: Elf32_Half;
}

enum E_TYPE {
    ET_NONE = 0,
    ET_REL = 1,
    ET_EXEC = 2,
    ET_DYN = 3,
    ET_CORE = 4,
    ET_LOPROC = 0xff00,
    ET_HIPROC = 0xffff,
}

enum E_MACHINE {
    ET_NONE = 0,
    EM_M32 = 1,
    EM_SPARC = 2,
    EM_386 = 3,
    EM_68K = 4,
    EM_88K = 5,
    EM_860 = 7,
    EM_MIPS = 8,
    EM_MIPS_RS4_BE = 10,

    EM_RESERVED_START = 11,
    EM_RESERVED_END = 16,
}

enum E_VERSION {
    EV_NONE = 0,
    EV_CURRENT = 1,
}

enum E_IDENT_NDX {
    EI_MAG0 = 0,
    EI_MAG1 = 1,
    EI_MAG2 = 2,
    EI_MAG3 = 3,
    EI_CLASS = 4,
    EI_DATA = 5,
    EI_VERSION = 6,
    EI_PAD = 7,
    EI_NIDENT = 16,
}

enum EI_CLASS {
    ELFCLASSNONE = 0,
    ELFCLASS32 = 1,
    ELFCLASS64 = 2,
}

enum EI_DATA {
    ELFDATANONE = 0,
    ELFDATA2LSB = 1,
    ELFDATA2MSB = 2,
}

type EI_VERSION = E_VERSION;

enum SH_NDX {
    SHN_UNDEF = 0,
    SHN_LORESERVE = 0xff00,
    SHN_LOPROC = 0xff00,
    SHN_HIPROC = 0xff1f,
    SHN_ABS = 0xfff1,
    SHN_COMMON = 0xfff2,
    SHN_HIRESERVE = 0xffff,
}

export interface SecHeader {
    sh_name: Elf32_Word;
    sh_type: Elf32_Word;
    sh_flags: Elf32_Word;
    sh_addr: Elf32_Addr;
    sh_offset: Elf32_Off;
    sh_size: Elf32_Word;
    sh_link: Elf32_Word;
    sh_info: Elf32_Word;
    sh_addralign: Elf32_Word;
    sh_entsize: Elf32_Word;
}

export enum SH_TYPE {
    SHT_NULL = 0,
    SHT_PROGBITS = 1,
    SHT_SYMTAB = 2,
    SHT_STRTAB = 3,
    SHT_RELA = 4,
    SHT_HASH = 5,
    SHT_DYNAMIC = 6,
    SHT_NOTE = 7,
    SHT_NOBITS = 8,
    SHT_REL = 9,
    SHT_SHLIB = 10,
    SHT_DYNSYM = 11,

    // extended section type: group
    SHT_GROUP = 17,
    // tasking specified: .tasking.callinfo
    SHT_TSK_CALL = 0x7f000001,

    SHT_LOPROC = 0x70000000,
    SHT_HIPROC = 0x7fffffff,
    SHT_LOUSER = 0x80000000,
    SHT_HIUSER = 0xffffffff,
}

export enum SH_FLAGS {
    SHF_WRITE = 0x1,
    SHF_ALLOC = 0x2,
    SHF_EXECINSTR = 0x4,
    SHF_MASKPROC = 0xf0000000,
}

export interface SymEntry {
    st_name: Elf32_Word;
    st_value: Elf32_Addr;
    st_size: Elf32_Word;
    st_info: Elf32_Byte;
    st_other: Elf32_Byte;
    st_shndx: Elf32_Half;
}

enum ST_BIND {
    STB_LOCAL = 0,
    STB_GLOBAL = 1,
    STB_WEAK = 2,
    STB_LOPROC = 13,
    STB_HIPROC = 15,
}

enum ST_TYPE {
    STT_NOTYPE = 0,
    STT_OBJECT = 1,
    STT_FUNC = 2,
    STT_SECTION = 3,
    STT_FILE = 4,
    STT_LOPROC = 13,
    STT_HIPROC = 15,
}

interface RelaEntry {
    r_offset: Elf32_Addr;
    r_info: Elf32_Word;
    r_addend: Elf32_Sword;
}

interface GroupEntry {
    g_comdat: uHalf;
    g_size: uWord;
    g_sections: uWord[];
}

const ELFMAG: string = String.fromCodePoint(0x7f) + 'ELF';

export class ElfReader {
    protected reader: BytesReader = new BytesReader();
    protected _file_content: Uint8Array;
    protected header: ElfHeader;
    protected sh_table: SecHeader[];
    protected sh_str_table: Addr;
    protected str_table: Addr;
    protected sym_table: SymEntry[] = [];
    protected rel_maps: Record<string, RelaEntry[]> = {};
    protected group_entries: GroupEntry[] = [];

    public readElf(file_content: Uint8Array) {
        this.reader.bind(file_content);
        const magics = this.reader.read(4).toString();
        assert(magics === ELFMAG);
        this._file_content = file_content;
        this.readElfHeader();
        this.readElfSecTable();
        this.auxSections();
    }
    protected readElfHeader() {
        this.reader.seek(0);
        this.header = {
            e_ident: this.reader.read(16),
            e_type: this.reader.readHalf(),
            e_machine: this.reader.readHalf(),
            e_version: this.reader.readWord(),
            e_entry: BigInt.asUintN(32, BigInt(this.reader.readWord())),
            e_phoff: BigInt.asUintN(32, BigInt(this.reader.readWord())),
            e_shoff: BigInt.asUintN(32, BigInt(this.reader.readWord())),
            e_flags: this.reader.readWord(),
            e_ehsize: this.reader.readHalf(),
            e_phentsize: this.reader.readHalf(),
            e_phnum: this.reader.readHalf(),
            e_shentsize: this.reader.readHalf(),
            e_shnum: this.reader.readHalf(),
            e_shstrndx: this.reader.readHalf(),
        };
    }
    protected readElfSecTable() {
        this.reader.seek(Number(this.header.e_shoff));
        const sh_table: SecHeader[] = [];
        for (let i = 0; i < this.header.e_shnum; i++) {
            sh_table.push(this.readSecHeader());
        }
        this.sh_table = sh_table;
        this.setSecNameTable();
    }
    protected readSecHeader(seek?: number) {
        if (seek) {
            this.reader.seek(seek);
        }
        const sec_header: SecHeader = {
            sh_name: this.reader.readWord(),
            sh_type: this.reader.readWord(),
            sh_flags: this.reader.readWord(),
            sh_addr: BigInt.asUintN(32, BigInt(this.reader.readWord())),
            sh_offset: BigInt.asUintN(32, BigInt(this.reader.readWord())),
            sh_size: this.reader.readWord(),
            sh_link: this.reader.readWord(),
            sh_info: this.reader.readWord(),
            sh_addralign: this.reader.readWord(),
            sh_entsize: this.reader.readWord(),
        };
        return sec_header;
    }
    protected auxSections() {
        for (const sec_header of this.sh_table) {
            switch (sec_header.sh_type) {
                case SH_TYPE.SHT_NULL: {
                    continue;
                }
                case SH_TYPE.SHT_PROGBITS: {
                    continue;
                }
                case SH_TYPE.SHT_SYMTAB: {
                    this.saveSymbols(sec_header);
                    break;
                }
                case SH_TYPE.SHT_STRTAB: {
                    this.setStrTable(sec_header);
                    break;
                }
                case SH_TYPE.SHT_RELA: {
                    this.addAddendRelocations(sec_header);
                    break;
                }
                case SH_TYPE.SHT_HASH: {
                    continue;
                }
                case SH_TYPE.SHT_DYNAMIC: {
                    continue;
                }
                case SH_TYPE.SHT_NOTE: {
                    continue;
                }
                case SH_TYPE.SHT_NOBITS: {
                    continue;
                }
                case SH_TYPE.SHT_REL: {
                    this.addRelocations(sec_header);
                    break;
                }
                case SH_TYPE.SHT_SHLIB: {
                    continue;
                }
                case SH_TYPE.SHT_DYNSYM: {
                    continue;
                }
                case SH_TYPE.SHT_GROUP: {
                    this.addGroup(sec_header);
                    break;
                }
                case SH_TYPE.SHT_LOPROC: {
                    continue;
                }
                case SH_TYPE.SHT_HIPROC: {
                    continue;
                }
                case SH_TYPE.SHT_LOUSER: {
                    continue;
                }
                case SH_TYPE.SHT_HIUSER: {
                    continue;
                }
                case SH_TYPE.SHT_TSK_CALL: {
                    continue;
                }
                default: {
                    const type = sec_header.sh_type;
                    const name = this.sh_str_table ? this.readSecName(sec_header) : '';
                    warn(`Unknown section: {name: ${name}, type: 0x${type.toString(16)}}`);
                }
            }
        }
    }
    protected addGroup(sec_header: SecHeader) {
        const start = sec_header.sh_offset;
        const size = sec_header.sh_size;
        const ent_size = sec_header.sh_entsize;
        const ent_count = size / ent_size;
        this.reader.seek(Number(start));
        const sections: uWord[] = [];
        for (let i = 0; i < ent_count; i++) {
            sections.push(this.reader.readWord());
        }
        const grp_entry: GroupEntry = {
            g_comdat: 0,
            g_size: ent_size,
            g_sections: sections,
        };
        this.group_entries.push(grp_entry);
    }
    protected addRelToMap(rela_table: SecHeader) {}
    protected addRelocations(rel_table: SecHeader) {
        const rel_name = this.readSecName(rel_table).substring(5);
        let rel_list = this.rel_maps[rel_name];
        if (!rel_list) {
            rel_list = [];
            this.rel_maps[rel_name] = rel_list;
        }
        const start = Number(rel_table.sh_offset);
        const steps = rel_table.sh_entsize;
        for (let i = 0; i < rel_table.sh_size; i += steps) {
            this.reader.seek(start + i);
            rel_list.push(this.readRelEntry());
        }
    }
    protected addAddendRelocations(rela_table: SecHeader) {
        const rela_name = this.readSecName(rela_table).substring(5);
        let rela_list = this.rel_maps[rela_name];
        if (!rela_list) {
            rela_list = [];
            this.rel_maps[rela_name] = rela_list;
        }
        const start = Number(rela_table.sh_offset);
        const steps = rela_table.sh_entsize;
        for (let i = 0; i < rela_table.sh_size; i += steps) {
            this.reader.seek(start + i);
            rela_list.push(this.readRelaEntry());
        }
    }
    protected readRelaEntry() {
        const rela_entry: RelaEntry = {
            r_offset: BigInt.asUintN(32, BigInt(this.reader.readWord())),
            r_info: this.reader.readWord(),
            r_addend: this.reader.readSWord(),
        };
        return rela_entry;
    }
    protected readRelEntry() {
        const rel_entry: RelaEntry = {
            r_offset: BigInt.asUintN(32, BigInt(this.reader.readWord())),
            r_info: this.reader.readWord(),
            r_addend: 0,
        };
        return rel_entry;
    }
    protected saveSymbols(sym_table: SecHeader) {
        const start = Number(sym_table.sh_offset);
        const steps = sym_table.sh_entsize;
        for (let i = 0; i < sym_table.sh_size; i += steps) {
            this.reader.seek(start + i);
            const sym_entry = this.readSymEntry();
            this.sym_table.push(sym_entry);
        }
    }
    protected readSymEntry() {
        const sym_entry: SymEntry = {
            st_name: this.reader.readWord(),
            st_value: BigInt.asUintN(32, BigInt(this.reader.readWord())),
            st_size: this.reader.readWord(),
            st_info: this.reader.readByte(),
            st_other: this.reader.readByte(),
            st_shndx: this.reader.readHalf(),
        };
        return sym_entry;
    }
    protected setSecNameTable() {
        const sh_str_table = this.sh_table[this.header.e_shstrndx];
        this.sh_str_table = sh_str_table.sh_offset;
    }
    protected setStrTable(str_table: SecHeader) {
        this.str_table = str_table.sh_offset;
    }
    public secHeader(shndx: uWord) {
        return this.sh_table[shndx];
    }
    public readSecName(sec_header: SecHeader) {
        this.reader.seek(Number(this.sh_str_table) + sec_header.sh_name);
        return this.reader.readString();
    }
    public readSymName(sym_entry: SymEntry) {
        this.reader.seek(Number(this.str_table) + sym_entry.st_name);
        return this.reader.readString();
    }
    public readRelTargetName(rel_entry: RelaEntry) {
        const sym_ndx = this.elf32_r_sym(rel_entry.r_info);
        const sym_entry = this.sym_table[sym_ndx];
        return this.readSymName(sym_entry);
    }
    public getContentsOf(sec_headers: SecHeader[]) {
        const contents: ByteArray[] = [];
        for (const sec_header of sec_headers) {
            const size = sec_header.sh_size;
            this.reader.seek(Number(sec_header.sh_offset));
            contents.push(this.reader.read(size));
        }
        return contents;
    }
    public findSecsBy(filter: (sec_header: SecHeader) => boolean) {
        const headers: SecHeader[] = [];
        for (const sec_header of this.sh_table) {
            if (filter(sec_header)) {
                headers.push(sec_header);
            }
        }
        return headers;
    }
    public getSecsOf(group: GroupEntry, filter: (shndx: uWord) => boolean) {
        const shndxs = group.g_sections.filter(filter);
        const sec_headers: SecHeader[] = [];
        for (const shndx of shndxs) {
            sec_headers.push(this.sh_table[shndx]);
        }
        return sec_headers;
    }
    public getDbgLineSecsOf(group: GroupEntry) {
        return this.getSecsOf(group, (shndx: uWord) => {
            const sec_name = this.readSecName(this.sh_table[shndx]);
            return sec_name.startsWith('.debug_line');
        });
    }
    public getTextSecsOf(group: GroupEntry) {
        return this.getSecsOf(group, (shndx: uWord) => {
            const sec_name = this.readSecName(this.sh_table[shndx]);
            return sec_name.startsWith('.text');
        });
    }
    public getRelaocations() {
        return this.rel_maps;
    }
    public readRelaEnties(rela: SecHeader) {
        const target = rela.sh_info;
        const symtab = rela.sh_link;
        const entries: RelaEntry[] = [];
        const start = Number(rela.sh_offset);
        const steps = rela.sh_entsize;
        this.reader.seek(start);
        let readMethord: () => RelaEntry;
        if (rela.sh_type === SH_TYPE.SHT_REL) {
            readMethord = this.readRelEntry;
        } else if (rela.sh_type === SH_TYPE.SHT_RELA) {
            readMethord = this.readRelaEntry;
        } else {
            return {target: target, symtab: symtab, entries};
        }
        for (let i = 0; i < rela.sh_size; i += steps) {
            entries.push(readMethord());
        }
        return {target: target, symtab: symtab, entries: entries};
    }
    public readSymEntries(symtab: SecHeader) {
        const symbols: SymEntry[] = [];
        const start = Number(symtab.sh_offset);
        const steps = symtab.sh_entsize;
        for (let i = 0; i < symtab.sh_size; i += steps) {
            this.reader.seek(start + i);
            const sym_entry = this.readSymEntry();
            symbols.push(sym_entry);
        }
        return symbols;
    }
    public getRelaSymbols(secs: SecHeader[]) {
        const relas = this.findSecsBy((sec: SecHeader) => {
            return sec.sh_type === SH_TYPE.SHT_REL || sec.sh_type === SH_TYPE.SHT_RELA;
        });
        const sec_with_rela_symbols: {sec: SecHeader; symbols: SymEntry[]}[] = [];
        for (const sec of secs) {
            const relas_of_sec = relas.filter((sec: SecHeader) => {
                const target = this.secHeader(sec.sh_info);
                return target.sh_offset === sec.sh_offset;
            });
            const symbols: SymEntry[] = [];
            for (const rela of relas_of_sec) {
                const record = this.readRelaEnties(rela);
                const st_header = this.secHeader(record.symtab);
                const symtab = this.readSymEntries(st_header);
                for (const entry of record.entries) {
                    const symbol = symtab[this.elf32_r_sym(entry.r_info)];
                    symbols.push(symbol);
                }
            }
            sec_with_rela_symbols.push({sec: sec, symbols: symbols});
        }
        return sec_with_rela_symbols;
    }
    getSecHeaders(filter: (sec: SecHeader) => boolean) {
        return this.sh_table.filter(filter);
    }

    public getGroups() {
        return this.group_entries;
    }

    public elf32_st_bind(i: Elf32_Byte) {
        return i >> 4;
    }
    public elf32_st_type(i: Elf32_Byte) {
        return i & 0xf;
    }
    public elf32_st_info(b: Elf32_Byte, t: Elf32_Byte) {
        return (b << 4) | (t & 0xf);
    }

    public elf32_r_sym(i: Elf32_Byte) {
        return i >> 8;
    }
    public elf32_r_type(i: Elf32_Byte) {
        return i & 0xff;
    }
    public elf32_r_info(b: Elf32_Byte, t: Elf32_Byte) {
        return (b << 8) | (t & 0xff);
    }
}
