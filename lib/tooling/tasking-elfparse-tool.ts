import * as fs from 'fs';

import {stringify} from 'qs';

//elf_header struct
class Elf_Header {
    e_ident: Buffer;
    e_shoff: number;
    e_shentsize: number;
    e_shnum: number;
    e_shstrndx: number;
}

//elf_header const off
const elf_header_off = {
    e_identoff: 0,
    e_shoffoff: 32,
    e_shentsizeoff: 46,
    e_shnumoff: 48,
    e_shstrndxoff: 50,
};

class Elf_symtab {
    st_name: number;
    st_value: number;
    st_size: number;
    st_info: number;
    st_other: number;
    st_shndx: number;
}

//elf_section struct
class Elf_Section {
    sh_name: number;
    sh_strname: string;
    sh_offset: number;
    sh_size: number;
    sh_end: number;
}

//elf_section const off
const elf_section_off = {
    sh_nameoff: 0,
    sh_offsetoff: 16,
    sh_sizeoff: 20,
};

//elf_section _Rela
class Elf32_Rela {
    r_offset: number; //Elf32_Addr
    r_info: number; //Elf32_Word
    r_addend: number; //Elf32_Sword
}

//standopcode
enum StandardOpCode {
    DW_LNS_copy = 1,
    DW_LNS_advance_pc = 2,
    DW_LNS_advance_line = 3,
    DW_LNS_set_file = 4,
    DW_LNS_set_column = 5,
    DW_LNS_negate_stmt = 6,
    DW_LNS_set_basic_block = 7,
    DW_LNS_const_add_pc = 8,
    DW_LNS_fixed_advance_pc = 9,
    DW_LNS_set_prologue_end = 10,
    DW_LNS_set_epilogue_begin = 11,
    DW_LNS_set_isa = 12,
}

enum ExtendedOpCode {
    DW_LNE_end_sequence = 1,
    DW_LNE_set_address = 2,
    DW_LNE_define_file = 3,
    DW_LNE_set_discriminator = 4,
}

export class elf_Parse {
    _filename: string;
    _filecontent: any;
    _elf_header: any;
    _elf_section: Elf_Section[];
    _elf_group: Elf_Section[];
    _elf_include_directories: any;
    _elf_fileNameTable: any;
    _elf_opcode_test: Array<number>;
    begin: number;
    _elf_debugLineMap: Map<string, Map<string, number>>;
    _elf_debugLine: Elf_Section[];
    _elf_debugLineSet: Set<string>;
    _elf_examplepathc: string;
    _elf_examplepathcpp: string;
    _elf_RelaMap: Map<string, Map<number, string>>;
    _elf_SymSection: Elf_Section;
    _elf_SymTable: Array<string>;
    _elf_StrSection: Elf_Section;

    constructor(filename: string) {
        this._filename = filename;
        this._filecontent = fs.readFileSync(this._filename);
        this._elf_header = this.parse_elf_header();
        this._elf_section = this.parse_elf_section();
        this._elf_group = [];
        this._elf_include_directories = [];
        this._elf_fileNameTable = [];
        this._elf_opcode_test = [];
        this.begin = 0;
        this._elf_debugLineMap = new Map<string, Map<string, number>>();
        this._elf_debugLine = [];
        this._elf_debugLineSet = new Set<string>();
        this._elf_examplepathc = filename.slice(0, -1) + 'c';
        this._elf_examplepathcpp = filename.slice(0, -1) + 'cpp';
        this._elf_RelaMap = new Map<string, Map<number, string>>();
        this._elf_SymTable = new Array<string>();
    }

    start() {
        this.parse_elf_header();
        this.parse_elf_section();
        this.parse_elf_section_name();
        this.parse_elf_symtab();
        this.parse_elf_group();
        this.parse_debugRela();
    }

    find_endindex(str: Buffer): number {
        const size = str.length;
        let i = 0;
        for (; i < size; i++) {
            if (str[i] === 0x00) {
                return i + 1;
            }
        }
        return i;
    }

    //"./files/cpp_demo.o"
    parse_elf_header(): Elf_Header {
        const elf_header = new Elf_Header();
        elf_header.e_ident = this._filecontent.slice(0, 16);
        elf_header.e_shoff = this._filecontent.readUInt32LE(elf_header_off.e_shoffoff);
        elf_header.e_shentsize = this._filecontent.readUInt16LE(elf_header_off.e_shentsizeoff);
        elf_header.e_shnum = this._filecontent.readUInt16LE(elf_header_off.e_shnumoff);
        elf_header.e_shstrndx = this._filecontent.readUInt16LE(elf_header_off.e_shstrndxoff);

        return elf_header;
    }

    parse_elf_section(): Elf_Section[] {
        const elf_section: Elf_Section[] = [];
        for (let i = 0; i < this._elf_header.e_shnum; i++) {
            const temp = new Elf_Section();
            temp.sh_name = this._filecontent.readUInt32LE(
                this._elf_header.e_shoff + this._elf_header.e_shentsize * i + elf_section_off.sh_nameoff,
            );
            temp.sh_offset = this._filecontent.readUInt32LE(
                this._elf_header.e_shoff + this._elf_header.e_shentsize * i + elf_section_off.sh_offsetoff,
            );
            temp.sh_size = this._filecontent.readUInt32LE(
                this._elf_header.e_shoff + this._elf_header.e_shentsize * i + elf_section_off.sh_sizeoff,
            );
            elf_section.push(temp);
        }
        return elf_section;
    }

    parse_elf_section_name() {
        for (let i = 1; i < this._elf_header.e_shnum; i++) {
            const index = this._elf_section[this._elf_header.e_shstrndx];
            const begining = index.sh_offset + this._elf_section[i].sh_name;
            const end = begining + this.find_endindex(this._filecontent.slice(begining + 1));
            const temp_name = this._filecontent.slice(begining, end).toString();
            if (temp_name === 'group') this._elf_group.push(this._elf_section[i]);
            if (temp_name === '.symtab') this._elf_SymSection = this._elf_section[i];
            if (temp_name === '.strtab') this._elf_StrSection = this._elf_section[i];
            this._elf_section[i].sh_strname = temp_name;
        }
    }

    parse_elf_symtab() {
        this.begin = this._elf_SymSection.sh_offset;
        const size = this._elf_SymSection.sh_size / 16;
        for (let i = 0; i < size; i++) {
            const temp: Elf_symtab = new Elf_symtab();
            temp.st_name = this._filecontent.readUInt32LE(this.begin);
            this.begin += 4;
            temp.st_value = this._filecontent.readUInt32LE(this.begin);
            this.begin += 4;
            temp.st_size = this._filecontent.readUInt32LE(this.begin);
            this.begin += 4;
            temp.st_info = this._filecontent.readUInt8(this.begin++);
            temp.st_other = this._filecontent.readUInt8(this.begin++);
            temp.st_shndx = this._filecontent.readUInt16LE(this.begin);
            this.begin += 2;

            const index = this._elf_StrSection;
            const begining = index.sh_offset + temp.st_name;
            const end = begining + this.find_endindex(this._filecontent.slice(begining + 1));
            const temp_name = this._filecontent.slice(begining, end).toString();

            this._elf_SymTable.push(temp_name);
        }
    }

    parse_elf_group() {
        for (let i = 0; i < this._elf_group.length; i++) {
            //text-section
            this.begin = this._elf_group[i].sh_offset + 4;
            const text_number = this._filecontent.readUInt32LE(this.begin);
            const text_string = this._elf_section[text_number].sh_strname;
            //debug_line-section
            this.begin = this._elf_group[i].sh_offset + 16;
            const line_number = this._filecontent.readUInt32LE(this.begin);
            this._elf_section[line_number].sh_end =
                this._elf_section[line_number].sh_offset + this._elf_section[line_number].sh_size - 1;
            this.parse_debugLine(line_number, text_string);
        }
    }

    parse_debugRela() {
        for (const setion of this._elf_section) {
            if (!setion.sh_strname) continue;
            if (setion.sh_strname.startsWith('.rela.text', 0)) {
                const tempname = setion.sh_strname.substring(5);
                if (this._elf_debugLineSet.has(tempname)) {
                    this.begin = setion.sh_offset;
                    const size = setion.sh_size / 12;
                    this._elf_RelaMap.set(tempname, new Map<number, string>());
                    for (let i = 0; i < size; i++) {
                        const rela = new Elf32_Rela();
                        rela.r_offset = this._filecontent.readUInt32LE(this.begin);
                        this.begin += 4;
                        rela.r_info = this._filecontent.readUInt32LE(this.begin);
                        this.begin += 4;
                        rela.r_addend = this._filecontent.readInt32LE(this.begin);
                        this.begin += 4;

                        const R_SYM = rela.r_info >> 8;
                        const R_TYPE = stringify(rela.r_info);
                        const name_string = this._elf_SymTable[R_SYM];
                        const map = this._elf_RelaMap.get(tempname);
                        if (map) {
                            map.set(rela.r_offset, name_string);
                        }
                    }
                }
            }
        }
    }

    parse_debugLine(line_number: number, text_string: string) {
        this._elf_fileNameTable = [];
        const debugline = this._elf_section[line_number];
        this._elf_debugLine.push(debugline);
        this.begin = debugline.sh_offset;
        const length = this._filecontent.readUInt32LE(this.begin);
        this.begin += 4;
        const version = this._filecontent.readUInt16LE(this.begin);
        this.begin += 2;
        const header_length = this._filecontent.readUInt32LE(this.begin);
        this.begin += 4;
        const minimum_instruction_length = this._filecontent.readInt8(this.begin++);
        const default_is_stmt = this._filecontent.readInt8(this.begin++);
        const line_base = this._filecontent.readInt8(this.begin++);
        const line_range = this._filecontent.readInt8(this.begin++);
        const opcode_base = this._filecontent.readInt8(this.begin++);
        const standard_opcode_lengths = new Array<any>();
        for (let i = 1; i < opcode_base; i++) {
            const opCodeArgumentLength = this._filecontent.readUInt8(this.begin++);
            standard_opcode_lengths.push(opCodeArgumentLength);
        }
        let x = 0;
        do {
            x = this.find_endindex(this._filecontent.slice(this.begin));
            const str = this._filecontent.slice(this.begin, this.begin + x).toString();
            this.begin += x;
            this._elf_include_directories.push(str);
        } while (x !== 1);
        //include_files
        x = 0;
        while (x !== 1) {
            x = this.find_endindex(this._filecontent.slice(this.begin));
            const str = this._filecontent.slice(this.begin, this.begin + x - 1).toString();
            if (x !== 1) {
                this._elf_fileNameTable.push(str);
                this.begin += x;
                const dirIndex = this.readLEB128(false);
                const modifyTime = this.readLEB128(false);
                const fileSize = this.readLEB128(false);
            }
        }
        this.begin++;
        //opcode
        let address = 0;
        let lineNumber = 1;
        let file = 1;
        const end = debugline.sh_end;
        let filename: string = this._elf_fileNameTable[file - 1];
        if (filename === this._elf_examplepathc || filename === this._elf_examplepathcpp) {
            this._elf_debugLineSet.add(text_string);
        }

        while (this.begin <= end) {
            //debugline.sh_offset + length + 1
            if (this.begin > debugline.sh_offset + length + 1) {
                this._elf_section[line_number].sh_offset = this.begin;
                this.parse_debugLine(line_number, text_string);
                return;
            }
            const opcode = this._filecontent.readUInt8(this.begin++);
            let addressIncrease = 0;
            let lineIncrease = 0;
            if (opcode >= opcode_base) {
                addressIncrease = ((opcode - opcode_base) / line_range) * minimum_instruction_length;
                lineIncrease = line_base + ((opcode - opcode_base) % line_range);
                this._elf_opcode_test.push(opcode);
            } else if (opcode > 0) {
                this._elf_opcode_test.push(opcode);
                const opCodeArgumentLength = standard_opcode_lengths[opcode - 1];
                const standardopcode: StandardOpCode = opcode;
                switch (standardopcode) {
                    case StandardOpCode.DW_LNS_copy: {
                        // The opcode is not needed in current example, skip it
                        break;
                    }
                    case StandardOpCode.DW_LNS_advance_pc: {
                        const operand = this.readLEB128(false);
                        addressIncrease = operand * minimum_instruction_length;
                        break;
                    }
                    case StandardOpCode.DW_LNS_advance_line: {
                        const operand = this.readLEB128(true);
                        lineIncrease = operand;
                        break;
                    }
                    case StandardOpCode.DW_LNS_set_file: {
                        file = this.readLEB128(false);
                        filename = this._elf_fileNameTable[file - 1];
                        if (filename === this._elf_examplepathc || filename === this._elf_examplepathcpp) {
                            this._elf_debugLineSet.add(text_string);
                        }
                        break;
                    }
                    case StandardOpCode.DW_LNS_set_column: {
                        const column = this.readLEB128(false);
                        break;
                    }
                    case StandardOpCode.DW_LNS_negate_stmt: {
                        // The opcode is not needed in current example, skip it
                        break;
                    }
                    case StandardOpCode.DW_LNS_set_basic_block: {
                        // The opcode is not needed in current example, skip it
                        break;
                    }
                    case StandardOpCode.DW_LNS_const_add_pc: {
                        addressIncrease = ((255 - opcode_base) / line_range) * minimum_instruction_length;
                        break;
                    }
                    case StandardOpCode.DW_LNS_fixed_advance_pc: {
                        const operand = this._filecontent.readUInt16LE(this.begin);
                        this.begin += 2;
                        addressIncrease = operand;
                        break;
                    }
                }
            } else {
                const commandLength = this.readLEB128(false);
                const subOpcode = this._filecontent.readUInt8(this.begin++);
                const extendedOpCode: ExtendedOpCode = subOpcode;
                this._elf_opcode_test.push(subOpcode);
                switch (extendedOpCode) {
                    case ExtendedOpCode.DW_LNE_end_sequence: {
                        address = 0;
                        lineNumber = 1;
                        file = 1;
                        continue;
                    }
                    case ExtendedOpCode.DW_LNE_set_address: {
                        address = this._filecontent.readUInt32LE(this.begin);
                        this.begin += 4;
                        break;
                    }
                    case ExtendedOpCode.DW_LNE_set_discriminator: {
                        const discriminator = this.readLEB128(false);
                        break;
                    }
                }
            }
            if (lineIncrease !== 0 || addressIncrease !== 0) {
                address = address + addressIncrease;
                lineNumber = lineNumber + lineIncrease;
            }
            if (lineIncrease !== 0) {
                this.set_map(filename, address, text_string, lineNumber);
            }
        }
    }

    set_map(filename: string, address: number, text_string: string, lineNumber: number) {
        //filename === this._elf_examplepathc || filename === this._elf_examplepathcpp
        if (filename === this._elf_examplepathc || filename === this._elf_examplepathcpp) {
            let tempaddress = address.toString(16);
            for (let i = tempaddress.length; i < 8; i++) {
                tempaddress = '0' + tempaddress;
            }
            if (text_string === '') {
                text_string = filename;
            }
            if (!this._elf_debugLineMap.has(text_string)) {
                this._elf_debugLineMap.set(text_string, new Map<string, number>());
            }
            const map = this._elf_debugLineMap.get(text_string);
            if (map) {
                map.set(tempaddress, lineNumber);
            }
        }
    }

    readLEB128(signedInt: boolean, maxBits = 64) {
        // assert(maxBits <= 64U && "maxBits longer than 64 bits"); // GCOVR_EXCL_LINE
        let result = 0;
        let bitsWritten = 0;
        let byte = 0xff;
        while ((byte & 0x80) !== 0) {
            // byte = getNumber<uint8_t>();
            byte = this._filecontent.readUInt8(this.begin++);
            const lowByte = byte & 0x7f;
            result |= lowByte << bitsWritten;
            bitsWritten = bitsWritten + 7;
            if (bitsWritten > maxBits) {
                // More bits written than allowed
                if (signedInt && (byte & (1 << (6 - (bitsWritten - maxBits)))) !== 0) {
                    // If it is signed and negative (sign bit set) "1" padding allowed
                    const bitMask = (0xff << (6 - (bitsWritten - maxBits) + 1)) & 0b01111111;
                    if ((byte & bitMask) !== bitMask) {
                        // throw std::runtime_error("Malformed LEB128 signed integer (Wrong padding)\n");
                    }
                } else {
                    // Zero padding allowed if unsigned or positive signed integer
                    const bitMask = (0xff << (6 - (bitsWritten - maxBits) + 1)) & 0b01111111;
                    if ((byte & bitMask) !== 0) {
                        // throw std::runtime_error("Malformed LEB128 unsigned integer (Wrong padding)\n");
                    }
                }
            }
        }
        if (signedInt && (byte & 0x40) !== 0 && bitsWritten < 64) {
            // bug...
            const signExtensionMask = 0xfffffffffffff << bitsWritten;
            result |= signExtensionMask;
        }
        return result;
    }
}
