import {assert} from 'console';

import {BytesReader} from './byte-reader';
import {Addr, ByteArray, sByte, uByte, uHalf, uLeb128, uWord} from './elf-types';

export interface FileItem {
    filename: string;
    include_dir: string;
    inc_dir_index: uLeb128;
    modified_time: uLeb128;
    file_length: uLeb128;
}

export interface LineNumProgHeader {
    unit_len: uWord;
    version: uHalf;
    header_len: uWord;
    min_inst_len: uByte;
    dft_is_stmt: uByte;
    line_base: sByte;
    line_range: uByte;
    opcode_base: uByte;
    std_op_lens: ByteArray;
    include_dirs: Array<string>;
    file_names: Array<FileItem>;
    op_begin: number;
    op_length: number;
}

enum DW_LNS {
    DW_LNS_copy = 1,
    DW_LNS_advance_pc = 2,
    DW_LNS_advance_line = 3,
    DW_LNS_set_file = 4,
    DW_LNS_set_column = 5,
    DW_LNS_neg_stmt = 6,
    DW_LNS_set_basic_block = 7,
    DW_LNS_const_add_pc = 8,
    DW_LNS_fix_advance_pc = 9,
    DW_LNS_set_prologue_end = 10,
    DW_LNS_set_epilogue_begin = 11,
    DW_LNS_set_isa = 12,
}

enum DW_LNE {
    DW_LNE_end_sequence = 1,
    DW_LNE_set_address = 2,
    DW_LNE_define_file = 3,
    DW_LNE_set_discriminator = 4,
}

export interface LineInfoItem {
    address_start: bigint;
    address_end: bigint;
    inc_dir: string;
    srcpath: string;
    line: number;
    column: number;
}

const DEFAULT_IS_STMT = false;

interface LineRegisters {
    address: Addr;
    file: uWord;
    line: uWord;
    column: uWord;
    is_stmt: boolean;
    basic_block: boolean;
    end_sequence: boolean;
    prologue_end: boolean;
    epilogue_begin: boolean;
    isa: uWord;
    descreminator: uWord;
}

export class DwarfLineReader {
    protected header: LineNumProgHeader;
    protected registers: LineRegisters = {
        address: 0n,
        file: 1,
        line: 1,
        column: 0,
        is_stmt: DEFAULT_IS_STMT,
        basic_block: false,
        end_sequence: false,
        prologue_end: false,
        epilogue_begin: false,
        isa: 0,
        descreminator: 0,
    };
    protected lineItemList: LineInfoItem[] = [];
    protected reader: BytesReader = new BytesReader();
    protected entries: {data: Uint8Array; size: uWord}[] = [];

    readEntries(data: Uint8Array) {
        assert(data.length > 0);
        this.reader.bind(data);
        while (!this.reader.isEnd()) {
            const unit_len = this.reader.readWord();
            const entry_data = this.reader.read(unit_len);
            this.entries.push({data: entry_data, size: unit_len});
        }
        for (const entry of this.entries) {
            this.readEntry(entry);
        }
    }

    protected readEntry(entry: {data: Uint8Array; size: uWord}) {
        assert(entry.data.length > 0);
        this.reader.bind(entry.data);
        const lino_header: LineNumProgHeader = {
            unit_len: entry.size,
            version: this.reader.readHalf(),
            header_len: this.reader.readWord(),
            min_inst_len: this.reader.readByte(),
            dft_is_stmt: this.reader.readByte(),
            line_base: this.reader.readSByte(),
            line_range: this.reader.readByte(),
            opcode_base: this.reader.readByte(),
            std_op_lens: new Uint8Array(),
            include_dirs: new Array<string>(),
            file_names: new Array<FileItem>(),
            op_begin: 0,
            op_length: 0,
        };
        this.header = lino_header;
        const op_cnt_list = this.reader.read(lino_header.opcode_base - 1);
        lino_header.std_op_lens = op_cnt_list;
        let path: string = this.reader.readString();
        while (path) {
            lino_header.include_dirs.push(path);
            path = this.reader.readString();
        }
        let file_item: FileItem | null = this.readFileItem();
        while (file_item) {
            lino_header.file_names.push(file_item);
            file_item = this.readFileItem();
        }
        lino_header.op_begin = lino_header.header_len + 4 + 2;
        lino_header.op_length = lino_header.unit_len - lino_header.op_begin;
        if (lino_header.op_begin === 0) {
            return;
        }
        this.execLineProgram();
    }

    protected readFileItem(): FileItem | null {
        const name = this.reader.readString();
        if (!name) {
            return null;
        }
        const idx = this.reader.readULeb128();
        const dirs = this.header.include_dirs;
        const inc_dir = idx > dirs.length ? dirs[Number(idx)] : '';
        const file_item: FileItem = {
            filename: name,
            include_dir: inc_dir,
            inc_dir_index: idx,
            modified_time: this.reader.readULeb128(),
            file_length: this.reader.readULeb128(),
        };
        return file_item;
    }

    protected resetRegisters() {
        this.registers.address = 0n;
        this.registers.file = 1;
        this.registers.line = 1;
        this.registers.column = 0;
        this.registers.is_stmt = DEFAULT_IS_STMT;
        this.registers.basic_block = false;
        this.registers.end_sequence = false;
        this.registers.prologue_end = false;
        this.registers.epilogue_begin = false;
        this.registers.isa = 0;
        this.registers.descreminator = 0;
    }
    protected composeLineItem() {
        if (this.lineItemList.length > 0) {
            const item = this.lineItemList[this.lineItemList.length - 1];
            item.address_end = BigInt.asUintN(32, this.registers.address);
        }
        const file_item = this.header.file_names[this.registers.file - 1];
        const inc_dir = file_item.inc_dir_index ? this.header.include_dirs[Number(file_item.inc_dir_index)] : '';
        const item: LineInfoItem = {
            address_start: BigInt.asUintN(32, this.registers.address),
            address_end: 0n,
            line: this.registers.line,
            srcpath: file_item.filename,
            inc_dir: inc_dir,
            column: this.registers.column,
        };
        this.lineItemList.push(item);
    }

    protected execExtOperation(inst_len: uByte, opcode: uByte) {
        switch (opcode) {
            case DW_LNE.DW_LNE_end_sequence: {
                this.composeLineItem();
                this.registers.end_sequence = true;
                this.resetRegisters();
                return 0;
            }
            case DW_LNE.DW_LNE_set_address: {
                const operand = inst_len === 4 ? this.reader.readWord() : this.reader.readWord(); // readLong
                this.registers.address = BigInt.asUintN(32, BigInt(operand));
                return inst_len;
            }
            case DW_LNE.DW_LNE_define_file: {
                const file = this.readFileItem();
                if (file) {
                    this.header.file_names.push(file);
                }
                return inst_len;
            }
            case DW_LNE.DW_LNE_set_discriminator: {
                const operand = this.reader.readULeb128();
                this.registers.descreminator = Number(operand);
                return inst_len;
            }
        }
        throw new Error('it should not be reached.');
    }

    protected execStdOperation(opcode: uByte) {
        // const operand_count = this.header.std_op_lens[opcode - 1];
        switch (opcode) {
            case DW_LNS.DW_LNS_copy: {
                this.composeLineItem();
                this.registers.basic_block = false;
                this.registers.prologue_end = false;
                this.registers.epilogue_begin = false;
                this.registers.descreminator = 0;
                return 1;
            }
            case DW_LNS.DW_LNS_advance_pc: {
                const adv_pc = this.reader.readULeb128();
                const __num = BigInt.asUintN(32, BigInt(this.header.min_inst_len));
                this.registers.address += adv_pc * __num;
                return 1 + this.reader.readed_size();
            }
            case DW_LNS.DW_LNS_advance_line: {
                this.registers.line += Number(this.reader.readSLeb128());
                return 1 + this.reader.readed_size();
            }
            case DW_LNS.DW_LNS_set_file: {
                this.registers.file = Number(this.reader.readULeb128());
                return 1 + this.reader.readed_size();
            }
            case DW_LNS.DW_LNS_set_column: {
                this.registers.column = Number(this.reader.readULeb128());
                return 1 + this.reader.readed_size();
            }
            case DW_LNS.DW_LNS_neg_stmt: {
                this.registers.is_stmt = !this.registers.is_stmt;
                return 1;
            }
            case DW_LNS.DW_LNS_set_basic_block: {
                this.registers.basic_block = true;
                return 1;
            }
            case DW_LNS.DW_LNS_const_add_pc: {
                this.execSpeOperation(255);
                return 1;
            }
            case DW_LNS.DW_LNS_fix_advance_pc: {
                const operand = this.reader.readHalf();
                this.registers.address += BigInt.asUintN(32, BigInt(operand));
                return 1 + 2;
            }
            case DW_LNS.DW_LNS_set_prologue_end: {
                this.registers.prologue_end = true;
                return 1;
            }
            case DW_LNS.DW_LNS_set_epilogue_begin: {
                this.registers.epilogue_begin = true;
                return 1;
            }
            case DW_LNS.DW_LNS_set_isa: {
                this.registers.isa = Number(this.reader.readULeb128());
                return 1 + this.reader.readed_size();
            }
        }
        throw new Error('it should not be reached.');
    }

    protected execSpeOperation(opcode: uByte) {
        const adjust_opcode = opcode - this.header.opcode_base;
        const addr_advance = Math.floor(adjust_opcode / this.header.line_range) * this.header.min_inst_len;
        const line_increment = this.header.line_base + (adjust_opcode % this.header.line_range);

        this.registers.line += line_increment;
        this.registers.address += BigInt.asUintN(32, BigInt(addr_advance));
        this.registers.basic_block = false;
        this.registers.prologue_end = false;
        this.registers.epilogue_begin = false;

        this.composeLineItem();
        return 1;
    }

    protected execLineProgram() {
        this.reader.seek(this.header.op_begin);
        for (let i = 0; i < this.header.op_length; ) {
            const inst_header = this.reader.readByte();
            if (inst_header === 0) {
                let inst_len = this.reader.readByte();
                const opcode = this.reader.readByte();
                inst_len -= 1;
                const len = this.execExtOperation(inst_len, opcode);
                if (len <= 0) {
                    break;
                }
                i += 1 + inst_len;
            } else if (inst_header < this.header.opcode_base) {
                const opcode = inst_header;
                i += this.execStdOperation(opcode);
            } else {
                const opcode = inst_header;
                i += this.execSpeOperation(opcode);
            }
        }
    }
    public resetAll() {
        this.resetRegisters();
        this.clearItems();
        this.entries = [];
    }

    public clearItems() {
        this.lineItemList = [];
    }

    public lineInfo() {
        return this.lineItemList;
    }

    public fileInfo() {
        return this.header.file_names;
    }

    public getHeader() {
        const header: LineNumProgHeader = {
            unit_len: this.header.unit_len,
            version: this.header.version,
            header_len: this.header.header_len,
            min_inst_len: this.header.min_inst_len,
            dft_is_stmt: this.header.dft_is_stmt,
            line_base: this.header.line_base,
            line_range: this.header.line_range,
            opcode_base: this.header.opcode_base,
            std_op_lens: this.header.std_op_lens,
            include_dirs: this.header.include_dirs,
            file_names: this.header.file_names,
            op_begin: this.header.op_begin,
            op_length: this.header.op_length,
        };
        return header;
    }
}
