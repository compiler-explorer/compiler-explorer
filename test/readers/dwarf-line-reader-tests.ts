import {assert} from '../../lib/assert';
import {DwarfLineReader, LineInfoItem} from '../../lib/tooling/readers/dwarf-line-reader';

describe('test DwarfLineReader', () => {
    const lineReader = new DwarfLineReader();

    it('test lineReader.header', () => {
        lineReader.resetAll();
        const file_name = Array.from('test-cpp.cpp').map(letter => {
            const val = letter.codePointAt(0);
            assert(val !== undefined && val !== null);
            return val;
        });
        const entry = new Uint8Array(
            [72, 0, 0, 0].concat(
                // unit_len <32-bit>
                [3, 0], // version <16-bit>
                [32, 0, 0, 0], // head_len <32-bit>
                [2, 1, -4, 9, 10], // machine initial value <byte[4]>
                [0, 1, 1, 1, 1, 0, 0, 0, 1], // std opcode args <byte[opbase - 1]>
                [0], // include directories <strings with endle 0>
                file_name,
                [0], // filepath <strings with endle 0>
                [0, 0, 0, 0], // file attributes <byte[4]>
                [
                    5, 56, 7, 0, 5, 2, 0, 0, 0, 0, 1, 5, 88, 1, 5, 21, 9, 6, 0, 3, 1, 1, 5, 1, 3, 1, 1, 7, 9, 6, 0, 0,
                    1, 1,
                ], // opcode <byte[?]>
            ),
        );
        lineReader.readEntries(entry);

        const header = lineReader.getHeader();

        header.unit_len.should.equal(72);
        header.version.should.equal(3);
        header.header_len.should.equal(32);

        header.min_inst_len.should.equal(2);
        header.dft_is_stmt.should.equal(1);
        header.line_base.should.equal(-4);
        header.line_range.should.equal(9);
        header.opcode_base.should.equal(10);

        const std_op_argcs = [0, 1, 1, 1, 1, 0, 0, 0, 1];
        header.std_op_lens.length.should.equal(std_op_argcs.length);
        for (let i = 0; i < header.std_op_lens.length; i++) {
            header.std_op_lens[i].should.equal(std_op_argcs[i]);
        }

        header.include_dirs.length.should.equal(0);

        header.file_names.length.should.equal(1);
        for (const file of header.file_names) {
            file.filename.should.equal('test-cpp.cpp');
            file.include_dir.should.equal('');
            file.inc_dir_index.should.equal(0n);
            file.modified_time.should.equal(0n);
            file.file_length.should.equal(0n);
        }

        header.op_begin.should.equal(38);
        header.op_length.should.equal(34);
    });

    it('test lineReader read opcode', () => {
        lineReader.resetAll();
        const file_name = Array.from('test-cpp.cpp').map(letter => {
            const val = letter.codePointAt(0);
            assert(val !== undefined && val !== null);
            return val;
        });
        const opcodes: Array<number> = [
            0,
            5,
            2,
            1,
            2,
            3,
            4, // extop<5>: SET_ADDRESS   0x04030201<uWord>
        ].concat(
            [0, 8, 3, 97, 98, 99, 0, 0, 0, 21], // extop<8> DEFINE_FILE  {name: "abc", dir_index: 0<uLeb128>, mod_time: 0<uLeb128>, length: 21<uLeb128>}
            [
                0,
                2,
                4,
                5, // extop<2>: SET_DEISCRIMINATOR  5<uLeb128>
            ],
            [
                4,
                1, // stdop: SET_FILE    1<uLeb128> {'test-cpp.cpp'}
            ],
            [
                5,
                12, // stdop: SET_COLUMN     12<uLeb128>
                1, // stdop: COPY
            ],
            [
                2,
                4, // stdop: ADVANCE_PC     4<uLeb128>
                1, // stdop: COPY
            ],
            [
                2,
                4, // stdop: ADVANCE_PC       4<uLeb128>
                3,
                1, // stdop: ADVANCE_LINE     1<sLeb128>
                1, // stdop: COPY
            ],
            [
                4,
                2, // stdop: SET_FILE    2<uLeb128> {'abc'}
            ],
            [
                6, // stdop: NEG_STMT
                7, //stdop: SET_BASIC_BLOCK
            ],
            [
                8, // stdop: SPECIAL_255
                1, // stdop: COPY
            ],
            [
                9,
                0,
                1, // stdop: FIX_ADVANCE_PC   256 <uHalf>
            ],
            [
                10, // stdop: SET_PROLOGUE_END
                11, // stdop: SET_EPILOGUE_BEGIN
                12, // stdop: SET_ISA
            ],
            [
                0,
                1,
                1, // extop<1>: END_SEQUENCE
            ],
        );
        const entry = new Uint8Array(
            [38 + opcodes.length, 0, 0, 0].concat(
                // unit_len <32-bit>
                [3, 0], // version <16-bit>
                [32, 0, 0, 0], // head_len <32-bit>
                [2, 1, -4, 9, 10], // machine initial value <byte[4]>
                [0, 1, 1, 1, 1, 0, 0, 0, 1], // std opcode args <byte[opbase - 1]>
                [0], // include directories <strings with endle 0>
                file_name,
                [0], // filepath <strings with endle 0>
                [0, 0, 0, 0], // file attributes <byte[4]>
                // opcode <byte[?]>
                opcodes,
            ),
        );
        lineReader.readEntries(entry);
        const lineInfoItems_expected: LineInfoItem[] = [
            {
                address_start: 0x04030201n,
                address_end: 0x04030209n,
                line: 1,
                column: 12,
                inc_dir: '',
                srcpath: 'test-cpp.cpp',
            },
            {
                address_start: 0x04030209n,
                address_end: 0x04030211n,
                line: 1,
                column: 12,
                inc_dir: '',
                srcpath: 'test-cpp.cpp',
            },
            {
                address_start: 0x04030211n,
                address_end: 0x04030247n,
                line: 2,
                column: 12,
                inc_dir: '',
                srcpath: 'test-cpp.cpp',
            },
            {address_start: 0x04030247n, address_end: 0x04030247n, line: 0, column: 12, inc_dir: '', srcpath: 'abc'},
            {address_start: 0x04030247n, address_end: 0x04030347n, line: 0, column: 12, inc_dir: '', srcpath: 'abc'},
            {address_start: 0x04030347n, address_end: 0x04030347n, line: -4, column: 12, inc_dir: '', srcpath: 'abc'},
            {address_start: 0x04030347n, address_end: 0x04030347n, line: -7, column: 12, inc_dir: '', srcpath: 'abc'},
            {address_start: 0x04030347n, address_end: 0x04030347n, line: -9, column: 12, inc_dir: '', srcpath: 'abc'},
            {address_start: 0x04030347n, address_end: 0x00000000n, line: -9, column: 12, inc_dir: '', srcpath: 'abc'},
        ];
        const lineInfoItems_actual = lineReader.lineInfo();
        lineInfoItems_actual.length.should.equal(lineInfoItems_expected.length);
        for (const [i, lineInfo_actual] of lineInfoItems_actual.entries()) {
            const lineInfo_expected = lineInfoItems_expected[i];
            lineInfo_actual.address_start.should.equal(lineInfo_expected.address_start);
            lineInfo_actual.address_end.should.equal(lineInfo_expected.address_end);
            lineInfo_actual.line.should.equal(lineInfo_expected.line);
            lineInfo_actual.column.should.equal(lineInfo_expected.column);
            lineInfo_actual.inc_dir.should.equal(lineInfo_expected.inc_dir);
            lineInfo_actual.srcpath.should.equal(lineInfo_expected.srcpath);
        }
    });
});
