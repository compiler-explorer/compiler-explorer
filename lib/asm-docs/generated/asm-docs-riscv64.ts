import {AssemblyInstructionInfo} from '../base.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
        case "ADD":
            return {
                "html": "<div><span class=\"opcode\"><b>ADD</b> rd, rs1, rs2</span><br><div><b>ADD</b> performs the addition of rs1 and rs2. SUB performs the subtraction of rs2 from rs1. Overflows are ignored and the low XLEN bits of results are written to the destination rd. SLT and SLTU perform signed and unsigned compares respectively, writing 1 to rd if rs1 < rs2, 0 otherwise. Note, SLTU rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs). AND, OR, and XOR perform bitwise logical operations.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "ADD performs the addition of rs1 and rs2. SUB performs the subtraction of rs2 from rs1. Overflows are ignored and the low XLEN bits of results are written to the destination rd. SLT and SLTU perform signed and unsigned compares respectively, writing 1 to rd if rs1 < rs2, 0 otherwise. Note, SLTU rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs). AND, OR, and XOR perform bitwise logical operations.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-register-operations"
            };

        case "ADD.UW":
            return {
                "html": "<div><span class=\"opcode\"><b>ADD.UW</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "ADDI":
            return {
                "html": "<div><span class=\"opcode\"><b>ADDI</b> rd, rs1, imm12</span><br><div><b>ADDI</b> adds the sign-extended 12-bit immediate to register rs1. Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result. <b>ADDI</b> rd, rs1, 0 is used to implement the MV rd, rs1 assembler pseudoinstruction.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "ADDI adds the sign-extended 12-bit immediate to register rs1. Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result. ADDI rd, rs1, 0 is used to implement the MV rd, rs1 assembler pseudoinstruction.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-immediate-instructions"
            };

        case "ADDIW":
            return {
                "html": "<div><span class=\"opcode\"><b>ADDIW</b> rd, rs1, imm12</span><br><div><b>ADDIW</b> is an RV64I instruction that adds the sign-extended 12-bit immediate to register rs1 and produces the proper sign-extension of a 32-bit result in rd. Overflows are ignored and the result is the low 32 bits of the result sign-extended to 64 bits. Note, <b>ADDIW</b> rd, rs1, 0 writes the sign-extension of the lower 32 bits of register rs1 into register rd (assembler pseudoinstruction SEXT.W).</div><br><div><b>ISA</b>: rv64</div></div>",
                "tooltip": "ADDIW is an RV64I instruction that adds the sign-extended 12-bit immediate to register rs1 and produces the proper sign-extension of a 32-bit result in rd. Overflows are ignored and the result is the low 32 bits of the result sign-extended to 64 bits. Note, ADDIW rd, rs1, 0 writes the sign-extension of the lower 32 bits of register rs1 into register rd (assembler pseudoinstruction SEXT.W).\n\n\n\n(ISA: rv64)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv64.html#integer-register-immediate-instructions"
            };

        case "ADDW":
            return {
                "html": "<div><span class=\"opcode\"><b>ADDW</b> rd, rs1, rs2</span><br><div><b>ADDW</b> and SUBW are RV64I-only instructions that are defined analogously to ADD and SUB but operate on 32-bit values and produce signed 32-bit results. Overflows are ignored, and the low 32-bits of the result is sign-extended to 64-bits and written to the destination register.</div><br><div><b>ISA</b>: rv64</div></div>",
                "tooltip": "ADDW and SUBW are RV64I-only instructions that are defined analogously to ADD and SUB but operate on 32-bit values and produce signed 32-bit results. Overflows are ignored, and the low 32-bits of the result is sign-extended to 64-bits and written to the destination register.\n\n\n\n(ISA: rv64)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-register-operations"
            };

        case "AES32DSI":
            return {
                "html": "<div><span class=\"opcode\"><b>AES32DSI</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "AES32DSMI":
            return {
                "html": "<div><span class=\"opcode\"><b>AES32DSMI</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "AES32ESI":
            return {
                "html": "<div><span class=\"opcode\"><b>AES32ESI</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "AES32ESMI":
            return {
                "html": "<div><span class=\"opcode\"><b>AES32ESMI</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "AES64DS":
            return {
                "html": "<div><span class=\"opcode\"><b>AES64DS</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "AES64DSM":
            return {
                "html": "<div><span class=\"opcode\"><b>AES64DSM</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "AES64ES":
            return {
                "html": "<div><span class=\"opcode\"><b>AES64ES</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "AES64ESM":
            return {
                "html": "<div><span class=\"opcode\"><b>AES64ESM</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "AES64IM":
            return {
                "html": "<div><span class=\"opcode\"><b>AES64IM</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "AES64KS1I":
            return {
                "html": "<div><span class=\"opcode\"><b>AES64KS1I</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "AES64KS2":
            return {
                "html": "<div><span class=\"opcode\"><b>AES64KS2</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "AMOADD.D":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOADD.D</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOADD.W":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOADD.W</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOAND.D":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOAND.D</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOAND.W":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOAND.W</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOMAX.D":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOMAX.D</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOMAX.W":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOMAX.W</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOMAXU.D":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOMAXU.D</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOMAXU.W":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOMAXU.W</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOMIN.D":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOMIN.D</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOMIN.W":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOMIN.W</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOMINU.D":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOMINU.D</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOMINU.W":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOMINU.W</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOOR.D":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOOR.D</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOOR.W":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOOR.W</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOSWAP.D":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOSWAP.D</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOSWAP.W":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOSWAP.W</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOXOR.D":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOXOR.D</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AMOXOR.W":
            return {
                "html": "<div><span class=\"opcode\"><b>AMOXOR.W</b> rd, rs1, rs2</span><br><div>The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "The atomic memory operation (AMO) instructions perform read-modify-write operations for multiprocessor synchronization and are encoded with an R-type instruction format. These AMO instructions atomically load a data value from the address in rs1, place the value into register rd, apply a binary operator to the loaded value and the original value in rs2, then store the result back to the original address in rs1. AMOs can either operate on 64-bit (RV64 only) or 32-bit words in memory. For RV64, 32-bit AMOs always sign-extend the value placed in rd, and ignore the upper 32 bits of the original value of rs2.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:amo"
            };

        case "AND":
            return {
                "html": "<div><span class=\"opcode\"><b>AND</b> rd, rs1, rs2</span><br><div>ADD performs the addition of rs1 <b>and</b> rs2. SUB performs the subtraction of rs2 from rs1. Overflows are ignored <b>and</b> the low XLEN bits of results are written to the destination rd. SLT <b>and</b> SLTU perform signed <b>and</b> unsigned compares respectively, writing 1 to rd if rs1 < rs2, 0 otherwise. Note, SLTU rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs). <b>AND</b>, OR, <b>and</b> XOR perform bitwise logical operations.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "ADD performs the addition of rs1 and rs2. SUB performs the subtraction of rs2 from rs1. Overflows are ignored and the low XLEN bits of results are written to the destination rd. SLT and SLTU perform signed and unsigned compares respectively, writing 1 to rd if rs1 < rs2, 0 otherwise. Note, SLTU rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs). AND, OR, and XOR perform bitwise logical operations.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-register-operations"
            };

        case "ANDI":
            return {
                "html": "<div><span class=\"opcode\"><b>ANDI</b> rd, rs1, imm12</span><br><div><b>ANDI</b>, ORI, XORI are logical operations that perform bitwise AND, OR, and XOR on register rs1 and the sign-extended 12-bit immediate and place the result in rd. Note, XORI rd, rs1, -1 performs a bitwise logical inversion of register rs1 (assembler pseudoinstruction NOT rd, rs).</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "ANDI, ORI, XORI are logical operations that perform bitwise AND, OR, and XOR on register rs1 and the sign-extended 12-bit immediate and place the result in rd. Note, XORI rd, rs1, -1 performs a bitwise logical inversion of register rs1 (assembler pseudoinstruction NOT rd, rs).\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-immediate-instructions"
            };

        case "ANDN":
            return {
                "html": "<div><span class=\"opcode\"><b>ANDN</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "AUIPC":
            return {
                "html": "<div><span class=\"opcode\"><b>AUIPC</b> rd, imm20</span><br><div><b>AUIPC</b> (add upper immediate to pc) is used to build pc-relative addresses and uses the U-type format. <b>AUIPC</b> forms a 32-bit offset from the U-immediate, filling in the lowest 12 bits with zeros, adds this offset to the address of the <b>AUIPC</b> instruction, then places the result in register rd.<br>The <b>AUIPC</b> instruction supports two-instruction sequences to access arbitrary offsets from the PC for both control-flow transfers and data accesses. The combination of an <b>AUIPC</b> and the 12-bit immediate in a JALR can transfer control to any 32-bit PC-relative address, while an <b>AUIPC</b> plus the 12-bit immediate offset in regular load or store instructions can access any 32-bit PC-relative data address.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "AUIPC (add upper immediate to pc) is used to build pc-relative addresses and uses the U-type format. AUIPC forms a 32-bit offset from the U-immediate, filling in the lowest 12 bits with zeros, adds this offset to the address of the AUIPC instruction, then places the result in register rd.\nThe AUIPC instruction supports two-instruction sequences to access arbitrary offsets from the PC for both control-flow transfers and data accesses. The combination of an AUIPC and the 12-bit immediate in a JALR can transfer control to any 32-bit PC-relative address, while an AUIPC plus the 12-bit immediate offset in regular load or store instructions can access any 32-bit PC-relative data address.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-immediate-instructions"
            };

        case "BCLR":
            return {
                "html": "<div><span class=\"opcode\"><b>BCLR</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "BCLRI":
            return {
                "html": "<div><span class=\"opcode\"><b>BCLRI</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "BEQ":
            return {
                "html": "<div><span class=\"opcode\"><b>BEQ</b> rs1, rs2, bimm12</span><br><div>Branch instructions compare two registers. <b>BEQ</b> and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#conditional-branches"
            };

        case "BEQZ":
            return {
                "html": "<div><span class=\"opcode\"><b>BEQZ</b> rs, offset</span><br><div><b>Equivalent ASM:</b><pre>beq rs, x0, offset</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nbeq rs, x0, offset\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "BEXT":
            return {
                "html": "<div><span class=\"opcode\"><b>BEXT</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "BEXTI":
            return {
                "html": "<div><span class=\"opcode\"><b>BEXTI</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "BGE":
            return {
                "html": "<div><span class=\"opcode\"><b>BGE</b> rs1, rs2, bimm12</span><br><div>Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. <b>BGE</b> and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, <b>BGE</b>, and BGEU, respectively.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#conditional-branches"
            };

        case "BGEU":
            return {
                "html": "<div><span class=\"opcode\"><b>BGEU</b> rs1, rs2, bimm12</span><br><div>Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and <b>BGEU</b> take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and <b>BGEU</b>, respectively.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#conditional-branches"
            };

        case "BGEZ":
            return {
                "html": "<div><span class=\"opcode\"><b>BGEZ</b> rs, offset</span><br><div><b>Equivalent ASM:</b><pre>bge rs, x0, offset</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nbge rs, x0, offset\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "BGT":
            return {
                "html": "<div><span class=\"opcode\"><b>BGT</b> rs, rt, offset</span><br><div>Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, <b>BGT</b>, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.</div><br><div><b>Equivalent ASM:</b><pre>blt rt, rs, offset</pre></div><br><div><b>ISA</b>: rv32(pseudo)</div></div>",
                "tooltip": "Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.\n\nEquivalent ASM:\n\nblt rt, rs, offset\n\n(ISA: rv32(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#conditional-branches"
            };

        case "BGTU":
            return {
                "html": "<div><span class=\"opcode\"><b>BGTU</b> rs, rt, offset</span><br><div>Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, <b>BGTU</b>, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.</div><br><div><b>Equivalent ASM:</b><pre>bltu rt, rs, offset</pre></div><br><div><b>ISA</b>: rv32(pseudo)</div></div>",
                "tooltip": "Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.\n\nEquivalent ASM:\n\nbltu rt, rs, offset\n\n(ISA: rv32(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#conditional-branches"
            };

        case "BGTZ":
            return {
                "html": "<div><span class=\"opcode\"><b>BGTZ</b> rs, offset</span><br><div><b>Equivalent ASM:</b><pre>blt x0, rs, offset</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nblt x0, rs, offset\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "BINV":
            return {
                "html": "<div><span class=\"opcode\"><b>BINV</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "BINVI":
            return {
                "html": "<div><span class=\"opcode\"><b>BINVI</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "BLE":
            return {
                "html": "<div><span class=\"opcode\"><b>BLE</b> rs, rt, offset</span><br><div>Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, <b>BLE</b>, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.</div><br><div><b>Equivalent ASM:</b><pre>bge rt, rs, offset</pre></div><br><div><b>ISA</b>: rv32(pseudo)</div></div>",
                "tooltip": "Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.\n\nEquivalent ASM:\n\nbge rt, rs, offset\n\n(ISA: rv32(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#conditional-branches"
            };

        case "BLEU":
            return {
                "html": "<div><span class=\"opcode\"><b>BLEU</b> rs, rt, offset</span><br><div>Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and <b>BLEU</b> can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.</div><br><div><b>Equivalent ASM:</b><pre>bgeu rt, rs, offset</pre></div><br><div><b>ISA</b>: rv32(pseudo)</div></div>",
                "tooltip": "Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.\n\nEquivalent ASM:\n\nbgeu rt, rs, offset\n\n(ISA: rv32(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#conditional-branches"
            };

        case "BLEZ":
            return {
                "html": "<div><span class=\"opcode\"><b>BLEZ</b> rs, offset</span><br><div><b>Equivalent ASM:</b><pre>bge x0, rs, offset</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nbge x0, rs, offset\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "BLT":
            return {
                "html": "<div><span class=\"opcode\"><b>BLT</b> rs1, rs2, bimm12</span><br><div>Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. <b>BLT</b> and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to <b>BLT</b>, BLTU, BGE, and BGEU, respectively.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#conditional-branches"
            };

        case "BLTU":
            return {
                "html": "<div><span class=\"opcode\"><b>BLTU</b> rs1, rs2, bimm12</span><br><div>Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and <b>BLTU</b> take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, <b>BLTU</b>, BGE, and BGEU, respectively.<br>Signed array bounds may be checked with a single <b>BLTU</b> instruction, since any negative index will compare greater than any nonnegative bound.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.\nSigned array bounds may be checked with a single BLTU instruction, since any negative index will compare greater than any nonnegative bound.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#conditional-branches"
            };

        case "BLTZ":
            return {
                "html": "<div><span class=\"opcode\"><b>BLTZ</b> rs, offset</span><br><div><b>Equivalent ASM:</b><pre>blt rs, x0, offset</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nblt rs, x0, offset\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "BNE":
            return {
                "html": "<div><span class=\"opcode\"><b>BNE</b> rs1, rs2, bimm12</span><br><div>Branch instructions compare two registers. BEQ and <b>BNE</b> take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "Branch instructions compare two registers. BEQ and BNE take the branch if registers rs1 and rs2 are equal or unequal respectively. BLT and BLTU take the branch if rs1 is less than rs2, using signed and unsigned comparison respectively. BGE and BGEU take the branch if rs1 is greater than or equal to rs2, using signed and unsigned comparison respectively. Note, BGT, BGTU, BLE, and BLEU can be synthesized by reversing the operands to BLT, BLTU, BGE, and BGEU, respectively.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#conditional-branches"
            };

        case "BNEZ":
            return {
                "html": "<div><span class=\"opcode\"><b>BNEZ</b> rs, offset</span><br><div><b>Equivalent ASM:</b><pre>bne rs, x0, offset</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nbne rs, x0, offset\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "BREV8":
            return {
                "html": "<div><span class=\"opcode\"><b>BREV8</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "BSET":
            return {
                "html": "<div><span class=\"opcode\"><b>BSET</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "BSETI":
            return {
                "html": "<div><span class=\"opcode\"><b>BSETI</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "C.ADD":
            return {
                "html": "<div><span class=\"opcode\"><b>C.ADD</b> rd_rs1, c_rs2_n0</span><br><div><b>C.ADD</b> adds the values in registers rd and rs2 and writes the result to register rd. <b>C.ADD</b> expands into add rd, rd, rs2. <b>C.ADD</b> is only valid when rs2 x0; the code points with rs2 = x0 correspond to the C.JALR and C.EBREAK instructions. The code points with rs2 x0 and rd = x0 are HINTs.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.ADD adds the values in registers rd and rs2 and writes the result to register rd. C.ADD expands into add rd, rd, rs2. C.ADD is only valid when rs2 x0; the code points with rs2 = x0 correspond to the C.JALR and C.EBREAK instructions. The code points with rs2 x0 and rd = x0 are HINTs.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-register-operations"
            };

        case "C.ADDI":
            return {
                "html": "<div><span class=\"opcode\"><b>C.ADDI</b> rd_rs1_n0, c_nzimm6, c_nzimm6</span><br><div><b>C.ADDI</b> adds the non-zero sign-extended 6-bit immediate to the value in register rd then writes the result to rd. <b>C.ADDI</b> expands into addi rd, rd, nzimm. <b>C.ADDI</b> is only valid when rd x0 and nzimm 0. The code points with rd=x0 encode the C.NOP instruction; the remaining code points with nzimm=0 encode HINTs.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.ADDI adds the non-zero sign-extended 6-bit immediate to the value in register rd then writes the result to rd. C.ADDI expands into addi rd, rd, nzimm. C.ADDI is only valid when rd x0 and nzimm 0. The code points with rd=x0 encode the C.NOP instruction; the remaining code points with nzimm=0 encode HINTs.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-immediate-operations"
            };

        case "C.ADDI16SP":
        case "ADDI16SP":
            return {
                "html": "<div><span class=\"opcode\"><b>C.ADDI16SP</b> c_nzimm10</span><br><div>C.LUI loads the non-zero 6-bit immediate field into bits 17-12 of the destination register, clears the bottom 12 bits, and sign-extends bit 17 into all higher bits of the destination. C.LUI expands into lui rd, nzimm. C.LUI is only valid when rd {x0,x2}, and when the immediate is not equal to zero. The code points with nzimm=0 are reserved; the remaining code points with rd=x0 are HINTs; and the remaining code points with rd=x2 correspond to the <b>C.ADDI16SP</b> instruction.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.LUI loads the non-zero 6-bit immediate field into bits 17-12 of the destination register, clears the bottom 12 bits, and sign-extends bit 17 into all higher bits of the destination. C.LUI expands into lui rd, nzimm. C.LUI is only valid when rd {x0,x2}, and when the immediate is not equal to zero. The code points with nzimm=0 are reserved; the remaining code points with rd=x0 are HINTs; and the remaining code points with rd=x2 correspond to the C.ADDI16SP instruction.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-constant-generation-instructions"
            };

        case "C.ADDI4SPN":
        case "ADDI4SPN":
            return {
                "html": "<div><span class=\"opcode\"><b>C.ADDI4SPN</b> rd_p, c_nzuimm10</span><br><div><b>C.ADDI4SPN</b> is a CIW-format instruction that adds a zero-extended non-zero immediate, scaled by 4, to the stack pointer, x2, and writes the result to rd'. This instruction is used to generate pointers to stack-allocated variables, and expands to addi rd', x2, nzuimm. <b>C.ADDI4SPN</b> is only valid when nzuimm 0; the code points with nzuimm=0 are reserved.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.ADDI4SPN is a CIW-format instruction that adds a zero-extended non-zero immediate, scaled by 4, to the stack pointer, x2, and writes the result to rd'. This instruction is used to generate pointers to stack-allocated variables, and expands to addi rd', x2, nzuimm. C.ADDI4SPN is only valid when nzuimm 0; the code points with nzuimm=0 are reserved.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-immediate-operations"
            };

        case "C.ADDIW":
            return {
                "html": "<div><span class=\"opcode\"><b>C.ADDIW</b> rd_rs1_n0, c_imm6</span><br><div><b>C.ADDIW</b> is an RV64C/RV128C-only instruction that performs the same computation but produces a 32-bit result, then sign-extends result to 64 bits. <b>C.ADDIW</b> expands into addiw rd, rd, imm. The immediate can be zero for <b>C.ADDIW</b>, where this corresponds to sext.w rd. <b>C.ADDIW</b> is only valid when rd x0; the code points with rd=x0 are reserved.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.ADDIW is an RV64C/RV128C-only instruction that performs the same computation but produces a 32-bit result, then sign-extends result to 64 bits. C.ADDIW expands into addiw rd, rd, imm. The immediate can be zero for C.ADDIW, where this corresponds to sext.w rd. C.ADDIW is only valid when rd x0; the code points with rd=x0 are reserved.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-immediate-operations"
            };

        case "C.ADDW":
            return {
                "html": "<div><span class=\"opcode\"><b>C.ADDW</b> rd_rs1_p, rs2_p</span><br><div><b>C.ADDW</b> is an RV64C/RV128C-only instruction that adds the values in registers rd' and rs2', then sign-extends the lower 32 bits of the sum before writing the result to register rd'. <b>C.ADDW</b> expands into addw rd', rd', rs2'.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.ADDW is an RV64C/RV128C-only instruction that adds the values in registers rd' and rs2', then sign-extends the lower 32 bits of the sum before writing the result to register rd'. C.ADDW expands into addw rd', rd', rs2'.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-register-operations"
            };

        case "C.AND":
            return {
                "html": "<div><span class=\"opcode\"><b>C.AND</b> rd_rs1_p, rs2_p</span><br><div><b>C.AND</b> computes the bitwise AND of the values in registers rd' and rs2', then writes the result to register rd'. <b>C.AND</b> expands into and rd', rd', rs2'.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.AND computes the bitwise AND of the values in registers rd' and rs2', then writes the result to register rd'. C.AND expands into and rd', rd', rs2'.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-register-operations"
            };

        case "C.ANDI":
            return {
                "html": "<div><span class=\"opcode\"><b>C.ANDI</b> rd_rs1_p, c_imm6</span><br><div><b>C.ANDI</b> is a CB-format instruction that computes the bitwise AND of the value in register rd' and the sign-extended 6-bit immediate, then writes the result to rd'. <b>C.ANDI</b> expands to andi rd', rd', imm.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.ANDI is a CB-format instruction that computes the bitwise AND of the value in register rd' and the sign-extended 6-bit immediate, then writes the result to rd'. C.ANDI expands to andi rd', rd', imm.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-immediate-operations"
            };

        case "C.BEQZ":
            return {
                "html": "<div><span class=\"opcode\"><b>C.BEQZ</b> rs1_p, c_bimm9</span><br><div><b>C.BEQZ</b> performs conditional control transfers. The offset is sign-extended and added to the pc to form the branch target address. It can therefore target a \u00b1256B range. <b>C.BEQZ</b> takes the branch if the value in register rs1' is zero. It expands to beq rs1', x0, offset.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.BEQZ performs conditional control transfers. The offset is sign-extended and added to the pc to form the branch target address. It can therefore target a \u00b1256B range. C.BEQZ takes the branch if the value in register rs1' is zero. It expands to beq rs1', x0, offset.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#control-transfer-instructions"
            };

        case "C.BNEZ":
            return {
                "html": "<div><span class=\"opcode\"><b>C.BNEZ</b> rs1_p, c_bimm9</span><br><div><b>C.BNEZ</b> is defined analogously, but it takes the branch if rs1' contains a nonzero value. It expands to bne rs1', x0, offset.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.BNEZ is defined analogously, but it takes the branch if rs1' contains a nonzero value. It expands to bne rs1', x0, offset.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#control-transfer-instructions"
            };

        case "C.EBREAK":
            return {
                "html": "<div><span class=\"opcode\"><b>C.EBREAK</b> </span><br><div>C.ADD adds the values in registers rd and rs2 and writes the result to register rd. C.ADD expands into add rd, rd, rs2. C.ADD is only valid when rs2 x0; the code points with rs2 = x0 correspond to the C.JALR and <b>C.EBREAK</b> instructions. The code points with rs2 x0 and rd = x0 are HINTs.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.ADD adds the values in registers rd and rs2 and writes the result to register rd. C.ADD expands into add rd, rd, rs2. C.ADD is only valid when rs2 x0; the code points with rs2 = x0 correspond to the C.JALR and C.EBREAK instructions. The code points with rs2 x0 and rd = x0 are HINTs.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-register-operations"
            };

        case "C.FLD":
            return {
                "html": "<div><span class=\"opcode\"><b>C.FLD</b> rd_p, rs1_p, c_uimm8</span><br><div><b>C.FLD</b> is an RV32DC/RV64DC-only instruction that loads a double-precision floating-point value from memory into floating-point register rd'. It computes an effective address by adding the zero-extended offset, scaled by 8, to the base address in register rs1'. It expands to fld rd', offset(rs1').</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.FLD is an RV32DC/RV64DC-only instruction that loads a double-precision floating-point value from memory into floating-point register rd'. It computes an effective address by adding the zero-extended offset, scaled by 8, to the base address in register rs1'. It expands to fld rd', offset(rs1').\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#register-based-loads-and-stores"
            };

        case "C.FLDSP":
        case "FLDSP":
            return {
                "html": "<div><span class=\"opcode\"><b>C.FLDSP</b> rd, c_uimm9sp</span><br><div><b>C.FLDSP</b> is an RV32DC/RV64DC-only instruction that loads a double-precision floating-point value from memory into floating-point register rd. It computes its effective address by adding the zero-extended offset, scaled by 8, to the stack pointer, x2. It expands to fld rd, offset(x2).</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.FLDSP is an RV32DC/RV64DC-only instruction that loads a double-precision floating-point value from memory into floating-point register rd. It computes its effective address by adding the zero-extended offset, scaled by 8, to the stack pointer, x2. It expands to fld rd, offset(x2).\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#stack-pointer-based-loads-and-stores"
            };

        case "C.FLW":
            return {
                "html": "<div><span class=\"opcode\"><b>C.FLW</b> rd_p, rs1_p, c_uimm7</span><br><div><b>C.FLW</b> is an RV32FC-only instruction that loads a single-precision floating-point value from memory into floating-point register rd'. It computes an effective address by adding the zero-extended offset, scaled by 4, to the base address in register rs1'. It expands to flw rd', offset(rs1').</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.FLW is an RV32FC-only instruction that loads a single-precision floating-point value from memory into floating-point register rd'. It computes an effective address by adding the zero-extended offset, scaled by 4, to the base address in register rs1'. It expands to flw rd', offset(rs1').\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#register-based-loads-and-stores"
            };

        case "C.FLWSP":
        case "FLWSP":
            return {
                "html": "<div><span class=\"opcode\"><b>C.FLWSP</b> rd, c_uimm8sp</span><br><div><b>C.FLWSP</b> is an RV32FC-only instruction that loads a single-precision floating-point value from memory into floating-point register rd. It computes its effective address by adding the zero-extended offset, scaled by 4, to the stack pointer, x2. It expands to flw rd, offset(x2).</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.FLWSP is an RV32FC-only instruction that loads a single-precision floating-point value from memory into floating-point register rd. It computes its effective address by adding the zero-extended offset, scaled by 4, to the stack pointer, x2. It expands to flw rd, offset(x2).\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#stack-pointer-based-loads-and-stores"
            };

        case "C.FSD":
            return {
                "html": "<div><span class=\"opcode\"><b>C.FSD</b> rs1_p, rs2_p, c_uimm8</span><br><div><b>C.FSD</b> is an RV32DC/RV64DC-only instruction that stores a double-precision floating-point value in floating-point register rs2' to memory. It computes an effective address by adding the zero-extended offset, scaled by 8, to the base address in register rs1'. It expands to fsd rs2', offset(rs1').</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.FSD is an RV32DC/RV64DC-only instruction that stores a double-precision floating-point value in floating-point register rs2' to memory. It computes an effective address by adding the zero-extended offset, scaled by 8, to the base address in register rs1'. It expands to fsd rs2', offset(rs1').\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#register-based-loads-and-stores"
            };

        case "C.FSDSP":
        case "FSDSP":
            return {
                "html": "<div><span class=\"opcode\"><b>C.FSDSP</b> c_rs2, c_uimm9sp_s</span><br><div><b>C.FSDSP</b> is an RV32DC/RV64DC-only instruction that stores a double-precision floating-point value in floating-point register rs2 to memory. It computes an effective address by adding the zero-extended offset, scaled by 8, to the stack pointer, x2. It expands to fsd rs2, offset(x2).</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.FSDSP is an RV32DC/RV64DC-only instruction that stores a double-precision floating-point value in floating-point register rs2 to memory. It computes an effective address by adding the zero-extended offset, scaled by 8, to the stack pointer, x2. It expands to fsd rs2, offset(x2).\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#stack-pointer-based-loads-and-stores"
            };

        case "C.FSW":
            return {
                "html": "<div><span class=\"opcode\"><b>C.FSW</b> rs1_p, rs2_p, c_uimm7</span><br><div><b>C.FSW</b> is an RV32FC-only instruction that stores a single-precision floating-point value in floating-point register rs2' to memory. It computes an effective address by adding the zero-extended offset, scaled by 4, to the base address in register rs1'. It expands to fsw rs2', offset(rs1').</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.FSW is an RV32FC-only instruction that stores a single-precision floating-point value in floating-point register rs2' to memory. It computes an effective address by adding the zero-extended offset, scaled by 4, to the base address in register rs1'. It expands to fsw rs2', offset(rs1').\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#register-based-loads-and-stores"
            };

        case "C.FSWSP":
        case "FSWSP":
            return {
                "html": "<div><span class=\"opcode\"><b>C.FSWSP</b> c_rs2, c_uimm8sp_s</span><br><div><b>C.FSWSP</b> is an RV32FC-only instruction that stores a single-precision floating-point value in floating-point register rs2 to memory. It computes an effective address by adding the zero-extended offset, scaled by 4, to the stack pointer, x2. It expands to fsw rs2, offset(x2).</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.FSWSP is an RV32FC-only instruction that stores a single-precision floating-point value in floating-point register rs2 to memory. It computes an effective address by adding the zero-extended offset, scaled by 4, to the stack pointer, x2. It expands to fsw rs2, offset(x2).\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#stack-pointer-based-loads-and-stores"
            };

        case "C.J":
            return {
                "html": "<div><span class=\"opcode\"><b>C.J</b> c_imm12</span><br><div><b>C.J</b> performs an unconditional control transfer. The offset is sign-extended and added to the pc to form the jump target address. <b>C.J</b> can therefore target a \u00b12KiB range. <b>C.J</b> expands to jal x0, offset.<br>C.JAL is an RV32C-only instruction that performs the same operation as <b>C.J</b>, but additionally writes the address of the instruction following the jump (pc+2) to the link register, x1. C.JAL expands to jal x1, offset.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.J performs an unconditional control transfer. The offset is sign-extended and added to the pc to form the jump target address. C.J can therefore target a \u00b12KiB range. C.J expands to jal x0, offset.\nC.JAL is an RV32C-only instruction that performs the same operation as C.J, but additionally writes the address of the instruction following the jump (pc+2) to the link register, x1. C.JAL expands to jal x1, offset.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#control-transfer-instructions"
            };

        case "C.JAL":
            return {
                "html": "<div><span class=\"opcode\"><b>C.JAL</b> c_imm12</span><br><div><b>C.JAL</b> is an RV32C-only instruction that performs the same operation as C.J, but additionally writes the address of the instruction following the jump (pc+2) to the link register, x1. <b>C.JAL</b> expands to jal x1, offset.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.JAL is an RV32C-only instruction that performs the same operation as C.J, but additionally writes the address of the instruction following the jump (pc+2) to the link register, x1. C.JAL expands to jal x1, offset.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#control-transfer-instructions"
            };

        case "C.JALR":
            return {
                "html": "<div><span class=\"opcode\"><b>C.JALR</b> c_rs1_n0</span><br><div>C.ADD adds the values in registers rd and rs2 and writes the result to register rd. C.ADD expands into add rd, rd, rs2. C.ADD is only valid when rs2 x0; the code points with rs2 = x0 correspond to the <b>C.JALR</b> and C.EBREAK instructions. The code points with rs2 x0 and rd = x0 are HINTs.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.ADD adds the values in registers rd and rs2 and writes the result to register rd. C.ADD expands into add rd, rd, rs2. C.ADD is only valid when rs2 x0; the code points with rs2 = x0 correspond to the C.JALR and C.EBREAK instructions. The code points with rs2 x0 and rd = x0 are HINTs.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-register-operations"
            };

        case "C.JR":
            return {
                "html": "<div><span class=\"opcode\"><b>C.JR</b> rs1_n0</span><br><div>C.MV copies the value in register rs2 into register rd. C.MV expands into add rd, x0, rs2. C.MV is only valid when rs2 x0; the code points with rs2 = x0 correspond to the <b>C.JR</b> instruction. The code points with rs2 x0 and rd = x0 are HINTs.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.MV copies the value in register rs2 into register rd. C.MV expands into add rd, x0, rs2. C.MV is only valid when rs2 x0; the code points with rs2 = x0 correspond to the C.JR instruction. The code points with rs2 x0 and rd = x0 are HINTs.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-register-operations"
            };

        case "C.LBU":
            return {
                "html": "<div><span class=\"opcode\"><b>C.LBU</b> rd_p, rs1_p, c_uimm2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "C.LD":
            return {
                "html": "<div><span class=\"opcode\"><b>C.LD</b> rd_p, rs1_p, c_uimm8</span><br><div><b>C.LD</b> is an RV64C/RV128C-only instruction that loads a 64-bit value from memory into register rd'. It computes an effective address by adding the zero-extended offset, scaled by 8, to the base address in register rs1'. It expands to ld rd', offset(rs1').</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.LD is an RV64C/RV128C-only instruction that loads a 64-bit value from memory into register rd'. It computes an effective address by adding the zero-extended offset, scaled by 8, to the base address in register rs1'. It expands to ld rd', offset(rs1').\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#register-based-loads-and-stores"
            };

        case "C.LDSP":
        case "LDSP":
            return {
                "html": "<div><span class=\"opcode\"><b>C.LDSP</b> rd_n0, c_uimm9sp</span><br><div><b>C.LDSP</b> is an RV64C/RV128C-only instruction that loads a 64-bit value from memory into register rd. It computes its effective address by adding the zero-extended offset, scaled by 8, to the stack pointer, x2. It expands to ld rd, offset(x2). <b>C.LDSP</b> is only valid when rd x0; the code points with rd = x0 are reserved.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.LDSP is an RV64C/RV128C-only instruction that loads a 64-bit value from memory into register rd. It computes its effective address by adding the zero-extended offset, scaled by 8, to the stack pointer, x2. It expands to ld rd, offset(x2). C.LDSP is only valid when rd x0; the code points with rd = x0 are reserved.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#stack-pointer-based-loads-and-stores"
            };

        case "C.LH":
            return {
                "html": "<div><span class=\"opcode\"><b>C.LH</b> rd_p, rs1_p, c_uimm1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "C.LHU":
            return {
                "html": "<div><span class=\"opcode\"><b>C.LHU</b> rd_p, rs1_p, c_uimm1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "C.LI":
            return {
                "html": "<div><span class=\"opcode\"><b>C.LI</b> rd, c_imm6</span><br><div><b>C.LI</b> loads the sign-extended 6-bit immediate, imm, into register rd. <b>C.LI</b> expands into addi rd, x0, imm. <b>C.LI</b> is only valid when rd x0; the code points with rd=x0 encode HINTs.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.LI loads the sign-extended 6-bit immediate, imm, into register rd. C.LI expands into addi rd, x0, imm. C.LI is only valid when rd x0; the code points with rd=x0 encode HINTs.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-constant-generation-instructions"
            };

        case "C.LUI":
            return {
                "html": "<div><span class=\"opcode\"><b>C.LUI</b> rd_n2, c_nzimm18</span><br><div><b>C.LUI</b> loads the non-zero 6-bit immediate field into bits 17-12 of the destination register, clears the bottom 12 bits, and sign-extends bit 17 into all higher bits of the destination. <b>C.LUI</b> expands into lui rd, nzimm. <b>C.LUI</b> is only valid when rd {x0,x2}, and when the immediate is not equal to zero. The code points with nzimm=0 are reserved; the remaining code points with rd=x0 are HINTs; and the remaining code points with rd=x2 correspond to the C.ADDI16SP instruction.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.LUI loads the non-zero 6-bit immediate field into bits 17-12 of the destination register, clears the bottom 12 bits, and sign-extends bit 17 into all higher bits of the destination. C.LUI expands into lui rd, nzimm. C.LUI is only valid when rd {x0,x2}, and when the immediate is not equal to zero. The code points with nzimm=0 are reserved; the remaining code points with rd=x0 are HINTs; and the remaining code points with rd=x2 correspond to the C.ADDI16SP instruction.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-constant-generation-instructions"
            };

        case "C.LW":
            return {
                "html": "<div><span class=\"opcode\"><b>C.LW</b> rd_p, rs1_p, c_uimm7</span><br><div><b>C.LW</b> loads a 32-bit value from memory into register rd'. It computes an effective address by adding the zero-extended offset, scaled by 4, to the base address in register rs1'. It expands to lw rd', offset(rs1').</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.LW loads a 32-bit value from memory into register rd'. It computes an effective address by adding the zero-extended offset, scaled by 4, to the base address in register rs1'. It expands to lw rd', offset(rs1').\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#register-based-loads-and-stores"
            };

        case "C.LWSP":
        case "LWSP":
            return {
                "html": "<div><span class=\"opcode\"><b>C.LWSP</b> rd_n0, c_uimm8sp</span><br><div><b>C.LWSP</b> loads a 32-bit value from memory into register rd. It computes an effective address by adding the zero-extended offset, scaled by 4, to the stack pointer, x2. It expands to lw rd, offset(x2). <b>C.LWSP</b> is only valid when rd x0; the code points with rd = x0 are reserved.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.LWSP loads a 32-bit value from memory into register rd. It computes an effective address by adding the zero-extended offset, scaled by 4, to the stack pointer, x2. It expands to lw rd, offset(x2). C.LWSP is only valid when rd x0; the code points with rd = x0 are reserved.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#stack-pointer-based-loads-and-stores"
            };

        case "C.MUL":
            return {
                "html": "<div><span class=\"opcode\"><b>C.MUL</b> rd_rs1_p, rs2_p</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "C.MV":
            return {
                "html": "<div><span class=\"opcode\"><b>C.MV</b> rd, c_rs2_n0</span><br><div><b>C.MV</b> copies the value in register rs2 into register rd. <b>C.MV</b> expands into add rd, x0, rs2. <b>C.MV</b> is only valid when rs2 x0; the code points with rs2 = x0 correspond to the C.JR instruction. The code points with rs2 x0 and rd = x0 are HINTs.<br><b>C.MV</b> expands to a different instruction than the canonical MV pseudoinstruction, which instead uses ADDI. Implementations that handle MV specially, e.g. using register-renaming hardware, may find it more convenient to expand <b>C.MV</b> to MV instead of ADD, at slight additional hardware cost.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.MV copies the value in register rs2 into register rd. C.MV expands into add rd, x0, rs2. C.MV is only valid when rs2 x0; the code points with rs2 = x0 correspond to the C.JR instruction. The code points with rs2 x0 and rd = x0 are HINTs.\nC.MV expands to a different instruction than the canonical MV pseudoinstruction, which instead uses ADDI. Implementations that handle MV specially, e.g. using register-renaming hardware, may find it more convenient to expand C.MV to MV instead of ADD, at slight additional hardware cost.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-register-operations"
            };

        case "C.NOP":
            return {
                "html": "<div><span class=\"opcode\"><b>C.NOP</b> c_nzimm6</span><br><div><b>C.NOP</b> is a CI-format instruction that does not change any user-visible state, except for advancing the pc and incrementing any applicable performance counters. <b>C.NOP</b> expands to nop. <b>C.NOP</b> is only valid when imm=0; the code points with imm 0 encode HINTs.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.NOP is a CI-format instruction that does not change any user-visible state, except for advancing the pc and incrementing any applicable performance counters. C.NOP expands to nop. C.NOP is only valid when imm=0; the code points with imm 0 encode HINTs.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#nop-instruction"
            };

        case "C.NOT":
            return {
                "html": "<div><span class=\"opcode\"><b>C.NOT</b> rd_rs1_p</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "C.OR":
            return {
                "html": "<div><span class=\"opcode\"><b>C.OR</b> rd_rs1_p, rs2_p</span><br><div><b>C.OR</b> computes the bitwise OR of the values in registers rd' and rs2', then writes the result to register rd'. <b>C.OR</b> expands into or rd', rd', rs2'.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.OR computes the bitwise OR of the values in registers rd' and rs2', then writes the result to register rd'. C.OR expands into or rd', rd', rs2'.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-register-operations"
            };

        case "C.SB":
            return {
                "html": "<div><span class=\"opcode\"><b>C.SB</b> rs2_p, rs1_p, c_uimm2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "C.SD":
            return {
                "html": "<div><span class=\"opcode\"><b>C.SD</b> rs1_p, rs2_p, c_uimm8</span><br><div><b>C.SD</b> is an RV64C/RV128C-only instruction that stores a 64-bit value in register rs2' to memory. It computes an effective address by adding the zero-extended offset, scaled by 8, to the base address in register rs1'. It expands to sd rs2', offset(rs1').</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.SD is an RV64C/RV128C-only instruction that stores a 64-bit value in register rs2' to memory. It computes an effective address by adding the zero-extended offset, scaled by 8, to the base address in register rs1'. It expands to sd rs2', offset(rs1').\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#register-based-loads-and-stores"
            };

        case "C.SDSP":
        case "SDSP":
            return {
                "html": "<div><span class=\"opcode\"><b>C.SDSP</b> c_rs2, c_uimm9sp_s</span><br><div><b>C.SDSP</b> is an RV64C/RV128C-only instruction that stores a 64-bit value in register rs2 to memory. It computes an effective address by adding the zero-extended offset, scaled by 8, to the stack pointer, x2. It expands to sd rs2, offset(x2).</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.SDSP is an RV64C/RV128C-only instruction that stores a 64-bit value in register rs2 to memory. It computes an effective address by adding the zero-extended offset, scaled by 8, to the stack pointer, x2. It expands to sd rs2, offset(x2).\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#stack-pointer-based-loads-and-stores"
            };

        case "C.SEXT.B":
            return {
                "html": "<div><span class=\"opcode\"><b>C.SEXT.B</b> rd_rs1_p</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "C.SEXT.H":
            return {
                "html": "<div><span class=\"opcode\"><b>C.SEXT.H</b> rd_rs1_p</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "C.SH":
            return {
                "html": "<div><span class=\"opcode\"><b>C.SH</b> rs2_p, rs1_p, c_uimm1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "C.SLLI":
            return {
                "html": "<div><span class=\"opcode\"><b>C.SLLI</b> rd_rs1_n0, c_nzuimm6</span><br><div><b>C.SLLI</b> is a CI-format instruction that performs a logical left shift of the value in register rd then writes the result to rd. The shift amount is encoded in the shamt field. For RV128C, a shift amount of zero is used to encode a shift of 64. <b>C.SLLI</b> expands into slli rd, rd, shamt, except for RV128C with shamt=0, which expands to slli rd, rd, 64.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.SLLI is a CI-format instruction that performs a logical left shift of the value in register rd then writes the result to rd. The shift amount is encoded in the shamt field. For RV128C, a shift amount of zero is used to encode a shift of 64. C.SLLI expands into slli rd, rd, shamt, except for RV128C with shamt=0, which expands to slli rd, rd, 64.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-immediate-operations"
            };

        case "C.SLLI_RV32":
            return {
                "html": "<div><span class=\"opcode\"><b>C.SLLI_RV32</b> rd_rs1_n0, c_nzuimm6lo</span><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "C.SRAI":
            return {
                "html": "<div><span class=\"opcode\"><b>C.SRAI</b> rd_rs1_p, c_nzuimm6</span><br><div><b>C.SRAI</b> is defined analogously to C.SRLI, but instead performs an arithmetic right shift. <b>C.SRAI</b> expands to srai rd', rd', shamt.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.SRAI is defined analogously to C.SRLI, but instead performs an arithmetic right shift. C.SRAI expands to srai rd', rd', shamt.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-immediate-operations"
            };

        case "C.SRAI_RV32":
            return {
                "html": "<div><span class=\"opcode\"><b>C.SRAI_RV32</b> rd_rs1_p, c_nzuimm5</span><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "C.SRLI":
            return {
                "html": "<div><span class=\"opcode\"><b>C.SRLI</b> rd_rs1_p, c_nzuimm6</span><br><div><b>C.SRLI</b> is a CB-format instruction that performs a logical right shift of the value in register rd' then writes the result to rd'. The shift amount is encoded in the shamt field. For RV128C, a shift amount of zero is used to encode a shift of 64. Furthermore, the shift amount is sign-extended for RV128C, and so the legal shift amounts are 1-31, 64, and 96-127. <b>C.SRLI</b> expands into srli rd', rd', shamt, except for RV128C with shamt=0, which expands to srli rd', rd', 64.<br>C.SRAI is defined analogously to <b>C.SRLI</b>, but instead performs an arithmetic right shift. C.SRAI expands to srai rd', rd', shamt.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.SRLI is a CB-format instruction that performs a logical right shift of the value in register rd' then writes the result to rd'. The shift amount is encoded in the shamt field. For RV128C, a shift amount of zero is used to encode a shift of 64. Furthermore, the shift amount is sign-extended for RV128C, and so the legal shift amounts are 1-31, 64, and 96-127. C.SRLI expands into srli rd', rd', shamt, except for RV128C with shamt=0, which expands to srli rd', rd', 64.\nC.SRAI is defined analogously to C.SRLI, but instead performs an arithmetic right shift. C.SRAI expands to srai rd', rd', shamt.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-immediate-operations"
            };

        case "C.SRLI_RV32":
            return {
                "html": "<div><span class=\"opcode\"><b>C.SRLI_RV32</b> rd_rs1_p, c_nzuimm5</span><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "C.SUB":
            return {
                "html": "<div><span class=\"opcode\"><b>C.SUB</b> rd_rs1_p, rs2_p</span><br><div><b>C.SUB</b> subtracts the value in register rs2' from the value in register rd', then writes the result to register rd'. <b>C.SUB</b> expands into sub rd', rd', rs2'.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.SUB subtracts the value in register rs2' from the value in register rd', then writes the result to register rd'. C.SUB expands into sub rd', rd', rs2'.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-register-operations"
            };

        case "C.SUBW":
            return {
                "html": "<div><span class=\"opcode\"><b>C.SUBW</b> rd_rs1_p, rs2_p</span><br><div><b>C.SUBW</b> is an RV64C/RV128C-only instruction that subtracts the value in register rs2' from the value in register rd', then sign-extends the lower 32 bits of the difference before writing the result to register rd'. <b>C.SUBW</b> expands into subw rd', rd', rs2'.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.SUBW is an RV64C/RV128C-only instruction that subtracts the value in register rs2' from the value in register rd', then sign-extends the lower 32 bits of the difference before writing the result to register rd'. C.SUBW expands into subw rd', rd', rs2'.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-register-operations"
            };

        case "C.SW":
            return {
                "html": "<div><span class=\"opcode\"><b>C.SW</b> rs1_p, rs2_p, c_uimm7</span><br><div><b>C.SW</b> stores a 32-bit value in register rs2' to memory. It computes an effective address by adding the zero-extended offset, scaled by 4, to the base address in register rs1'. It expands to sw rs2', offset(rs1').</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.SW stores a 32-bit value in register rs2' to memory. It computes an effective address by adding the zero-extended offset, scaled by 4, to the base address in register rs1'. It expands to sw rs2', offset(rs1').\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#register-based-loads-and-stores"
            };

        case "C.SWSP":
        case "SWSP":
            return {
                "html": "<div><span class=\"opcode\"><b>C.SWSP</b> c_rs2, c_uimm8sp_s</span><br><div><b>C.SWSP</b> stores a 32-bit value in register rs2 to memory. It computes an effective address by adding the zero-extended offset, scaled by 4, to the stack pointer, x2. It expands to sw rs2, offset(x2).</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.SWSP stores a 32-bit value in register rs2 to memory. It computes an effective address by adding the zero-extended offset, scaled by 4, to the stack pointer, x2. It expands to sw rs2, offset(x2).\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#stack-pointer-based-loads-and-stores"
            };

        case "C.XOR":
            return {
                "html": "<div><span class=\"opcode\"><b>C.XOR</b> rd_rs1_p, rs2_p</span><br><div><b>C.XOR</b> computes the bitwise XOR of the values in registers rd' and rs2', then writes the result to register rd'. <b>C.XOR</b> expands into xor rd', rd', rs2'.</div><br><div><b>ISA</b>: c</div></div>",
                "tooltip": "C.XOR computes the bitwise XOR of the values in registers rd' and rs2', then writes the result to register rd'. C.XOR expands into xor rd', rd', rs2'.\n\n\n\n(ISA: c)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/c.html#integer-register-register-operations"
            };

        case "C.ZEXT.B":
            return {
                "html": "<div><span class=\"opcode\"><b>C.ZEXT.B</b> rd_rs1_p</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "C.ZEXT.H":
            return {
                "html": "<div><span class=\"opcode\"><b>C.ZEXT.H</b> rd_rs1_p</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "C.ZEXT.W":
            return {
                "html": "<div><span class=\"opcode\"><b>C.ZEXT.W</b> rd_rs1_p</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CALL":
            return {
                "html": "<div><span class=\"opcode\"><b>CALL</b> offset</span><br><div><b>Equivalent ASM:</b><pre>auipc x1, offset[31:12]\njalr x1, x1, offset[11:0]</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nauipc x1, offset[31:12]\njalr x1, x1, offset[11:0]\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CBO.CLEAN":
            return {
                "html": "<div><span class=\"opcode\"><b>CBO.CLEAN</b> rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CBO.FLUSH":
            return {
                "html": "<div><span class=\"opcode\"><b>CBO.FLUSH</b> rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CBO.INVAL":
            return {
                "html": "<div><span class=\"opcode\"><b>CBO.INVAL</b> rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CBO.ZERO":
            return {
                "html": "<div><span class=\"opcode\"><b>CBO.ZERO</b> rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CLMUL":
            return {
                "html": "<div><span class=\"opcode\"><b>CLMUL</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CLMULH":
            return {
                "html": "<div><span class=\"opcode\"><b>CLMULH</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CLMULR":
            return {
                "html": "<div><span class=\"opcode\"><b>CLMULR</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CLZ":
            return {
                "html": "<div><span class=\"opcode\"><b>CLZ</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CLZW":
            return {
                "html": "<div><span class=\"opcode\"><b>CLZW</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CM.JALT":
            return {
                "html": "<div><span class=\"opcode\"><b>CM.JALT</b> </span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CM.MVA01S":
            return {
                "html": "<div><span class=\"opcode\"><b>CM.MVA01S</b> </span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CM.MVSA01":
            return {
                "html": "<div><span class=\"opcode\"><b>CM.MVSA01</b> </span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CM.POP":
            return {
                "html": "<div><span class=\"opcode\"><b>CM.POP</b> c_spimm</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CM.POPRET":
            return {
                "html": "<div><span class=\"opcode\"><b>CM.POPRET</b> c_spimm</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CM.POPRETZ":
            return {
                "html": "<div><span class=\"opcode\"><b>CM.POPRETZ</b> c_spimm</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CM.PUSH":
            return {
                "html": "<div><span class=\"opcode\"><b>CM.PUSH</b> c_spimm</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CPOP":
            return {
                "html": "<div><span class=\"opcode\"><b>CPOP</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CPOPW":
            return {
                "html": "<div><span class=\"opcode\"><b>CPOPW</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CSRC":
            return {
                "html": "<div><span class=\"opcode\"><b>CSRC</b> csr, rs</span><br><div>Further assembler pseudoinstructions are defined to set and clear bits in the CSR when the old value is not required: CSRS/<b>CSRC</b> csr, rs1; CSRSI/CSRCI csr, uimm.</div><br><div><b>Equivalent ASM:</b><pre>csrrc x0, csr, rs</pre></div><br><div><b>ISA</b>: csr(pseudo)</div></div>",
                "tooltip": "Further assembler pseudoinstructions are defined to set and clear bits in the CSR when the old value is not required: CSRS/CSRC csr, rs1; CSRSI/CSRCI csr, uimm.\n\nEquivalent ASM:\n\ncsrrc x0, csr, rs\n\n(ISA: csr(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/csr.html#csr-instructions"
            };

        case "CSRCI":
            return {
                "html": "<div><span class=\"opcode\"><b>CSRCI</b> csr, imm</span><br><div>Further assembler pseudoinstructions are defined to set and clear bits in the CSR when the old value is not required: CSRS/CSRC csr, rs1; CSRSI/<b>CSRCI</b> csr, uimm.</div><br><div><b>Equivalent ASM:</b><pre>csrrci x0, csr, imm</pre></div><br><div><b>ISA</b>: csr(pseudo)</div></div>",
                "tooltip": "Further assembler pseudoinstructions are defined to set and clear bits in the CSR when the old value is not required: CSRS/CSRC csr, rs1; CSRSI/CSRCI csr, uimm.\n\nEquivalent ASM:\n\ncsrrci x0, csr, imm\n\n(ISA: csr(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/csr.html#csr-instructions"
            };

        case "CSRR":
            return {
                "html": "<div><span class=\"opcode\"><b>CSRR</b> rd, csr</span><br><div>The assembler pseudoinstruction to read a CSR, <b>CSRR</b> rd, csr, is encoded as CSRRS rd, csr, x0. The assembler pseudoinstruction to write a CSR, CSRW csr, rs1, is encoded as CSRRW x0, csr, rs1, while CSRWI csr, uimm, is encoded as CSRRWI x0, csr, uimm.</div><br><div><b>Equivalent ASM:</b><pre>csrrs rd, csr, x0</pre></div><br><div><b>ISA</b>: csr(pseudo)</div></div>",
                "tooltip": "The assembler pseudoinstruction to read a CSR, CSRR rd, csr, is encoded as CSRRS rd, csr, x0. The assembler pseudoinstruction to write a CSR, CSRW csr, rs1, is encoded as CSRRW x0, csr, rs1, while CSRWI csr, uimm, is encoded as CSRRWI x0, csr, uimm.\n\nEquivalent ASM:\n\ncsrrs rd, csr, x0\n\n(ISA: csr(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/csr.html#csr-instructions"
            };

        case "CSRRC":
            return {
                "html": "<div><span class=\"opcode\"><b>CSRRC</b> rd, rs1</span><br><div>The <b>CSRRC</b> (Atomic Read and Clear Bits in CSR) instruction reads the value of the CSR, zero-extends the value to XLEN bits, and writes it to integer register rd. The initial value in integer register rs1 is treated as a bit mask that specifies bit positions to be cleared in the CSR. Any bit that is high in rs1 will cause the corresponding bit to be cleared in the CSR, if that CSR bit is writable. Other bits in the CSR are not explicitly written.<br>For both CSRRS and <b>CSRRC</b>, if rs1=x0, then the instruction will not write to the CSR at all, and so shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. Both CSRRS and <b>CSRRC</b> always read the addressed CSR and cause any read side effects regardless of rs1 and rd fields. Note that if rs1 specifies a register holding a zero value other than x0, the instruction will still attempt to write the unmodified value back to the CSR and will cause any attendant side effects. A CSRRW with rs1=x0 will attempt to write zero to the destination CSR.<br>The CSRRWI, CSRRSI, and CSRRCI variants are similar to CSRRW, CSRRS, and <b>CSRRC</b> respectively, except they update the CSR using an XLEN-bit value obtained by zero-extending a 5-bit unsigned immediate (uimm[4:0]) field encoded in the rs1 field instead of a value from an integer register. For CSRRSI and CSRRCI, if the uimm[4:0] field is zero, then these instructions will not write to the CSR, and shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. For CSRRWI, if rd=x0, then the instruction shall not read the CSR and shall not cause any of the side effects that might occur on a CSR read. Both CSRRSI and CSRRCI will always read the CSR and cause any read side effects regardless of rd and rs1 fields.</div><br><div><b>ISA</b>: csr</div></div>",
                "tooltip": "The CSRRC (Atomic Read and Clear Bits in CSR) instruction reads the value of the CSR, zero-extends the value to XLEN bits, and writes it to integer register rd. The initial value in integer register rs1 is treated as a bit mask that specifies bit positions to be cleared in the CSR. Any bit that is high in rs1 will cause the corresponding bit to be cleared in the CSR, if that CSR bit is writable. Other bits in the CSR are not explicitly written.\nFor both CSRRS and CSRRC, if rs1=x0, then the instruction will not write to the CSR at all, and so shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. Both CSRRS and CSRRC always read the addressed CSR and cause any read side effects regardless of rs1 and rd fields. Note that if rs1 specifies a register holding a zero value other than x0, the instruction will still attempt to write the unmodified value back to the CSR and will cause any attendant side effects. A CSRRW with rs1=x0 will attempt to write zero to the destination CSR.\nThe CSRRWI, CSRRSI, and CSRRCI variants are similar to CSRRW, CSRRS, and CSRRC respectively, except they update the CSR using an XLEN-bit value obtained by zero-extending a 5-bit unsigned immediate (uimm[4:0]) field encoded in the rs1 field instead of a value from an integer register. For CSRRSI and CSRRCI, if the uimm[4:0] field is zero, then these instructions will not write to the CSR, and shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. For CSRRWI, if rd=x0, then the instruction shall not read the CSR and shall not cause any of the side effects that might occur on a CSR read. Both CSRRSI and CSRRCI will always read the CSR and cause any read side effects regardless of rd and rs1 fields.\n\n\n\n(ISA: csr)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/csr.html#csr-instructions"
            };

        case "CSRRCI":
            return {
                "html": "<div><span class=\"opcode\"><b>CSRRCI</b> rd, zimm</span><br><div>The CSRRWI, CSRRSI, and <b>CSRRCI</b> variants are similar to CSRRW, CSRRS, and CSRRC respectively, except they update the CSR using an XLEN-bit value obtained by zero-extending a 5-bit unsigned immediate (uimm[4:0]) field encoded in the rs1 field instead of a value from an integer register. For CSRRSI and <b>CSRRCI</b>, if the uimm[4:0] field is zero, then these instructions will not write to the CSR, and shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. For CSRRWI, if rd=x0, then the instruction shall not read the CSR and shall not cause any of the side effects that might occur on a CSR read. Both CSRRSI and <b>CSRRCI</b> will always read the CSR and cause any read side effects regardless of rd and rs1 fields.</div><br><div><b>ISA</b>: csr</div></div>",
                "tooltip": "The CSRRWI, CSRRSI, and CSRRCI variants are similar to CSRRW, CSRRS, and CSRRC respectively, except they update the CSR using an XLEN-bit value obtained by zero-extending a 5-bit unsigned immediate (uimm[4:0]) field encoded in the rs1 field instead of a value from an integer register. For CSRRSI and CSRRCI, if the uimm[4:0] field is zero, then these instructions will not write to the CSR, and shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. For CSRRWI, if rd=x0, then the instruction shall not read the CSR and shall not cause any of the side effects that might occur on a CSR read. Both CSRRSI and CSRRCI will always read the CSR and cause any read side effects regardless of rd and rs1 fields.\n\n\n\n(ISA: csr)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/csr.html#csr-instructions"
            };

        case "CSRRS":
            return {
                "html": "<div><span class=\"opcode\"><b>CSRRS</b> rd, rs1</span><br><div>The <b>CSRRS</b> (Atomic Read and Set Bits in CSR) instruction reads the value of the CSR, zero-extends the value to XLEN bits, and writes it to integer register rd. The initial value in integer register rs1 is treated as a bit mask that specifies bit positions to be set in the CSR. Any bit that is high in rs1 will cause the corresponding bit to be set in the CSR, if that CSR bit is writable. Other bits in the CSR are not explicitly written.<br>For both <b>CSRRS</b> and CSRRC, if rs1=x0, then the instruction will not write to the CSR at all, and so shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. Both <b>CSRRS</b> and CSRRC always read the addressed CSR and cause any read side effects regardless of rs1 and rd fields. Note that if rs1 specifies a register holding a zero value other than x0, the instruction will still attempt to write the unmodified value back to the CSR and will cause any attendant side effects. A CSRRW with rs1=x0 will attempt to write zero to the destination CSR.<br>The CSRRWI, CSRRSI, and CSRRCI variants are similar to CSRRW, <b>CSRRS</b>, and CSRRC respectively, except they update the CSR using an XLEN-bit value obtained by zero-extending a 5-bit unsigned immediate (uimm[4:0]) field encoded in the rs1 field instead of a value from an integer register. For CSRRSI and CSRRCI, if the uimm[4:0] field is zero, then these instructions will not write to the CSR, and shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. For CSRRWI, if rd=x0, then the instruction shall not read the CSR and shall not cause any of the side effects that might occur on a CSR read. Both CSRRSI and CSRRCI will always read the CSR and cause any read side effects regardless of rd and rs1 fields.<br>The assembler pseudoinstruction to read a CSR, CSRR rd, csr, is encoded as <b>CSRRS</b> rd, csr, x0. The assembler pseudoinstruction to write a CSR, CSRW csr, rs1, is encoded as CSRRW x0, csr, rs1, while CSRWI csr, uimm, is encoded as CSRRWI x0, csr, uimm.</div><br><div><b>ISA</b>: csr</div></div>",
                "tooltip": "The CSRRS (Atomic Read and Set Bits in CSR) instruction reads the value of the CSR, zero-extends the value to XLEN bits, and writes it to integer register rd. The initial value in integer register rs1 is treated as a bit mask that specifies bit positions to be set in the CSR. Any bit that is high in rs1 will cause the corresponding bit to be set in the CSR, if that CSR bit is writable. Other bits in the CSR are not explicitly written.\nFor both CSRRS and CSRRC, if rs1=x0, then the instruction will not write to the CSR at all, and so shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. Both CSRRS and CSRRC always read the addressed CSR and cause any read side effects regardless of rs1 and rd fields. Note that if rs1 specifies a register holding a zero value other than x0, the instruction will still attempt to write the unmodified value back to the CSR and will cause any attendant side effects. A CSRRW with rs1=x0 will attempt to write zero to the destination CSR.\nThe CSRRWI, CSRRSI, and CSRRCI variants are similar to CSRRW, CSRRS, and CSRRC respectively, except they update the CSR using an XLEN-bit value obtained by zero-extending a 5-bit unsigned immediate (uimm[4:0]) field encoded in the rs1 field instead of a value from an integer register. For CSRRSI and CSRRCI, if the uimm[4:0] field is zero, then these instructions will not write to the CSR, and shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. For CSRRWI, if rd=x0, then the instruction shall not read the CSR and shall not cause any of the side effects that might occur on a CSR read. Both CSRRSI and CSRRCI will always read the CSR and cause any read side effects regardless of rd and rs1 fields.\nThe assembler pseudoinstruction to read a CSR, CSRR rd, csr, is encoded as CSRRS rd, csr, x0. The assembler pseudoinstruction to write a CSR, CSRW csr, rs1, is encoded as CSRRW x0, csr, rs1, while CSRWI csr, uimm, is encoded as CSRRWI x0, csr, uimm.\n\n\n\n(ISA: csr)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/csr.html#csr-instructions"
            };

        case "CSRRSI":
            return {
                "html": "<div><span class=\"opcode\"><b>CSRRSI</b> rd, zimm</span><br><div>The CSRRWI, <b>CSRRSI</b>, and CSRRCI variants are similar to CSRRW, CSRRS, and CSRRC respectively, except they update the CSR using an XLEN-bit value obtained by zero-extending a 5-bit unsigned immediate (uimm[4:0]) field encoded in the rs1 field instead of a value from an integer register. For <b>CSRRSI</b> and CSRRCI, if the uimm[4:0] field is zero, then these instructions will not write to the CSR, and shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. For CSRRWI, if rd=x0, then the instruction shall not read the CSR and shall not cause any of the side effects that might occur on a CSR read. Both <b>CSRRSI</b> and CSRRCI will always read the CSR and cause any read side effects regardless of rd and rs1 fields.</div><br><div><b>ISA</b>: csr</div></div>",
                "tooltip": "The CSRRWI, CSRRSI, and CSRRCI variants are similar to CSRRW, CSRRS, and CSRRC respectively, except they update the CSR using an XLEN-bit value obtained by zero-extending a 5-bit unsigned immediate (uimm[4:0]) field encoded in the rs1 field instead of a value from an integer register. For CSRRSI and CSRRCI, if the uimm[4:0] field is zero, then these instructions will not write to the CSR, and shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. For CSRRWI, if rd=x0, then the instruction shall not read the CSR and shall not cause any of the side effects that might occur on a CSR read. Both CSRRSI and CSRRCI will always read the CSR and cause any read side effects regardless of rd and rs1 fields.\n\n\n\n(ISA: csr)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/csr.html#csr-instructions"
            };

        case "CSRRW":
            return {
                "html": "<div><span class=\"opcode\"><b>CSRRW</b> rd, rs1</span><br><div>The <b>CSRRW</b> (Atomic Read/Write CSR) instruction atomically swaps values in the CSRs and integer registers. <b>CSRRW</b> reads the old value of the CSR, zero-extends the value to XLEN bits, then writes it to integer register rd. The initial value in rs1 is written to the CSR. If rd=x0, then the instruction shall not read the CSR and shall not cause any of the side effects that might occur on a CSR read.<br>For both CSRRS and CSRRC, if rs1=x0, then the instruction will not write to the CSR at all, and so shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. Both CSRRS and CSRRC always read the addressed CSR and cause any read side effects regardless of rs1 and rd fields. Note that if rs1 specifies a register holding a zero value other than x0, the instruction will still attempt to write the unmodified value back to the CSR and will cause any attendant side effects. A <b>CSRRW</b> with rs1=x0 will attempt to write zero to the destination CSR.<br>The CSRRWI, CSRRSI, and CSRRCI variants are similar to <b>CSRRW</b>, CSRRS, and CSRRC respectively, except they update the CSR using an XLEN-bit value obtained by zero-extending a 5-bit unsigned immediate (uimm[4:0]) field encoded in the rs1 field instead of a value from an integer register. For CSRRSI and CSRRCI, if the uimm[4:0] field is zero, then these instructions will not write to the CSR, and shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. For CSRRWI, if rd=x0, then the instruction shall not read the CSR and shall not cause any of the side effects that might occur on a CSR read. Both CSRRSI and CSRRCI will always read the CSR and cause any read side effects regardless of rd and rs1 fields.<br>The assembler pseudoinstruction to read a CSR, CSRR rd, csr, is encoded as CSRRS rd, csr, x0. The assembler pseudoinstruction to write a CSR, CSRW csr, rs1, is encoded as <b>CSRRW</b> x0, csr, rs1, while CSRWI csr, uimm, is encoded as CSRRWI x0, csr, uimm.</div><br><div><b>ISA</b>: csr</div></div>",
                "tooltip": "The CSRRW (Atomic Read/Write CSR) instruction atomically swaps values in the CSRs and integer registers. CSRRW reads the old value of the CSR, zero-extends the value to XLEN bits, then writes it to integer register rd. The initial value in rs1 is written to the CSR. If rd=x0, then the instruction shall not read the CSR and shall not cause any of the side effects that might occur on a CSR read.\nFor both CSRRS and CSRRC, if rs1=x0, then the instruction will not write to the CSR at all, and so shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. Both CSRRS and CSRRC always read the addressed CSR and cause any read side effects regardless of rs1 and rd fields. Note that if rs1 specifies a register holding a zero value other than x0, the instruction will still attempt to write the unmodified value back to the CSR and will cause any attendant side effects. A CSRRW with rs1=x0 will attempt to write zero to the destination CSR.\nThe CSRRWI, CSRRSI, and CSRRCI variants are similar to CSRRW, CSRRS, and CSRRC respectively, except they update the CSR using an XLEN-bit value obtained by zero-extending a 5-bit unsigned immediate (uimm[4:0]) field encoded in the rs1 field instead of a value from an integer register. For CSRRSI and CSRRCI, if the uimm[4:0] field is zero, then these instructions will not write to the CSR, and shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. For CSRRWI, if rd=x0, then the instruction shall not read the CSR and shall not cause any of the side effects that might occur on a CSR read. Both CSRRSI and CSRRCI will always read the CSR and cause any read side effects regardless of rd and rs1 fields.\nThe assembler pseudoinstruction to read a CSR, CSRR rd, csr, is encoded as CSRRS rd, csr, x0. The assembler pseudoinstruction to write a CSR, CSRW csr, rs1, is encoded as CSRRW x0, csr, rs1, while CSRWI csr, uimm, is encoded as CSRRWI x0, csr, uimm.\n\n\n\n(ISA: csr)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/csr.html#csr-instructions"
            };

        case "CSRRWI":
            return {
                "html": "<div><span class=\"opcode\"><b>CSRRWI</b> rd, zimm</span><br><div>The <b>CSRRWI</b>, CSRRSI, and CSRRCI variants are similar to CSRRW, CSRRS, and CSRRC respectively, except they update the CSR using an XLEN-bit value obtained by zero-extending a 5-bit unsigned immediate (uimm[4:0]) field encoded in the rs1 field instead of a value from an integer register. For CSRRSI and CSRRCI, if the uimm[4:0] field is zero, then these instructions will not write to the CSR, and shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. For <b>CSRRWI</b>, if rd=x0, then the instruction shall not read the CSR and shall not cause any of the side effects that might occur on a CSR read. Both CSRRSI and CSRRCI will always read the CSR and cause any read side effects regardless of rd and rs1 fields.<br>The assembler pseudoinstruction to read a CSR, CSRR rd, csr, is encoded as CSRRS rd, csr, x0. The assembler pseudoinstruction to write a CSR, CSRW csr, rs1, is encoded as CSRRW x0, csr, rs1, while CSRWI csr, uimm, is encoded as <b>CSRRWI</b> x0, csr, uimm.</div><br><div><b>ISA</b>: csr</div></div>",
                "tooltip": "The CSRRWI, CSRRSI, and CSRRCI variants are similar to CSRRW, CSRRS, and CSRRC respectively, except they update the CSR using an XLEN-bit value obtained by zero-extending a 5-bit unsigned immediate (uimm[4:0]) field encoded in the rs1 field instead of a value from an integer register. For CSRRSI and CSRRCI, if the uimm[4:0] field is zero, then these instructions will not write to the CSR, and shall not cause any of the side effects that might otherwise occur on a CSR write, nor raise illegal instruction exceptions on accesses to read-only CSRs. For CSRRWI, if rd=x0, then the instruction shall not read the CSR and shall not cause any of the side effects that might occur on a CSR read. Both CSRRSI and CSRRCI will always read the CSR and cause any read side effects regardless of rd and rs1 fields.\nThe assembler pseudoinstruction to read a CSR, CSRR rd, csr, is encoded as CSRRS rd, csr, x0. The assembler pseudoinstruction to write a CSR, CSRW csr, rs1, is encoded as CSRRW x0, csr, rs1, while CSRWI csr, uimm, is encoded as CSRRWI x0, csr, uimm.\n\n\n\n(ISA: csr)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/csr.html#csr-instructions"
            };

        case "CSRS":
            return {
                "html": "<div><span class=\"opcode\"><b>CSRS</b> csr, rs</span><br><div>Further assembler pseudoinstructions are defined to set and clear bits in the CSR when the old value is not required: <b>CSRS</b>/CSRC csr, rs1; CSRSI/CSRCI csr, uimm.</div><br><div><b>Equivalent ASM:</b><pre>csrrs x0, csr, rs</pre></div><br><div><b>ISA</b>: csr(pseudo)</div></div>",
                "tooltip": "Further assembler pseudoinstructions are defined to set and clear bits in the CSR when the old value is not required: CSRS/CSRC csr, rs1; CSRSI/CSRCI csr, uimm.\n\nEquivalent ASM:\n\ncsrrs x0, csr, rs\n\n(ISA: csr(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/csr.html#csr-instructions"
            };

        case "CSRSI":
            return {
                "html": "<div><span class=\"opcode\"><b>CSRSI</b> csr, imm</span><br><div>Further assembler pseudoinstructions are defined to set and clear bits in the CSR when the old value is not required: CSRS/CSRC csr, rs1; <b>CSRSI</b>/CSRCI csr, uimm.</div><br><div><b>Equivalent ASM:</b><pre>csrrsi x0, csr, imm</pre></div><br><div><b>ISA</b>: csr(pseudo)</div></div>",
                "tooltip": "Further assembler pseudoinstructions are defined to set and clear bits in the CSR when the old value is not required: CSRS/CSRC csr, rs1; CSRSI/CSRCI csr, uimm.\n\nEquivalent ASM:\n\ncsrrsi x0, csr, imm\n\n(ISA: csr(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/csr.html#csr-instructions"
            };

        case "CSRW":
            return {
                "html": "<div><span class=\"opcode\"><b>CSRW</b> csr, rs</span><br><div>The assembler pseudoinstruction to read a CSR, CSRR rd, csr, is encoded as CSRRS rd, csr, x0. The assembler pseudoinstruction to write a CSR, <b>CSRW</b> csr, rs1, is encoded as CSRRW x0, csr, rs1, while CSRWI csr, uimm, is encoded as CSRRWI x0, csr, uimm.</div><br><div><b>Equivalent ASM:</b><pre>csrrw x0, csr, rs</pre></div><br><div><b>ISA</b>: csr(pseudo)</div></div>",
                "tooltip": "The assembler pseudoinstruction to read a CSR, CSRR rd, csr, is encoded as CSRRS rd, csr, x0. The assembler pseudoinstruction to write a CSR, CSRW csr, rs1, is encoded as CSRRW x0, csr, rs1, while CSRWI csr, uimm, is encoded as CSRRWI x0, csr, uimm.\n\nEquivalent ASM:\n\ncsrrw x0, csr, rs\n\n(ISA: csr(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/csr.html#csr-instructions"
            };

        case "CSRWI":
            return {
                "html": "<div><span class=\"opcode\"><b>CSRWI</b> csr, imm</span><br><div>The assembler pseudoinstruction to read a CSR, CSRR rd, csr, is encoded as CSRRS rd, csr, x0. The assembler pseudoinstruction to write a CSR, CSRW csr, rs1, is encoded as CSRRW x0, csr, rs1, while <b>CSRWI</b> csr, uimm, is encoded as CSRRWI x0, csr, uimm.</div><br><div><b>Equivalent ASM:</b><pre>csrrwi x0, csr, imm</pre></div><br><div><b>ISA</b>: csr(pseudo)</div></div>",
                "tooltip": "The assembler pseudoinstruction to read a CSR, CSRR rd, csr, is encoded as CSRRS rd, csr, x0. The assembler pseudoinstruction to write a CSR, CSRW csr, rs1, is encoded as CSRRW x0, csr, rs1, while CSRWI csr, uimm, is encoded as CSRRWI x0, csr, uimm.\n\nEquivalent ASM:\n\ncsrrwi x0, csr, imm\n\n(ISA: csr(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/csr.html#csr-instructions"
            };

        case "CTZ":
            return {
                "html": "<div><span class=\"opcode\"><b>CTZ</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "CTZW":
            return {
                "html": "<div><span class=\"opcode\"><b>CTZW</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "DIV":
            return {
                "html": "<div><span class=\"opcode\"><b>DIV</b> rd, rs1, rs2</span><br><div><b>DIV</b> and DIVU perform an XLEN bits by XLEN bits signed and unsigned integer division of rs1 by rs2, rounding towards zero. REM and REMU provide the remainder of the corresponding division operation. For REM, the sign of the result equals the sign of the dividend.<br>If both the quotient and remainder are required from the same division, the recommended code sequence is: <b>DIV</b>[U] rdq, rs1, rs2; REM[U] rdr, rs1, rs2 (rdq cannot be the same as rs1 or rs2). Microarchitectures can then fuse these into a single divide operation instead of performing two separate divides.<br><b>DIV</b>[W]</div><br><div><b>ISA</b>: m</div></div>",
                "tooltip": "DIV and DIVU perform an XLEN bits by XLEN bits signed and unsigned integer division of rs1 by rs2, rounding towards zero. REM and REMU provide the remainder of the corresponding division operation. For REM, the sign of the result equals the sign of the dividend.\nIf both the quotient and remainder are required from the same division, the recommended code sequence is: DIV[U] rdq, rs1, rs2; REM[U] rdr, rs1, rs2 (rdq cannot be the same as rs1 or rs2). Microarchitectures can then fuse these into a single divide operation instead of performing two separate divides.\nDIV[W]\n\n\n\n(ISA: m)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/m.html#division-operations"
            };

        case "DIVU":
            return {
                "html": "<div><span class=\"opcode\"><b>DIVU</b> rd, rs1, rs2</span><br><div>DIV and <b>DIVU</b> perform an XLEN bits by XLEN bits signed and unsigned integer division of rs1 by rs2, rounding towards zero. REM and REMU provide the remainder of the corresponding division operation. For REM, the sign of the result equals the sign of the dividend.<br><b>DIVU</b>[W]</div><br><div><b>ISA</b>: m</div></div>",
                "tooltip": "DIV and DIVU perform an XLEN bits by XLEN bits signed and unsigned integer division of rs1 by rs2, rounding towards zero. REM and REMU provide the remainder of the corresponding division operation. For REM, the sign of the result equals the sign of the dividend.\nDIVU[W]\n\n\n\n(ISA: m)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/m.html#division-operations"
            };

        case "DIVUW":
            return {
                "html": "<div><span class=\"opcode\"><b>DIVUW</b> rd, rs1, rs2</span><br><div>DIVW and <b>DIVUW</b> are RV64 instructions that divide the lower 32 bits of rs1 by the lower 32 bits of rs2, treating them as signed and unsigned integers respectively, placing the 32-bit quotient in rd, sign-extended to 64 bits. REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned remainder operations respectively. Both REMW and REMUW always sign-extend the 32-bit result to 64 bits, including on a divide by zero.</div><br><div><b>ISA</b>: m</div></div>",
                "tooltip": "DIVW and DIVUW are RV64 instructions that divide the lower 32 bits of rs1 by the lower 32 bits of rs2, treating them as signed and unsigned integers respectively, placing the 32-bit quotient in rd, sign-extended to 64 bits. REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned remainder operations respectively. Both REMW and REMUW always sign-extend the 32-bit result to 64 bits, including on a divide by zero.\n\n\n\n(ISA: m)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/m.html#division-operations"
            };

        case "DIVW":
            return {
                "html": "<div><span class=\"opcode\"><b>DIVW</b> rd, rs1, rs2</span><br><div><b>DIVW</b> and DIVUW are RV64 instructions that divide the lower 32 bits of rs1 by the lower 32 bits of rs2, treating them as signed and unsigned integers respectively, placing the 32-bit quotient in rd, sign-extended to 64 bits. REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned remainder operations respectively. Both REMW and REMUW always sign-extend the 32-bit result to 64 bits, including on a divide by zero.</div><br><div><b>ISA</b>: m</div></div>",
                "tooltip": "DIVW and DIVUW are RV64 instructions that divide the lower 32 bits of rs1 by the lower 32 bits of rs2, treating them as signed and unsigned integers respectively, placing the 32-bit quotient in rd, sign-extended to 64 bits. REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned remainder operations respectively. Both REMW and REMUW always sign-extend the 32-bit result to 64 bits, including on a divide by zero.\n\n\n\n(ISA: m)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/m.html#division-operations"
            };

        case "DRET":
            return {
                "html": "<div><span class=\"opcode\"><b>DRET</b> </span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "EBREAK":
            return {
                "html": "<div><span class=\"opcode\"><b>EBREAK</b> </span><br><div>RV32I was designed to be sufficient to form a compiler target and to support modern operating system environments. The ISA was also designed to reduce the hardware required in a minimal implementation. RV32I contains 40 unique instructions, though a simple implementation might cover the ECALL/<b>EBREAK</b> instructions with a single SYSTEM hardware instruction that always traps and might be able to implement the FENCE instruction as a NOP, reducing base instruction count to 38 total. RV32I can emulate almost any other ISA extension (except the A extension, which requires additional hardware support for atomicity).</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "RV32I was designed to be sufficient to form a compiler target and to support modern operating system environments. The ISA was also designed to reduce the hardware required in a minimal implementation. RV32I contains 40 unique instructions, though a simple implementation might cover the ECALL/EBREAK instructions with a single SYSTEM hardware instruction that always traps and might be able to implement the FENCE instruction as a NOP, reducing base instruction count to 38 total. RV32I can emulate almost any other ISA extension (except the A extension, which requires additional hardware support for atomicity).\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#rv32"
            };

        case "ECALL":
            return {
                "html": "<div><span class=\"opcode\"><b>ECALL</b> </span><br><div>RV32I was designed to be sufficient to form a compiler target and to support modern operating system environments. The ISA was also designed to reduce the hardware required in a minimal implementation. RV32I contains 40 unique instructions, though a simple implementation might cover the <b>ECALL</b>/EBREAK instructions with a single SYSTEM hardware instruction that always traps and might be able to implement the FENCE instruction as a NOP, reducing base instruction count to 38 total. RV32I can emulate almost any other ISA extension (except the A extension, which requires additional hardware support for atomicity).</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "RV32I was designed to be sufficient to form a compiler target and to support modern operating system environments. The ISA was also designed to reduce the hardware required in a minimal implementation. RV32I contains 40 unique instructions, though a simple implementation might cover the ECALL/EBREAK instructions with a single SYSTEM hardware instruction that always traps and might be able to implement the FENCE instruction as a NOP, reducing base instruction count to 38 total. RV32I can emulate almost any other ISA extension (except the A extension, which requires additional hardware support for atomicity).\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#rv32"
            };

        case "FABS.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FABS.D</b> rd, rs</span><br><div><b>Equivalent ASM:</b><pre>fsgnjx.d rd, rs, rs</pre></div><br><div><b>ISA</b>: d(pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nfsgnjx.d rd, rs, rs\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FABS.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FABS.S</b> rd, rs</span><br><div>Floating-point to floating-point sign-injection instructions, FSGNJ.S, FSGNJN.S, and FSGNJX.S, produce a result that takes all bits except the sign bit from rs1. For FSGNJ, the result's sign bit is rs2's sign bit; for FSGNJN, the result's sign bit is the opposite of rs2's sign bit; and for FSGNJX, the sign bit is the XOR of the sign bits of rs1 and rs2. Sign-injection instructions do not set floating-point exception flags, nor do they canonicalize NaNs. Note, FSGNJ.S rx, ry, ry moves ry to rx (assembler pseudoinstruction FMV.S rx, ry); FSGNJN.S rx, ry, ry moves the negation of ry to rx (assembler pseudoinstruction FNEG.S rx, ry); and FSGNJX.S rx, ry, ry moves the absolute value of ry to rx (assembler pseudoinstruction <b>FABS.S</b> rx, ry).</div><br><div><b>Equivalent ASM:</b><pre>fsgnjx.s rd, rs, rs</pre></div><br><div><b>ISA</b>: f(pseudo)</div></div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.S, FSGNJN.S, and FSGNJX.S, produce a result that takes all bits except the sign bit from rs1. For FSGNJ, the result's sign bit is rs2's sign bit; for FSGNJN, the result's sign bit is the opposite of rs2's sign bit; and for FSGNJX, the sign bit is the XOR of the sign bits of rs1 and rs2. Sign-injection instructions do not set floating-point exception flags, nor do they canonicalize NaNs. Note, FSGNJ.S rx, ry, ry moves ry to rx (assembler pseudoinstruction FMV.S rx, ry); FSGNJN.S rx, ry, ry moves the negation of ry to rx (assembler pseudoinstruction FNEG.S rx, ry); and FSGNJX.S rx, ry, ry moves the absolute value of ry to rx (assembler pseudoinstruction FABS.S rx, ry).\n\nEquivalent ASM:\n\nfsgnjx.s rd, rs, rs\n\n(ISA: f(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FADD.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FADD.D</b> rd, rs1, rs2</span><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FADD.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FADD.H</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FADD.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FADD.Q</b> rd, rs1, rs2</span><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FADD.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FADD.S</b> rd, rs1, rs2</span><br><div>Floating-point arithmetic instructions with one or two source operands use the R-type format with the OP-FP major opcode. <b>FADD.S</b> and FMUL.S perform single-precision floating-point addition and multiplication respectively, between rs1 and rs2. FSUB.S performs the single-precision floating-point subtraction of rs2 from rs1. FDIV.S performs the single-precision floating-point division of rs1 by rs2. FSQRT.S computes the square root of rs1. In each case, the result is written to rd.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point arithmetic instructions with one or two source operands use the R-type format with the OP-FP major opcode. FADD.S and FMUL.S perform single-precision floating-point addition and multiplication respectively, between rs1 and rs2. FSUB.S performs the single-precision floating-point subtraction of rs2 from rs1. FDIV.S performs the single-precision floating-point division of rs1 by rs2. FSQRT.S computes the square root of rs1. In each case, the result is written to rd.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#sec:single-float-compute"
            };

        case "FCLASS.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FCLASS.D</b> rd, rs1</span><br><div>The double-precision floating-point classify instruction, <b>FCLASS.D</b>, is defined analogously to its single-precision counterpart, but operates on double-precision operands.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "The double-precision floating-point classify instruction, FCLASS.D, is defined analogously to its single-precision counterpart, but operates on double-precision operands.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#double-precision-floating-point-classify-instruction"
            };

        case "FCLASS.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FCLASS.H</b> rd, rs1</span><br><div>The half-precision floating-point classify instruction, <b>FCLASS.H</b>, is defined analogously to its single-precision counterpart, but operates on half-precision operands.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "The half-precision floating-point classify instruction, FCLASS.H, is defined analogously to its single-precision counterpart, but operates on half-precision operands.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-floating-point-classify-instruction"
            };

        case "FCLASS.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FCLASS.Q</b> rd, rs1</span><br><div>The quad-precision floating-point classify instruction, <b>FCLASS.Q</b>, is defined analogously to its double-precision counterpart, but operates on quad-precision operands.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "The quad-precision floating-point classify instruction, FCLASS.Q, is defined analogously to its double-precision counterpart, but operates on quad-precision operands.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-floating-point-classify-instruction"
            };

        case "FCLASS.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FCLASS.S</b> rd, rs1</span><br><div>The <b>FCLASS.S</b> instruction examines the value in floating-point register rs1 and writes to integer register rd a 10-bit mask that indicates the class of the floating-point number. The format of the mask is described in Table [tab:fclass] . The corresponding bit in rd will be set if the property is true and clear otherwise. All other bits in rd are cleared. Note that exactly one bit in rd will be set. <b>FCLASS.S</b> does not set the floating-point exception flags.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "The FCLASS.S instruction examines the value in floating-point register rs1 and writes to integer register rd a 10-bit mask that indicates the class of the floating-point number. The format of the mask is described in Table [tab:fclass] . The corresponding bit in rd will be set if the property is true and clear otherwise. All other bits in rd are cleared. Note that exactly one bit in rd will be set. FCLASS.S does not set the floating-point exception flags.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-classify-instruction"
            };

        case "FCVT.D.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.D.H</b> rd, rs1</span><br><div>New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.H or FCVT.H.S converts a half-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. If the D extension is present, <b>FCVT.D.H</b> or FCVT.H.D converts a half-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively. If the Q extension is present, FCVT.Q.H or FCVT.H.Q converts a half-precision floating-point number to a quad-precision floating-point number, or vice-versa, respectively.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.H or FCVT.H.S converts a half-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. If the D extension is present, FCVT.D.H or FCVT.H.D converts a half-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively. If the Q extension is present, FCVT.Q.H or FCVT.H.Q converts a half-precision floating-point number to a quad-precision floating-point number, or vice-versa, respectively.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FCVT.D.L":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.D.L</b> rd, rs1</span><br><div>Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.D or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.D.W or <b>FCVT.D.L</b> converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd. FCVT.WU.D, FCVT.LU.D, FCVT.D.WU, and FCVT.D.LU variants convert to or from unsigned integer values. For RV64, FCVT.W[U].D sign-extends the 32-bit result. FCVT.L[U].D and <b>FCVT.D.L</b>[U] are RV64-only instructions. The range of valid inputs for FCVT.int.D and the behavior for invalid inputs are the same as for FCVT.int.S.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.D or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.D.W or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd. FCVT.WU.D, FCVT.LU.D, FCVT.D.WU, and FCVT.D.LU variants convert to or from unsigned integer values. For RV64, FCVT.W[U].D sign-extends the 32-bit result. FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions. The range of valid inputs for FCVT.int.D and the behavior for invalid inputs are the same as for FCVT.int.S.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.D.LU":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.D.LU</b> rd, rs1</span><br><div>Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.D or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.D.W or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd. FCVT.WU.D, FCVT.LU.D, FCVT.D.WU, and <b>FCVT.D.LU</b> variants convert to or from unsigned integer values. For RV64, FCVT.W[U].D sign-extends the 32-bit result. FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions. The range of valid inputs for FCVT.int.D and the behavior for invalid inputs are the same as for FCVT.int.S.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.D or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.D.W or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd. FCVT.WU.D, FCVT.LU.D, FCVT.D.WU, and FCVT.D.LU variants convert to or from unsigned integer values. For RV64, FCVT.W[U].D sign-extends the 32-bit result. FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions. The range of valid inputs for FCVT.int.D and the behavior for invalid inputs are the same as for FCVT.int.S.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.D.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.D.Q</b> rd, rs1</span><br><div>New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.Q or FCVT.Q.S converts a quad-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. <b>FCVT.D.Q</b> or FCVT.Q.D converts a quad-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.Q or FCVT.Q.S converts a quad-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. FCVT.D.Q or FCVT.Q.D converts a quad-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.D.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.D.S</b> rd, rs1</span><br><div>The double-precision to single-precision and single-precision to double-precision conversion instructions, FCVT.S.D and <b>FCVT.D.S</b>, are encoded in the OP-FP major opcode space and both the source and destination are floating-point registers. The rs2 field encodes the datatype of the source, and the fmt field encodes the datatype of the destination. FCVT.S.D rounds according to the RM field; <b>FCVT.D.S</b> will never round.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "The double-precision to single-precision and single-precision to double-precision conversion instructions, FCVT.S.D and FCVT.D.S, are encoded in the OP-FP major opcode space and both the source and destination are floating-point registers. The rs2 field encodes the datatype of the source, and the fmt field encodes the datatype of the destination. FCVT.S.D rounds according to the RM field; FCVT.D.S will never round.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.D.W":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.D.W</b> rd, rs1</span><br><div>Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.D or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. <b>FCVT.D.W</b> or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd. FCVT.WU.D, FCVT.LU.D, FCVT.D.WU, and FCVT.D.LU variants convert to or from unsigned integer values. For RV64, FCVT.W[U].D sign-extends the 32-bit result. FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions. The range of valid inputs for FCVT.int.D and the behavior for invalid inputs are the same as for FCVT.int.S.<br>All floating-point to integer and integer to floating-point conversion instructions round according to the rm field. Note <b>FCVT.D.W</b>[U] always produces an exact result and is unaffected by rounding mode.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.D or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.D.W or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd. FCVT.WU.D, FCVT.LU.D, FCVT.D.WU, and FCVT.D.LU variants convert to or from unsigned integer values. For RV64, FCVT.W[U].D sign-extends the 32-bit result. FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions. The range of valid inputs for FCVT.int.D and the behavior for invalid inputs are the same as for FCVT.int.S.\nAll floating-point to integer and integer to floating-point conversion instructions round according to the rm field. Note FCVT.D.W[U] always produces an exact result and is unaffected by rounding mode.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.D.WU":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.D.WU</b> rd, rs1</span><br><div>Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.D or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.D.W or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd. FCVT.WU.D, FCVT.LU.D, <b>FCVT.D.WU</b>, and FCVT.D.LU variants convert to or from unsigned integer values. For RV64, FCVT.W[U].D sign-extends the 32-bit result. FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions. The range of valid inputs for FCVT.int.D and the behavior for invalid inputs are the same as for FCVT.int.S.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.D or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.D.W or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd. FCVT.WU.D, FCVT.LU.D, FCVT.D.WU, and FCVT.D.LU variants convert to or from unsigned integer values. For RV64, FCVT.W[U].D sign-extends the 32-bit result. FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions. The range of valid inputs for FCVT.int.D and the behavior for invalid inputs are the same as for FCVT.int.S.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.H.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.H.D</b> rd, rs1</span><br><div>New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.H or FCVT.H.S converts a half-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. If the D extension is present, FCVT.D.H or <b>FCVT.H.D</b> converts a half-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively. If the Q extension is present, FCVT.Q.H or FCVT.H.Q converts a half-precision floating-point number to a quad-precision floating-point number, or vice-versa, respectively.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.H or FCVT.H.S converts a half-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. If the D extension is present, FCVT.D.H or FCVT.H.D converts a half-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively. If the Q extension is present, FCVT.Q.H or FCVT.H.Q converts a half-precision floating-point number to a quad-precision floating-point number, or vice-versa, respectively.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FCVT.H.L":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.H.L</b> rd, rs1</span><br><div>New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the single-precision-to-integer and integer-to-single-precision conversion instructions. FCVT.W.H or FCVT.L.H converts a half-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.H.W or <b>FCVT.H.L</b> converts a 32-bit or 64-bit signed integer, respectively, into a half-precision floating-point number. FCVT.WU.H, FCVT.LU.H, FCVT.H.WU, and FCVT.H.LU variants convert to or from unsigned integer values. FCVT.L[U].H and <b>FCVT.H.L</b>[U] are RV64-only instructions.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the single-precision-to-integer and integer-to-single-precision conversion instructions. FCVT.W.H or FCVT.L.H converts a half-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.H.W or FCVT.H.L converts a 32-bit or 64-bit signed integer, respectively, into a half-precision floating-point number. FCVT.WU.H, FCVT.LU.H, FCVT.H.WU, and FCVT.H.LU variants convert to or from unsigned integer values. FCVT.L[U].H and FCVT.H.L[U] are RV64-only instructions.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FCVT.H.LU":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.H.LU</b> rd, rs1</span><br><div>New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the single-precision-to-integer and integer-to-single-precision conversion instructions. FCVT.W.H or FCVT.L.H converts a half-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.H.W or FCVT.H.L converts a 32-bit or 64-bit signed integer, respectively, into a half-precision floating-point number. FCVT.WU.H, FCVT.LU.H, FCVT.H.WU, and <b>FCVT.H.LU</b> variants convert to or from unsigned integer values. FCVT.L[U].H and FCVT.H.L[U] are RV64-only instructions.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the single-precision-to-integer and integer-to-single-precision conversion instructions. FCVT.W.H or FCVT.L.H converts a half-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.H.W or FCVT.H.L converts a 32-bit or 64-bit signed integer, respectively, into a half-precision floating-point number. FCVT.WU.H, FCVT.LU.H, FCVT.H.WU, and FCVT.H.LU variants convert to or from unsigned integer values. FCVT.L[U].H and FCVT.H.L[U] are RV64-only instructions.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FCVT.H.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.H.Q</b> rd, rs1</span><br><div>New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.H or FCVT.H.S converts a half-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. If the D extension is present, FCVT.D.H or FCVT.H.D converts a half-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively. If the Q extension is present, FCVT.Q.H or <b>FCVT.H.Q</b> converts a half-precision floating-point number to a quad-precision floating-point number, or vice-versa, respectively.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.H or FCVT.H.S converts a half-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. If the D extension is present, FCVT.D.H or FCVT.H.D converts a half-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively. If the Q extension is present, FCVT.Q.H or FCVT.H.Q converts a half-precision floating-point number to a quad-precision floating-point number, or vice-versa, respectively.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FCVT.H.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.H.S</b> rd, rs1</span><br><div>New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.H or <b>FCVT.H.S</b> converts a half-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. If the D extension is present, FCVT.D.H or FCVT.H.D converts a half-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively. If the Q extension is present, FCVT.Q.H or FCVT.H.Q converts a half-precision floating-point number to a quad-precision floating-point number, or vice-versa, respectively.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.H or FCVT.H.S converts a half-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. If the D extension is present, FCVT.D.H or FCVT.H.D converts a half-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively. If the Q extension is present, FCVT.Q.H or FCVT.H.Q converts a half-precision floating-point number to a quad-precision floating-point number, or vice-versa, respectively.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FCVT.H.W":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.H.W</b> rd, rs1</span><br><div>New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the single-precision-to-integer and integer-to-single-precision conversion instructions. FCVT.W.H or FCVT.L.H converts a half-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. <b>FCVT.H.W</b> or FCVT.H.L converts a 32-bit or 64-bit signed integer, respectively, into a half-precision floating-point number. FCVT.WU.H, FCVT.LU.H, FCVT.H.WU, and FCVT.H.LU variants convert to or from unsigned integer values. FCVT.L[U].H and FCVT.H.L[U] are RV64-only instructions.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the single-precision-to-integer and integer-to-single-precision conversion instructions. FCVT.W.H or FCVT.L.H converts a half-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.H.W or FCVT.H.L converts a 32-bit or 64-bit signed integer, respectively, into a half-precision floating-point number. FCVT.WU.H, FCVT.LU.H, FCVT.H.WU, and FCVT.H.LU variants convert to or from unsigned integer values. FCVT.L[U].H and FCVT.H.L[U] are RV64-only instructions.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FCVT.H.WU":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.H.WU</b> rd, rs1</span><br><div>New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the single-precision-to-integer and integer-to-single-precision conversion instructions. FCVT.W.H or FCVT.L.H converts a half-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.H.W or FCVT.H.L converts a 32-bit or 64-bit signed integer, respectively, into a half-precision floating-point number. FCVT.WU.H, FCVT.LU.H, <b>FCVT.H.WU</b>, and FCVT.H.LU variants convert to or from unsigned integer values. FCVT.L[U].H and FCVT.H.L[U] are RV64-only instructions.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the single-precision-to-integer and integer-to-single-precision conversion instructions. FCVT.W.H or FCVT.L.H converts a half-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.H.W or FCVT.H.L converts a 32-bit or 64-bit signed integer, respectively, into a half-precision floating-point number. FCVT.WU.H, FCVT.LU.H, FCVT.H.WU, and FCVT.H.LU variants convert to or from unsigned integer values. FCVT.L[U].H and FCVT.H.L[U] are RV64-only instructions.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FCVT.L.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.L.D</b> rd, rs1</span><br><div>Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.D or <b>FCVT.L.D</b> converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.D.W or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd. FCVT.WU.D, FCVT.LU.D, FCVT.D.WU, and FCVT.D.LU variants convert to or from unsigned integer values. For RV64, FCVT.W[U].D sign-extends the 32-bit result. FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions. The range of valid inputs for FCVT.int.D and the behavior for invalid inputs are the same as for FCVT.int.S.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.D or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.D.W or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd. FCVT.WU.D, FCVT.LU.D, FCVT.D.WU, and FCVT.D.LU variants convert to or from unsigned integer values. For RV64, FCVT.W[U].D sign-extends the 32-bit result. FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions. The range of valid inputs for FCVT.int.D and the behavior for invalid inputs are the same as for FCVT.int.S.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.L.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.L.H</b> rd, rs1</span><br><div>New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the single-precision-to-integer and integer-to-single-precision conversion instructions. FCVT.W.H or <b>FCVT.L.H</b> converts a half-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.H.W or FCVT.H.L converts a 32-bit or 64-bit signed integer, respectively, into a half-precision floating-point number. FCVT.WU.H, FCVT.LU.H, FCVT.H.WU, and FCVT.H.LU variants convert to or from unsigned integer values. FCVT.L[U].H and FCVT.H.L[U] are RV64-only instructions.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the single-precision-to-integer and integer-to-single-precision conversion instructions. FCVT.W.H or FCVT.L.H converts a half-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.H.W or FCVT.H.L converts a 32-bit or 64-bit signed integer, respectively, into a half-precision floating-point number. FCVT.WU.H, FCVT.LU.H, FCVT.H.WU, and FCVT.H.LU variants convert to or from unsigned integer values. FCVT.L[U].H and FCVT.H.L[U] are RV64-only instructions.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FCVT.L.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.L.Q</b> rd, rs1</span><br><div>New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision-to-integer and integer-to-double-precision conversion instructions. FCVT.W.Q or <b>FCVT.L.Q</b> converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.Q.W or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number. FCVT.WU.Q, FCVT.LU.Q, FCVT.Q.WU, and FCVT.Q.LU variants convert to or from unsigned integer values. FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision-to-integer and integer-to-double-precision conversion instructions. FCVT.W.Q or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.Q.W or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number. FCVT.WU.Q, FCVT.LU.Q, FCVT.Q.WU, and FCVT.Q.LU variants convert to or from unsigned integer values. FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.L.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.L.S</b> rd, rs1</span><br><div>Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.S or <b>FCVT.L.S</b> converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.S.W or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd. FCVT.WU.S, FCVT.LU.S, FCVT.S.WU, and FCVT.S.LU variants convert to or from unsigned integer values. For XLEN > 32, FCVT.W[U].S sign-extends the 32-bit result to the destination register width. FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions. If the rounded result is not representable in the destination format, it is clipped to the nearest value and the invalid flag is set. Table [tab:int_conv] gives the range of valid inputs for FCVT.int.S and the behavior for invalid inputs.<br><b>FCVT.L.S</b></div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.S or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.S.W or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd. FCVT.WU.S, FCVT.LU.S, FCVT.S.WU, and FCVT.S.LU variants convert to or from unsigned integer values. For XLEN > 32, FCVT.W[U].S sign-extends the 32-bit result to the destination register width. FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions. If the rounded result is not representable in the destination format, it is clipped to the nearest value and the invalid flag is set. Table [tab:int_conv] gives the range of valid inputs for FCVT.int.S and the behavior for invalid inputs.\nFCVT.L.S\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.LU.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.LU.D</b> rd, rs1</span><br><div>Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.D or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.D.W or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd. FCVT.WU.D, <b>FCVT.LU.D</b>, FCVT.D.WU, and FCVT.D.LU variants convert to or from unsigned integer values. For RV64, FCVT.W[U].D sign-extends the 32-bit result. FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions. The range of valid inputs for FCVT.int.D and the behavior for invalid inputs are the same as for FCVT.int.S.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.D or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.D.W or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd. FCVT.WU.D, FCVT.LU.D, FCVT.D.WU, and FCVT.D.LU variants convert to or from unsigned integer values. For RV64, FCVT.W[U].D sign-extends the 32-bit result. FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions. The range of valid inputs for FCVT.int.D and the behavior for invalid inputs are the same as for FCVT.int.S.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.LU.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.LU.H</b> rd, rs1</span><br><div>New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the single-precision-to-integer and integer-to-single-precision conversion instructions. FCVT.W.H or FCVT.L.H converts a half-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.H.W or FCVT.H.L converts a 32-bit or 64-bit signed integer, respectively, into a half-precision floating-point number. FCVT.WU.H, <b>FCVT.LU.H</b>, FCVT.H.WU, and FCVT.H.LU variants convert to or from unsigned integer values. FCVT.L[U].H and FCVT.H.L[U] are RV64-only instructions.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the single-precision-to-integer and integer-to-single-precision conversion instructions. FCVT.W.H or FCVT.L.H converts a half-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.H.W or FCVT.H.L converts a 32-bit or 64-bit signed integer, respectively, into a half-precision floating-point number. FCVT.WU.H, FCVT.LU.H, FCVT.H.WU, and FCVT.H.LU variants convert to or from unsigned integer values. FCVT.L[U].H and FCVT.H.L[U] are RV64-only instructions.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FCVT.LU.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.LU.Q</b> rd, rs1</span><br><div>New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision-to-integer and integer-to-double-precision conversion instructions. FCVT.W.Q or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.Q.W or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number. FCVT.WU.Q, <b>FCVT.LU.Q</b>, FCVT.Q.WU, and FCVT.Q.LU variants convert to or from unsigned integer values. FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision-to-integer and integer-to-double-precision conversion instructions. FCVT.W.Q or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.Q.W or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number. FCVT.WU.Q, FCVT.LU.Q, FCVT.Q.WU, and FCVT.Q.LU variants convert to or from unsigned integer values. FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.LU.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.LU.S</b> rd, rs1</span><br><div>Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.S or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.S.W or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd. FCVT.WU.S, <b>FCVT.LU.S</b>, FCVT.S.WU, and FCVT.S.LU variants convert to or from unsigned integer values. For XLEN > 32, FCVT.W[U].S sign-extends the 32-bit result to the destination register width. FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions. If the rounded result is not representable in the destination format, it is clipped to the nearest value and the invalid flag is set. Table [tab:int_conv] gives the range of valid inputs for FCVT.int.S and the behavior for invalid inputs.<br><b>FCVT.LU.S</b></div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.S or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.S.W or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd. FCVT.WU.S, FCVT.LU.S, FCVT.S.WU, and FCVT.S.LU variants convert to or from unsigned integer values. For XLEN > 32, FCVT.W[U].S sign-extends the 32-bit result to the destination register width. FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions. If the rounded result is not representable in the destination format, it is clipped to the nearest value and the invalid flag is set. Table [tab:int_conv] gives the range of valid inputs for FCVT.int.S and the behavior for invalid inputs.\nFCVT.LU.S\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.Q.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.Q.D</b> rd, rs1</span><br><div>New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.Q or FCVT.Q.S converts a quad-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. FCVT.D.Q or <b>FCVT.Q.D</b> converts a quad-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.Q or FCVT.Q.S converts a quad-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. FCVT.D.Q or FCVT.Q.D converts a quad-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.Q.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.Q.H</b> rd, rs1</span><br><div>New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.H or FCVT.H.S converts a half-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. If the D extension is present, FCVT.D.H or FCVT.H.D converts a half-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively. If the Q extension is present, <b>FCVT.Q.H</b> or FCVT.H.Q converts a half-precision floating-point number to a quad-precision floating-point number, or vice-versa, respectively.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.H or FCVT.H.S converts a half-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. If the D extension is present, FCVT.D.H or FCVT.H.D converts a half-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively. If the Q extension is present, FCVT.Q.H or FCVT.H.Q converts a half-precision floating-point number to a quad-precision floating-point number, or vice-versa, respectively.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FCVT.Q.L":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.Q.L</b> rd, rs1</span><br><div>New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision-to-integer and integer-to-double-precision conversion instructions. FCVT.W.Q or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.Q.W or <b>FCVT.Q.L</b> converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number. FCVT.WU.Q, FCVT.LU.Q, FCVT.Q.WU, and FCVT.Q.LU variants convert to or from unsigned integer values. FCVT.L[U].Q and <b>FCVT.Q.L</b>[U] are RV64-only instructions.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision-to-integer and integer-to-double-precision conversion instructions. FCVT.W.Q or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.Q.W or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number. FCVT.WU.Q, FCVT.LU.Q, FCVT.Q.WU, and FCVT.Q.LU variants convert to or from unsigned integer values. FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.Q.LU":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.Q.LU</b> rd, rs1</span><br><div>New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision-to-integer and integer-to-double-precision conversion instructions. FCVT.W.Q or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.Q.W or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number. FCVT.WU.Q, FCVT.LU.Q, FCVT.Q.WU, and <b>FCVT.Q.LU</b> variants convert to or from unsigned integer values. FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision-to-integer and integer-to-double-precision conversion instructions. FCVT.W.Q or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.Q.W or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number. FCVT.WU.Q, FCVT.LU.Q, FCVT.Q.WU, and FCVT.Q.LU variants convert to or from unsigned integer values. FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.Q.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.Q.S</b> rd, rs1</span><br><div>New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.Q or <b>FCVT.Q.S</b> converts a quad-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. FCVT.D.Q or FCVT.Q.D converts a quad-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.Q or FCVT.Q.S converts a quad-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. FCVT.D.Q or FCVT.Q.D converts a quad-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.Q.W":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.Q.W</b> rd, rs1</span><br><div>New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision-to-integer and integer-to-double-precision conversion instructions. FCVT.W.Q or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. <b>FCVT.Q.W</b> or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number. FCVT.WU.Q, FCVT.LU.Q, FCVT.Q.WU, and FCVT.Q.LU variants convert to or from unsigned integer values. FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision-to-integer and integer-to-double-precision conversion instructions. FCVT.W.Q or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.Q.W or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number. FCVT.WU.Q, FCVT.LU.Q, FCVT.Q.WU, and FCVT.Q.LU variants convert to or from unsigned integer values. FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.Q.WU":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.Q.WU</b> rd, rs1</span><br><div>New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision-to-integer and integer-to-double-precision conversion instructions. FCVT.W.Q or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.Q.W or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number. FCVT.WU.Q, FCVT.LU.Q, <b>FCVT.Q.WU</b>, and FCVT.Q.LU variants convert to or from unsigned integer values. FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision-to-integer and integer-to-double-precision conversion instructions. FCVT.W.Q or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.Q.W or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number. FCVT.WU.Q, FCVT.LU.Q, FCVT.Q.WU, and FCVT.Q.LU variants convert to or from unsigned integer values. FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.S.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.S.D</b> rd, rs1</span><br><div>The double-precision to single-precision and single-precision to double-precision conversion instructions, <b>FCVT.S.D</b> and FCVT.D.S, are encoded in the OP-FP major opcode space and both the source and destination are floating-point registers. The rs2 field encodes the datatype of the source, and the fmt field encodes the datatype of the destination. <b>FCVT.S.D</b> rounds according to the RM field; FCVT.D.S will never round.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "The double-precision to single-precision and single-precision to double-precision conversion instructions, FCVT.S.D and FCVT.D.S, are encoded in the OP-FP major opcode space and both the source and destination are floating-point registers. The rs2 field encodes the datatype of the source, and the fmt field encodes the datatype of the destination. FCVT.S.D rounds according to the RM field; FCVT.D.S will never round.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.S.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.S.H</b> rd, rs1</span><br><div>New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. <b>FCVT.S.H</b> or FCVT.H.S converts a half-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. If the D extension is present, FCVT.D.H or FCVT.H.D converts a half-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively. If the Q extension is present, FCVT.Q.H or FCVT.H.Q converts a half-precision floating-point number to a quad-precision floating-point number, or vice-versa, respectively.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.H or FCVT.H.S converts a half-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. If the D extension is present, FCVT.D.H or FCVT.H.D converts a half-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively. If the Q extension is present, FCVT.Q.H or FCVT.H.Q converts a half-precision floating-point number to a quad-precision floating-point number, or vice-versa, respectively.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FCVT.S.L":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.S.L</b> rd, rs1</span><br><div>Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.S or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.S.W or <b>FCVT.S.L</b> converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd. FCVT.WU.S, FCVT.LU.S, FCVT.S.WU, and FCVT.S.LU variants convert to or from unsigned integer values. For XLEN > 32, FCVT.W[U].S sign-extends the 32-bit result to the destination register width. FCVT.L[U].S and <b>FCVT.S.L</b>[U] are RV64-only instructions. If the rounded result is not representable in the destination format, it is clipped to the nearest value and the invalid flag is set. Table [tab:int_conv] gives the range of valid inputs for FCVT.int.S and the behavior for invalid inputs.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.S or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.S.W or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd. FCVT.WU.S, FCVT.LU.S, FCVT.S.WU, and FCVT.S.LU variants convert to or from unsigned integer values. For XLEN > 32, FCVT.W[U].S sign-extends the 32-bit result to the destination register width. FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions. If the rounded result is not representable in the destination format, it is clipped to the nearest value and the invalid flag is set. Table [tab:int_conv] gives the range of valid inputs for FCVT.int.S and the behavior for invalid inputs.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.S.LU":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.S.LU</b> rd, rs1</span><br><div>Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.S or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.S.W or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd. FCVT.WU.S, FCVT.LU.S, FCVT.S.WU, and <b>FCVT.S.LU</b> variants convert to or from unsigned integer values. For XLEN > 32, FCVT.W[U].S sign-extends the 32-bit result to the destination register width. FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions. If the rounded result is not representable in the destination format, it is clipped to the nearest value and the invalid flag is set. Table [tab:int_conv] gives the range of valid inputs for FCVT.int.S and the behavior for invalid inputs.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.S or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.S.W or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd. FCVT.WU.S, FCVT.LU.S, FCVT.S.WU, and FCVT.S.LU variants convert to or from unsigned integer values. For XLEN > 32, FCVT.W[U].S sign-extends the 32-bit result to the destination register width. FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions. If the rounded result is not representable in the destination format, it is clipped to the nearest value and the invalid flag is set. Table [tab:int_conv] gives the range of valid inputs for FCVT.int.S and the behavior for invalid inputs.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.S.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.S.Q</b> rd, rs1</span><br><div>New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. <b>FCVT.S.Q</b> or FCVT.Q.S converts a quad-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. FCVT.D.Q or FCVT.Q.D converts a quad-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "New floating-point-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision floating-point-to-floating-point conversion instructions. FCVT.S.Q or FCVT.Q.S converts a quad-precision floating-point number to a single-precision floating-point number, or vice-versa, respectively. FCVT.D.Q or FCVT.Q.D converts a quad-precision floating-point number to a double-precision floating-point number, or vice-versa, respectively.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.S.W":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.S.W</b> rd, rs1</span><br><div>Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.S or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. <b>FCVT.S.W</b> or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd. FCVT.WU.S, FCVT.LU.S, FCVT.S.WU, and FCVT.S.LU variants convert to or from unsigned integer values. For XLEN > 32, FCVT.W[U].S sign-extends the 32-bit result to the destination register width. FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions. If the rounded result is not representable in the destination format, it is clipped to the nearest value and the invalid flag is set. Table [tab:int_conv] gives the range of valid inputs for FCVT.int.S and the behavior for invalid inputs.<br>All floating-point to integer and integer to floating-point conversion instructions round according to the rm field. A floating-point register can be initialized to floating-point positive zero using <b>FCVT.S.W</b> rd, x0, which will never set any exception flags.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.S or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.S.W or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd. FCVT.WU.S, FCVT.LU.S, FCVT.S.WU, and FCVT.S.LU variants convert to or from unsigned integer values. For XLEN > 32, FCVT.W[U].S sign-extends the 32-bit result to the destination register width. FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions. If the rounded result is not representable in the destination format, it is clipped to the nearest value and the invalid flag is set. Table [tab:int_conv] gives the range of valid inputs for FCVT.int.S and the behavior for invalid inputs.\nAll floating-point to integer and integer to floating-point conversion instructions round according to the rm field. A floating-point register can be initialized to floating-point positive zero using FCVT.S.W rd, x0, which will never set any exception flags.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.S.WU":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.S.WU</b> rd, rs1</span><br><div>Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.S or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.S.W or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd. FCVT.WU.S, FCVT.LU.S, <b>FCVT.S.WU</b>, and FCVT.S.LU variants convert to or from unsigned integer values. For XLEN > 32, FCVT.W[U].S sign-extends the 32-bit result to the destination register width. FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions. If the rounded result is not representable in the destination format, it is clipped to the nearest value and the invalid flag is set. Table [tab:int_conv] gives the range of valid inputs for FCVT.int.S and the behavior for invalid inputs.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.S or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.S.W or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd. FCVT.WU.S, FCVT.LU.S, FCVT.S.WU, and FCVT.S.LU variants convert to or from unsigned integer values. For XLEN > 32, FCVT.W[U].S sign-extends the 32-bit result to the destination register width. FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions. If the rounded result is not representable in the destination format, it is clipped to the nearest value and the invalid flag is set. Table [tab:int_conv] gives the range of valid inputs for FCVT.int.S and the behavior for invalid inputs.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.W.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.W.D</b> rd, rs1</span><br><div>Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. <b>FCVT.W.D</b> or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.D.W or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd. FCVT.WU.D, FCVT.LU.D, FCVT.D.WU, and FCVT.D.LU variants convert to or from unsigned integer values. For RV64, FCVT.W[U].D sign-extends the 32-bit result. FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions. The range of valid inputs for FCVT.int.D and the behavior for invalid inputs are the same as for FCVT.int.S.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.D or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.D.W or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd. FCVT.WU.D, FCVT.LU.D, FCVT.D.WU, and FCVT.D.LU variants convert to or from unsigned integer values. For RV64, FCVT.W[U].D sign-extends the 32-bit result. FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions. The range of valid inputs for FCVT.int.D and the behavior for invalid inputs are the same as for FCVT.int.S.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.W.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.W.H</b> rd, rs1</span><br><div>New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the single-precision-to-integer and integer-to-single-precision conversion instructions. <b>FCVT.W.H</b> or FCVT.L.H converts a half-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.H.W or FCVT.H.L converts a 32-bit or 64-bit signed integer, respectively, into a half-precision floating-point number. FCVT.WU.H, FCVT.LU.H, FCVT.H.WU, and FCVT.H.LU variants convert to or from unsigned integer values. FCVT.L[U].H and FCVT.H.L[U] are RV64-only instructions.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the single-precision-to-integer and integer-to-single-precision conversion instructions. FCVT.W.H or FCVT.L.H converts a half-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.H.W or FCVT.H.L converts a 32-bit or 64-bit signed integer, respectively, into a half-precision floating-point number. FCVT.WU.H, FCVT.LU.H, FCVT.H.WU, and FCVT.H.LU variants convert to or from unsigned integer values. FCVT.L[U].H and FCVT.H.L[U] are RV64-only instructions.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FCVT.W.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.W.Q</b> rd, rs1</span><br><div>New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision-to-integer and integer-to-double-precision conversion instructions. <b>FCVT.W.Q</b> or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.Q.W or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number. FCVT.WU.Q, FCVT.LU.Q, FCVT.Q.WU, and FCVT.Q.LU variants convert to or from unsigned integer values. FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision-to-integer and integer-to-double-precision conversion instructions. FCVT.W.Q or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.Q.W or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number. FCVT.WU.Q, FCVT.LU.Q, FCVT.Q.WU, and FCVT.Q.LU variants convert to or from unsigned integer values. FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.W.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.W.S</b> rd, rs1</span><br><div>Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. <b>FCVT.W.S</b> or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.S.W or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd. FCVT.WU.S, FCVT.LU.S, FCVT.S.WU, and FCVT.S.LU variants convert to or from unsigned integer values. For XLEN > 32, FCVT.W[U].S sign-extends the 32-bit result to the destination register width. FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions. If the rounded result is not representable in the destination format, it is clipped to the nearest value and the invalid flag is set. Table [tab:int_conv] gives the range of valid inputs for FCVT.int.S and the behavior for invalid inputs.<br><b>FCVT.W.S</b></div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.S or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.S.W or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd. FCVT.WU.S, FCVT.LU.S, FCVT.S.WU, and FCVT.S.LU variants convert to or from unsigned integer values. For XLEN > 32, FCVT.W[U].S sign-extends the 32-bit result to the destination register width. FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions. If the rounded result is not representable in the destination format, it is clipped to the nearest value and the invalid flag is set. Table [tab:int_conv] gives the range of valid inputs for FCVT.int.S and the behavior for invalid inputs.\nFCVT.W.S\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.WU.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.WU.D</b> rd, rs1</span><br><div>Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.D or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.D.W or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd. <b>FCVT.WU.D</b>, FCVT.LU.D, FCVT.D.WU, and FCVT.D.LU variants convert to or from unsigned integer values. For RV64, FCVT.W[U].D sign-extends the 32-bit result. FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions. The range of valid inputs for FCVT.int.D and the behavior for invalid inputs are the same as for FCVT.int.S.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.D or FCVT.L.D converts a double-precision floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.D.W or FCVT.D.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a double-precision floating-point number in floating-point register rd. FCVT.WU.D, FCVT.LU.D, FCVT.D.WU, and FCVT.D.LU variants convert to or from unsigned integer values. For RV64, FCVT.W[U].D sign-extends the 32-bit result. FCVT.L[U].D and FCVT.D.L[U] are RV64-only instructions. The range of valid inputs for FCVT.int.D and the behavior for invalid inputs are the same as for FCVT.int.S.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FCVT.WU.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.WU.H</b> rd, rs1</span><br><div>New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the single-precision-to-integer and integer-to-single-precision conversion instructions. FCVT.W.H or FCVT.L.H converts a half-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.H.W or FCVT.H.L converts a 32-bit or 64-bit signed integer, respectively, into a half-precision floating-point number. <b>FCVT.WU.H</b>, FCVT.LU.H, FCVT.H.WU, and FCVT.H.LU variants convert to or from unsigned integer values. FCVT.L[U].H and FCVT.H.L[U] are RV64-only instructions.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the single-precision-to-integer and integer-to-single-precision conversion instructions. FCVT.W.H or FCVT.L.H converts a half-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.H.W or FCVT.H.L converts a 32-bit or 64-bit signed integer, respectively, into a half-precision floating-point number. FCVT.WU.H, FCVT.LU.H, FCVT.H.WU, and FCVT.H.LU variants convert to or from unsigned integer values. FCVT.L[U].H and FCVT.H.L[U] are RV64-only instructions.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FCVT.WU.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.WU.Q</b> rd, rs1</span><br><div>New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision-to-integer and integer-to-double-precision conversion instructions. FCVT.W.Q or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.Q.W or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number. <b>FCVT.WU.Q</b>, FCVT.LU.Q, FCVT.Q.WU, and FCVT.Q.LU variants convert to or from unsigned integer values. FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "New floating-point-to-integer and integer-to-floating-point conversion instructions are added. These instructions are defined analogously to the double-precision-to-integer and integer-to-double-precision conversion instructions. FCVT.W.Q or FCVT.L.Q converts a quad-precision floating-point number to a signed 32-bit or 64-bit integer, respectively. FCVT.Q.W or FCVT.Q.L converts a 32-bit or 64-bit signed integer, respectively, into a quad-precision floating-point number. FCVT.WU.Q, FCVT.LU.Q, FCVT.Q.WU, and FCVT.Q.LU variants convert to or from unsigned integer values. FCVT.L[U].Q and FCVT.Q.L[U] are RV64-only instructions.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FCVT.WU.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FCVT.WU.S</b> rd, rs1</span><br><div>Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.S or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.S.W or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd. <b>FCVT.WU.S</b>, FCVT.LU.S, FCVT.S.WU, and FCVT.S.LU variants convert to or from unsigned integer values. For XLEN > 32, FCVT.W[U].S sign-extends the 32-bit result to the destination register width. FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions. If the rounded result is not representable in the destination format, it is clipped to the nearest value and the invalid flag is set. Table [tab:int_conv] gives the range of valid inputs for FCVT.int.S and the behavior for invalid inputs.<br><b>FCVT.WU.S</b></div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point-to-integer and integer-to-floating-point conversion instructions are encoded in the OP-FP major opcode space. FCVT.W.S or FCVT.L.S converts a floating-point number in floating-point register rs1 to a signed 32-bit or 64-bit integer, respectively, in integer register rd. FCVT.S.W or FCVT.S.L converts a 32-bit or 64-bit signed integer, respectively, in integer register rs1 into a floating-point number in floating-point register rd. FCVT.WU.S, FCVT.LU.S, FCVT.S.WU, and FCVT.S.LU variants convert to or from unsigned integer values. For XLEN > 32, FCVT.W[U].S sign-extends the 32-bit result to the destination register width. FCVT.L[U].S and FCVT.S.L[U] are RV64-only instructions. If the rounded result is not representable in the destination format, it is clipped to the nearest value and the invalid flag is set. Table [tab:int_conv] gives the range of valid inputs for FCVT.int.S and the behavior for invalid inputs.\nFCVT.WU.S\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FDIV.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FDIV.D</b> rd, rs1, rs2</span><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FDIV.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FDIV.H</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FDIV.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FDIV.Q</b> rd, rs1, rs2</span><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FDIV.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FDIV.S</b> rd, rs1, rs2</span><br><div>Floating-point arithmetic instructions with one or two source operands use the R-type format with the OP-FP major opcode. FADD.S and FMUL.S perform single-precision floating-point addition and multiplication respectively, between rs1 and rs2. FSUB.S performs the single-precision floating-point subtraction of rs2 from rs1. <b>FDIV.S</b> performs the single-precision floating-point division of rs1 by rs2. FSQRT.S computes the square root of rs1. In each case, the result is written to rd.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point arithmetic instructions with one or two source operands use the R-type format with the OP-FP major opcode. FADD.S and FMUL.S perform single-precision floating-point addition and multiplication respectively, between rs1 and rs2. FSUB.S performs the single-precision floating-point subtraction of rs2 from rs1. FDIV.S performs the single-precision floating-point division of rs1 by rs2. FSQRT.S computes the square root of rs1. In each case, the result is written to rd.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#sec:single-float-compute"
            };

        case "FENCE":
            return {
                "html": "<div><span class=\"opcode\"><b>FENCE</b> rs1, rd</span><br><div>RV32I was designed to be sufficient to form a compiler target and to support modern operating system environments. The ISA was also designed to reduce the hardware required in a minimal implementation. RV32I contains 40 unique instructions, though a simple implementation might cover the ECALL/EBREAK instructions with a single SYSTEM hardware instruction that always traps and might be able to implement the <b>FENCE</b> instruction as a NOP, reducing base instruction count to 38 total. RV32I can emulate almost any other ISA extension (except the A extension, which requires additional hardware support for atomicity).</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "RV32I was designed to be sufficient to form a compiler target and to support modern operating system environments. The ISA was also designed to reduce the hardware required in a minimal implementation. RV32I contains 40 unique instructions, though a simple implementation might cover the ECALL/EBREAK instructions with a single SYSTEM hardware instruction that always traps and might be able to implement the FENCE instruction as a NOP, reducing base instruction count to 38 total. RV32I can emulate almost any other ISA extension (except the A extension, which requires additional hardware support for atomicity).\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#rv32"
            };

        case "FENCE.I":
            return {
                "html": "<div><span class=\"opcode\"><b>FENCE.I</b> imm12, rs1, rd</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FENCE.TSO":
            return {
                "html": "<div><span class=\"opcode\"><b>FENCE.TSO</b> rs1, rd</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FEQ.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FEQ.D</b> rd, rs1, rs2</span><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FEQ.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FEQ.H</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FEQ.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FEQ.Q</b> rd, rs1, rs2</span><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FEQ.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FEQ.S</b> rd, rs1, rs2</span><br><div>Floating-point compare instructions (<b>FEQ.S</b>, FLT.S, FLE.S) perform the specified comparison between floating-point registers (rs1 = rs2, rs1 < rs2, rs1 \\leq rs2) writing 1 to the integer register rd if the condition holds, and 0 otherwise.<br>FLT.S and FLE.S perform what the IEEE 754-2008 standard refers to as signaling comparisons: that is, they set the invalid operation exception flag if either input is NaN. <b>FEQ.S</b> performs a quiet comparison: it only sets the invalid operation exception flag if either input is a signaling NaN. For all three instructions, the result is 0 if either operand is NaN.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point compare instructions (FEQ.S, FLT.S, FLE.S) perform the specified comparison between floating-point registers (rs1 = rs2, rs1 < rs2, rs1 \\leq rs2) writing 1 to the integer register rd if the condition holds, and 0 otherwise.\nFLT.S and FLE.S perform what the IEEE 754-2008 standard refers to as signaling comparisons: that is, they set the invalid operation exception flag if either input is NaN. FEQ.S performs a quiet comparison: it only sets the invalid operation exception flag if either input is a signaling NaN. For all three instructions, the result is 0 if either operand is NaN.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-compare-instructions"
            };

        case "FLD":
            return {
                "html": "<div><span class=\"opcode\"><b>FLD</b> rd, rs1, imm12</span><br><div>The <b>FLD</b> instruction loads a double-precision floating-point value from memory into floating-point register rd. FSD stores a double-precision value from the floating-point registers to memory.<br><b>FLD</b> and FSD are only guaranteed to execute atomically if the effective address is naturally aligned and XLEN>=64.<br><b>FLD</b> and FSD do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "The FLD instruction loads a double-precision floating-point value from memory into floating-point register rd. FSD stores a double-precision value from the floating-point registers to memory.\nFLD and FSD are only guaranteed to execute atomically if the effective address is naturally aligned and XLEN>=64.\nFLD and FSD do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#fld_fsd"
            };

        case "FLE.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FLE.D</b> rd, rs1, rs2</span><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FLE.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FLE.H</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FLE.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FLE.Q</b> rd, rs1, rs2</span><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FLE.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FLE.S</b> rd, rs1, rs2</span><br><div>Floating-point compare instructions (FEQ.S, FLT.S, <b>FLE.S</b>) perform the specified comparison between floating-point registers (rs1 = rs2, rs1 < rs2, rs1 \\leq rs2) writing 1 to the integer register rd if the condition holds, and 0 otherwise.<br>FLT.S and <b>FLE.S</b> perform what the IEEE 754-2008 standard refers to as signaling comparisons: that is, they set the invalid operation exception flag if either input is NaN. FEQ.S performs a quiet comparison: it only sets the invalid operation exception flag if either input is a signaling NaN. For all three instructions, the result is 0 if either operand is NaN.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point compare instructions (FEQ.S, FLT.S, FLE.S) perform the specified comparison between floating-point registers (rs1 = rs2, rs1 < rs2, rs1 \\leq rs2) writing 1 to the integer register rd if the condition holds, and 0 otherwise.\nFLT.S and FLE.S perform what the IEEE 754-2008 standard refers to as signaling comparisons: that is, they set the invalid operation exception flag if either input is NaN. FEQ.S performs a quiet comparison: it only sets the invalid operation exception flag if either input is a signaling NaN. For all three instructions, the result is 0 if either operand is NaN.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-compare-instructions"
            };

        case "FLH":
            return {
                "html": "<div><span class=\"opcode\"><b>FLH</b> rd, rs1, imm12</span><br><div><b>FLH</b> and FSH are only guaranteed to execute atomically if the effective address is naturally aligned.<br><b>FLH</b> and FSH do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved. <b>FLH</b> NaN-boxes the result written to rd, whereas FSH ignores all but the lower 16 bits in rs2.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "FLH and FSH are only guaranteed to execute atomically if the effective address is naturally aligned.\nFLH and FSH do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved. FLH NaN-boxes the result written to rd, whereas FSH ignores all but the lower 16 bits in rs2.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-load-and-store-instructions"
            };

        case "FLQ":
            return {
                "html": "<div><span class=\"opcode\"><b>FLQ</b> rd, rs1, imm12</span><br><div><b>FLQ</b> and FSQ are only guaranteed to execute atomically if the effective address is naturally aligned and XLEN=128.<br><b>FLQ</b> and FSQ do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "FLQ and FSQ are only guaranteed to execute atomically if the effective address is naturally aligned and XLEN=128.\nFLQ and FSQ do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-load-and-store-instructions"
            };

        case "FLT.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FLT.D</b> rd, rs1, rs2</span><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FLT.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FLT.H</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FLT.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FLT.Q</b> rd, rs1, rs2</span><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FLT.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FLT.S</b> rd, rs1, rs2</span><br><div>Floating-point compare instructions (FEQ.S, <b>FLT.S</b>, FLE.S) perform the specified comparison between floating-point registers (rs1 = rs2, rs1 < rs2, rs1 \\leq rs2) writing 1 to the integer register rd if the condition holds, and 0 otherwise.<br><b>FLT.S</b> and FLE.S perform what the IEEE 754-2008 standard refers to as signaling comparisons: that is, they set the invalid operation exception flag if either input is NaN. FEQ.S performs a quiet comparison: it only sets the invalid operation exception flag if either input is a signaling NaN. For all three instructions, the result is 0 if either operand is NaN.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point compare instructions (FEQ.S, FLT.S, FLE.S) perform the specified comparison between floating-point registers (rs1 = rs2, rs1 < rs2, rs1 \\leq rs2) writing 1 to the integer register rd if the condition holds, and 0 otherwise.\nFLT.S and FLE.S perform what the IEEE 754-2008 standard refers to as signaling comparisons: that is, they set the invalid operation exception flag if either input is NaN. FEQ.S performs a quiet comparison: it only sets the invalid operation exception flag if either input is a signaling NaN. For all three instructions, the result is 0 if either operand is NaN.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-compare-instructions"
            };

        case "FLW":
            return {
                "html": "<div><span class=\"opcode\"><b>FLW</b> rd, rs1, imm12</span><br><div>Floating-point loads and stores use the same base+offset addressing mode as the integer base ISAs, with a base address in register rs1 and a 12-bit signed byte offset. The <b>FLW</b> instruction loads a single-precision floating-point value from memory into floating-point register rd. FSW stores a single-precision value from floating-point register rs2 to memory.<br><b>FLW</b> and FSW are only guaranteed to execute atomically if the effective address is naturally aligned.<br><b>FLW</b> and FSW do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point loads and stores use the same base+offset addressing mode as the integer base ISAs, with a base address in register rs1 and a 12-bit signed byte offset. The FLW instruction loads a single-precision floating-point value from memory into floating-point register rd. FSW stores a single-precision value from floating-point register rs2 to memory.\nFLW and FSW are only guaranteed to execute atomically if the effective address is naturally aligned.\nFLW and FSW do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-load-and-store-instructions"
            };

        case "FMADD.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FMADD.D</b> rd, rs1, rs2, rs3</span><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMADD.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FMADD.H</b> rd, rs1, rs2, rs3</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMADD.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FMADD.Q</b> rd, rs1, rs2, rs3</span><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMADD.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FMADD.S</b> rd, rs1, rs2, rs3</span><br><div><b>FMADD.S</b> multiplies the values in rs1 and rs2, adds the value in rs3, and writes the final result to rd. <b>FMADD.S</b> computes (rs1\u00d7rs2)+rs3.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "FMADD.S multiplies the values in rs1 and rs2, adds the value in rs3, and writes the final result to rd. FMADD.S computes (rs1\u00d7rs2)+rs3.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#sec:single-float-compute"
            };

        case "FMAX.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FMAX.D</b> rd, rs1, rs2</span><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMAX.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FMAX.H</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMAX.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FMAX.Q</b> rd, rs1, rs2</span><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMAX.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FMAX.S</b> rd, rs1, rs2</span><br><div>Floating-point minimum-number and maximum-number instructions FMIN.S and <b>FMAX.S</b> write, respectively, the smaller or larger of rs1 and rs2 to rd. For the purposes of these instructions only, the value - 0.0 is considered to be less than the value + 0.0. If both inputs are NaNs, the result is the canonical NaN. If only one operand is a NaN, the result is the non-NaN operand. Signaling NaN inputs set the invalid operation exception flag, even when the result is not NaN.<br>Note that in version 2.2 of the F extension, the FMIN.S and <b>FMAX.S</b> instructions were amended to implement the proposed IEEE 754-201x minimumNumber and maximumNumber operations, rather than the IEEE 754-2008 minNum and maxNum operations. These operations differ in their handling of signaling NaNs.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point minimum-number and maximum-number instructions FMIN.S and FMAX.S write, respectively, the smaller or larger of rs1 and rs2 to rd. For the purposes of these instructions only, the value - 0.0 is considered to be less than the value + 0.0. If both inputs are NaNs, the result is the canonical NaN. If only one operand is a NaN, the result is the non-NaN operand. Signaling NaN inputs set the invalid operation exception flag, even when the result is not NaN.\nNote that in version 2.2 of the F extension, the FMIN.S and FMAX.S instructions were amended to implement the proposed IEEE 754-201x minimumNumber and maximumNumber operations, rather than the IEEE 754-2008 minNum and maxNum operations. These operations differ in their handling of signaling NaNs.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#sec:single-float-compute"
            };

        case "FMIN.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FMIN.D</b> rd, rs1, rs2</span><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMIN.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FMIN.H</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMIN.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FMIN.Q</b> rd, rs1, rs2</span><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMIN.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FMIN.S</b> rd, rs1, rs2</span><br><div>Floating-point minimum-number and maximum-number instructions <b>FMIN.S</b> and FMAX.S write, respectively, the smaller or larger of rs1 and rs2 to rd. For the purposes of these instructions only, the value - 0.0 is considered to be less than the value + 0.0. If both inputs are NaNs, the result is the canonical NaN. If only one operand is a NaN, the result is the non-NaN operand. Signaling NaN inputs set the invalid operation exception flag, even when the result is not NaN.<br>Note that in version 2.2 of the F extension, the <b>FMIN.S</b> and FMAX.S instructions were amended to implement the proposed IEEE 754-201x minimumNumber and maximumNumber operations, rather than the IEEE 754-2008 minNum and maxNum operations. These operations differ in their handling of signaling NaNs.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point minimum-number and maximum-number instructions FMIN.S and FMAX.S write, respectively, the smaller or larger of rs1 and rs2 to rd. For the purposes of these instructions only, the value - 0.0 is considered to be less than the value + 0.0. If both inputs are NaNs, the result is the canonical NaN. If only one operand is a NaN, the result is the non-NaN operand. Signaling NaN inputs set the invalid operation exception flag, even when the result is not NaN.\nNote that in version 2.2 of the F extension, the FMIN.S and FMAX.S instructions were amended to implement the proposed IEEE 754-201x minimumNumber and maximumNumber operations, rather than the IEEE 754-2008 minNum and maxNum operations. These operations differ in their handling of signaling NaNs.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#sec:single-float-compute"
            };

        case "FMSUB.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FMSUB.D</b> rd, rs1, rs2, rs3</span><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMSUB.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FMSUB.H</b> rd, rs1, rs2, rs3</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMSUB.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FMSUB.Q</b> rd, rs1, rs2, rs3</span><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMSUB.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FMSUB.S</b> rd, rs1, rs2, rs3</span><br><div><b>FMSUB.S</b> multiplies the values in rs1 and rs2, subtracts the value in rs3, and writes the final result to rd. <b>FMSUB.S</b> computes (rs1\u00d7rs2)-rs3.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "FMSUB.S multiplies the values in rs1 and rs2, subtracts the value in rs3, and writes the final result to rd. FMSUB.S computes (rs1\u00d7rs2)-rs3.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#sec:single-float-compute"
            };

        case "FMUL.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FMUL.D</b> rd, rs1, rs2</span><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMUL.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FMUL.H</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMUL.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FMUL.Q</b> rd, rs1, rs2</span><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMUL.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FMUL.S</b> rd, rs1, rs2</span><br><div>Floating-point arithmetic instructions with one or two source operands use the R-type format with the OP-FP major opcode. FADD.S and <b>FMUL.S</b> perform single-precision floating-point addition and multiplication respectively, between rs1 and rs2. FSUB.S performs the single-precision floating-point subtraction of rs2 from rs1. FDIV.S performs the single-precision floating-point division of rs1 by rs2. FSQRT.S computes the square root of rs1. In each case, the result is written to rd.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point arithmetic instructions with one or two source operands use the R-type format with the OP-FP major opcode. FADD.S and FMUL.S perform single-precision floating-point addition and multiplication respectively, between rs1 and rs2. FSUB.S performs the single-precision floating-point subtraction of rs2 from rs1. FDIV.S performs the single-precision floating-point division of rs1 by rs2. FSQRT.S computes the square root of rs1. In each case, the result is written to rd.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#sec:single-float-compute"
            };

        case "FMV.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FMV.D</b> rd, rs</span><br><div>For XLEN>=64 only, instructions are provided to move bit patterns between the floating-point and integer registers. FMV.X.D moves the double-precision value in floating-point register rs1 to a representation in IEEE 754-2008 standard encoding in integer register rd. <b>FMV.D</b>.X moves the double-precision value encoded in IEEE 754-2008 standard encoding from the integer register rs1 to the floating-point register rd.<br>FMV.X.D and <b>FMV.D</b>.X do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.</div><br><div><b>Equivalent ASM:</b><pre>fsgnj.d rd, rs, rs</pre></div><br><div><b>ISA</b>: d(pseudo)</div></div>",
                "tooltip": "For XLEN>=64 only, instructions are provided to move bit patterns between the floating-point and integer registers. FMV.X.D moves the double-precision value in floating-point register rs1 to a representation in IEEE 754-2008 standard encoding in integer register rd. FMV.D.X moves the double-precision value encoded in IEEE 754-2008 standard encoding from the integer register rs1 to the floating-point register rd.\nFMV.X.D and FMV.D.X do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.\n\nEquivalent ASM:\n\nfsgnj.d rd, rs, rs\n\n(ISA: d(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FMV.D.X":
            return {
                "html": "<div><span class=\"opcode\"><b>FMV.D.X</b> rd, rs1</span><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMV.H.X":
            return {
                "html": "<div><span class=\"opcode\"><b>FMV.H.X</b> rd, rs1</span><br><div><b>FMV.H.X</b> moves the half-precision value encoded in IEEE 754-2008 standard encoding from the lower 16 bits of integer register rs1 to the floating-point register rd, NaN-boxing the result.<br>FMV.X.H and <b>FMV.H.X</b> do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "FMV.H.X moves the half-precision value encoded in IEEE 754-2008 standard encoding from the lower 16 bits of integer register rs1 to the floating-point register rd, NaN-boxing the result.\nFMV.X.H and FMV.H.X do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FMV.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FMV.S</b> rd, rs</span><br><div>Floating-point to floating-point sign-injection instructions, FSGNJ.S, FSGNJN.S, and FSGNJX.S, produce a result that takes all bits except the sign bit from rs1. For FSGNJ, the result's sign bit is rs2's sign bit; for FSGNJN, the result's sign bit is the opposite of rs2's sign bit; and for FSGNJX, the sign bit is the XOR of the sign bits of rs1 and rs2. Sign-injection instructions do not set floating-point exception flags, nor do they canonicalize NaNs. Note, FSGNJ.S rx, ry, ry moves ry to rx (assembler pseudoinstruction <b>FMV.S</b> rx, ry); FSGNJN.S rx, ry, ry moves the negation of ry to rx (assembler pseudoinstruction FNEG.S rx, ry); and FSGNJX.S rx, ry, ry moves the absolute value of ry to rx (assembler pseudoinstruction FABS.S rx, ry).<br>The FMV.W.X and FMV.X.W instructions were previously called <b>FMV.S</b>.X and FMV.X.S. The use of W is more consistent with their semantics as an instruction that moves 32 bits without interpreting them. This became clearer after defining NaN-boxing. To avoid disturbing existing code, both the W and S versions will be supported by tools.</div><br><div><b>Equivalent ASM:</b><pre>fsgnj.s rd, rs, rs</pre></div><br><div><b>ISA</b>: f(pseudo)</div></div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.S, FSGNJN.S, and FSGNJX.S, produce a result that takes all bits except the sign bit from rs1. For FSGNJ, the result's sign bit is rs2's sign bit; for FSGNJN, the result's sign bit is the opposite of rs2's sign bit; and for FSGNJX, the sign bit is the XOR of the sign bits of rs1 and rs2. Sign-injection instructions do not set floating-point exception flags, nor do they canonicalize NaNs. Note, FSGNJ.S rx, ry, ry moves ry to rx (assembler pseudoinstruction FMV.S rx, ry); FSGNJN.S rx, ry, ry moves the negation of ry to rx (assembler pseudoinstruction FNEG.S rx, ry); and FSGNJX.S rx, ry, ry moves the absolute value of ry to rx (assembler pseudoinstruction FABS.S rx, ry).\nThe FMV.W.X and FMV.X.W instructions were previously called FMV.S.X and FMV.X.S. The use of W is more consistent with their semantics as an instruction that moves 32 bits without interpreting them. This became clearer after defining NaN-boxing. To avoid disturbing existing code, both the W and S versions will be supported by tools.\n\nEquivalent ASM:\n\nfsgnj.s rd, rs, rs\n\n(ISA: f(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FMV.S.X":
            return {
                "html": "<div><span class=\"opcode\"><b>FMV.S.X</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FMV.W.X":
            return {
                "html": "<div><span class=\"opcode\"><b>FMV.W.X</b> rd, rs1</span><br><div><b>FMV.W.X</b> moves the single-precision value encoded in IEEE 754-2008 standard encoding from the lower 32 bits of integer register rs1 to the floating-point register rd. The bits are not modified in the transfer, and in particular, the payloads of non-canonical NaNs are preserved.<br>The <b>FMV.W.X</b> and FMV.X.W instructions were previously called FMV.S.X and FMV.X.S. The use of W is more consistent with their semantics as an instruction that moves 32 bits without interpreting them. This became clearer after defining NaN-boxing. To avoid disturbing existing code, both the W and S versions will be supported by tools.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "FMV.W.X moves the single-precision value encoded in IEEE 754-2008 standard encoding from the lower 32 bits of integer register rs1 to the floating-point register rd. The bits are not modified in the transfer, and in particular, the payloads of non-canonical NaNs are preserved.\nThe FMV.W.X and FMV.X.W instructions were previously called FMV.S.X and FMV.X.S. The use of W is more consistent with their semantics as an instruction that moves 32 bits without interpreting them. This became clearer after defining NaN-boxing. To avoid disturbing existing code, both the W and S versions will be supported by tools.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FMV.X.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FMV.X.D</b> rd, rs1</span><br><div>For XLEN>=64 only, instructions are provided to move bit patterns between the floating-point and integer registers. <b>FMV.X.D</b> moves the double-precision value in floating-point register rs1 to a representation in IEEE 754-2008 standard encoding in integer register rd. FMV.D.X moves the double-precision value encoded in IEEE 754-2008 standard encoding from the integer register rs1 to the floating-point register rd.<br><b>FMV.X.D</b> and FMV.D.X do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "For XLEN>=64 only, instructions are provided to move bit patterns between the floating-point and integer registers. FMV.X.D moves the double-precision value in floating-point register rs1 to a representation in IEEE 754-2008 standard encoding in integer register rd. FMV.D.X moves the double-precision value encoded in IEEE 754-2008 standard encoding from the integer register rs1 to the floating-point register rd.\nFMV.X.D and FMV.D.X do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FMV.X.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FMV.X.H</b> rd, rs1</span><br><div>Instructions are provided to move bit patterns between the floating-point and integer registers. <b>FMV.X.H</b> moves the half-precision value in floating-point register rs1 to a representation in IEEE 754-2008 standard encoding in integer register rd, filling the upper XLEN-16 bits with copies of the floating-point number's sign bit.<br><b>FMV.X.H</b> and FMV.H.X do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "Instructions are provided to move bit patterns between the floating-point and integer registers. FMV.X.H moves the half-precision value in floating-point register rs1 to a representation in IEEE 754-2008 standard encoding in integer register rd, filling the upper XLEN-16 bits with copies of the floating-point number's sign bit.\nFMV.X.H and FMV.H.X do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FMV.X.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FMV.X.S</b> rd, rs1</span><br><div>The FMV.W.X and FMV.X.W instructions were previously called FMV.S.X and <b>FMV.X.S</b>. The use of W is more consistent with their semantics as an instruction that moves 32 bits without interpreting them. This became clearer after defining NaN-boxing. To avoid disturbing existing code, both the W and S versions will be supported by tools.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "The FMV.W.X and FMV.X.W instructions were previously called FMV.S.X and FMV.X.S. The use of W is more consistent with their semantics as an instruction that moves 32 bits without interpreting them. This became clearer after defining NaN-boxing. To avoid disturbing existing code, both the W and S versions will be supported by tools.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FMV.X.W":
            return {
                "html": "<div><span class=\"opcode\"><b>FMV.X.W</b> rd, rs1</span><br><div>Instructions are provided to move bit patterns between the floating-point and integer registers. <b>FMV.X.W</b> moves the single-precision value in floating-point register rs1 represented in IEEE 754-2008 encoding to the lower 32 bits of integer register rd. The bits are not modified in the transfer, and in particular, the payloads of non-canonical NaNs are preserved. For RV64, the higher 32 bits of the destination register are filled with copies of the floating-point number's sign bit.<br>The FMV.W.X and <b>FMV.X.W</b> instructions were previously called FMV.S.X and FMV.X.S. The use of W is more consistent with their semantics as an instruction that moves 32 bits without interpreting them. This became clearer after defining NaN-boxing. To avoid disturbing existing code, both the W and S versions will be supported by tools.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Instructions are provided to move bit patterns between the floating-point and integer registers. FMV.X.W moves the single-precision value in floating-point register rs1 represented in IEEE 754-2008 encoding to the lower 32 bits of integer register rd. The bits are not modified in the transfer, and in particular, the payloads of non-canonical NaNs are preserved. For RV64, the higher 32 bits of the destination register are filled with copies of the floating-point number's sign bit.\nThe FMV.W.X and FMV.X.W instructions were previously called FMV.S.X and FMV.X.S. The use of W is more consistent with their semantics as an instruction that moves 32 bits without interpreting them. This became clearer after defining NaN-boxing. To avoid disturbing existing code, both the W and S versions will be supported by tools.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FNEG.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FNEG.D</b> rd, rs</span><br><div><b>Equivalent ASM:</b><pre>fsgnjn.d rd, rs, rs</pre></div><br><div><b>ISA</b>: d(pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nfsgnjn.d rd, rs, rs\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FNEG.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FNEG.S</b> rd, rs</span><br><div>Floating-point to floating-point sign-injection instructions, FSGNJ.S, FSGNJN.S, and FSGNJX.S, produce a result that takes all bits except the sign bit from rs1. For FSGNJ, the result's sign bit is rs2's sign bit; for FSGNJN, the result's sign bit is the opposite of rs2's sign bit; and for FSGNJX, the sign bit is the XOR of the sign bits of rs1 and rs2. Sign-injection instructions do not set floating-point exception flags, nor do they canonicalize NaNs. Note, FSGNJ.S rx, ry, ry moves ry to rx (assembler pseudoinstruction FMV.S rx, ry); FSGNJN.S rx, ry, ry moves the negation of ry to rx (assembler pseudoinstruction <b>FNEG.S</b> rx, ry); and FSGNJX.S rx, ry, ry moves the absolute value of ry to rx (assembler pseudoinstruction FABS.S rx, ry).</div><br><div><b>Equivalent ASM:</b><pre>fsgnjn.s rd, rs, rs</pre></div><br><div><b>ISA</b>: f(pseudo)</div></div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.S, FSGNJN.S, and FSGNJX.S, produce a result that takes all bits except the sign bit from rs1. For FSGNJ, the result's sign bit is rs2's sign bit; for FSGNJN, the result's sign bit is the opposite of rs2's sign bit; and for FSGNJX, the sign bit is the XOR of the sign bits of rs1 and rs2. Sign-injection instructions do not set floating-point exception flags, nor do they canonicalize NaNs. Note, FSGNJ.S rx, ry, ry moves ry to rx (assembler pseudoinstruction FMV.S rx, ry); FSGNJN.S rx, ry, ry moves the negation of ry to rx (assembler pseudoinstruction FNEG.S rx, ry); and FSGNJX.S rx, ry, ry moves the absolute value of ry to rx (assembler pseudoinstruction FABS.S rx, ry).\n\nEquivalent ASM:\n\nfsgnjn.s rd, rs, rs\n\n(ISA: f(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FNMADD.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FNMADD.D</b> rd, rs1, rs2, rs3</span><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FNMADD.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FNMADD.H</b> rd, rs1, rs2, rs3</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FNMADD.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FNMADD.Q</b> rd, rs1, rs2, rs3</span><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FNMADD.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FNMADD.S</b> rd, rs1, rs2, rs3</span><br><div><b>FNMADD.S</b> multiplies the values in rs1 and rs2, negates the product, subtracts the value in rs3, and writes the final result to rd. <b>FNMADD.S</b> computes -(rs1\u00d7rs2)-rs3.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "FNMADD.S multiplies the values in rs1 and rs2, negates the product, subtracts the value in rs3, and writes the final result to rd. FNMADD.S computes -(rs1\u00d7rs2)-rs3.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#sec:single-float-compute"
            };

        case "FNMSUB.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FNMSUB.D</b> rd, rs1, rs2, rs3</span><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FNMSUB.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FNMSUB.H</b> rd, rs1, rs2, rs3</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FNMSUB.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FNMSUB.Q</b> rd, rs1, rs2, rs3</span><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FNMSUB.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FNMSUB.S</b> rd, rs1, rs2, rs3</span><br><div><b>FNMSUB.S</b> multiplies the values in rs1 and rs2, negates the product, adds the value in rs3, and writes the final result to rd. <b>FNMSUB.S</b> computes -(rs1\u00d7rs2)+rs3.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "FNMSUB.S multiplies the values in rs1 and rs2, negates the product, adds the value in rs3, and writes the final result to rd. FNMSUB.S computes -(rs1\u00d7rs2)+rs3.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#sec:single-float-compute"
            };

        case "FRCSR":
            return {
                "html": "<div><span class=\"opcode\"><b>FRCSR</b> rd</span><br><div>The fcsr register can be read and written with the <b>FRCSR</b> and FSCSR instructions, which are assembler pseudoinstructions built on the underlying CSR access instructions. <b>FRCSR</b> reads fcsr by copying it into integer register rd. FSCSR swaps the value in fcsr by copying the original value into integer register rd, and then writing a new value obtained from integer register rs1 into fcsr.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "The fcsr register can be read and written with the FRCSR and FSCSR instructions, which are assembler pseudoinstructions built on the underlying CSR access instructions. FRCSR reads fcsr by copying it into integer register rd. FSCSR swaps the value in fcsr by copying the original value into integer register rd, and then writing a new value obtained from integer register rs1 into fcsr.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#floating-point-control-and-status-register"
            };

        case "FRFLAGS":
            return {
                "html": "<div><span class=\"opcode\"><b>FRFLAGS</b> rd</span><br><div>The fields within the fcsr can also be accessed individually through different CSR addresses, and separate assembler pseudoinstructions are defined for these accesses. The FRRM instruction reads the Rounding Mode field frm and copies it into the least-significant three bits of integer register rd, with zero in all other bits. FSRM swaps the value in frm by copying the original value into integer register rd, and then writing a new value obtained from the three least-significant bits of integer register rs1 into frm. <b>FRFLAGS</b> and FSFLAGS are defined analogously for the Accrued Exception Flags field fflags.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "The fields within the fcsr can also be accessed individually through different CSR addresses, and separate assembler pseudoinstructions are defined for these accesses. The FRRM instruction reads the Rounding Mode field frm and copies it into the least-significant three bits of integer register rd, with zero in all other bits. FSRM swaps the value in frm by copying the original value into integer register rd, and then writing a new value obtained from the three least-significant bits of integer register rs1 into frm. FRFLAGS and FSFLAGS are defined analogously for the Accrued Exception Flags field fflags.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#floating-point-control-and-status-register"
            };

        case "FRRM":
            return {
                "html": "<div><span class=\"opcode\"><b>FRRM</b> rd</span><br><div>The fields within the fcsr can also be accessed individually through different CSR addresses, and separate assembler pseudoinstructions are defined for these accesses. The <b>FRRM</b> instruction reads the Rounding Mode field frm and copies it into the least-significant three bits of integer register rd, with zero in all other bits. FSRM swaps the value in frm by copying the original value into integer register rd, and then writing a new value obtained from the three least-significant bits of integer register rs1 into frm. FRFLAGS and FSFLAGS are defined analogously for the Accrued Exception Flags field fflags.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "The fields within the fcsr can also be accessed individually through different CSR addresses, and separate assembler pseudoinstructions are defined for these accesses. The FRRM instruction reads the Rounding Mode field frm and copies it into the least-significant three bits of integer register rd, with zero in all other bits. FSRM swaps the value in frm by copying the original value into integer register rd, and then writing a new value obtained from the three least-significant bits of integer register rs1 into frm. FRFLAGS and FSFLAGS are defined analogously for the Accrued Exception Flags field fflags.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#floating-point-control-and-status-register"
            };

        case "FSCSR":
            return {
                "html": "<div><span class=\"opcode\"><b>FSCSR</b> rd, rs1</span><br><div>The fcsr register can be read and written with the FRCSR and <b>FSCSR</b> instructions, which are assembler pseudoinstructions built on the underlying CSR access instructions. FRCSR reads fcsr by copying it into integer register rd. <b>FSCSR</b> swaps the value in fcsr by copying the original value into integer register rd, and then writing a new value obtained from integer register rs1 into fcsr.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "The fcsr register can be read and written with the FRCSR and FSCSR instructions, which are assembler pseudoinstructions built on the underlying CSR access instructions. FRCSR reads fcsr by copying it into integer register rd. FSCSR swaps the value in fcsr by copying the original value into integer register rd, and then writing a new value obtained from integer register rs1 into fcsr.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#floating-point-control-and-status-register"
            };

        case "FSD":
            return {
                "html": "<div><span class=\"opcode\"><b>FSD</b> rs1, rs2, imm12</span><br><div>The FLD instruction loads a double-precision floating-point value from memory into floating-point register rd. <b>FSD</b> stores a double-precision value from the floating-point registers to memory.<br>FLD and <b>FSD</b> are only guaranteed to execute atomically if the effective address is naturally aligned and XLEN>=64.<br>FLD and <b>FSD</b> do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "The FLD instruction loads a double-precision floating-point value from memory into floating-point register rd. FSD stores a double-precision value from the floating-point registers to memory.\nFLD and FSD are only guaranteed to execute atomically if the effective address is naturally aligned and XLEN>=64.\nFLD and FSD do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#fld_fsd"
            };

        case "FSFLAGS":
            return {
                "html": "<div><span class=\"opcode\"><b>FSFLAGS</b> rd, rs1</span><br><div>The fields within the fcsr can also be accessed individually through different CSR addresses, and separate assembler pseudoinstructions are defined for these accesses. The FRRM instruction reads the Rounding Mode field frm and copies it into the least-significant three bits of integer register rd, with zero in all other bits. FSRM swaps the value in frm by copying the original value into integer register rd, and then writing a new value obtained from the three least-significant bits of integer register rs1 into frm. FRFLAGS and <b>FSFLAGS</b> are defined analogously for the Accrued Exception Flags field fflags.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "The fields within the fcsr can also be accessed individually through different CSR addresses, and separate assembler pseudoinstructions are defined for these accesses. The FRRM instruction reads the Rounding Mode field frm and copies it into the least-significant three bits of integer register rd, with zero in all other bits. FSRM swaps the value in frm by copying the original value into integer register rd, and then writing a new value obtained from the three least-significant bits of integer register rs1 into frm. FRFLAGS and FSFLAGS are defined analogously for the Accrued Exception Flags field fflags.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#floating-point-control-and-status-register"
            };

        case "FSFLAGSI":
            return {
                "html": "<div><span class=\"opcode\"><b>FSFLAGSI</b> rd, zimm</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FSGNJ.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FSGNJ.D</b> rd, rs1, rs2</span><br><div>Floating-point to floating-point sign-injection instructions, <b>FSGNJ.D</b>, FSGNJN.D, and FSGNJX.D are defined analogously to the single-precision sign-injection instruction.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.D, FSGNJN.D, and FSGNJX.D are defined analogously to the single-precision sign-injection instruction.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FSGNJ.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FSGNJ.H</b> rd, rs1, rs2</span><br><div>Floating-point to floating-point sign-injection instructions, <b>FSGNJ.H</b>, FSGNJN.H, and FSGNJX.H are defined analogously to the single-precision sign-injection instruction.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.H, FSGNJN.H, and FSGNJX.H are defined analogously to the single-precision sign-injection instruction.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FSGNJ.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FSGNJ.Q</b> rd, rs1, rs2</span><br><div>Floating-point to floating-point sign-injection instructions, <b>FSGNJ.Q</b>, FSGNJN.Q, and FSGNJX.Q are defined analogously to the double-precision sign-injection instruction.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.Q, FSGNJN.Q, and FSGNJX.Q are defined analogously to the double-precision sign-injection instruction.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FSGNJ.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FSGNJ.S</b> rd, rs1, rs2</span><br><div>Floating-point to floating-point sign-injection instructions, <b>FSGNJ.S</b>, FSGNJN.S, and FSGNJX.S, produce a result that takes all bits except the sign bit from rs1. For FSGNJ, the result's sign bit is rs2's sign bit; for FSGNJN, the result's sign bit is the opposite of rs2's sign bit; and for FSGNJX, the sign bit is the XOR of the sign bits of rs1 and rs2. Sign-injection instructions do not set floating-point exception flags, nor do they canonicalize NaNs. Note, <b>FSGNJ.S</b> rx, ry, ry moves ry to rx (assembler pseudoinstruction FMV.S rx, ry); FSGNJN.S rx, ry, ry moves the negation of ry to rx (assembler pseudoinstruction FNEG.S rx, ry); and FSGNJX.S rx, ry, ry moves the absolute value of ry to rx (assembler pseudoinstruction FABS.S rx, ry).</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.S, FSGNJN.S, and FSGNJX.S, produce a result that takes all bits except the sign bit from rs1. For FSGNJ, the result's sign bit is rs2's sign bit; for FSGNJN, the result's sign bit is the opposite of rs2's sign bit; and for FSGNJX, the sign bit is the XOR of the sign bits of rs1 and rs2. Sign-injection instructions do not set floating-point exception flags, nor do they canonicalize NaNs. Note, FSGNJ.S rx, ry, ry moves ry to rx (assembler pseudoinstruction FMV.S rx, ry); FSGNJN.S rx, ry, ry moves the negation of ry to rx (assembler pseudoinstruction FNEG.S rx, ry); and FSGNJX.S rx, ry, ry moves the absolute value of ry to rx (assembler pseudoinstruction FABS.S rx, ry).\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FSGNJN.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FSGNJN.D</b> rd, rs1, rs2</span><br><div>Floating-point to floating-point sign-injection instructions, FSGNJ.D, <b>FSGNJN.D</b>, and FSGNJX.D are defined analogously to the single-precision sign-injection instruction.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.D, FSGNJN.D, and FSGNJX.D are defined analogously to the single-precision sign-injection instruction.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FSGNJN.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FSGNJN.H</b> rd, rs1, rs2</span><br><div>Floating-point to floating-point sign-injection instructions, FSGNJ.H, <b>FSGNJN.H</b>, and FSGNJX.H are defined analogously to the single-precision sign-injection instruction.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.H, FSGNJN.H, and FSGNJX.H are defined analogously to the single-precision sign-injection instruction.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FSGNJN.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FSGNJN.Q</b> rd, rs1, rs2</span><br><div>Floating-point to floating-point sign-injection instructions, FSGNJ.Q, <b>FSGNJN.Q</b>, and FSGNJX.Q are defined analogously to the double-precision sign-injection instruction.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.Q, FSGNJN.Q, and FSGNJX.Q are defined analogously to the double-precision sign-injection instruction.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FSGNJN.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FSGNJN.S</b> rd, rs1, rs2</span><br><div>Floating-point to floating-point sign-injection instructions, FSGNJ.S, <b>FSGNJN.S</b>, and FSGNJX.S, produce a result that takes all bits except the sign bit from rs1. For FSGNJ, the result's sign bit is rs2's sign bit; for FSGNJN, the result's sign bit is the opposite of rs2's sign bit; and for FSGNJX, the sign bit is the XOR of the sign bits of rs1 and rs2. Sign-injection instructions do not set floating-point exception flags, nor do they canonicalize NaNs. Note, FSGNJ.S rx, ry, ry moves ry to rx (assembler pseudoinstruction FMV.S rx, ry); <b>FSGNJN.S</b> rx, ry, ry moves the negation of ry to rx (assembler pseudoinstruction FNEG.S rx, ry); and FSGNJX.S rx, ry, ry moves the absolute value of ry to rx (assembler pseudoinstruction FABS.S rx, ry).</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.S, FSGNJN.S, and FSGNJX.S, produce a result that takes all bits except the sign bit from rs1. For FSGNJ, the result's sign bit is rs2's sign bit; for FSGNJN, the result's sign bit is the opposite of rs2's sign bit; and for FSGNJX, the sign bit is the XOR of the sign bits of rs1 and rs2. Sign-injection instructions do not set floating-point exception flags, nor do they canonicalize NaNs. Note, FSGNJ.S rx, ry, ry moves ry to rx (assembler pseudoinstruction FMV.S rx, ry); FSGNJN.S rx, ry, ry moves the negation of ry to rx (assembler pseudoinstruction FNEG.S rx, ry); and FSGNJX.S rx, ry, ry moves the absolute value of ry to rx (assembler pseudoinstruction FABS.S rx, ry).\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FSGNJX.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FSGNJX.D</b> rd, rs1, rs2</span><br><div>Floating-point to floating-point sign-injection instructions, FSGNJ.D, FSGNJN.D, and <b>FSGNJX.D</b> are defined analogously to the single-precision sign-injection instruction.</div><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.D, FSGNJN.D, and FSGNJX.D are defined analogously to the single-precision sign-injection instruction.\n\n\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/d.html#double-precision-floating-point-conversion-and-move-instructions"
            };

        case "FSGNJX.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FSGNJX.H</b> rd, rs1, rs2</span><br><div>Floating-point to floating-point sign-injection instructions, FSGNJ.H, FSGNJN.H, and <b>FSGNJX.H</b> are defined analogously to the single-precision sign-injection instruction.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.H, FSGNJN.H, and FSGNJX.H are defined analogously to the single-precision sign-injection instruction.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-convert-and-move-instructions"
            };

        case "FSGNJX.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FSGNJX.Q</b> rd, rs1, rs2</span><br><div>Floating-point to floating-point sign-injection instructions, FSGNJ.Q, FSGNJN.Q, and <b>FSGNJX.Q</b> are defined analogously to the double-precision sign-injection instruction.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.Q, FSGNJN.Q, and FSGNJX.Q are defined analogously to the double-precision sign-injection instruction.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-convert-and-move-instructions"
            };

        case "FSGNJX.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FSGNJX.S</b> rd, rs1, rs2</span><br><div>Floating-point to floating-point sign-injection instructions, FSGNJ.S, FSGNJN.S, and <b>FSGNJX.S</b>, produce a result that takes all bits except the sign bit from rs1. For FSGNJ, the result's sign bit is rs2's sign bit; for FSGNJN, the result's sign bit is the opposite of rs2's sign bit; and for FSGNJX, the sign bit is the XOR of the sign bits of rs1 and rs2. Sign-injection instructions do not set floating-point exception flags, nor do they canonicalize NaNs. Note, FSGNJ.S rx, ry, ry moves ry to rx (assembler pseudoinstruction FMV.S rx, ry); FSGNJN.S rx, ry, ry moves the negation of ry to rx (assembler pseudoinstruction FNEG.S rx, ry); and <b>FSGNJX.S</b> rx, ry, ry moves the absolute value of ry to rx (assembler pseudoinstruction FABS.S rx, ry).</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point to floating-point sign-injection instructions, FSGNJ.S, FSGNJN.S, and FSGNJX.S, produce a result that takes all bits except the sign bit from rs1. For FSGNJ, the result's sign bit is rs2's sign bit; for FSGNJN, the result's sign bit is the opposite of rs2's sign bit; and for FSGNJX, the sign bit is the XOR of the sign bits of rs1 and rs2. Sign-injection instructions do not set floating-point exception flags, nor do they canonicalize NaNs. Note, FSGNJ.S rx, ry, ry moves ry to rx (assembler pseudoinstruction FMV.S rx, ry); FSGNJN.S rx, ry, ry moves the negation of ry to rx (assembler pseudoinstruction FNEG.S rx, ry); and FSGNJX.S rx, ry, ry moves the absolute value of ry to rx (assembler pseudoinstruction FABS.S rx, ry).\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "FSH":
            return {
                "html": "<div><span class=\"opcode\"><b>FSH</b> rs1, rs2, imm12</span><br><div>FLH and <b>FSH</b> are only guaranteed to execute atomically if the effective address is naturally aligned.<br>FLH and <b>FSH</b> do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved. FLH NaN-boxes the result written to rd, whereas <b>FSH</b> ignores all but the lower 16 bits in rs2.</div><br><div><b>ISA</b>: zfh</div></div>",
                "tooltip": "FLH and FSH are only guaranteed to execute atomically if the effective address is naturally aligned.\nFLH and FSH do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved. FLH NaN-boxes the result written to rd, whereas FSH ignores all but the lower 16 bits in rs2.\n\n\n\n(ISA: zfh)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zfh.html#half-precision-load-and-store-instructions"
            };

        case "FSQ":
            return {
                "html": "<div><span class=\"opcode\"><b>FSQ</b> rs1, rs2, imm12</span><br><div>FLQ and <b>FSQ</b> are only guaranteed to execute atomically if the effective address is naturally aligned and XLEN=128.<br>FLQ and <b>FSQ</b> do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.</div><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "FLQ and FSQ are only guaranteed to execute atomically if the effective address is naturally aligned and XLEN=128.\nFLQ and FSQ do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.\n\n\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/q.html#quad-precision-load-and-store-instructions"
            };

        case "FSQRT.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FSQRT.D</b> rd, rs1</span><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FSQRT.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FSQRT.H</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FSQRT.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FSQRT.Q</b> rd, rs1</span><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FSQRT.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FSQRT.S</b> rd, rs1</span><br><div>Floating-point arithmetic instructions with one or two source operands use the R-type format with the OP-FP major opcode. FADD.S and FMUL.S perform single-precision floating-point addition and multiplication respectively, between rs1 and rs2. FSUB.S performs the single-precision floating-point subtraction of rs2 from rs1. FDIV.S performs the single-precision floating-point division of rs1 by rs2. <b>FSQRT.S</b> computes the square root of rs1. In each case, the result is written to rd.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point arithmetic instructions with one or two source operands use the R-type format with the OP-FP major opcode. FADD.S and FMUL.S perform single-precision floating-point addition and multiplication respectively, between rs1 and rs2. FSUB.S performs the single-precision floating-point subtraction of rs2 from rs1. FDIV.S performs the single-precision floating-point division of rs1 by rs2. FSQRT.S computes the square root of rs1. In each case, the result is written to rd.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#sec:single-float-compute"
            };

        case "FSRM":
            return {
                "html": "<div><span class=\"opcode\"><b>FSRM</b> rd, rs1</span><br><div>The fields within the fcsr can also be accessed individually through different CSR addresses, and separate assembler pseudoinstructions are defined for these accesses. The FRRM instruction reads the Rounding Mode field frm and copies it into the least-significant three bits of integer register rd, with zero in all other bits. <b>FSRM</b> swaps the value in frm by copying the original value into integer register rd, and then writing a new value obtained from the three least-significant bits of integer register rs1 into frm. FRFLAGS and FSFLAGS are defined analogously for the Accrued Exception Flags field fflags.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "The fields within the fcsr can also be accessed individually through different CSR addresses, and separate assembler pseudoinstructions are defined for these accesses. The FRRM instruction reads the Rounding Mode field frm and copies it into the least-significant three bits of integer register rd, with zero in all other bits. FSRM swaps the value in frm by copying the original value into integer register rd, and then writing a new value obtained from the three least-significant bits of integer register rs1 into frm. FRFLAGS and FSFLAGS are defined analogously for the Accrued Exception Flags field fflags.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#floating-point-control-and-status-register"
            };

        case "FSRMI":
            return {
                "html": "<div><span class=\"opcode\"><b>FSRMI</b> rd, zimm</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FSUB.D":
            return {
                "html": "<div><span class=\"opcode\"><b>FSUB.D</b> rd, rs1, rs2</span><br><div><b>ISA</b>: d</div></div>",
                "tooltip": "\n\n(ISA: d)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FSUB.H":
            return {
                "html": "<div><span class=\"opcode\"><b>FSUB.H</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FSUB.Q":
            return {
                "html": "<div><span class=\"opcode\"><b>FSUB.Q</b> rd, rs1, rs2</span><br><div><b>ISA</b>: q</div></div>",
                "tooltip": "\n\n(ISA: q)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "FSUB.S":
            return {
                "html": "<div><span class=\"opcode\"><b>FSUB.S</b> rd, rs1, rs2</span><br><div>Floating-point arithmetic instructions with one or two source operands use the R-type format with the OP-FP major opcode. FADD.S and FMUL.S perform single-precision floating-point addition and multiplication respectively, between rs1 and rs2. <b>FSUB.S</b> performs the single-precision floating-point subtraction of rs2 from rs1. FDIV.S performs the single-precision floating-point division of rs1 by rs2. FSQRT.S computes the square root of rs1. In each case, the result is written to rd.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point arithmetic instructions with one or two source operands use the R-type format with the OP-FP major opcode. FADD.S and FMUL.S perform single-precision floating-point addition and multiplication respectively, between rs1 and rs2. FSUB.S performs the single-precision floating-point subtraction of rs2 from rs1. FDIV.S performs the single-precision floating-point division of rs1 by rs2. FSQRT.S computes the square root of rs1. In each case, the result is written to rd.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#sec:single-float-compute"
            };

        case "FSW":
            return {
                "html": "<div><span class=\"opcode\"><b>FSW</b> rs1, rs2, imm12</span><br><div>Floating-point loads and stores use the same base+offset addressing mode as the integer base ISAs, with a base address in register rs1 and a 12-bit signed byte offset. The FLW instruction loads a single-precision floating-point value from memory into floating-point register rd. <b>FSW</b> stores a single-precision value from floating-point register rs2 to memory.<br>FLW and <b>FSW</b> are only guaranteed to execute atomically if the effective address is naturally aligned.<br>FLW and <b>FSW</b> do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.</div><br><div><b>ISA</b>: f</div></div>",
                "tooltip": "Floating-point loads and stores use the same base+offset addressing mode as the integer base ISAs, with a base address in register rs1 and a 12-bit signed byte offset. The FLW instruction loads a single-precision floating-point value from memory into floating-point register rd. FSW stores a single-precision value from floating-point register rs2 to memory.\nFLW and FSW are only guaranteed to execute atomically if the effective address is naturally aligned.\nFLW and FSW do not modify the bits being transferred; in particular, the payloads of non-canonical NaNs are preserved.\n\n\n\n(ISA: f)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-load-and-store-instructions"
            };

        case "HFENCE.GVMA":
            return {
                "html": "<div><span class=\"opcode\"><b>HFENCE.GVMA</b> rs1, rs2</span><br><div>The Svinval extension splits SFENCE.VMA, HFENCE.VVMA, and <b>HFENCE.GVMA</b> instructions into finer-grained invalidation and ordering operations that can be more efficiently batched or pipelined on certain classes of high-performance implementation.<br>If the hypervisor extension is implemented, the Svinval extension also provides two additional instructions: HINVAL.VVMA and HINVAL.GVMA. These have the same semantics as SINVAL.VMA, except that they combine with SFENCE.W.INVAL and SFENCE.INVAL.IR to replace HFENCE.VVMA and <b>HFENCE.GVMA</b>, respectively, instead of SFENCE.VMA. In addition, HINVAL.GVMA uses VMIDs instead of ASIDs.<br>SINVAL.VMA, HINVAL.VVMA, and HINVAL.GVMA require the same permissions and raise the same exceptions as SFENCE.VMA, HFENCE.VVMA, and <b>HFENCE.GVMA</b>, respectively. In particular, an attempt to execute any of these instructions in U-mode always raises an illegal instruction exception, and an attempt to execute SINVAL.VMA or HINVAL.GVMA in S-mode or HS-mode when mstatus.TVM=1 also raises an illegal instruction exception. An attempt to execute HINVAL.VVMA or HINVAL.GVMA in VS-mode or VU-mode, or to execute SINVAL.VMA in VU-mode, raises a virtual instruction exception. When hstatus.VTVM=1, an attempt to execute SINVAL.VMA in VS-mode also raises a virtual instruction exception.<br>High-performance implementations will be able to pipeline the address-translation cache invalidation operations, and will defer any pipeline stalls or other memory ordering enforcement until an SFENCE.W.INVAL, SFENCE.INVAL.IR, SFENCE.VMA, <b>HFENCE.GVMA</b>, or HFENCE.VVMA instruction is executed.<br>Simpler implementations may implement SINVAL.VMA, HINVAL.VVMA, and HINVAL.GVMA identically to SFENCE.VMA, HFENCE.VVMA, and <b>HFENCE.GVMA</b>, respectively, while implementing SFENCE.W.INVAL and SFENCE.INVAL.IR instructions as no-ops.</div><br><div><b>ISA</b>: supervisor</div></div>",
                "tooltip": "The Svinval extension splits SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA instructions into finer-grained invalidation and ordering operations that can be more efficiently batched or pipelined on certain classes of high-performance implementation.\nIf the hypervisor extension is implemented, the Svinval extension also provides two additional instructions: HINVAL.VVMA and HINVAL.GVMA. These have the same semantics as SINVAL.VMA, except that they combine with SFENCE.W.INVAL and SFENCE.INVAL.IR to replace HFENCE.VVMA and HFENCE.GVMA, respectively, instead of SFENCE.VMA. In addition, HINVAL.GVMA uses VMIDs instead of ASIDs.\nSINVAL.VMA, HINVAL.VVMA, and HINVAL.GVMA require the same permissions and raise the same exceptions as SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA, respectively. In particular, an attempt to execute any of these instructions in U-mode always raises an illegal instruction exception, and an attempt to execute SINVAL.VMA or HINVAL.GVMA in S-mode or HS-mode when mstatus.TVM=1 also raises an illegal instruction exception. An attempt to execute HINVAL.VVMA or HINVAL.GVMA in VS-mode or VU-mode, or to execute SINVAL.VMA in VU-mode, raises a virtual instruction exception. When hstatus.VTVM=1, an attempt to execute SINVAL.VMA in VS-mode also raises a virtual instruction exception.\nHigh-performance implementations will be able to pipeline the address-translation cache invalidation operations, and will defer any pipeline stalls or other memory ordering enforcement until an SFENCE.W.INVAL, SFENCE.INVAL.IR, SFENCE.VMA, HFENCE.GVMA, or HFENCE.VVMA instruction is executed.\nSimpler implementations may implement SINVAL.VMA, HINVAL.VVMA, and HINVAL.GVMA identically to SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA, respectively, while implementing SFENCE.W.INVAL and SFENCE.INVAL.IR instructions as no-ops.\n\n\n\n(ISA: supervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/supervisor.html#svinval"
            };

        case "HFENCE.VVMA":
            return {
                "html": "<div><span class=\"opcode\"><b>HFENCE.VVMA</b> rs1, rs2</span><br><div>The Svinval extension splits SFENCE.VMA, <b>HFENCE.VVMA</b>, and HFENCE.GVMA instructions into finer-grained invalidation and ordering operations that can be more efficiently batched or pipelined on certain classes of high-performance implementation.<br>If the hypervisor extension is implemented, the Svinval extension also provides two additional instructions: HINVAL.VVMA and HINVAL.GVMA. These have the same semantics as SINVAL.VMA, except that they combine with SFENCE.W.INVAL and SFENCE.INVAL.IR to replace <b>HFENCE.VVMA</b> and HFENCE.GVMA, respectively, instead of SFENCE.VMA. In addition, HINVAL.GVMA uses VMIDs instead of ASIDs.<br>SINVAL.VMA, HINVAL.VVMA, and HINVAL.GVMA require the same permissions and raise the same exceptions as SFENCE.VMA, <b>HFENCE.VVMA</b>, and HFENCE.GVMA, respectively. In particular, an attempt to execute any of these instructions in U-mode always raises an illegal instruction exception, and an attempt to execute SINVAL.VMA or HINVAL.GVMA in S-mode or HS-mode when mstatus.TVM=1 also raises an illegal instruction exception. An attempt to execute HINVAL.VVMA or HINVAL.GVMA in VS-mode or VU-mode, or to execute SINVAL.VMA in VU-mode, raises a virtual instruction exception. When hstatus.VTVM=1, an attempt to execute SINVAL.VMA in VS-mode also raises a virtual instruction exception.<br>High-performance implementations will be able to pipeline the address-translation cache invalidation operations, and will defer any pipeline stalls or other memory ordering enforcement until an SFENCE.W.INVAL, SFENCE.INVAL.IR, SFENCE.VMA, HFENCE.GVMA, or <b>HFENCE.VVMA</b> instruction is executed.<br>Simpler implementations may implement SINVAL.VMA, HINVAL.VVMA, and HINVAL.GVMA identically to SFENCE.VMA, <b>HFENCE.VVMA</b>, and HFENCE.GVMA, respectively, while implementing SFENCE.W.INVAL and SFENCE.INVAL.IR instructions as no-ops.</div><br><div><b>ISA</b>: supervisor</div></div>",
                "tooltip": "The Svinval extension splits SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA instructions into finer-grained invalidation and ordering operations that can be more efficiently batched or pipelined on certain classes of high-performance implementation.\nIf the hypervisor extension is implemented, the Svinval extension also provides two additional instructions: HINVAL.VVMA and HINVAL.GVMA. These have the same semantics as SINVAL.VMA, except that they combine with SFENCE.W.INVAL and SFENCE.INVAL.IR to replace HFENCE.VVMA and HFENCE.GVMA, respectively, instead of SFENCE.VMA. In addition, HINVAL.GVMA uses VMIDs instead of ASIDs.\nSINVAL.VMA, HINVAL.VVMA, and HINVAL.GVMA require the same permissions and raise the same exceptions as SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA, respectively. In particular, an attempt to execute any of these instructions in U-mode always raises an illegal instruction exception, and an attempt to execute SINVAL.VMA or HINVAL.GVMA in S-mode or HS-mode when mstatus.TVM=1 also raises an illegal instruction exception. An attempt to execute HINVAL.VVMA or HINVAL.GVMA in VS-mode or VU-mode, or to execute SINVAL.VMA in VU-mode, raises a virtual instruction exception. When hstatus.VTVM=1, an attempt to execute SINVAL.VMA in VS-mode also raises a virtual instruction exception.\nHigh-performance implementations will be able to pipeline the address-translation cache invalidation operations, and will defer any pipeline stalls or other memory ordering enforcement until an SFENCE.W.INVAL, SFENCE.INVAL.IR, SFENCE.VMA, HFENCE.GVMA, or HFENCE.VVMA instruction is executed.\nSimpler implementations may implement SINVAL.VMA, HINVAL.VVMA, and HINVAL.GVMA identically to SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA, respectively, while implementing SFENCE.W.INVAL and SFENCE.INVAL.IR instructions as no-ops.\n\n\n\n(ISA: supervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/supervisor.html#svinval"
            };

        case "HINVAL.GVMA":
            return {
                "html": "<div><span class=\"opcode\"><b>HINVAL.GVMA</b> rs1, rs2</span><br><div>If the hypervisor extension is implemented, the Svinval extension also provides two additional instructions: HINVAL.VVMA and <b>HINVAL.GVMA</b>. These have the same semantics as SINVAL.VMA, except that they combine with SFENCE.W.INVAL and SFENCE.INVAL.IR to replace HFENCE.VVMA and HFENCE.GVMA, respectively, instead of SFENCE.VMA. In addition, <b>HINVAL.GVMA</b> uses VMIDs instead of ASIDs.<br>SINVAL.VMA, HINVAL.VVMA, and <b>HINVAL.GVMA</b> require the same permissions and raise the same exceptions as SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA, respectively. In particular, an attempt to execute any of these instructions in U-mode always raises an illegal instruction exception, and an attempt to execute SINVAL.VMA or <b>HINVAL.GVMA</b> in S-mode or HS-mode when mstatus.TVM=1 also raises an illegal instruction exception. An attempt to execute HINVAL.VVMA or <b>HINVAL.GVMA</b> in VS-mode or VU-mode, or to execute SINVAL.VMA in VU-mode, raises a virtual instruction exception. When hstatus.VTVM=1, an attempt to execute SINVAL.VMA in VS-mode also raises a virtual instruction exception.<br>In typical usage, software will invalidate a range of virtual addresses in the address-translation caches by executing an SFENCE.W.INVAL instruction, executing a series of SINVAL.VMA, HINVAL.VVMA, or <b>HINVAL.GVMA</b> instructions to the addresses (and optionally ASIDs or VMIDs) in question, and then executing an SFENCE.INVAL.IR instruction.<br>Simpler implementations may implement SINVAL.VMA, HINVAL.VVMA, and <b>HINVAL.GVMA</b> identically to SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA, respectively, while implementing SFENCE.W.INVAL and SFENCE.INVAL.IR instructions as no-ops.</div><br><div><b>ISA</b>: supervisor</div></div>",
                "tooltip": "If the hypervisor extension is implemented, the Svinval extension also provides two additional instructions: HINVAL.VVMA and HINVAL.GVMA. These have the same semantics as SINVAL.VMA, except that they combine with SFENCE.W.INVAL and SFENCE.INVAL.IR to replace HFENCE.VVMA and HFENCE.GVMA, respectively, instead of SFENCE.VMA. In addition, HINVAL.GVMA uses VMIDs instead of ASIDs.\nSINVAL.VMA, HINVAL.VVMA, and HINVAL.GVMA require the same permissions and raise the same exceptions as SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA, respectively. In particular, an attempt to execute any of these instructions in U-mode always raises an illegal instruction exception, and an attempt to execute SINVAL.VMA or HINVAL.GVMA in S-mode or HS-mode when mstatus.TVM=1 also raises an illegal instruction exception. An attempt to execute HINVAL.VVMA or HINVAL.GVMA in VS-mode or VU-mode, or to execute SINVAL.VMA in VU-mode, raises a virtual instruction exception. When hstatus.VTVM=1, an attempt to execute SINVAL.VMA in VS-mode also raises a virtual instruction exception.\nIn typical usage, software will invalidate a range of virtual addresses in the address-translation caches by executing an SFENCE.W.INVAL instruction, executing a series of SINVAL.VMA, HINVAL.VVMA, or HINVAL.GVMA instructions to the addresses (and optionally ASIDs or VMIDs) in question, and then executing an SFENCE.INVAL.IR instruction.\nSimpler implementations may implement SINVAL.VMA, HINVAL.VVMA, and HINVAL.GVMA identically to SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA, respectively, while implementing SFENCE.W.INVAL and SFENCE.INVAL.IR instructions as no-ops.\n\n\n\n(ISA: supervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/supervisor.html#svinval"
            };

        case "HINVAL.VVMA":
            return {
                "html": "<div><span class=\"opcode\"><b>HINVAL.VVMA</b> rs1, rs2</span><br><div>If the hypervisor extension is implemented, the Svinval extension also provides two additional instructions: <b>HINVAL.VVMA</b> and HINVAL.GVMA. These have the same semantics as SINVAL.VMA, except that they combine with SFENCE.W.INVAL and SFENCE.INVAL.IR to replace HFENCE.VVMA and HFENCE.GVMA, respectively, instead of SFENCE.VMA. In addition, HINVAL.GVMA uses VMIDs instead of ASIDs.<br>SINVAL.VMA, <b>HINVAL.VVMA</b>, and HINVAL.GVMA require the same permissions and raise the same exceptions as SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA, respectively. In particular, an attempt to execute any of these instructions in U-mode always raises an illegal instruction exception, and an attempt to execute SINVAL.VMA or HINVAL.GVMA in S-mode or HS-mode when mstatus.TVM=1 also raises an illegal instruction exception. An attempt to execute <b>HINVAL.VVMA</b> or HINVAL.GVMA in VS-mode or VU-mode, or to execute SINVAL.VMA in VU-mode, raises a virtual instruction exception. When hstatus.VTVM=1, an attempt to execute SINVAL.VMA in VS-mode also raises a virtual instruction exception.<br>In typical usage, software will invalidate a range of virtual addresses in the address-translation caches by executing an SFENCE.W.INVAL instruction, executing a series of SINVAL.VMA, <b>HINVAL.VVMA</b>, or HINVAL.GVMA instructions to the addresses (and optionally ASIDs or VMIDs) in question, and then executing an SFENCE.INVAL.IR instruction.<br>Simpler implementations may implement SINVAL.VMA, <b>HINVAL.VVMA</b>, and HINVAL.GVMA identically to SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA, respectively, while implementing SFENCE.W.INVAL and SFENCE.INVAL.IR instructions as no-ops.</div><br><div><b>ISA</b>: supervisor</div></div>",
                "tooltip": "If the hypervisor extension is implemented, the Svinval extension also provides two additional instructions: HINVAL.VVMA and HINVAL.GVMA. These have the same semantics as SINVAL.VMA, except that they combine with SFENCE.W.INVAL and SFENCE.INVAL.IR to replace HFENCE.VVMA and HFENCE.GVMA, respectively, instead of SFENCE.VMA. In addition, HINVAL.GVMA uses VMIDs instead of ASIDs.\nSINVAL.VMA, HINVAL.VVMA, and HINVAL.GVMA require the same permissions and raise the same exceptions as SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA, respectively. In particular, an attempt to execute any of these instructions in U-mode always raises an illegal instruction exception, and an attempt to execute SINVAL.VMA or HINVAL.GVMA in S-mode or HS-mode when mstatus.TVM=1 also raises an illegal instruction exception. An attempt to execute HINVAL.VVMA or HINVAL.GVMA in VS-mode or VU-mode, or to execute SINVAL.VMA in VU-mode, raises a virtual instruction exception. When hstatus.VTVM=1, an attempt to execute SINVAL.VMA in VS-mode also raises a virtual instruction exception.\nIn typical usage, software will invalidate a range of virtual addresses in the address-translation caches by executing an SFENCE.W.INVAL instruction, executing a series of SINVAL.VMA, HINVAL.VVMA, or HINVAL.GVMA instructions to the addresses (and optionally ASIDs or VMIDs) in question, and then executing an SFENCE.INVAL.IR instruction.\nSimpler implementations may implement SINVAL.VMA, HINVAL.VVMA, and HINVAL.GVMA identically to SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA, respectively, while implementing SFENCE.W.INVAL and SFENCE.INVAL.IR instructions as no-ops.\n\n\n\n(ISA: supervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/supervisor.html#svinval"
            };

        case "HLV.B":
            return {
                "html": "<div><span class=\"opcode\"><b>HLV.B</b> rd, rs1</span><br><div>For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: <b>HLV.B</b>, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.</div><br><div><b>ISA</b>: hypervisor</div></div>",
                "tooltip": "For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.\n\n\n\n(ISA: hypervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/hypervisor.html#hypervisor-virtual-machine-load-and-store-instructions"
            };

        case "HLV.BU":
            return {
                "html": "<div><span class=\"opcode\"><b>HLV.BU</b> rd, rs1</span><br><div>For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, <b>HLV.BU</b>, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.</div><br><div><b>ISA</b>: hypervisor</div></div>",
                "tooltip": "For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.\n\n\n\n(ISA: hypervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/hypervisor.html#hypervisor-virtual-machine-load-and-store-instructions"
            };

        case "HLV.D":
            return {
                "html": "<div><span class=\"opcode\"><b>HLV.D</b> rd, rs1</span><br><div>For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and <b>HLV.D</b>. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, <b>HLV.D</b>, and HSV.D are not valid for RV32, of course.</div><br><div><b>ISA</b>: hypervisor</div></div>",
                "tooltip": "For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.\n\n\n\n(ISA: hypervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/hypervisor.html#hypervisor-virtual-machine-load-and-store-instructions"
            };

        case "HLV.H":
            return {
                "html": "<div><span class=\"opcode\"><b>HLV.H</b> rd, rs1</span><br><div>For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, <b>HLV.H</b>, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.</div><br><div><b>ISA</b>: hypervisor</div></div>",
                "tooltip": "For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.\n\n\n\n(ISA: hypervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/hypervisor.html#hypervisor-virtual-machine-load-and-store-instructions"
            };

        case "HLV.HU":
            return {
                "html": "<div><span class=\"opcode\"><b>HLV.HU</b> rd, rs1</span><br><div>For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, <b>HLV.HU</b>, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.<br>Instructions HLVX.HU and HLVX.WU are the same as <b>HLV.HU</b> and HLV.WU, except that execute permission takes the place of read permission during address translation. That is, the memory being read must be executable in both stages of address translation, but read permission is not required. For the supervisor physical address that results from address translation, the supervisor physical memory attributes must grant both execute and read permissions. (The supervisor physical memory attributes are the machine's physical memory attributes as modified by physical memory protection, Section [sec:pmp] , for supervisor level.)</div><br><div><b>ISA</b>: hypervisor</div></div>",
                "tooltip": "For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.\nInstructions HLVX.HU and HLVX.WU are the same as HLV.HU and HLV.WU, except that execute permission takes the place of read permission during address translation. That is, the memory being read must be executable in both stages of address translation, but read permission is not required. For the supervisor physical address that results from address translation, the supervisor physical memory attributes must grant both execute and read permissions. (The supervisor physical memory attributes are the machine's physical memory attributes as modified by physical memory protection, Section [sec:pmp] , for supervisor level.)\n\n\n\n(ISA: hypervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/hypervisor.html#hypervisor-virtual-machine-load-and-store-instructions"
            };

        case "HLV.W":
            return {
                "html": "<div><span class=\"opcode\"><b>HLV.W</b> rd, rs1</span><br><div>For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, <b>HLV.W</b>, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.<br>HLVX.WU is valid for RV32, even though LWU and HLV.WU are not. (For RV32, HLVX.WU can be considered a variant of <b>HLV.W</b>, as sign extension is irrelevant for 32-bit values.)</div><br><div><b>ISA</b>: hypervisor</div></div>",
                "tooltip": "For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.\nHLVX.WU is valid for RV32, even though LWU and HLV.WU are not. (For RV32, HLVX.WU can be considered a variant of HLV.W, as sign extension is irrelevant for 32-bit values.)\n\n\n\n(ISA: hypervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/hypervisor.html#hypervisor-virtual-machine-load-and-store-instructions"
            };

        case "HLV.WU":
            return {
                "html": "<div><span class=\"opcode\"><b>HLV.WU</b> rd, rs1</span><br><div>For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, <b>HLV.WU</b>, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions <b>HLV.WU</b>, HLV.D, and HSV.D are not valid for RV32, of course.<br>Instructions HLVX.HU and HLVX.WU are the same as HLV.HU and <b>HLV.WU</b>, except that execute permission takes the place of read permission during address translation. That is, the memory being read must be executable in both stages of address translation, but read permission is not required. For the supervisor physical address that results from address translation, the supervisor physical memory attributes must grant both execute and read permissions. (The supervisor physical memory attributes are the machine's physical memory attributes as modified by physical memory protection, Section [sec:pmp] , for supervisor level.)<br>HLVX.WU is valid for RV32, even though LWU and <b>HLV.WU</b> are not. (For RV32, HLVX.WU can be considered a variant of HLV.W, as sign extension is irrelevant for 32-bit values.)</div><br><div><b>ISA</b>: hypervisor</div></div>",
                "tooltip": "For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.\nInstructions HLVX.HU and HLVX.WU are the same as HLV.HU and HLV.WU, except that execute permission takes the place of read permission during address translation. That is, the memory being read must be executable in both stages of address translation, but read permission is not required. For the supervisor physical address that results from address translation, the supervisor physical memory attributes must grant both execute and read permissions. (The supervisor physical memory attributes are the machine's physical memory attributes as modified by physical memory protection, Section [sec:pmp] , for supervisor level.)\nHLVX.WU is valid for RV32, even though LWU and HLV.WU are not. (For RV32, HLVX.WU can be considered a variant of HLV.W, as sign extension is irrelevant for 32-bit values.)\n\n\n\n(ISA: hypervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/hypervisor.html#hypervisor-virtual-machine-load-and-store-instructions"
            };

        case "HLVX.HU":
            return {
                "html": "<div><span class=\"opcode\"><b>HLVX.HU</b> rd, rs1</span><br><div>Instructions <b>HLVX.HU</b> and HLVX.WU are the same as HLV.HU and HLV.WU, except that execute permission takes the place of read permission during address translation. That is, the memory being read must be executable in both stages of address translation, but read permission is not required. For the supervisor physical address that results from address translation, the supervisor physical memory attributes must grant both execute and read permissions. (The supervisor physical memory attributes are the machine's physical memory attributes as modified by physical memory protection, Section [sec:pmp] , for supervisor level.)</div><br><div><b>ISA</b>: hypervisor</div></div>",
                "tooltip": "Instructions HLVX.HU and HLVX.WU are the same as HLV.HU and HLV.WU, except that execute permission takes the place of read permission during address translation. That is, the memory being read must be executable in both stages of address translation, but read permission is not required. For the supervisor physical address that results from address translation, the supervisor physical memory attributes must grant both execute and read permissions. (The supervisor physical memory attributes are the machine's physical memory attributes as modified by physical memory protection, Section [sec:pmp] , for supervisor level.)\n\n\n\n(ISA: hypervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/hypervisor.html#hypervisor-virtual-machine-load-and-store-instructions"
            };

        case "HLVX.WU":
            return {
                "html": "<div><span class=\"opcode\"><b>HLVX.WU</b> rd, rs1</span><br><div>Instructions HLVX.HU and <b>HLVX.WU</b> are the same as HLV.HU and HLV.WU, except that execute permission takes the place of read permission during address translation. That is, the memory being read must be executable in both stages of address translation, but read permission is not required. For the supervisor physical address that results from address translation, the supervisor physical memory attributes must grant both execute and read permissions. (The supervisor physical memory attributes are the machine's physical memory attributes as modified by physical memory protection, Section [sec:pmp] , for supervisor level.)<br><b>HLVX.WU</b> is valid for RV32, even though LWU and HLV.WU are not. (For RV32, <b>HLVX.WU</b> can be considered a variant of HLV.W, as sign extension is irrelevant for 32-bit values.)</div><br><div><b>ISA</b>: hypervisor</div></div>",
                "tooltip": "Instructions HLVX.HU and HLVX.WU are the same as HLV.HU and HLV.WU, except that execute permission takes the place of read permission during address translation. That is, the memory being read must be executable in both stages of address translation, but read permission is not required. For the supervisor physical address that results from address translation, the supervisor physical memory attributes must grant both execute and read permissions. (The supervisor physical memory attributes are the machine's physical memory attributes as modified by physical memory protection, Section [sec:pmp] , for supervisor level.)\nHLVX.WU is valid for RV32, even though LWU and HLV.WU are not. (For RV32, HLVX.WU can be considered a variant of HLV.W, as sign extension is irrelevant for 32-bit values.)\n\n\n\n(ISA: hypervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/hypervisor.html#hypervisor-virtual-machine-load-and-store-instructions"
            };

        case "HSV.B":
            return {
                "html": "<div><span class=\"opcode\"><b>HSV.B</b> rs1, rs2</span><br><div>For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: <b>HSV.B</b>, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.</div><br><div><b>ISA</b>: hypervisor</div></div>",
                "tooltip": "For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.\n\n\n\n(ISA: hypervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/hypervisor.html#hypervisor-virtual-machine-load-and-store-instructions"
            };

        case "HSV.D":
            return {
                "html": "<div><span class=\"opcode\"><b>HSV.D</b> rs1, rs2</span><br><div>For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and <b>HSV.D</b>. Instructions HLV.WU, HLV.D, and <b>HSV.D</b> are not valid for RV32, of course.</div><br><div><b>ISA</b>: hypervisor</div></div>",
                "tooltip": "For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.\n\n\n\n(ISA: hypervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/hypervisor.html#hypervisor-virtual-machine-load-and-store-instructions"
            };

        case "HSV.H":
            return {
                "html": "<div><span class=\"opcode\"><b>HSV.H</b> rs1, rs2</span><br><div>For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, <b>HSV.H</b>, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.</div><br><div><b>ISA</b>: hypervisor</div></div>",
                "tooltip": "For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.\n\n\n\n(ISA: hypervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/hypervisor.html#hypervisor-virtual-machine-load-and-store-instructions"
            };

        case "HSV.W":
            return {
                "html": "<div><span class=\"opcode\"><b>HSV.W</b> rs1, rs2</span><br><div>For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, <b>HSV.W</b>, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.</div><br><div><b>ISA</b>: hypervisor</div></div>",
                "tooltip": "For every RV32I or RV64I load instruction, LB, LBU, LH, LHU, LW, LWU, and LD, there is a corresponding virtual-machine load instruction: HLV.B, HLV.BU, HLV.H, HLV.HU, HLV.W, HLV.WU, and HLV.D. For every RV32I or RV64I store instruction, SB, SH, SW, and SD, there is a corresponding virtual-machine store instruction: HSV.B, HSV.H, HSV.W, and HSV.D. Instructions HLV.WU, HLV.D, and HSV.D are not valid for RV32, of course.\n\n\n\n(ISA: hypervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/hypervisor.html#hypervisor-virtual-machine-load-and-store-instructions"
            };

        case "J":
            return {
                "html": "<div><span class=\"opcode\"><b>J</b> offset</span><br><div>There are a further two variants of the instruction formats (B/<b>J</b>) based on the handling of immediates, as shown in Figure 1.3 .<br>Similarly, the only difference between the U and <b>J</b> formats is that the 20-bit immediate is shifted left by 12 bits to form U immediates and by 1 bit to form <b>J</b> immediates. The location of instruction bits in the U and <b>J</b> format immediates is chosen to maximize overlap with the other formats and with each other.<br>Although more complex implementations might have separate adders for branch and jump calculations and so would not benefit from keeping the location of immediate bits constant across types of instruction, we wanted to reduce the hardware cost of the simplest implementations. By rotating bits in the instruction encoding of B and <b>J</b> immediates instead of using dynamic hardware muxes to multiply the immediate by 2, we reduce instruction signal fanout and immediate mux costs by around a factor of 2. The scrambled immediate encoding will add negligible time to static or ahead-of-time compilation. For dynamic generation of instructions, there is some small additional overhead, but the most common short forward branches have straightforward immediate encodings.</div><br><div><b>Equivalent ASM:</b><pre>jal x0, offset</pre></div><br><div><b>ISA</b>: rv32(pseudo)</div></div>",
                "tooltip": "There are a further two variants of the instruction formats (B/J) based on the handling of immediates, as shown in Figure 1.3 .\nSimilarly, the only difference between the U and J formats is that the 20-bit immediate is shifted left by 12 bits to form U immediates and by 1 bit to form J immediates. The location of instruction bits in the U and J format immediates is chosen to maximize overlap with the other formats and with each other.\nAlthough more complex implementations might have separate adders for branch and jump calculations and so would not benefit from keeping the location of immediate bits constant across types of instruction, we wanted to reduce the hardware cost of the simplest implementations. By rotating bits in the instruction encoding of B and J immediates instead of using dynamic hardware muxes to multiply the immediate by 2, we reduce instruction signal fanout and immediate mux costs by around a factor of 2. The scrambled immediate encoding will add negligible time to static or ahead-of-time compilation. For dynamic generation of instructions, there is some small additional overhead, but the most common short forward branches have straightforward immediate encodings.\n\nEquivalent ASM:\n\njal x0, offset\n\n(ISA: rv32(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#immediate-encoding-variants"
            };

        case "JAL":
            return {
                "html": "<div><span class=\"opcode\"><b>JAL</b> rd, jimm20</span><br><div>The current PC can be obtained by setting the U-immediate to 0. Although a <b>JAL</b> +4 instruction could also be used to obtain the local PC (of the instruction following the <b>JAL</b>), it might cause pipeline breaks in simpler microarchitectures or pollute BTB structures in more complex microarchitectures.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "The current PC can be obtained by setting the U-immediate to 0. Although a JAL +4 instruction could also be used to obtain the local PC (of the instruction following the JAL), it might cause pipeline breaks in simpler microarchitectures or pollute BTB structures in more complex microarchitectures.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-immediate-instructions"
            };

        case "JALR":
            return {
                "html": "<div><span class=\"opcode\"><b>JALR</b> rd, rs1, imm12</span><br><div>The AUIPC instruction supports two-instruction sequences to access arbitrary offsets from the PC for both control-flow transfers and data accesses. The combination of an AUIPC and the 12-bit immediate in a <b>JALR</b> can transfer control to any 32-bit PC-relative address, while an AUIPC plus the 12-bit immediate offset in regular load or store instructions can access any 32-bit PC-relative data address.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "The AUIPC instruction supports two-instruction sequences to access arbitrary offsets from the PC for both control-flow transfers and data accesses. The combination of an AUIPC and the 12-bit immediate in a JALR can transfer control to any 32-bit PC-relative address, while an AUIPC plus the 12-bit immediate offset in regular load or store instructions can access any 32-bit PC-relative data address.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-immediate-instructions"
            };

        case "JR":
            return {
                "html": "<div><span class=\"opcode\"><b>JR</b> rs</span><br><div><b>Equivalent ASM:</b><pre>jalr x0, rs, 0</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\njalr x0, rs, 0\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "LA":
            return {
                "html": "<div><span class=\"opcode\"><b>LA</b> rd, symbol</span><br><div><b>Equivalent ASM:</b><pre>auipc rd, symbol@GOT[31:12]\nl{w\\</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nauipc rd, symbol@GOT[31:12]\nl{w\\\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "LB":
            return {
                "html": "<div><span class=\"opcode\"><b>LB</b> rd, rs1, imm12</span><br><div>The LW instruction loads a 32-bit value from memory into rd. LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd. LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd. <b>LB</b> and LBU are defined analogously for 8-bit values. The SW, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "The LW instruction loads a 32-bit value from memory into rd. LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd. LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd. LB and LBU are defined analogously for 8-bit values. The SW, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#sec:rv32:ldst"
            };

        case "LBU":
            return {
                "html": "<div><span class=\"opcode\"><b>LBU</b> rd, rs1, imm12</span><br><div>The LW instruction loads a 32-bit value from memory into rd. LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd. LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd. LB and <b>LBU</b> are defined analogously for 8-bit values. The SW, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "The LW instruction loads a 32-bit value from memory into rd. LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd. LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd. LB and LBU are defined analogously for 8-bit values. The SW, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#sec:rv32:ldst"
            };

        case "LD":
            return {
                "html": "<div><span class=\"opcode\"><b>LD</b> rd, rs1, imm12</span><br><div>Note that the set of address offsets that can be formed by pairing LUI with <b>LD</b>, AUIPC with JALR, etc.in RV64I is [ - 231 - 211, 231 - 211 - 1].</div><br><div><b>ISA</b>: rv64</div></div>",
                "tooltip": "Note that the set of address offsets that can be formed by pairing LUI with LD, AUIPC with JALR, etc.in RV64I is [ - 231 - 211, 231 - 211 - 1].\n\n\n\n(ISA: rv64)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv64.html#integer-register-immediate-instructions"
            };

        case "LGA":
            return {
                "html": "<div><span class=\"opcode\"><b>LGA</b> rd, symbol</span><br><div><b>Equivalent ASM:</b><pre>auipc rd, symbol@GOT[31:12]\nl{w\\</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nauipc rd, symbol@GOT[31:12]\nl{w\\\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "LH":
            return {
                "html": "<div><span class=\"opcode\"><b>LH</b> rd, rs1, imm12</span><br><div>The LW instruction loads a 32-bit value from memory into rd. <b>LH</b> loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd. LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd. LB and LBU are defined analogously for 8-bit values. The SW, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "The LW instruction loads a 32-bit value from memory into rd. LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd. LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd. LB and LBU are defined analogously for 8-bit values. The SW, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#sec:rv32:ldst"
            };

        case "LHU":
            return {
                "html": "<div><span class=\"opcode\"><b>LHU</b> rd, rs1, imm12</span><br><div>The LW instruction loads a 32-bit value from memory into rd. LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd. <b>LHU</b> loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd. LB and LBU are defined analogously for 8-bit values. The SW, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "The LW instruction loads a 32-bit value from memory into rd. LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd. LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd. LB and LBU are defined analogously for 8-bit values. The SW, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#sec:rv32:ldst"
            };

        case "LI":
            return {
                "html": "<div><span class=\"opcode\"><b>LI</b> rd, immediate</span><br><div><b>Equivalent ASM:</b><pre>*Myriad sequences*</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\n*Myriad sequences*\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "LLA":
            return {
                "html": "<div><span class=\"opcode\"><b>LLA</b> rd, symbol</span><br><div><b>Equivalent ASM:</b><pre>auipc rd, symbol[31:12]\naddi rd, rd, symbol[11:0]</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nauipc rd, symbol[31:12]\naddi rd, rd, symbol[11:0]\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "LR.D":
            return {
                "html": "<div><span class=\"opcode\"><b>LR.D</b> rd, rs1</span><br><div>Complex atomic memory operations on a single memory word or doubleword are performed with the load-reserved (LR) and store-conditional (SC) instructions. LR.W loads a word from the address in rs1, places the sign-extended value in rd, and registers a reservation set--a set of bytes that subsumes the bytes in the addressed word. SC.W conditionally writes a word in rs2 to the address in rs1: the SC.W succeeds only if the reservation is still valid and the reservation set contains the bytes being written. If the SC.W succeeds, the instruction writes the word in rs2 to memory, and it writes zero to rd. If the SC.W fails, the instruction does not write to memory, and it writes a nonzero value to rd. Regardless of success or failure, executing an SC.W instruction invalidates any reservation held by this hart. <b>LR.D</b> and SC.D act analogously on doublewords and are only available on RV64. For RV64, LR.W and SC.W sign-extend the value placed in rd.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "Complex atomic memory operations on a single memory word or doubleword are performed with the load-reserved (LR) and store-conditional (SC) instructions. LR.W loads a word from the address in rs1, places the sign-extended value in rd, and registers a reservation set--a set of bytes that subsumes the bytes in the addressed word. SC.W conditionally writes a word in rs2 to the address in rs1: the SC.W succeeds only if the reservation is still valid and the reservation set contains the bytes being written. If the SC.W succeeds, the instruction writes the word in rs2 to memory, and it writes zero to rd. If the SC.W fails, the instruction does not write to memory, and it writes a nonzero value to rd. Regardless of success or failure, executing an SC.W instruction invalidates any reservation held by this hart. LR.D and SC.D act analogously on doublewords and are only available on RV64. For RV64, LR.W and SC.W sign-extend the value placed in rd.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:lrsc"
            };

        case "LR.W":
            return {
                "html": "<div><span class=\"opcode\"><b>LR.W</b> rd, rs1</span><br><div>Complex atomic memory operations on a single memory word or doubleword are performed with the load-reserved (LR) and store-conditional (SC) instructions. <b>LR.W</b> loads a word from the address in rs1, places the sign-extended value in rd, and registers a reservation set--a set of bytes that subsumes the bytes in the addressed word. SC.W conditionally writes a word in rs2 to the address in rs1: the SC.W succeeds only if the reservation is still valid and the reservation set contains the bytes being written. If the SC.W succeeds, the instruction writes the word in rs2 to memory, and it writes zero to rd. If the SC.W fails, the instruction does not write to memory, and it writes a nonzero value to rd. Regardless of success or failure, executing an SC.W instruction invalidates any reservation held by this hart. LR.D and SC.D act analogously on doublewords and are only available on RV64. For RV64, <b>LR.W</b> and SC.W sign-extend the value placed in rd.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "Complex atomic memory operations on a single memory word or doubleword are performed with the load-reserved (LR) and store-conditional (SC) instructions. LR.W loads a word from the address in rs1, places the sign-extended value in rd, and registers a reservation set--a set of bytes that subsumes the bytes in the addressed word. SC.W conditionally writes a word in rs2 to the address in rs1: the SC.W succeeds only if the reservation is still valid and the reservation set contains the bytes being written. If the SC.W succeeds, the instruction writes the word in rs2 to memory, and it writes zero to rd. If the SC.W fails, the instruction does not write to memory, and it writes a nonzero value to rd. Regardless of success or failure, executing an SC.W instruction invalidates any reservation held by this hart. LR.D and SC.D act analogously on doublewords and are only available on RV64. For RV64, LR.W and SC.W sign-extend the value placed in rd.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:lrsc"
            };

        case "LUI":
            return {
                "html": "<div><span class=\"opcode\"><b>LUI</b> rd, imm20</span><br><div><b>LUI</b> (load upper immediate) is used to build 32-bit constants and uses the U-type format. <b>LUI</b> places the 32-bit U-immediate value into the destination register rd, filling in the lowest 12 bits with zeros.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "LUI (load upper immediate) is used to build 32-bit constants and uses the U-type format. LUI places the 32-bit U-immediate value into the destination register rd, filling in the lowest 12 bits with zeros.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-immediate-instructions"
            };

        case "LW":
            return {
                "html": "<div><span class=\"opcode\"><b>LW</b> rd, rs1, imm12</span><br><div>The <b>LW</b> instruction loads a 32-bit value from memory into rd. LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd. LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd. LB and LBU are defined analogously for 8-bit values. The SW, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "The LW instruction loads a 32-bit value from memory into rd. LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd. LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd. LB and LBU are defined analogously for 8-bit values. The SW, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#sec:rv32:ldst"
            };

        case "LWU":
            return {
                "html": "<div><span class=\"opcode\"><b>LWU</b> rd, rs1, imm12</span><br><div>The LW instruction loads a 32-bit value from memory and sign-extends this to 64 bits before storing it in register rd for RV64I. The <b>LWU</b> instruction, on the other hand, zero-extends the 32-bit value from memory for RV64I. LH and LHU are defined analogously for 16-bit values, as are LB and LBU for 8-bit values. The SD, SW, SH, and SB instructions store 64-bit, 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory respectively.</div><br><div><b>ISA</b>: rv64</div></div>",
                "tooltip": "The LW instruction loads a 32-bit value from memory and sign-extends this to 64 bits before storing it in register rd for RV64I. The LWU instruction, on the other hand, zero-extends the 32-bit value from memory for RV64I. LH and LHU are defined analogously for 16-bit values, as are LB and LBU for 8-bit values. The SD, SW, SH, and SB instructions store 64-bit, 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory respectively.\n\n\n\n(ISA: rv64)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv64.html#load-and-store-instructions"
            };

        case "MAX":
            return {
                "html": "<div><span class=\"opcode\"><b>MAX</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "MAXU":
            return {
                "html": "<div><span class=\"opcode\"><b>MAXU</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "MIN":
            return {
                "html": "<div><span class=\"opcode\"><b>MIN</b> rd, rs1, rs2</span><br><div><b>MIN</b></div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "MIN\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_register_grouping_vlmul20"
            };

        case "MINU":
            return {
                "html": "<div><span class=\"opcode\"><b>MINU</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "MRET":
            return {
                "html": "<div><span class=\"opcode\"><b>MRET</b> </span><br><div>An <b>MRET</b> or SRET instruction is used to return from a trap in M-mode or S-mode respectively. When executing an xRET instruction, supposing xPP holds the value y, xIE is set to xPIE; the privilege mode is changed to y; xPIE is set to 1; and xPP is set to the least-privileged supported mode (U if U-mode is implemented, else M). If xPP M, xRET also sets MPRV=0.</div><br><div><b>ISA</b>: machine</div></div>",
                "tooltip": "An MRET or SRET instruction is used to return from a trap in M-mode or S-mode respectively. When executing an xRET instruction, supposing xPP holds the value y, xIE is set to xPIE; the privilege mode is changed to y; xPIE is set to 1; and xPP is set to the least-privileged supported mode (U if U-mode is implemented, else M). If xPP M, xRET also sets MPRV=0.\n\n\n\n(ISA: machine)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/machine.html#privstack"
            };

        case "MUL":
            return {
                "html": "<div><span class=\"opcode\"><b>MUL</b> rd, rs1, rs2</span><br><div><b>MUL</b> performs an XLEN-bit\u00d7XLEN-bit multiplication of rs1 by rs2 and places the lower XLEN bits in the destination register. MULH, MULHU, and MULHSU perform the same multiplication but return the upper XLEN bits of the full 2\u00d7XLEN-bit product, for signed\u00d7signed, unsigned\u00d7unsigned, and signedrs1\u00d7unsignedrs2 multiplication, respectively. If both the high and low bits of the same product are required, then the recommended code sequence is: MULH[[S]U] rdh, rs1, rs2; <b>MUL</b> rdl, rs1, rs2 (source register specifiers must be in same order and rdh cannot be the same as rs1 or rs2). Microarchitectures can then fuse these into a single multiply operation instead of performing two separate multiplies.<br>In RV64, <b>MUL</b> can be used to obtain the upper 32 bits of the 64-bit product, but signed arguments must be proper 32-bit signed values, whereas unsigned arguments must have their upper 32 bits clear. If the arguments are not known to be sign- or zero-extended, an alternative is to shift both arguments left by 32 bits, then use MULH[[S]U].</div><br><div><b>ISA</b>: m</div></div>",
                "tooltip": "MUL performs an XLEN-bit\u00d7XLEN-bit multiplication of rs1 by rs2 and places the lower XLEN bits in the destination register. MULH, MULHU, and MULHSU perform the same multiplication but return the upper XLEN bits of the full 2\u00d7XLEN-bit product, for signed\u00d7signed, unsigned\u00d7unsigned, and signedrs1\u00d7unsignedrs2 multiplication, respectively. If both the high and low bits of the same product are required, then the recommended code sequence is: MULH[[S]U] rdh, rs1, rs2; MUL rdl, rs1, rs2 (source register specifiers must be in same order and rdh cannot be the same as rs1 or rs2). Microarchitectures can then fuse these into a single multiply operation instead of performing two separate multiplies.\nIn RV64, MUL can be used to obtain the upper 32 bits of the 64-bit product, but signed arguments must be proper 32-bit signed values, whereas unsigned arguments must have their upper 32 bits clear. If the arguments are not known to be sign- or zero-extended, an alternative is to shift both arguments left by 32 bits, then use MULH[[S]U].\n\n\n\n(ISA: m)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/m.html#multiplication-operations"
            };

        case "MULH":
            return {
                "html": "<div><span class=\"opcode\"><b>MULH</b> rd, rs1, rs2</span><br><div>MUL performs an XLEN-bit\u00d7XLEN-bit multiplication of rs1 by rs2 and places the lower XLEN bits in the destination register. <b>MULH</b>, MULHU, and MULHSU perform the same multiplication but return the upper XLEN bits of the full 2\u00d7XLEN-bit product, for signed\u00d7signed, unsigned\u00d7unsigned, and signedrs1\u00d7unsignedrs2 multiplication, respectively. If both the high and low bits of the same product are required, then the recommended code sequence is: <b>MULH</b>[[S]U] rdh, rs1, rs2; MUL rdl, rs1, rs2 (source register specifiers must be in same order and rdh cannot be the same as rs1 or rs2). Microarchitectures can then fuse these into a single multiply operation instead of performing two separate multiplies.<br>In RV64, MUL can be used to obtain the upper 32 bits of the 64-bit product, but signed arguments must be proper 32-bit signed values, whereas unsigned arguments must have their upper 32 bits clear. If the arguments are not known to be sign- or zero-extended, an alternative is to shift both arguments left by 32 bits, then use <b>MULH</b>[[S]U].</div><br><div><b>ISA</b>: m</div></div>",
                "tooltip": "MUL performs an XLEN-bit\u00d7XLEN-bit multiplication of rs1 by rs2 and places the lower XLEN bits in the destination register. MULH, MULHU, and MULHSU perform the same multiplication but return the upper XLEN bits of the full 2\u00d7XLEN-bit product, for signed\u00d7signed, unsigned\u00d7unsigned, and signedrs1\u00d7unsignedrs2 multiplication, respectively. If both the high and low bits of the same product are required, then the recommended code sequence is: MULH[[S]U] rdh, rs1, rs2; MUL rdl, rs1, rs2 (source register specifiers must be in same order and rdh cannot be the same as rs1 or rs2). Microarchitectures can then fuse these into a single multiply operation instead of performing two separate multiplies.\nIn RV64, MUL can be used to obtain the upper 32 bits of the 64-bit product, but signed arguments must be proper 32-bit signed values, whereas unsigned arguments must have their upper 32 bits clear. If the arguments are not known to be sign- or zero-extended, an alternative is to shift both arguments left by 32 bits, then use MULH[[S]U].\n\n\n\n(ISA: m)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/m.html#multiplication-operations"
            };

        case "MULHSU":
            return {
                "html": "<div><span class=\"opcode\"><b>MULHSU</b> rd, rs1, rs2</span><br><div>MUL performs an XLEN-bit\u00d7XLEN-bit multiplication of rs1 by rs2 and places the lower XLEN bits in the destination register. MULH, MULHU, and <b>MULHSU</b> perform the same multiplication but return the upper XLEN bits of the full 2\u00d7XLEN-bit product, for signed\u00d7signed, unsigned\u00d7unsigned, and signedrs1\u00d7unsignedrs2 multiplication, respectively. If both the high and low bits of the same product are required, then the recommended code sequence is: MULH[[S]U] rdh, rs1, rs2; MUL rdl, rs1, rs2 (source register specifiers must be in same order and rdh cannot be the same as rs1 or rs2). Microarchitectures can then fuse these into a single multiply operation instead of performing two separate multiplies.<br><b>MULHSU</b> is used in multi-word signed multiplication to multiply the most-significant word of the multiplicand (which contains the sign bit) with the less-significant words of the multiplier (which are unsigned).</div><br><div><b>ISA</b>: m</div></div>",
                "tooltip": "MUL performs an XLEN-bit\u00d7XLEN-bit multiplication of rs1 by rs2 and places the lower XLEN bits in the destination register. MULH, MULHU, and MULHSU perform the same multiplication but return the upper XLEN bits of the full 2\u00d7XLEN-bit product, for signed\u00d7signed, unsigned\u00d7unsigned, and signedrs1\u00d7unsignedrs2 multiplication, respectively. If both the high and low bits of the same product are required, then the recommended code sequence is: MULH[[S]U] rdh, rs1, rs2; MUL rdl, rs1, rs2 (source register specifiers must be in same order and rdh cannot be the same as rs1 or rs2). Microarchitectures can then fuse these into a single multiply operation instead of performing two separate multiplies.\nMULHSU is used in multi-word signed multiplication to multiply the most-significant word of the multiplicand (which contains the sign bit) with the less-significant words of the multiplier (which are unsigned).\n\n\n\n(ISA: m)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/m.html#multiplication-operations"
            };

        case "MULHU":
            return {
                "html": "<div><span class=\"opcode\"><b>MULHU</b> rd, rs1, rs2</span><br><div>MUL performs an XLEN-bit\u00d7XLEN-bit multiplication of rs1 by rs2 and places the lower XLEN bits in the destination register. MULH, <b>MULHU</b>, and MULHSU perform the same multiplication but return the upper XLEN bits of the full 2\u00d7XLEN-bit product, for signed\u00d7signed, unsigned\u00d7unsigned, and signedrs1\u00d7unsignedrs2 multiplication, respectively. If both the high and low bits of the same product are required, then the recommended code sequence is: MULH[[S]U] rdh, rs1, rs2; MUL rdl, rs1, rs2 (source register specifiers must be in same order and rdh cannot be the same as rs1 or rs2). Microarchitectures can then fuse these into a single multiply operation instead of performing two separate multiplies.</div><br><div><b>ISA</b>: m</div></div>",
                "tooltip": "MUL performs an XLEN-bit\u00d7XLEN-bit multiplication of rs1 by rs2 and places the lower XLEN bits in the destination register. MULH, MULHU, and MULHSU perform the same multiplication but return the upper XLEN bits of the full 2\u00d7XLEN-bit product, for signed\u00d7signed, unsigned\u00d7unsigned, and signedrs1\u00d7unsignedrs2 multiplication, respectively. If both the high and low bits of the same product are required, then the recommended code sequence is: MULH[[S]U] rdh, rs1, rs2; MUL rdl, rs1, rs2 (source register specifiers must be in same order and rdh cannot be the same as rs1 or rs2). Microarchitectures can then fuse these into a single multiply operation instead of performing two separate multiplies.\n\n\n\n(ISA: m)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/m.html#multiplication-operations"
            };

        case "MULW":
            return {
                "html": "<div><span class=\"opcode\"><b>MULW</b> rd, rs1, rs2</span><br><div><b>MULW</b> is an RV64 instruction that multiplies the lower 32 bits of the source registers, placing the sign-extension of the lower 32 bits of the result into the destination register.</div><br><div><b>ISA</b>: m</div></div>",
                "tooltip": "MULW is an RV64 instruction that multiplies the lower 32 bits of the source registers, placing the sign-extension of the lower 32 bits of the result into the destination register.\n\n\n\n(ISA: m)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/m.html#multiplication-operations"
            };

        case "MV":
            return {
                "html": "<div><span class=\"opcode\"><b>MV</b> rd, rs</span><br><div>ADDI adds the sign-extended 12-bit immediate to register rs1. Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result. ADDI rd, rs1, 0 is used to implement the <b>MV</b> rd, rs1 assembler pseudoinstruction.</div><br><div><b>Equivalent ASM:</b><pre>addi rd, rs, 0</pre></div><br><div><b>ISA</b>: rv32(pseudo)</div></div>",
                "tooltip": "ADDI adds the sign-extended 12-bit immediate to register rs1. Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result. ADDI rd, rs1, 0 is used to implement the MV rd, rs1 assembler pseudoinstruction.\n\nEquivalent ASM:\n\naddi rd, rs, 0\n\n(ISA: rv32(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-immediate-instructions"
            };

        case "NEG":
            return {
                "html": "<div><span class=\"opcode\"><b>NEG</b> rd, rs</span><br><div>The sign-injection instructions provide floating-point MV, ABS, and <b>NEG</b>, as well as supporting a few other operations, including the IEEE copySign operation and sign manipulation in transcendental math function libraries. Although MV, ABS, and <b>NEG</b> only need a single register operand, whereas FSGNJ instructions need two, it is unlikely most microarchitectures would add optimizations to benefit from the reduced number of register reads for these relatively infrequent instructions. Even in this case, a microarchitecture can simply detect when both source registers are the same for FSGNJ instructions and only read a single copy.</div><br><div><b>Equivalent ASM:</b><pre>sub rd, x0, rs</pre></div><br><div><b>ISA</b>: f(pseudo)</div></div>",
                "tooltip": "The sign-injection instructions provide floating-point MV, ABS, and NEG, as well as supporting a few other operations, including the IEEE copySign operation and sign manipulation in transcendental math function libraries. Although MV, ABS, and NEG only need a single register operand, whereas FSGNJ instructions need two, it is unlikely most microarchitectures would add optimizations to benefit from the reduced number of register reads for these relatively infrequent instructions. Even in this case, a microarchitecture can simply detect when both source registers are the same for FSGNJ instructions and only read a single copy.\n\nEquivalent ASM:\n\nsub rd, x0, rs\n\n(ISA: f(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/f.html#single-precision-floating-point-conversion-and-move-instructions"
            };

        case "NEGW":
            return {
                "html": "<div><span class=\"opcode\"><b>NEGW</b> rd, rs</span><br><div><b>Equivalent ASM:</b><pre>subw rd, x0, rs</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nsubw rd, x0, rs\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "NOP":
            return {
                "html": "<div><span class=\"opcode\"><b>NOP</b> </span><br><div>The <b>NOP</b> instruction does not change any architecturally visible state, except for advancing the pc and incrementing any applicable performance counters. <b>NOP</b> is encoded as ADDI x0, x0, 0.<br>NOPs can be used to align code segments to microarchitecturally significant address boundaries, or to leave space for inline code modifications. Although there are many possible ways to encode a <b>NOP</b>, we define a canonical <b>NOP</b> encoding to allow microarchitectural optimizations as well as for more readable disassembly output. The other <b>NOP</b> encodings are made available for HINT instructions (Section 1.9 ).<br>ADDI was chosen for the <b>NOP</b> encoding as this is most likely to take fewest resources to execute across a range of systems (if not optimized away in decode). In particular, the instruction only reads one register. Also, an ADDI functional unit is more likely to be available in a superscalar design as adds are the most common operation. In particular, address-generation functional units can execute ADDI using the same hardware needed for base+offset address calculations, while register-register ADD or logical/shift operations require additional hardware.</div><br><div><b>Equivalent ASM:</b><pre>addi x0, x0, 0</pre></div><br><div><b>ISA</b>: rv32(pseudo)</div></div>",
                "tooltip": "The NOP instruction does not change any architecturally visible state, except for advancing the pc and incrementing any applicable performance counters. NOP is encoded as ADDI x0, x0, 0.\nNOPs can be used to align code segments to microarchitecturally significant address boundaries, or to leave space for inline code modifications. Although there are many possible ways to encode a NOP, we define a canonical NOP encoding to allow microarchitectural optimizations as well as for more readable disassembly output. The other NOP encodings are made available for HINT instructions (Section 1.9 ).\nADDI was chosen for the NOP encoding as this is most likely to take fewest resources to execute across a range of systems (if not optimized away in decode). In particular, the instruction only reads one register. Also, an ADDI functional unit is more likely to be available in a superscalar design as adds are the most common operation. In particular, address-generation functional units can execute ADDI using the same hardware needed for base+offset address calculations, while register-register ADD or logical/shift operations require additional hardware.\n\nEquivalent ASM:\n\naddi x0, x0, 0\n\n(ISA: rv32(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#nop-instruction"
            };

        case "NOT":
            return {
                "html": "<div><span class=\"opcode\"><b>NOT</b> rd, rs</span><br><div>ANDI, ORI, XORI are logical operations that perform bitwise AND, OR, and XOR on register rs1 and the sign-extended 12-bit immediate and place the result in rd. Note, XORI rd, rs1, -1 performs a bitwise logical inversion of register rs1 (assembler pseudoinstruction <b>NOT</b> rd, rs).</div><br><div><b>Equivalent ASM:</b><pre>xori rd, rs, -1</pre></div><br><div><b>ISA</b>: rv32(pseudo)</div></div>",
                "tooltip": "ANDI, ORI, XORI are logical operations that perform bitwise AND, OR, and XOR on register rs1 and the sign-extended 12-bit immediate and place the result in rd. Note, XORI rd, rs1, -1 performs a bitwise logical inversion of register rs1 (assembler pseudoinstruction NOT rd, rs).\n\nEquivalent ASM:\n\nxori rd, rs, -1\n\n(ISA: rv32(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-immediate-instructions"
            };

        case "OR":
            return {
                "html": "<div><span class=\"opcode\"><b>OR</b> rd, rs1, rs2</span><br><div>ADD performs the addition of rs1 and rs2. SUB performs the subtraction of rs2 from rs1. Overflows are ignored and the low XLEN bits of results are written to the destination rd. SLT and SLTU perform signed and unsigned compares respectively, writing 1 to rd if rs1 < rs2, 0 otherwise. Note, SLTU rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs). AND, <b>OR</b>, and XOR perform bitwise logical operations.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "ADD performs the addition of rs1 and rs2. SUB performs the subtraction of rs2 from rs1. Overflows are ignored and the low XLEN bits of results are written to the destination rd. SLT and SLTU perform signed and unsigned compares respectively, writing 1 to rd if rs1 < rs2, 0 otherwise. Note, SLTU rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs). AND, OR, and XOR perform bitwise logical operations.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-register-operations"
            };

        case "ORC.B":
            return {
                "html": "<div><span class=\"opcode\"><b>ORC.B</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "ORI":
            return {
                "html": "<div><span class=\"opcode\"><b>ORI</b> rd, rs1, imm12</span><br><div>ANDI, <b>ORI</b>, XORI are logical operations that perform bitwise AND, OR, and XOR on register rs1 and the sign-extended 12-bit immediate and place the result in rd. Note, XORI rd, rs1, -1 performs a bitwise logical inversion of register rs1 (assembler pseudoinstruction NOT rd, rs).</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "ANDI, ORI, XORI are logical operations that perform bitwise AND, OR, and XOR on register rs1 and the sign-extended 12-bit immediate and place the result in rd. Note, XORI rd, rs1, -1 performs a bitwise logical inversion of register rs1 (assembler pseudoinstruction NOT rd, rs).\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-immediate-instructions"
            };

        case "ORN":
            return {
                "html": "<div><span class=\"opcode\"><b>ORN</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "PAUSE":
            return {
                "html": "<div><span class=\"opcode\"><b>PAUSE</b> </span><br><div>The <b>PAUSE</b> instruction is a HINT that indicates the current hart's rate of instruction retirement should be temporarily reduced or paused. The duration of its effect must be bounded and may be zero. No architectural state is changed.<br>Software can use the <b>PAUSE</b> instruction to reduce energy consumption while executing spin-wait code sequences. Multithreaded cores might temporarily relinquish execution resources to other harts when <b>PAUSE</b> is executed. It is recommended that a <b>PAUSE</b> instruction generally be included in the code sequence for a spin-wait loop.<br>A future extension might add primitives similar to the x86 MONITOR/MWAIT instructions, which provide a more efficient mechanism to wait on writes to a specific memory location. However, these instructions would not supplant <b>PAUSE</b>. <b>PAUSE</b> is more appropriate when polling for non-memory events, when polling for multiple events, or when software does not know precisely what events it is polling for.<br>The duration of a <b>PAUSE</b> instruction's effect may vary significantly within and among implementations. In typical implementations this duration should be much less than the time to perform a context switch, probably more on the rough order of an on-chip cache miss latency or a cacheless access to main memory.<br>A series of <b>PAUSE</b> instructions can be used to create a cumulative delay loosely proportional to the number of <b>PAUSE</b> instructions. In spin-wait loops in portable code, however, only one <b>PAUSE</b> instruction should be used before re-evaluating loop conditions, else the hart might stall longer than optimal on some implementations, degrading system performance.<br><b>PAUSE</b> is encoded as a FENCE instruction with pred=W, succ=0, fm=0, rd=x0, and rs1=x0.<br><b>PAUSE</b> is encoded as a hint within the FENCE opcode because some implementations are expected to deliberately stall the <b>PAUSE</b> instruction until outstanding memory transactions have completed. Because the successor set is null, however, <b>PAUSE</b> does not mandate any particular memory ordering--hence, it truly is a HINT.<br>Like other FENCE instructions, <b>PAUSE</b> cannot be used within LR/SC sequences without voiding the forward-progress guarantee.<br>The choice of a predecessor set of W is arbitrary, since the successor set is null. Other HINTs similar to <b>PAUSE</b> might be encoded with other predecessor sets.</div><br><div><b>ISA</b>: zihintpause</div></div>",
                "tooltip": "The PAUSE instruction is a HINT that indicates the current hart's rate of instruction retirement should be temporarily reduced or paused. The duration of its effect must be bounded and may be zero. No architectural state is changed.\nSoftware can use the PAUSE instruction to reduce energy consumption while executing spin-wait code sequences. Multithreaded cores might temporarily relinquish execution resources to other harts when PAUSE is executed. It is recommended that a PAUSE instruction generally be included in the code sequence for a spin-wait loop.\nA future extension might add primitives similar to the x86 MONITOR/MWAIT instructions, which provide a more efficient mechanism to wait on writes to a specific memory location. However, these instructions would not supplant PAUSE. PAUSE is more appropriate when polling for non-memory events, when polling for multiple events, or when software does not know precisely what events it is polling for.\nThe duration of a PAUSE instruction's effect may vary significantly within and among implementations. In typical implementations this duration should be much less than the time to perform a context switch, probably more on the rough order of an on-chip cache miss latency or a cacheless access to main memory.\nA series of PAUSE instructions can be used to create a cumulative delay loosely proportional to the number of PAUSE instructions. In spin-wait loops in portable code, however, only one PAUSE instruction should be used before re-evaluating loop conditions, else the hart might stall longer than optimal on some implementations, degrading system performance.\nPAUSE is encoded as a FENCE instruction with pred=W, succ=0, fm=0, rd=x0, and rs1=x0.\nPAUSE is encoded as a hint within the FENCE opcode because some implementations are expected to deliberately stall the PAUSE instruction until outstanding memory transactions have completed. Because the successor set is null, however, PAUSE does not mandate any particular memory ordering--hence, it truly is a HINT.\nLike other FENCE instructions, PAUSE cannot be used within LR/SC sequences without voiding the forward-progress guarantee.\nThe choice of a predecessor set of W is arbitrary, since the successor set is null. Other HINTs similar to PAUSE might be encoded with other predecessor sets.\n\n\n\n(ISA: zihintpause)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/zihintpause.html#chap:zihintpause"
            };

        case "PREFETCH.I":
            return {
                "html": "<div><span class=\"opcode\"><b>PREFETCH.I</b> rs1, imm12hi</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "PREFETCH.R":
            return {
                "html": "<div><span class=\"opcode\"><b>PREFETCH.R</b> rs1, imm12hi</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "PREFETCH.W":
            return {
                "html": "<div><span class=\"opcode\"><b>PREFETCH.W</b> rs1, imm12hi</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "RDCYCLE":
            return {
                "html": "<div><span class=\"opcode\"><b>RDCYCLE</b> rd</span><br><div>RV32I provides a number of 64-bit read-only user-level counters, which are mapped into the 12-bit CSR address space and accessed in 32-bit pieces using CSRRS instructions. In RV64I, the CSR instructions can manipulate 64-bit CSRs. In particular, the <b>RDCYCLE</b>, RDTIME, and RDINSTRET pseudoinstructions read the full 64 bits of the cycle, time, and instret counters. Hence, the RDCYCLEH, RDTIMEH, and RDINSTRETH instructions are RV32I-only.<br>The <b>RDCYCLE</b> pseudoinstruction reads the low XLEN bits of the cycle CSR which holds a count of the number of clock cycles executed by the processor core on which the hart is running from an arbitrary start time in the past. RDCYCLEH is an RV32I-only instruction that reads bits 63-32 of the same cycle counter. The underlying 64-bit counter should never overflow in practice. The rate at which the cycle counter advances will depend on the implementation and operating environment. The execution environment should provide a means to determine the current rate (cycles/second) at which the cycle counter is incrementing.<br><b>RDCYCLE</b> is intended to return the number of cycles executed by the processor core, not the hart. Precisely defining what is a \"core\" is difficult given some implementation choices (e.g., AMD Bulldozer). Precisely defining what is a \"clock cycle\" is also difficult given the range of implementations (including software emulations), but the intent is that <b>RDCYCLE</b> is used for performance monitoring along with the other performance counters. In particular, where there is one hart/core, one would expect cycle-count/instructions-retired to measure CPI for a hart.<br>Even though there is no precise definition that works for all platforms, this is still a useful facility for most platforms, and an imprecise, common, \"usually correct\" standard here is better than no standard. The intent of <b>RDCYCLE</b> was primarily performance monitoring/tuning, and the specification was written with that goal in mind.<br>On some simple platforms, cycle count might represent a valid implementation of RDTIME, in which case RDTIME and <b>RDCYCLE</b> may return the same result.</div><br><div><b>ISA</b>: counters</div></div>",
                "tooltip": "RV32I provides a number of 64-bit read-only user-level counters, which are mapped into the 12-bit CSR address space and accessed in 32-bit pieces using CSRRS instructions. In RV64I, the CSR instructions can manipulate 64-bit CSRs. In particular, the RDCYCLE, RDTIME, and RDINSTRET pseudoinstructions read the full 64 bits of the cycle, time, and instret counters. Hence, the RDCYCLEH, RDTIMEH, and RDINSTRETH instructions are RV32I-only.\nThe RDCYCLE pseudoinstruction reads the low XLEN bits of the cycle CSR which holds a count of the number of clock cycles executed by the processor core on which the hart is running from an arbitrary start time in the past. RDCYCLEH is an RV32I-only instruction that reads bits 63-32 of the same cycle counter. The underlying 64-bit counter should never overflow in practice. The rate at which the cycle counter advances will depend on the implementation and operating environment. The execution environment should provide a means to determine the current rate (cycles/second) at which the cycle counter is incrementing.\nRDCYCLE is intended to return the number of cycles executed by the processor core, not the hart. Precisely defining what is a \"core\" is difficult given some implementation choices (e.g., AMD Bulldozer). Precisely defining what is a \"clock cycle\" is also difficult given the range of implementations (including software emulations), but the intent is that RDCYCLE is used for performance monitoring along with the other performance counters. In particular, where there is one hart/core, one would expect cycle-count/instructions-retired to measure CPI for a hart.\nEven though there is no precise definition that works for all platforms, this is still a useful facility for most platforms, and an imprecise, common, \"usually correct\" standard here is better than no standard. The intent of RDCYCLE was primarily performance monitoring/tuning, and the specification was written with that goal in mind.\nOn some simple platforms, cycle count might represent a valid implementation of RDTIME, in which case RDTIME and RDCYCLE may return the same result.\n\n\n\n(ISA: counters)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/counters.html#zicntr-standard-extension-for-base-counters-and-timers"
            };

        case "RDCYCLEH":
            return {
                "html": "<div><span class=\"opcode\"><b>RDCYCLEH</b> rd</span><br><div>RV32I provides a number of 64-bit read-only user-level counters, which are mapped into the 12-bit CSR address space and accessed in 32-bit pieces using CSRRS instructions. In RV64I, the CSR instructions can manipulate 64-bit CSRs. In particular, the RDCYCLE, RDTIME, and RDINSTRET pseudoinstructions read the full 64 bits of the cycle, time, and instret counters. Hence, the <b>RDCYCLEH</b>, RDTIMEH, and RDINSTRETH instructions are RV32I-only.<br>The RDCYCLE pseudoinstruction reads the low XLEN bits of the cycle CSR which holds a count of the number of clock cycles executed by the processor core on which the hart is running from an arbitrary start time in the past. <b>RDCYCLEH</b> is an RV32I-only instruction that reads bits 63-32 of the same cycle counter. The underlying 64-bit counter should never overflow in practice. The rate at which the cycle counter advances will depend on the implementation and operating environment. The execution environment should provide a means to determine the current rate (cycles/second) at which the cycle counter is incrementing.</div><br><div><b>ISA</b>: counters</div></div>",
                "tooltip": "RV32I provides a number of 64-bit read-only user-level counters, which are mapped into the 12-bit CSR address space and accessed in 32-bit pieces using CSRRS instructions. In RV64I, the CSR instructions can manipulate 64-bit CSRs. In particular, the RDCYCLE, RDTIME, and RDINSTRET pseudoinstructions read the full 64 bits of the cycle, time, and instret counters. Hence, the RDCYCLEH, RDTIMEH, and RDINSTRETH instructions are RV32I-only.\nThe RDCYCLE pseudoinstruction reads the low XLEN bits of the cycle CSR which holds a count of the number of clock cycles executed by the processor core on which the hart is running from an arbitrary start time in the past. RDCYCLEH is an RV32I-only instruction that reads bits 63-32 of the same cycle counter. The underlying 64-bit counter should never overflow in practice. The rate at which the cycle counter advances will depend on the implementation and operating environment. The execution environment should provide a means to determine the current rate (cycles/second) at which the cycle counter is incrementing.\n\n\n\n(ISA: counters)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/counters.html#zicntr-standard-extension-for-base-counters-and-timers"
            };

        case "RDINSTRET":
            return {
                "html": "<div><span class=\"opcode\"><b>RDINSTRET</b> rd</span><br><div>RV32I provides a number of 64-bit read-only user-level counters, which are mapped into the 12-bit CSR address space and accessed in 32-bit pieces using CSRRS instructions. In RV64I, the CSR instructions can manipulate 64-bit CSRs. In particular, the RDCYCLE, RDTIME, and <b>RDINSTRET</b> pseudoinstructions read the full 64 bits of the cycle, time, and instret counters. Hence, the RDCYCLEH, RDTIMEH, and RDINSTRETH instructions are RV32I-only.<br>The <b>RDINSTRET</b> pseudoinstruction reads the low XLEN bits of the instret CSR, which counts the number of instructions retired by this hart from some arbitrary start point in the past. RDINSTRETH is an RV32I-only instruction that reads bits 63-32 of the same instruction counter. The underlying 64-bit counter should never overflow in practice.</div><br><div><b>ISA</b>: counters</div></div>",
                "tooltip": "RV32I provides a number of 64-bit read-only user-level counters, which are mapped into the 12-bit CSR address space and accessed in 32-bit pieces using CSRRS instructions. In RV64I, the CSR instructions can manipulate 64-bit CSRs. In particular, the RDCYCLE, RDTIME, and RDINSTRET pseudoinstructions read the full 64 bits of the cycle, time, and instret counters. Hence, the RDCYCLEH, RDTIMEH, and RDINSTRETH instructions are RV32I-only.\nThe RDINSTRET pseudoinstruction reads the low XLEN bits of the instret CSR, which counts the number of instructions retired by this hart from some arbitrary start point in the past. RDINSTRETH is an RV32I-only instruction that reads bits 63-32 of the same instruction counter. The underlying 64-bit counter should never overflow in practice.\n\n\n\n(ISA: counters)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/counters.html#zicntr-standard-extension-for-base-counters-and-timers"
            };

        case "RDINSTRETH":
            return {
                "html": "<div><span class=\"opcode\"><b>RDINSTRETH</b> rd</span><br><div>RV32I provides a number of 64-bit read-only user-level counters, which are mapped into the 12-bit CSR address space and accessed in 32-bit pieces using CSRRS instructions. In RV64I, the CSR instructions can manipulate 64-bit CSRs. In particular, the RDCYCLE, RDTIME, and RDINSTRET pseudoinstructions read the full 64 bits of the cycle, time, and instret counters. Hence, the RDCYCLEH, RDTIMEH, and <b>RDINSTRETH</b> instructions are RV32I-only.<br>The RDINSTRET pseudoinstruction reads the low XLEN bits of the instret CSR, which counts the number of instructions retired by this hart from some arbitrary start point in the past. <b>RDINSTRETH</b> is an RV32I-only instruction that reads bits 63-32 of the same instruction counter. The underlying 64-bit counter should never overflow in practice.</div><br><div><b>ISA</b>: counters</div></div>",
                "tooltip": "RV32I provides a number of 64-bit read-only user-level counters, which are mapped into the 12-bit CSR address space and accessed in 32-bit pieces using CSRRS instructions. In RV64I, the CSR instructions can manipulate 64-bit CSRs. In particular, the RDCYCLE, RDTIME, and RDINSTRET pseudoinstructions read the full 64 bits of the cycle, time, and instret counters. Hence, the RDCYCLEH, RDTIMEH, and RDINSTRETH instructions are RV32I-only.\nThe RDINSTRET pseudoinstruction reads the low XLEN bits of the instret CSR, which counts the number of instructions retired by this hart from some arbitrary start point in the past. RDINSTRETH is an RV32I-only instruction that reads bits 63-32 of the same instruction counter. The underlying 64-bit counter should never overflow in practice.\n\n\n\n(ISA: counters)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/counters.html#zicntr-standard-extension-for-base-counters-and-timers"
            };

        case "RDTIME":
            return {
                "html": "<div><span class=\"opcode\"><b>RDTIME</b> rd</span><br><div>RV32I provides a number of 64-bit read-only user-level counters, which are mapped into the 12-bit CSR address space and accessed in 32-bit pieces using CSRRS instructions. In RV64I, the CSR instructions can manipulate 64-bit CSRs. In particular, the RDCYCLE, <b>RDTIME</b>, and RDINSTRET pseudoinstructions read the full 64 bits of the cycle, time, and instret counters. Hence, the RDCYCLEH, RDTIMEH, and RDINSTRETH instructions are RV32I-only.<br>The <b>RDTIME</b> pseudoinstruction reads the low XLEN bits of the time CSR, which counts wall-clock real time that has passed from an arbitrary start time in the past. RDTIMEH is an RV32I-only instruction that reads bits 63-32 of the same real-time counter. The underlying 64-bit counter increments by one with each tick of the real-time clock, and, for realistic real-time clock frequencies, should never overflow in practice. The execution environment should provide a means of determining the period of a counter tick (seconds/tick). The period must be constant. The real-time clocks of all harts in a single user application should be synchronized to within one tick of the real-time clock. The environment should provide a means to determine the accuracy of the clock (i.e., the maximum relative error between the nominal and actual real-time clock periods).<br>On some simple platforms, cycle count might represent a valid implementation of <b>RDTIME</b>, in which case <b>RDTIME</b> and RDCYCLE may return the same result.</div><br><div><b>ISA</b>: counters</div></div>",
                "tooltip": "RV32I provides a number of 64-bit read-only user-level counters, which are mapped into the 12-bit CSR address space and accessed in 32-bit pieces using CSRRS instructions. In RV64I, the CSR instructions can manipulate 64-bit CSRs. In particular, the RDCYCLE, RDTIME, and RDINSTRET pseudoinstructions read the full 64 bits of the cycle, time, and instret counters. Hence, the RDCYCLEH, RDTIMEH, and RDINSTRETH instructions are RV32I-only.\nThe RDTIME pseudoinstruction reads the low XLEN bits of the time CSR, which counts wall-clock real time that has passed from an arbitrary start time in the past. RDTIMEH is an RV32I-only instruction that reads bits 63-32 of the same real-time counter. The underlying 64-bit counter increments by one with each tick of the real-time clock, and, for realistic real-time clock frequencies, should never overflow in practice. The execution environment should provide a means of determining the period of a counter tick (seconds/tick). The period must be constant. The real-time clocks of all harts in a single user application should be synchronized to within one tick of the real-time clock. The environment should provide a means to determine the accuracy of the clock (i.e., the maximum relative error between the nominal and actual real-time clock periods).\nOn some simple platforms, cycle count might represent a valid implementation of RDTIME, in which case RDTIME and RDCYCLE may return the same result.\n\n\n\n(ISA: counters)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/counters.html#zicntr-standard-extension-for-base-counters-and-timers"
            };

        case "RDTIMEH":
            return {
                "html": "<div><span class=\"opcode\"><b>RDTIMEH</b> rd</span><br><div>RV32I provides a number of 64-bit read-only user-level counters, which are mapped into the 12-bit CSR address space and accessed in 32-bit pieces using CSRRS instructions. In RV64I, the CSR instructions can manipulate 64-bit CSRs. In particular, the RDCYCLE, RDTIME, and RDINSTRET pseudoinstructions read the full 64 bits of the cycle, time, and instret counters. Hence, the RDCYCLEH, <b>RDTIMEH</b>, and RDINSTRETH instructions are RV32I-only.<br>The RDTIME pseudoinstruction reads the low XLEN bits of the time CSR, which counts wall-clock real time that has passed from an arbitrary start time in the past. <b>RDTIMEH</b> is an RV32I-only instruction that reads bits 63-32 of the same real-time counter. The underlying 64-bit counter increments by one with each tick of the real-time clock, and, for realistic real-time clock frequencies, should never overflow in practice. The execution environment should provide a means of determining the period of a counter tick (seconds/tick). The period must be constant. The real-time clocks of all harts in a single user application should be synchronized to within one tick of the real-time clock. The environment should provide a means to determine the accuracy of the clock (i.e., the maximum relative error between the nominal and actual real-time clock periods).</div><br><div><b>ISA</b>: counters</div></div>",
                "tooltip": "RV32I provides a number of 64-bit read-only user-level counters, which are mapped into the 12-bit CSR address space and accessed in 32-bit pieces using CSRRS instructions. In RV64I, the CSR instructions can manipulate 64-bit CSRs. In particular, the RDCYCLE, RDTIME, and RDINSTRET pseudoinstructions read the full 64 bits of the cycle, time, and instret counters. Hence, the RDCYCLEH, RDTIMEH, and RDINSTRETH instructions are RV32I-only.\nThe RDTIME pseudoinstruction reads the low XLEN bits of the time CSR, which counts wall-clock real time that has passed from an arbitrary start time in the past. RDTIMEH is an RV32I-only instruction that reads bits 63-32 of the same real-time counter. The underlying 64-bit counter increments by one with each tick of the real-time clock, and, for realistic real-time clock frequencies, should never overflow in practice. The execution environment should provide a means of determining the period of a counter tick (seconds/tick). The period must be constant. The real-time clocks of all harts in a single user application should be synchronized to within one tick of the real-time clock. The environment should provide a means to determine the accuracy of the clock (i.e., the maximum relative error between the nominal and actual real-time clock periods).\n\n\n\n(ISA: counters)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/counters.html#zicntr-standard-extension-for-base-counters-and-timers"
            };

        case "REM":
            return {
                "html": "<div><span class=\"opcode\"><b>REM</b> rd, rs1, rs2</span><br><div>DIV and DIVU perform an XLEN bits by XLEN bits signed and unsigned integer division of rs1 by rs2, rounding towards zero. <b>REM</b> and REMU provide the remainder of the corresponding division operation. For <b>REM</b>, the sign of the result equals the sign of the dividend.<br>If both the quotient and remainder are required from the same division, the recommended code sequence is: DIV[U] rdq, rs1, rs2; <b>REM</b>[U] rdr, rs1, rs2 (rdq cannot be the same as rs1 or rs2). Microarchitectures can then fuse these into a single divide operation instead of performing two separate divides.<br><b>REM</b>[W]</div><br><div><b>ISA</b>: m</div></div>",
                "tooltip": "DIV and DIVU perform an XLEN bits by XLEN bits signed and unsigned integer division of rs1 by rs2, rounding towards zero. REM and REMU provide the remainder of the corresponding division operation. For REM, the sign of the result equals the sign of the dividend.\nIf both the quotient and remainder are required from the same division, the recommended code sequence is: DIV[U] rdq, rs1, rs2; REM[U] rdr, rs1, rs2 (rdq cannot be the same as rs1 or rs2). Microarchitectures can then fuse these into a single divide operation instead of performing two separate divides.\nREM[W]\n\n\n\n(ISA: m)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/m.html#division-operations"
            };

        case "REMU":
            return {
                "html": "<div><span class=\"opcode\"><b>REMU</b> rd, rs1, rs2</span><br><div>DIV and DIVU perform an XLEN bits by XLEN bits signed and unsigned integer division of rs1 by rs2, rounding towards zero. REM and <b>REMU</b> provide the remainder of the corresponding division operation. For REM, the sign of the result equals the sign of the dividend.<br><b>REMU</b>[W]</div><br><div><b>ISA</b>: m</div></div>",
                "tooltip": "DIV and DIVU perform an XLEN bits by XLEN bits signed and unsigned integer division of rs1 by rs2, rounding towards zero. REM and REMU provide the remainder of the corresponding division operation. For REM, the sign of the result equals the sign of the dividend.\nREMU[W]\n\n\n\n(ISA: m)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/m.html#division-operations"
            };

        case "REMUW":
            return {
                "html": "<div><span class=\"opcode\"><b>REMUW</b> rd, rs1, rs2</span><br><div>DIVW and DIVUW are RV64 instructions that divide the lower 32 bits of rs1 by the lower 32 bits of rs2, treating them as signed and unsigned integers respectively, placing the 32-bit quotient in rd, sign-extended to 64 bits. REMW and <b>REMUW</b> are RV64 instructions that provide the corresponding signed and unsigned remainder operations respectively. Both REMW and <b>REMUW</b> always sign-extend the 32-bit result to 64 bits, including on a divide by zero.</div><br><div><b>ISA</b>: m</div></div>",
                "tooltip": "DIVW and DIVUW are RV64 instructions that divide the lower 32 bits of rs1 by the lower 32 bits of rs2, treating them as signed and unsigned integers respectively, placing the 32-bit quotient in rd, sign-extended to 64 bits. REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned remainder operations respectively. Both REMW and REMUW always sign-extend the 32-bit result to 64 bits, including on a divide by zero.\n\n\n\n(ISA: m)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/m.html#division-operations"
            };

        case "REMW":
            return {
                "html": "<div><span class=\"opcode\"><b>REMW</b> rd, rs1, rs2</span><br><div>DIVW and DIVUW are RV64 instructions that divide the lower 32 bits of rs1 by the lower 32 bits of rs2, treating them as signed and unsigned integers respectively, placing the 32-bit quotient in rd, sign-extended to 64 bits. <b>REMW</b> and REMUW are RV64 instructions that provide the corresponding signed and unsigned remainder operations respectively. Both <b>REMW</b> and REMUW always sign-extend the 32-bit result to 64 bits, including on a divide by zero.</div><br><div><b>ISA</b>: m</div></div>",
                "tooltip": "DIVW and DIVUW are RV64 instructions that divide the lower 32 bits of rs1 by the lower 32 bits of rs2, treating them as signed and unsigned integers respectively, placing the 32-bit quotient in rd, sign-extended to 64 bits. REMW and REMUW are RV64 instructions that provide the corresponding signed and unsigned remainder operations respectively. Both REMW and REMUW always sign-extend the 32-bit result to 64 bits, including on a divide by zero.\n\n\n\n(ISA: m)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/m.html#division-operations"
            };

        case "RET":
            return {
                "html": "<div><span class=\"opcode\"><b>RET</b> </span><br><div><b>Equivalent ASM:</b><pre>jalr x0, x1, 0</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\njalr x0, x1, 0\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "REV8":
            return {
                "html": "<div><span class=\"opcode\"><b>REV8</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "ROL":
            return {
                "html": "<div><span class=\"opcode\"><b>ROL</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "ROLW":
            return {
                "html": "<div><span class=\"opcode\"><b>ROLW</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "ROR":
            return {
                "html": "<div><span class=\"opcode\"><b>ROR</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "RORI":
            return {
                "html": "<div><span class=\"opcode\"><b>RORI</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "RORIW":
            return {
                "html": "<div><span class=\"opcode\"><b>RORIW</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "RORW":
            return {
                "html": "<div><span class=\"opcode\"><b>RORW</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SB":
            return {
                "html": "<div><span class=\"opcode\"><b>SB</b> rs1, rs2, imm12</span><br><div>The LW instruction loads a 32-bit value from memory into rd. LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd. LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd. LB and LBU are defined analogously for 8-bit values. The SW, SH, and <b>SB</b> instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "The LW instruction loads a 32-bit value from memory into rd. LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd. LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd. LB and LBU are defined analogously for 8-bit values. The SW, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#sec:rv32:ldst"
            };

        case "SBREAK":
            return {
                "html": "<div><span class=\"opcode\"><b>SBREAK</b> </span><br><div>ECALL and EBREAK were previously named SCALL and <b>SBREAK</b>. The instructions have the same functionality and encoding, but were renamed to reflect that they can be used more generally than to call a supervisor-level operating system or debugger.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "ECALL and EBREAK were previously named SCALL and SBREAK. The instructions have the same functionality and encoding, but were renamed to reflect that they can be used more generally than to call a supervisor-level operating system or debugger.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#environment-call-and-breakpoints"
            };

        case "SC.D":
            return {
                "html": "<div><span class=\"opcode\"><b>SC.D</b> rd, rs1, rs2</span><br><div>Complex atomic memory operations on a single memory word or doubleword are performed with the load-reserved (LR) and store-conditional (SC) instructions. LR.W loads a word from the address in rs1, places the sign-extended value in rd, and registers a reservation set--a set of bytes that subsumes the bytes in the addressed word. SC.W conditionally writes a word in rs2 to the address in rs1: the SC.W succeeds only if the reservation is still valid and the reservation set contains the bytes being written. If the SC.W succeeds, the instruction writes the word in rs2 to memory, and it writes zero to rd. If the SC.W fails, the instruction does not write to memory, and it writes a nonzero value to rd. Regardless of success or failure, executing an SC.W instruction invalidates any reservation held by this hart. LR.D and <b>SC.D</b> act analogously on doublewords and are only available on RV64. For RV64, LR.W and SC.W sign-extend the value placed in rd.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "Complex atomic memory operations on a single memory word or doubleword are performed with the load-reserved (LR) and store-conditional (SC) instructions. LR.W loads a word from the address in rs1, places the sign-extended value in rd, and registers a reservation set--a set of bytes that subsumes the bytes in the addressed word. SC.W conditionally writes a word in rs2 to the address in rs1: the SC.W succeeds only if the reservation is still valid and the reservation set contains the bytes being written. If the SC.W succeeds, the instruction writes the word in rs2 to memory, and it writes zero to rd. If the SC.W fails, the instruction does not write to memory, and it writes a nonzero value to rd. Regardless of success or failure, executing an SC.W instruction invalidates any reservation held by this hart. LR.D and SC.D act analogously on doublewords and are only available on RV64. For RV64, LR.W and SC.W sign-extend the value placed in rd.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:lrsc"
            };

        case "SC.W":
            return {
                "html": "<div><span class=\"opcode\"><b>SC.W</b> rd, rs1, rs2</span><br><div>Complex atomic memory operations on a single memory word or doubleword are performed with the load-reserved (LR) and store-conditional (SC) instructions. LR.W loads a word from the address in rs1, places the sign-extended value in rd, and registers a reservation set--a set of bytes that subsumes the bytes in the addressed word. <b>SC.W</b> conditionally writes a word in rs2 to the address in rs1: the <b>SC.W</b> succeeds only if the reservation is still valid and the reservation set contains the bytes being written. If the <b>SC.W</b> succeeds, the instruction writes the word in rs2 to memory, and it writes zero to rd. If the <b>SC.W</b> fails, the instruction does not write to memory, and it writes a nonzero value to rd. Regardless of success or failure, executing an <b>SC.W</b> instruction invalidates any reservation held by this hart. LR.D and SC.D act analogously on doublewords and are only available on RV64. For RV64, LR.W and <b>SC.W</b> sign-extend the value placed in rd.</div><br><div><b>ISA</b>: a</div></div>",
                "tooltip": "Complex atomic memory operations on a single memory word or doubleword are performed with the load-reserved (LR) and store-conditional (SC) instructions. LR.W loads a word from the address in rs1, places the sign-extended value in rd, and registers a reservation set--a set of bytes that subsumes the bytes in the addressed word. SC.W conditionally writes a word in rs2 to the address in rs1: the SC.W succeeds only if the reservation is still valid and the reservation set contains the bytes being written. If the SC.W succeeds, the instruction writes the word in rs2 to memory, and it writes zero to rd. If the SC.W fails, the instruction does not write to memory, and it writes a nonzero value to rd. Regardless of success or failure, executing an SC.W instruction invalidates any reservation held by this hart. LR.D and SC.D act analogously on doublewords and are only available on RV64. For RV64, LR.W and SC.W sign-extend the value placed in rd.\n\n\n\n(ISA: a)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/a.html#sec:lrsc"
            };

        case "SCALL":
            return {
                "html": "<div><span class=\"opcode\"><b>SCALL</b> </span><br><div>ECALL and EBREAK were previously named <b>SCALL</b> and SBREAK. The instructions have the same functionality and encoding, but were renamed to reflect that they can be used more generally than to call a supervisor-level operating system or debugger.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "ECALL and EBREAK were previously named SCALL and SBREAK. The instructions have the same functionality and encoding, but were renamed to reflect that they can be used more generally than to call a supervisor-level operating system or debugger.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#environment-call-and-breakpoints"
            };

        case "SD":
            return {
                "html": "<div><span class=\"opcode\"><b>SD</b> rs1, rs2, imm12</span><br><div>The LW instruction loads a 32-bit value from memory and sign-extends this to 64 bits before storing it in register rd for RV64I. The LWU instruction, on the other hand, zero-extends the 32-bit value from memory for RV64I. LH and LHU are defined analogously for 16-bit values, as are LB and LBU for 8-bit values. The <b>SD</b>, SW, SH, and SB instructions store 64-bit, 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory respectively.</div><br><div><b>ISA</b>: rv64</div></div>",
                "tooltip": "The LW instruction loads a 32-bit value from memory and sign-extends this to 64 bits before storing it in register rd for RV64I. The LWU instruction, on the other hand, zero-extends the 32-bit value from memory for RV64I. LH and LHU are defined analogously for 16-bit values, as are LB and LBU for 8-bit values. The SD, SW, SH, and SB instructions store 64-bit, 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory respectively.\n\n\n\n(ISA: rv64)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv64.html#load-and-store-instructions"
            };

        case "SEQZ":
            return {
                "html": "<div><span class=\"opcode\"><b>SEQZ</b> rd, rs</span><br><div>SLTI (set less than immediate) places the value 1 in register rd if register rs1 is less than the sign-extended immediate when both are treated as signed numbers, else 0 is written to rd. SLTIU is similar but compares the values as unsigned numbers (i.e., the immediate is first sign-extended to XLEN bits then treated as an unsigned number). Note, SLTIU rd, rs1, 1 sets rd to 1 if rs1 equals zero, otherwise sets rd to 0 (assembler pseudoinstruction <b>SEQZ</b> rd, rs).</div><br><div><b>Equivalent ASM:</b><pre>sltiu rd, rs, 1</pre></div><br><div><b>ISA</b>: rv32(pseudo)</div></div>",
                "tooltip": "SLTI (set less than immediate) places the value 1 in register rd if register rs1 is less than the sign-extended immediate when both are treated as signed numbers, else 0 is written to rd. SLTIU is similar but compares the values as unsigned numbers (i.e., the immediate is first sign-extended to XLEN bits then treated as an unsigned number). Note, SLTIU rd, rs1, 1 sets rd to 1 if rs1 equals zero, otherwise sets rd to 0 (assembler pseudoinstruction SEQZ rd, rs).\n\nEquivalent ASM:\n\nsltiu rd, rs, 1\n\n(ISA: rv32(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-immediate-instructions"
            };

        case "SEXT.B":
            return {
                "html": "<div><span class=\"opcode\"><b>SEXT.B</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SEXT.H":
            return {
                "html": "<div><span class=\"opcode\"><b>SEXT.H</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SEXT.W":
            return {
                "html": "<div><span class=\"opcode\"><b>SEXT.W</b> rd, rs</span><br><div>ADDIW is an RV64I instruction that adds the sign-extended 12-bit immediate to register rs1 and produces the proper sign-extension of a 32-bit result in rd. Overflows are ignored and the result is the low 32 bits of the result sign-extended to 64 bits. Note, ADDIW rd, rs1, 0 writes the sign-extension of the lower 32 bits of register rs1 into register rd (assembler pseudoinstruction <b>SEXT.W</b>).</div><br><div><b>Equivalent ASM:</b><pre>addiw rd, rs, 0</pre></div><br><div><b>ISA</b>: rv64(pseudo)</div></div>",
                "tooltip": "ADDIW is an RV64I instruction that adds the sign-extended 12-bit immediate to register rs1 and produces the proper sign-extension of a 32-bit result in rd. Overflows are ignored and the result is the low 32 bits of the result sign-extended to 64 bits. Note, ADDIW rd, rs1, 0 writes the sign-extension of the lower 32 bits of register rs1 into register rd (assembler pseudoinstruction SEXT.W).\n\nEquivalent ASM:\n\naddiw rd, rs, 0\n\n(ISA: rv64(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv64.html#integer-register-immediate-instructions"
            };

        case "SFENCE.INVAL.IR":
            return {
                "html": "<div><span class=\"opcode\"><b>SFENCE.INVAL.IR</b> </span><br><div>The SINVAL.VMA instruction invalidates any address-translation cache entries that an SFENCE.VMA instruction with the same values of rs1 and rs2 would invalidate. However, unlike SFENCE.VMA, SINVAL.VMA instructions are only ordered with respect to SFENCE.VMA, SFENCE.W.INVAL, and <b>SFENCE.INVAL.IR</b> instructions as defined below.<br>The SFENCE.W.INVAL instruction guarantees that any previous stores already visible to the current RISC-V hart are ordered before subsequent SINVAL.VMA instructions executed by the same hart. The <b>SFENCE.INVAL.IR</b> instruction guarantees that any previous SINVAL.VMA instructions executed by the current hart are ordered before subsequent implicit references by that hart to the memory-management data structures.<br>When executed in order (but not necessarily consecutively) by a single hart, the sequence SFENCE.W.INVAL, SINVAL.VMA, and <b>SFENCE.INVAL.IR</b> has the same effect as a hypothetical SFENCE.VMA instruction in which:<br>reads and writes following the <b>SFENCE.INVAL.IR</b> are considered to be those subsequent to the SFENCE.VMA.<br>If the hypervisor extension is implemented, the Svinval extension also provides two additional instructions: HINVAL.VVMA and HINVAL.GVMA. These have the same semantics as SINVAL.VMA, except that they combine with SFENCE.W.INVAL and <b>SFENCE.INVAL.IR</b> to replace HFENCE.VVMA and HFENCE.GVMA, respectively, instead of SFENCE.VMA. In addition, HINVAL.GVMA uses VMIDs instead of ASIDs.<br>SFENCE.W.INVAL and <b>SFENCE.INVAL.IR</b> instructions do not need to be trapped when mstatus.TVM=1 or when hstatus.VTVM=1, as they only have ordering effects but no visible side effects. Trapping of the SINVAL.VMA instruction is sufficient to enable emulation of the intended overall TLB maintenance functionality.<br>In typical usage, software will invalidate a range of virtual addresses in the address-translation caches by executing an SFENCE.W.INVAL instruction, executing a series of SINVAL.VMA, HINVAL.VVMA, or HINVAL.GVMA instructions to the addresses (and optionally ASIDs or VMIDs) in question, and then executing an <b>SFENCE.INVAL.IR</b> instruction.<br>High-performance implementations will be able to pipeline the address-translation cache invalidation operations, and will defer any pipeline stalls or other memory ordering enforcement until an SFENCE.W.INVAL, <b>SFENCE.INVAL.IR</b>, SFENCE.VMA, HFENCE.GVMA, or HFENCE.VVMA instruction is executed.<br>Simpler implementations may implement SINVAL.VMA, HINVAL.VVMA, and HINVAL.GVMA identically to SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA, respectively, while implementing SFENCE.W.INVAL and <b>SFENCE.INVAL.IR</b> instructions as no-ops.</div><br><div><b>ISA</b>: supervisor</div></div>",
                "tooltip": "The SINVAL.VMA instruction invalidates any address-translation cache entries that an SFENCE.VMA instruction with the same values of rs1 and rs2 would invalidate. However, unlike SFENCE.VMA, SINVAL.VMA instructions are only ordered with respect to SFENCE.VMA, SFENCE.W.INVAL, and SFENCE.INVAL.IR instructions as defined below.\nThe SFENCE.W.INVAL instruction guarantees that any previous stores already visible to the current RISC-V hart are ordered before subsequent SINVAL.VMA instructions executed by the same hart. The SFENCE.INVAL.IR instruction guarantees that any previous SINVAL.VMA instructions executed by the current hart are ordered before subsequent implicit references by that hart to the memory-management data structures.\nWhen executed in order (but not necessarily consecutively) by a single hart, the sequence SFENCE.W.INVAL, SINVAL.VMA, and SFENCE.INVAL.IR has the same effect as a hypothetical SFENCE.VMA instruction in which:\nreads and writes following the SFENCE.INVAL.IR are considered to be those subsequent to the SFENCE.VMA.\nIf the hypervisor extension is implemented, the Svinval extension also provides two additional instructions: HINVAL.VVMA and HINVAL.GVMA. These have the same semantics as SINVAL.VMA, except that they combine with SFENCE.W.INVAL and SFENCE.INVAL.IR to replace HFENCE.VVMA and HFENCE.GVMA, respectively, instead of SFENCE.VMA. In addition, HINVAL.GVMA uses VMIDs instead of ASIDs.\nSFENCE.W.INVAL and SFENCE.INVAL.IR instructions do not need to be trapped when mstatus.TVM=1 or when hstatus.VTVM=1, as they only have ordering effects but no visible side effects. Trapping of the SINVAL.VMA instruction is sufficient to enable emulation of the intended overall TLB maintenance functionality.\nIn typical usage, software will invalidate a range of virtual addresses in the address-translation caches by executing an SFENCE.W.INVAL instruction, executing a series of SINVAL.VMA, HINVAL.VVMA, or HINVAL.GVMA instructions to the addresses (and optionally ASIDs or VMIDs) in question, and then executing an SFENCE.INVAL.IR instruction.\nHigh-performance implementations will be able to pipeline the address-translation cache invalidation operations, and will defer any pipeline stalls or other memory ordering enforcement until an SFENCE.W.INVAL, SFENCE.INVAL.IR, SFENCE.VMA, HFENCE.GVMA, or HFENCE.VVMA instruction is executed.\nSimpler implementations may implement SINVAL.VMA, HINVAL.VVMA, and HINVAL.GVMA identically to SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA, respectively, while implementing SFENCE.W.INVAL and SFENCE.INVAL.IR instructions as no-ops.\n\n\n\n(ISA: supervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/supervisor.html#svinval"
            };

        case "SFENCE.VMA":
            return {
                "html": "<div><span class=\"opcode\"><b>SFENCE.VMA</b> rs1, rs2</span><br><div>This chapter defines the memory model for regular main memory operations. The interaction of the memory model with I/O memory, instruction fetches, FENCE.I, page table walks, and <b>SFENCE.VMA</b> is not (yet) formalized. Some or all of the above may be formalized in a future revision of this specification. The RV128 base ISA and future ISA extensions such as the \"V\" vector and \"J\" JIT extensions will need to be incorporated into a future revision as well.</div><br><div><b>ISA</b>: rvwmo</div></div>",
                "tooltip": "This chapter defines the memory model for regular main memory operations. The interaction of the memory model with I/O memory, instruction fetches, FENCE.I, page table walks, and SFENCE.VMA is not (yet) formalized. Some or all of the above may be formalized in a future revision of this specification. The RV128 base ISA and future ISA extensions such as the \"V\" vector and \"J\" JIT extensions will need to be incorporated into a future revision as well.\n\n\n\n(ISA: rvwmo)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rvwmo.html#ch:memorymodel"
            };

        case "SFENCE.W.INVAL":
            return {
                "html": "<div><span class=\"opcode\"><b>SFENCE.W.INVAL</b> </span><br><div>The SINVAL.VMA instruction invalidates any address-translation cache entries that an SFENCE.VMA instruction with the same values of rs1 and rs2 would invalidate. However, unlike SFENCE.VMA, SINVAL.VMA instructions are only ordered with respect to SFENCE.VMA, <b>SFENCE.W.INVAL</b>, and SFENCE.INVAL.IR instructions as defined below.<br>The <b>SFENCE.W.INVAL</b> instruction guarantees that any previous stores already visible to the current RISC-V hart are ordered before subsequent SINVAL.VMA instructions executed by the same hart. The SFENCE.INVAL.IR instruction guarantees that any previous SINVAL.VMA instructions executed by the current hart are ordered before subsequent implicit references by that hart to the memory-management data structures.<br>When executed in order (but not necessarily consecutively) by a single hart, the sequence <b>SFENCE.W.INVAL</b>, SINVAL.VMA, and SFENCE.INVAL.IR has the same effect as a hypothetical SFENCE.VMA instruction in which:<br>reads and writes prior to the <b>SFENCE.W.INVAL</b> are considered to be those prior to the SFENCE.VMA, and<br>If the hypervisor extension is implemented, the Svinval extension also provides two additional instructions: HINVAL.VVMA and HINVAL.GVMA. These have the same semantics as SINVAL.VMA, except that they combine with <b>SFENCE.W.INVAL</b> and SFENCE.INVAL.IR to replace HFENCE.VVMA and HFENCE.GVMA, respectively, instead of SFENCE.VMA. In addition, HINVAL.GVMA uses VMIDs instead of ASIDs.<br><b>SFENCE.W.INVAL</b> and SFENCE.INVAL.IR instructions do not need to be trapped when mstatus.TVM=1 or when hstatus.VTVM=1, as they only have ordering effects but no visible side effects. Trapping of the SINVAL.VMA instruction is sufficient to enable emulation of the intended overall TLB maintenance functionality.<br>In typical usage, software will invalidate a range of virtual addresses in the address-translation caches by executing an <b>SFENCE.W.INVAL</b> instruction, executing a series of SINVAL.VMA, HINVAL.VVMA, or HINVAL.GVMA instructions to the addresses (and optionally ASIDs or VMIDs) in question, and then executing an SFENCE.INVAL.IR instruction.<br>High-performance implementations will be able to pipeline the address-translation cache invalidation operations, and will defer any pipeline stalls or other memory ordering enforcement until an <b>SFENCE.W.INVAL</b>, SFENCE.INVAL.IR, SFENCE.VMA, HFENCE.GVMA, or HFENCE.VVMA instruction is executed.<br>Simpler implementations may implement SINVAL.VMA, HINVAL.VVMA, and HINVAL.GVMA identically to SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA, respectively, while implementing <b>SFENCE.W.INVAL</b> and SFENCE.INVAL.IR instructions as no-ops.</div><br><div><b>ISA</b>: supervisor</div></div>",
                "tooltip": "The SINVAL.VMA instruction invalidates any address-translation cache entries that an SFENCE.VMA instruction with the same values of rs1 and rs2 would invalidate. However, unlike SFENCE.VMA, SINVAL.VMA instructions are only ordered with respect to SFENCE.VMA, SFENCE.W.INVAL, and SFENCE.INVAL.IR instructions as defined below.\nThe SFENCE.W.INVAL instruction guarantees that any previous stores already visible to the current RISC-V hart are ordered before subsequent SINVAL.VMA instructions executed by the same hart. The SFENCE.INVAL.IR instruction guarantees that any previous SINVAL.VMA instructions executed by the current hart are ordered before subsequent implicit references by that hart to the memory-management data structures.\nWhen executed in order (but not necessarily consecutively) by a single hart, the sequence SFENCE.W.INVAL, SINVAL.VMA, and SFENCE.INVAL.IR has the same effect as a hypothetical SFENCE.VMA instruction in which:\nreads and writes prior to the SFENCE.W.INVAL are considered to be those prior to the SFENCE.VMA, and\nIf the hypervisor extension is implemented, the Svinval extension also provides two additional instructions: HINVAL.VVMA and HINVAL.GVMA. These have the same semantics as SINVAL.VMA, except that they combine with SFENCE.W.INVAL and SFENCE.INVAL.IR to replace HFENCE.VVMA and HFENCE.GVMA, respectively, instead of SFENCE.VMA. In addition, HINVAL.GVMA uses VMIDs instead of ASIDs.\nSFENCE.W.INVAL and SFENCE.INVAL.IR instructions do not need to be trapped when mstatus.TVM=1 or when hstatus.VTVM=1, as they only have ordering effects but no visible side effects. Trapping of the SINVAL.VMA instruction is sufficient to enable emulation of the intended overall TLB maintenance functionality.\nIn typical usage, software will invalidate a range of virtual addresses in the address-translation caches by executing an SFENCE.W.INVAL instruction, executing a series of SINVAL.VMA, HINVAL.VVMA, or HINVAL.GVMA instructions to the addresses (and optionally ASIDs or VMIDs) in question, and then executing an SFENCE.INVAL.IR instruction.\nHigh-performance implementations will be able to pipeline the address-translation cache invalidation operations, and will defer any pipeline stalls or other memory ordering enforcement until an SFENCE.W.INVAL, SFENCE.INVAL.IR, SFENCE.VMA, HFENCE.GVMA, or HFENCE.VVMA instruction is executed.\nSimpler implementations may implement SINVAL.VMA, HINVAL.VVMA, and HINVAL.GVMA identically to SFENCE.VMA, HFENCE.VVMA, and HFENCE.GVMA, respectively, while implementing SFENCE.W.INVAL and SFENCE.INVAL.IR instructions as no-ops.\n\n\n\n(ISA: supervisor)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/supervisor.html#svinval"
            };

        case "SGTZ":
            return {
                "html": "<div><span class=\"opcode\"><b>SGTZ</b> rd, rs</span><br><div><b>Equivalent ASM:</b><pre>slt rd, x0, rs</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nslt rd, x0, rs\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SH":
            return {
                "html": "<div><span class=\"opcode\"><b>SH</b> rs1, rs2, imm12</span><br><div>The LW instruction loads a 32-bit value from memory into rd. LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd. LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd. LB and LBU are defined analogously for 8-bit values. The SW, <b>SH</b>, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "The LW instruction loads a 32-bit value from memory into rd. LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd. LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd. LB and LBU are defined analogously for 8-bit values. The SW, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#sec:rv32:ldst"
            };

        case "SH1ADD":
            return {
                "html": "<div><span class=\"opcode\"><b>SH1ADD</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SH1ADD.UW":
            return {
                "html": "<div><span class=\"opcode\"><b>SH1ADD.UW</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SH2ADD":
            return {
                "html": "<div><span class=\"opcode\"><b>SH2ADD</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SH2ADD.UW":
            return {
                "html": "<div><span class=\"opcode\"><b>SH2ADD.UW</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SH3ADD":
            return {
                "html": "<div><span class=\"opcode\"><b>SH3ADD</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SH3ADD.UW":
            return {
                "html": "<div><span class=\"opcode\"><b>SH3ADD.UW</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SHA256SIG0":
            return {
                "html": "<div><span class=\"opcode\"><b>SHA256SIG0</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SHA256SIG1":
            return {
                "html": "<div><span class=\"opcode\"><b>SHA256SIG1</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SHA256SUM0":
            return {
                "html": "<div><span class=\"opcode\"><b>SHA256SUM0</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SHA256SUM1":
            return {
                "html": "<div><span class=\"opcode\"><b>SHA256SUM1</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SHA512SIG0":
            return {
                "html": "<div><span class=\"opcode\"><b>SHA512SIG0</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SHA512SIG0H":
            return {
                "html": "<div><span class=\"opcode\"><b>SHA512SIG0H</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SHA512SIG0L":
            return {
                "html": "<div><span class=\"opcode\"><b>SHA512SIG0L</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SHA512SIG1":
            return {
                "html": "<div><span class=\"opcode\"><b>SHA512SIG1</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SHA512SIG1H":
            return {
                "html": "<div><span class=\"opcode\"><b>SHA512SIG1H</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SHA512SIG1L":
            return {
                "html": "<div><span class=\"opcode\"><b>SHA512SIG1L</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SHA512SUM0":
            return {
                "html": "<div><span class=\"opcode\"><b>SHA512SUM0</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SHA512SUM0R":
            return {
                "html": "<div><span class=\"opcode\"><b>SHA512SUM0R</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SHA512SUM1":
            return {
                "html": "<div><span class=\"opcode\"><b>SHA512SUM1</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SHA512SUM1R":
            return {
                "html": "<div><span class=\"opcode\"><b>SHA512SUM1R</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SINVAL.VMA":
            return {
                "html": "<div><span class=\"opcode\"><b>SINVAL.VMA</b> rs1, rs2</span><br><div>field that supports intercepting supervisor virtual-memory management operations. When TVM=1, attempts to read or write the satp CSR or execute an SFENCE.VMA or <b>SINVAL.VMA</b> instruction while executing in S-mode will raise an illegal instruction exception. When TVM=0, these operations are permitted in S-mode. TVM is read-only 0 when S-mode is not supported.<br>Trapping satp accesses and the SFENCE.VMA and <b>SINVAL.VMA</b> instructions provides the hooks necessary to lazily populate shadow page tables.</div><br><div><b>ISA</b>: machine</div></div>",
                "tooltip": "field that supports intercepting supervisor virtual-memory management operations. When TVM=1, attempts to read or write the satp CSR or execute an SFENCE.VMA or SINVAL.VMA instruction while executing in S-mode will raise an illegal instruction exception. When TVM=0, these operations are permitted in S-mode. TVM is read-only 0 when S-mode is not supported.\nTrapping satp accesses and the SFENCE.VMA and SINVAL.VMA instructions provides the hooks necessary to lazily populate shadow page tables.\n\n\n\n(ISA: machine)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/machine.html#virt-control"
            };

        case "SLL":
            return {
                "html": "<div><span class=\"opcode\"><b>SLL</b> rd, rs1, rs2</span><br><div><b>SLL</b>, SRL, and SRA perform logical left, logical right, and arithmetic right shifts on the value in register rs1 by the shift amount held in the lower 5 bits of register rs2.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "SLL, SRL, and SRA perform logical left, logical right, and arithmetic right shifts on the value in register rs1 by the shift amount held in the lower 5 bits of register rs2.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-register-operations"
            };

        case "SLLI":
            return {
                "html": "<div><span class=\"opcode\"><b>SLLI</b> rd, rs1</span><br><div>Shifts by a constant are encoded as a specialization of the I-type format. The operand to be shifted is in rs1, and the shift amount is encoded in the lower 5 bits of the I-immediate field. The right shift type is encoded in bit 30. <b>SLLI</b> is a logical left shift (zeros are shifted into the lower bits); SRLI is a logical right shift (zeros are shifted into the upper bits); and SRAI is an arithmetic right shift (the original sign bit is copied into the vacated upper bits).</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "Shifts by a constant are encoded as a specialization of the I-type format. The operand to be shifted is in rs1, and the shift amount is encoded in the lower 5 bits of the I-immediate field. The right shift type is encoded in bit 30. SLLI is a logical left shift (zeros are shifted into the lower bits); SRLI is a logical right shift (zeros are shifted into the upper bits); and SRAI is an arithmetic right shift (the original sign bit is copied into the vacated upper bits).\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-immediate-instructions"
            };

        case "SLLI.UW":
            return {
                "html": "<div><span class=\"opcode\"><b>SLLI.UW</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SLLI_RV32":
            return {
                "html": "<div><span class=\"opcode\"><b>SLLI_RV32</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SLLIW":
            return {
                "html": "<div><span class=\"opcode\"><b>SLLIW</b> rd, rs1</span><br><div><b>SLLIW</b>, SRLIW, and SRAIW are RV64I-only instructions that are analogously defined but operate on 32-bit values and sign-extend their 32-bit results to 64 bits. <b>SLLIW</b>, SRLIW, and SRAIW encodings with imm[5] 0 are reserved.<br>Previously, <b>SLLIW</b>, SRLIW, and SRAIW with imm[5] 0 were defined to cause illegal instruction exceptions, whereas now they are marked as reserved. This is a backwards-compatible change.</div><br><div><b>ISA</b>: rv64</div></div>",
                "tooltip": "SLLIW, SRLIW, and SRAIW are RV64I-only instructions that are analogously defined but operate on 32-bit values and sign-extend their 32-bit results to 64 bits. SLLIW, SRLIW, and SRAIW encodings with imm[5] 0 are reserved.\nPreviously, SLLIW, SRLIW, and SRAIW with imm[5] 0 were defined to cause illegal instruction exceptions, whereas now they are marked as reserved. This is a backwards-compatible change.\n\n\n\n(ISA: rv64)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv64.html#integer-register-immediate-instructions"
            };

        case "SLLW":
            return {
                "html": "<div><span class=\"opcode\"><b>SLLW</b> rd, rs1, rs2</span><br><div><b>SLLW</b>, SRLW, and SRAW are RV64I-only instructions that are analogously defined but operate on 32-bit values and sign-extend their 32-bit results to 64 bits. The shift amount is given by rs2[4:0].</div><br><div><b>ISA</b>: rv64</div></div>",
                "tooltip": "SLLW, SRLW, and SRAW are RV64I-only instructions that are analogously defined but operate on 32-bit values and sign-extend their 32-bit results to 64 bits. The shift amount is given by rs2[4:0].\n\n\n\n(ISA: rv64)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv64.html#integer-register-register-operations"
            };

        case "SLT":
            return {
                "html": "<div><span class=\"opcode\"><b>SLT</b> rd, rs1, rs2</span><br><div>ADD performs the addition of rs1 and rs2. SUB performs the subtraction of rs2 from rs1. Overflows are ignored and the low XLEN bits of results are written to the destination rd. <b>SLT</b> and SLTU perform signed and unsigned compares respectively, writing 1 to rd if rs1 < rs2, 0 otherwise. Note, SLTU rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs). AND, OR, and XOR perform bitwise logical operations.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "ADD performs the addition of rs1 and rs2. SUB performs the subtraction of rs2 from rs1. Overflows are ignored and the low XLEN bits of results are written to the destination rd. SLT and SLTU perform signed and unsigned compares respectively, writing 1 to rd if rs1 < rs2, 0 otherwise. Note, SLTU rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs). AND, OR, and XOR perform bitwise logical operations.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-register-operations"
            };

        case "SLTI":
            return {
                "html": "<div><span class=\"opcode\"><b>SLTI</b> rd, rs1, imm12</span><br><div><b>SLTI</b> (set less than immediate) places the value 1 in register rd if register rs1 is less than the sign-extended immediate when both are treated as signed numbers, else 0 is written to rd. SLTIU is similar but compares the values as unsigned numbers (i.e., the immediate is first sign-extended to XLEN bits then treated as an unsigned number). Note, SLTIU rd, rs1, 1 sets rd to 1 if rs1 equals zero, otherwise sets rd to 0 (assembler pseudoinstruction SEQZ rd, rs).</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "SLTI (set less than immediate) places the value 1 in register rd if register rs1 is less than the sign-extended immediate when both are treated as signed numbers, else 0 is written to rd. SLTIU is similar but compares the values as unsigned numbers (i.e., the immediate is first sign-extended to XLEN bits then treated as an unsigned number). Note, SLTIU rd, rs1, 1 sets rd to 1 if rs1 equals zero, otherwise sets rd to 0 (assembler pseudoinstruction SEQZ rd, rs).\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-immediate-instructions"
            };

        case "SLTIU":
            return {
                "html": "<div><span class=\"opcode\"><b>SLTIU</b> rd, rs1, imm12</span><br><div>SLTI (set less than immediate) places the value 1 in register rd if register rs1 is less than the sign-extended immediate when both are treated as signed numbers, else 0 is written to rd. <b>SLTIU</b> is similar but compares the values as unsigned numbers (i.e., the immediate is first sign-extended to XLEN bits then treated as an unsigned number). Note, <b>SLTIU</b> rd, rs1, 1 sets rd to 1 if rs1 equals zero, otherwise sets rd to 0 (assembler pseudoinstruction SEQZ rd, rs).</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "SLTI (set less than immediate) places the value 1 in register rd if register rs1 is less than the sign-extended immediate when both are treated as signed numbers, else 0 is written to rd. SLTIU is similar but compares the values as unsigned numbers (i.e., the immediate is first sign-extended to XLEN bits then treated as an unsigned number). Note, SLTIU rd, rs1, 1 sets rd to 1 if rs1 equals zero, otherwise sets rd to 0 (assembler pseudoinstruction SEQZ rd, rs).\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-immediate-instructions"
            };

        case "SLTU":
            return {
                "html": "<div><span class=\"opcode\"><b>SLTU</b> rd, rs1, rs2</span><br><div>ADD performs the addition of rs1 and rs2. SUB performs the subtraction of rs2 from rs1. Overflows are ignored and the low XLEN bits of results are written to the destination rd. SLT and <b>SLTU</b> perform signed and unsigned compares respectively, writing 1 to rd if rs1 < rs2, 0 otherwise. Note, <b>SLTU</b> rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs). AND, OR, and XOR perform bitwise logical operations.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "ADD performs the addition of rs1 and rs2. SUB performs the subtraction of rs2 from rs1. Overflows are ignored and the low XLEN bits of results are written to the destination rd. SLT and SLTU perform signed and unsigned compares respectively, writing 1 to rd if rs1 < rs2, 0 otherwise. Note, SLTU rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs). AND, OR, and XOR perform bitwise logical operations.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-register-operations"
            };

        case "SLTZ":
            return {
                "html": "<div><span class=\"opcode\"><b>SLTZ</b> rd, rs</span><br><div><b>Equivalent ASM:</b><pre>slt rd, rs, x0</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nslt rd, rs, x0\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SM3P0":
            return {
                "html": "<div><span class=\"opcode\"><b>SM3P0</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SM3P1":
            return {
                "html": "<div><span class=\"opcode\"><b>SM3P1</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SM4ED":
            return {
                "html": "<div><span class=\"opcode\"><b>SM4ED</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SM4KS":
            return {
                "html": "<div><span class=\"opcode\"><b>SM4KS</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SNEZ":
            return {
                "html": "<div><span class=\"opcode\"><b>SNEZ</b> rd, rs</span><br><div>ADD performs the addition of rs1 and rs2. SUB performs the subtraction of rs2 from rs1. Overflows are ignored and the low XLEN bits of results are written to the destination rd. SLT and SLTU perform signed and unsigned compares respectively, writing 1 to rd if rs1 < rs2, 0 otherwise. Note, SLTU rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction <b>SNEZ</b> rd, rs). AND, OR, and XOR perform bitwise logical operations.</div><br><div><b>Equivalent ASM:</b><pre>sltu rd, x0, rs</pre></div><br><div><b>ISA</b>: rv32(pseudo)</div></div>",
                "tooltip": "ADD performs the addition of rs1 and rs2. SUB performs the subtraction of rs2 from rs1. Overflows are ignored and the low XLEN bits of results are written to the destination rd. SLT and SLTU perform signed and unsigned compares respectively, writing 1 to rd if rs1 < rs2, 0 otherwise. Note, SLTU rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs). AND, OR, and XOR perform bitwise logical operations.\n\nEquivalent ASM:\n\nsltu rd, x0, rs\n\n(ISA: rv32(pseudo))",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-register-operations"
            };

        case "SRA":
            return {
                "html": "<div><span class=\"opcode\"><b>SRA</b> rd, rs1, rs2</span><br><div>SLL, SRL, and <b>SRA</b> perform logical left, logical right, and arithmetic right shifts on the value in register rs1 by the shift amount held in the lower 5 bits of register rs2.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "SLL, SRL, and SRA perform logical left, logical right, and arithmetic right shifts on the value in register rs1 by the shift amount held in the lower 5 bits of register rs2.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-register-operations"
            };

        case "SRAI":
            return {
                "html": "<div><span class=\"opcode\"><b>SRAI</b> rd, rs1</span><br><div>Shifts by a constant are encoded as a specialization of the I-type format. The operand to be shifted is in rs1, and the shift amount is encoded in the lower 5 bits of the I-immediate field. The right shift type is encoded in bit 30. SLLI is a logical left shift (zeros are shifted into the lower bits); SRLI is a logical right shift (zeros are shifted into the upper bits); and <b>SRAI</b> is an arithmetic right shift (the original sign bit is copied into the vacated upper bits).</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "Shifts by a constant are encoded as a specialization of the I-type format. The operand to be shifted is in rs1, and the shift amount is encoded in the lower 5 bits of the I-immediate field. The right shift type is encoded in bit 30. SLLI is a logical left shift (zeros are shifted into the lower bits); SRLI is a logical right shift (zeros are shifted into the upper bits); and SRAI is an arithmetic right shift (the original sign bit is copied into the vacated upper bits).\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-immediate-instructions"
            };

        case "SRAI_RV32":
            return {
                "html": "<div><span class=\"opcode\"><b>SRAI_RV32</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SRAIW":
            return {
                "html": "<div><span class=\"opcode\"><b>SRAIW</b> rd, rs1</span><br><div>SLLIW, SRLIW, and <b>SRAIW</b> are RV64I-only instructions that are analogously defined but operate on 32-bit values and sign-extend their 32-bit results to 64 bits. SLLIW, SRLIW, and <b>SRAIW</b> encodings with imm[5] 0 are reserved.<br>Previously, SLLIW, SRLIW, and <b>SRAIW</b> with imm[5] 0 were defined to cause illegal instruction exceptions, whereas now they are marked as reserved. This is a backwards-compatible change.</div><br><div><b>ISA</b>: rv64</div></div>",
                "tooltip": "SLLIW, SRLIW, and SRAIW are RV64I-only instructions that are analogously defined but operate on 32-bit values and sign-extend their 32-bit results to 64 bits. SLLIW, SRLIW, and SRAIW encodings with imm[5] 0 are reserved.\nPreviously, SLLIW, SRLIW, and SRAIW with imm[5] 0 were defined to cause illegal instruction exceptions, whereas now they are marked as reserved. This is a backwards-compatible change.\n\n\n\n(ISA: rv64)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv64.html#integer-register-immediate-instructions"
            };

        case "SRAW":
            return {
                "html": "<div><span class=\"opcode\"><b>SRAW</b> rd, rs1, rs2</span><br><div>SLLW, SRLW, and <b>SRAW</b> are RV64I-only instructions that are analogously defined but operate on 32-bit values and sign-extend their 32-bit results to 64 bits. The shift amount is given by rs2[4:0].</div><br><div><b>ISA</b>: rv64</div></div>",
                "tooltip": "SLLW, SRLW, and SRAW are RV64I-only instructions that are analogously defined but operate on 32-bit values and sign-extend their 32-bit results to 64 bits. The shift amount is given by rs2[4:0].\n\n\n\n(ISA: rv64)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv64.html#integer-register-register-operations"
            };

        case "SRET":
            return {
                "html": "<div><span class=\"opcode\"><b>SRET</b> </span><br><div>An MRET or <b>SRET</b> instruction is used to return from a trap in M-mode or S-mode respectively. When executing an xRET instruction, supposing xPP holds the value y, xIE is set to xPIE; the privilege mode is changed to y; xPIE is set to 1; and xPP is set to the least-privileged supported mode (U if U-mode is implemented, else M). If xPP M, xRET also sets MPRV=0.</div><br><div><b>ISA</b>: machine</div></div>",
                "tooltip": "An MRET or SRET instruction is used to return from a trap in M-mode or S-mode respectively. When executing an xRET instruction, supposing xPP holds the value y, xIE is set to xPIE; the privilege mode is changed to y; xPIE is set to 1; and xPP is set to the least-privileged supported mode (U if U-mode is implemented, else M). If xPP M, xRET also sets MPRV=0.\n\n\n\n(ISA: machine)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/machine.html#privstack"
            };

        case "SRL":
            return {
                "html": "<div><span class=\"opcode\"><b>SRL</b> rd, rs1, rs2</span><br><div>SLL, <b>SRL</b>, and SRA perform logical left, logical right, and arithmetic right shifts on the value in register rs1 by the shift amount held in the lower 5 bits of register rs2.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "SLL, SRL, and SRA perform logical left, logical right, and arithmetic right shifts on the value in register rs1 by the shift amount held in the lower 5 bits of register rs2.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-register-operations"
            };

        case "SRLI":
            return {
                "html": "<div><span class=\"opcode\"><b>SRLI</b> rd, rs1</span><br><div>Shifts by a constant are encoded as a specialization of the I-type format. The operand to be shifted is in rs1, and the shift amount is encoded in the lower 5 bits of the I-immediate field. The right shift type is encoded in bit 30. SLLI is a logical left shift (zeros are shifted into the lower bits); <b>SRLI</b> is a logical right shift (zeros are shifted into the upper bits); and SRAI is an arithmetic right shift (the original sign bit is copied into the vacated upper bits).</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "Shifts by a constant are encoded as a specialization of the I-type format. The operand to be shifted is in rs1, and the shift amount is encoded in the lower 5 bits of the I-immediate field. The right shift type is encoded in bit 30. SLLI is a logical left shift (zeros are shifted into the lower bits); SRLI is a logical right shift (zeros are shifted into the upper bits); and SRAI is an arithmetic right shift (the original sign bit is copied into the vacated upper bits).\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-immediate-instructions"
            };

        case "SRLI_RV32":
            return {
                "html": "<div><span class=\"opcode\"><b>SRLI_RV32</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "SRLIW":
            return {
                "html": "<div><span class=\"opcode\"><b>SRLIW</b> rd, rs1</span><br><div>SLLIW, <b>SRLIW</b>, and SRAIW are RV64I-only instructions that are analogously defined but operate on 32-bit values and sign-extend their 32-bit results to 64 bits. SLLIW, <b>SRLIW</b>, and SRAIW encodings with imm[5] 0 are reserved.<br>Previously, SLLIW, <b>SRLIW</b>, and SRAIW with imm[5] 0 were defined to cause illegal instruction exceptions, whereas now they are marked as reserved. This is a backwards-compatible change.</div><br><div><b>ISA</b>: rv64</div></div>",
                "tooltip": "SLLIW, SRLIW, and SRAIW are RV64I-only instructions that are analogously defined but operate on 32-bit values and sign-extend their 32-bit results to 64 bits. SLLIW, SRLIW, and SRAIW encodings with imm[5] 0 are reserved.\nPreviously, SLLIW, SRLIW, and SRAIW with imm[5] 0 were defined to cause illegal instruction exceptions, whereas now they are marked as reserved. This is a backwards-compatible change.\n\n\n\n(ISA: rv64)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv64.html#integer-register-immediate-instructions"
            };

        case "SRLW":
            return {
                "html": "<div><span class=\"opcode\"><b>SRLW</b> rd, rs1, rs2</span><br><div>SLLW, <b>SRLW</b>, and SRAW are RV64I-only instructions that are analogously defined but operate on 32-bit values and sign-extend their 32-bit results to 64 bits. The shift amount is given by rs2[4:0].</div><br><div><b>ISA</b>: rv64</div></div>",
                "tooltip": "SLLW, SRLW, and SRAW are RV64I-only instructions that are analogously defined but operate on 32-bit values and sign-extend their 32-bit results to 64 bits. The shift amount is given by rs2[4:0].\n\n\n\n(ISA: rv64)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv64.html#integer-register-register-operations"
            };

        case "SUB":
            return {
                "html": "<div><span class=\"opcode\"><b>SUB</b> rd, rs1, rs2</span><br><div>ADD performs the addition of rs1 and rs2. <b>SUB</b> performs the subtraction of rs2 from rs1. Overflows are ignored and the low XLEN bits of results are written to the destination rd. SLT and SLTU perform signed and unsigned compares respectively, writing 1 to rd if rs1 < rs2, 0 otherwise. Note, SLTU rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs). AND, OR, and XOR perform bitwise logical operations.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "ADD performs the addition of rs1 and rs2. SUB performs the subtraction of rs2 from rs1. Overflows are ignored and the low XLEN bits of results are written to the destination rd. SLT and SLTU perform signed and unsigned compares respectively, writing 1 to rd if rs1 < rs2, 0 otherwise. Note, SLTU rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs). AND, OR, and XOR perform bitwise logical operations.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-register-operations"
            };

        case "SUBW":
            return {
                "html": "<div><span class=\"opcode\"><b>SUBW</b> rd, rs1, rs2</span><br><div>ADDW and <b>SUBW</b> are RV64I-only instructions that are defined analogously to ADD and SUB but operate on 32-bit values and produce signed 32-bit results. Overflows are ignored, and the low 32-bits of the result is sign-extended to 64-bits and written to the destination register.</div><br><div><b>ISA</b>: rv64</div></div>",
                "tooltip": "ADDW and SUBW are RV64I-only instructions that are defined analogously to ADD and SUB but operate on 32-bit values and produce signed 32-bit results. Overflows are ignored, and the low 32-bits of the result is sign-extended to 64-bits and written to the destination register.\n\n\n\n(ISA: rv64)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv64.html#integer-register-register-operations"
            };

        case "SW":
            return {
                "html": "<div><span class=\"opcode\"><b>SW</b> rs1, rs2, imm12</span><br><div>The LW instruction loads a 32-bit value from memory into rd. LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd. LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd. LB and LBU are defined analogously for 8-bit values. The <b>SW</b>, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "The LW instruction loads a 32-bit value from memory into rd. LH loads a 16-bit value from memory, then sign-extends to 32-bits before storing in rd. LHU loads a 16-bit value from memory but then zero extends to 32-bits before storing in rd. LB and LBU are defined analogously for 8-bit values. The SW, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#sec:rv32:ldst"
            };

        case "TAIL":
            return {
                "html": "<div><span class=\"opcode\"><b>TAIL</b> offset</span><br><div><b>Equivalent ASM:</b><pre>auipc x6, offset[31:12]\njalr x0, x6, offset[11:0]</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nauipc x6, offset[31:12]\njalr x0, x6, offset[11:0]\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "UNZIP":
            return {
                "html": "<div><span class=\"opcode\"><b>UNZIP</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAADD.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VAADD.VV</b> vs2, vs1, vd</span><br><div>vaadd</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vaadd\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VAADD.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VAADD.VX</b> vs2, rs1, vd</span><br><div>vaadd</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vaadd\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VAADDU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VAADDU.VV</b> vs2, vs1, vd</span><br><div>The averaging add and subtract instructions right shift the result by one bit and round off the result according to the setting in vxrm. Both unsigned and signed versions are provided. For vaaddu and vaadd there can be no overflow in the result. For vasub and vasubu, overflow is ignored and the result wraps around.<br># Averaging add # Averaging adds of unsigned integers. <b>vaaddu.vv</b> vd, vs2, vs1, vm # roundoff_unsigned(vs2[i] + vs1[i], 1) vaaddu.vx vd, vs2, rs1, vm # roundoff_unsigned(vs2[i] + x[rs1], 1) # Averaging adds of signed integers. vaadd.vv vd, vs2, vs1, vm # roundoff_signed(vs2[i] + vs1[i], 1) vaadd.vx vd, vs2, rs1, vm # roundoff_signed(vs2[i] + x[rs1], 1) # Averaging subtract # Averaging subtract of unsigned integers. vasubu.vv vd, vs2, vs1, vm # roundoff_unsigned(vs2[i] - vs1[i], 1) vasubu.vx vd, vs2, rs1, vm # roundoff_unsigned(vs2[i] - x[rs1], 1) # Averaging subtract of signed integers. vasub.vv vd, vs2, vs1, vm # roundoff_signed(vs2[i] - vs1[i], 1) vasub.vx vd, vs2, rs1, vm # roundoff_signed(vs2[i] - x[rs1], 1)</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The averaging add and subtract instructions right shift the result by one bit and round off the result according to the setting in vxrm. Both unsigned and signed versions are provided. For vaaddu and vaadd there can be no overflow in the result. For vasub and vasubu, overflow is ignored and the result wraps around.\n# Averaging add # Averaging adds of unsigned integers. vaaddu.vv vd, vs2, vs1, vm # roundoff_unsigned(vs2[i] + vs1[i], 1) vaaddu.vx vd, vs2, rs1, vm # roundoff_unsigned(vs2[i] + x[rs1], 1) # Averaging adds of signed integers. vaadd.vv vd, vs2, vs1, vm # roundoff_signed(vs2[i] + vs1[i], 1) vaadd.vx vd, vs2, rs1, vm # roundoff_signed(vs2[i] + x[rs1], 1) # Averaging subtract # Averaging subtract of unsigned integers. vasubu.vv vd, vs2, vs1, vm # roundoff_unsigned(vs2[i] - vs1[i], 1) vasubu.vx vd, vs2, rs1, vm # roundoff_unsigned(vs2[i] - x[rs1], 1) # Averaging subtract of signed integers. vasub.vv vd, vs2, vs1, vm # roundoff_signed(vs2[i] - vs1[i], 1) vasub.vx vd, vs2, rs1, vm # roundoff_signed(vs2[i] - x[rs1], 1)\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_averaging_add_and_subtract"
            };

        case "VAADDU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VAADDU.VX</b> vs2, rs1, vd</span><br><div>The averaging add and subtract instructions right shift the result by one bit and round off the result according to the setting in vxrm. Both unsigned and signed versions are provided. For vaaddu and vaadd there can be no overflow in the result. For vasub and vasubu, overflow is ignored and the result wraps around.<br># Averaging add # Averaging adds of unsigned integers. vaaddu.vv vd, vs2, vs1, vm # roundoff_unsigned(vs2[i] + vs1[i], 1) <b>vaaddu.vx</b> vd, vs2, rs1, vm # roundoff_unsigned(vs2[i] + x[rs1], 1) # Averaging adds of signed integers. vaadd.vv vd, vs2, vs1, vm # roundoff_signed(vs2[i] + vs1[i], 1) vaadd.vx vd, vs2, rs1, vm # roundoff_signed(vs2[i] + x[rs1], 1) # Averaging subtract # Averaging subtract of unsigned integers. vasubu.vv vd, vs2, vs1, vm # roundoff_unsigned(vs2[i] - vs1[i], 1) vasubu.vx vd, vs2, rs1, vm # roundoff_unsigned(vs2[i] - x[rs1], 1) # Averaging subtract of signed integers. vasub.vv vd, vs2, vs1, vm # roundoff_signed(vs2[i] - vs1[i], 1) vasub.vx vd, vs2, rs1, vm # roundoff_signed(vs2[i] - x[rs1], 1)</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The averaging add and subtract instructions right shift the result by one bit and round off the result according to the setting in vxrm. Both unsigned and signed versions are provided. For vaaddu and vaadd there can be no overflow in the result. For vasub and vasubu, overflow is ignored and the result wraps around.\n# Averaging add # Averaging adds of unsigned integers. vaaddu.vv vd, vs2, vs1, vm # roundoff_unsigned(vs2[i] + vs1[i], 1) vaaddu.vx vd, vs2, rs1, vm # roundoff_unsigned(vs2[i] + x[rs1], 1) # Averaging adds of signed integers. vaadd.vv vd, vs2, vs1, vm # roundoff_signed(vs2[i] + vs1[i], 1) vaadd.vx vd, vs2, rs1, vm # roundoff_signed(vs2[i] + x[rs1], 1) # Averaging subtract # Averaging subtract of unsigned integers. vasubu.vv vd, vs2, vs1, vm # roundoff_unsigned(vs2[i] - vs1[i], 1) vasubu.vx vd, vs2, rs1, vm # roundoff_unsigned(vs2[i] - x[rs1], 1) # Averaging subtract of signed integers. vasub.vv vd, vs2, vs1, vm # roundoff_signed(vs2[i] - vs1[i], 1) vasub.vx vd, vs2, rs1, vm # roundoff_signed(vs2[i] - x[rs1], 1)\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_averaging_add_and_subtract"
            };

        case "VADC.VIM":
            return {
                "html": "<div><span class=\"opcode\"><b>VADC.VIM</b> vs2, simm5, vd</span><br><div>vadc and vsbc add or subtract the source operands and the carry-in or borrow-in, and write the result to vector register vd. These instructions are encoded as masked instructions (vm=0), but they operate on and write back all body elements. Encodings corresponding to the unmasked versions (vm=1) are reserved.<br>For vadc and vsbc, the instruction encoding is reserved if the destination vector register is v0.<br># Produce sum with carry. # vd[i] = vs2[i] + vs1[i] + v0.mask[i] vadc.vvm vd, vs2, vs1, v0 # Vector-vector # vd[i] = vs2[i] + x[rs1] + v0.mask[i] vadc.vxm vd, vs2, rs1, v0 # Vector-scalar # vd[i] = vs2[i] + imm + v0.mask[i] <b>vadc.vim</b> vd, vs2, imm, v0 # Vector-immediate # Produce carry out in mask register format # vd.mask[i] = carry_out(vs2[i] + vs1[i] + v0.mask[i]) vmadc.vvm vd, vs2, vs1, v0 # Vector-vector # vd.mask[i] = carry_out(vs2[i] + x[rs1] + v0.mask[i]) vmadc.vxm vd, vs2, rs1, v0 # Vector-scalar # vd.mask[i] = carry_out(vs2[i] + imm + v0.mask[i]) vmadc.vim vd, vs2, imm, v0 # Vector-immediate # vd.mask[i] = carry_out(vs2[i] + vs1[i]) vmadc.vv vd, vs2, vs1 # Vector-vector, no carry-in # vd.mask[i] = carry_out(vs2[i] + x[rs1]) vmadc.vx vd, vs2, rs1 # Vector-scalar, no carry-in # vd.mask[i] = carry_out(vs2[i] + imm) vmadc.vi vd, vs2, imm # Vector-immediate, no carry-in</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vadc and vsbc add or subtract the source operands and the carry-in or borrow-in, and write the result to vector register vd. These instructions are encoded as masked instructions (vm=0), but they operate on and write back all body elements. Encodings corresponding to the unmasked versions (vm=1) are reserved.\nFor vadc and vsbc, the instruction encoding is reserved if the destination vector register is v0.\n# Produce sum with carry. # vd[i] = vs2[i] + vs1[i] + v0.mask[i] vadc.vvm vd, vs2, vs1, v0 # Vector-vector # vd[i] = vs2[i] + x[rs1] + v0.mask[i] vadc.vxm vd, vs2, rs1, v0 # Vector-scalar # vd[i] = vs2[i] + imm + v0.mask[i] vadc.vim vd, vs2, imm, v0 # Vector-immediate # Produce carry out in mask register format # vd.mask[i] = carry_out(vs2[i] + vs1[i] + v0.mask[i]) vmadc.vvm vd, vs2, vs1, v0 # Vector-vector # vd.mask[i] = carry_out(vs2[i] + x[rs1] + v0.mask[i]) vmadc.vxm vd, vs2, rs1, v0 # Vector-scalar # vd.mask[i] = carry_out(vs2[i] + imm + v0.mask[i]) vmadc.vim vd, vs2, imm, v0 # Vector-immediate # vd.mask[i] = carry_out(vs2[i] + vs1[i]) vmadc.vv vd, vs2, vs1 # Vector-vector, no carry-in # vd.mask[i] = carry_out(vs2[i] + x[rs1]) vmadc.vx vd, vs2, rs1 # Vector-scalar, no carry-in # vd.mask[i] = carry_out(vs2[i] + imm) vmadc.vi vd, vs2, imm # Vector-immediate, no carry-in\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VADC.VVM":
            return {
                "html": "<div><span class=\"opcode\"><b>VADC.VVM</b> vs2, vs1, vd</span><br><div>vadc and vsbc add or subtract the source operands and the carry-in or borrow-in, and write the result to vector register vd. These instructions are encoded as masked instructions (vm=0), but they operate on and write back all body elements. Encodings corresponding to the unmasked versions (vm=1) are reserved.<br>For vadc and vsbc, the instruction encoding is reserved if the destination vector register is v0.<br># Produce sum with carry. # vd[i] = vs2[i] + vs1[i] + v0.mask[i] <b>vadc.vvm</b> vd, vs2, vs1, v0 # Vector-vector # vd[i] = vs2[i] + x[rs1] + v0.mask[i] vadc.vxm vd, vs2, rs1, v0 # Vector-scalar # vd[i] = vs2[i] + imm + v0.mask[i] vadc.vim vd, vs2, imm, v0 # Vector-immediate # Produce carry out in mask register format # vd.mask[i] = carry_out(vs2[i] + vs1[i] + v0.mask[i]) vmadc.vvm vd, vs2, vs1, v0 # Vector-vector # vd.mask[i] = carry_out(vs2[i] + x[rs1] + v0.mask[i]) vmadc.vxm vd, vs2, rs1, v0 # Vector-scalar # vd.mask[i] = carry_out(vs2[i] + imm + v0.mask[i]) vmadc.vim vd, vs2, imm, v0 # Vector-immediate # vd.mask[i] = carry_out(vs2[i] + vs1[i]) vmadc.vv vd, vs2, vs1 # Vector-vector, no carry-in # vd.mask[i] = carry_out(vs2[i] + x[rs1]) vmadc.vx vd, vs2, rs1 # Vector-scalar, no carry-in # vd.mask[i] = carry_out(vs2[i] + imm) vmadc.vi vd, vs2, imm # Vector-immediate, no carry-in</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vadc and vsbc add or subtract the source operands and the carry-in or borrow-in, and write the result to vector register vd. These instructions are encoded as masked instructions (vm=0), but they operate on and write back all body elements. Encodings corresponding to the unmasked versions (vm=1) are reserved.\nFor vadc and vsbc, the instruction encoding is reserved if the destination vector register is v0.\n# Produce sum with carry. # vd[i] = vs2[i] + vs1[i] + v0.mask[i] vadc.vvm vd, vs2, vs1, v0 # Vector-vector # vd[i] = vs2[i] + x[rs1] + v0.mask[i] vadc.vxm vd, vs2, rs1, v0 # Vector-scalar # vd[i] = vs2[i] + imm + v0.mask[i] vadc.vim vd, vs2, imm, v0 # Vector-immediate # Produce carry out in mask register format # vd.mask[i] = carry_out(vs2[i] + vs1[i] + v0.mask[i]) vmadc.vvm vd, vs2, vs1, v0 # Vector-vector # vd.mask[i] = carry_out(vs2[i] + x[rs1] + v0.mask[i]) vmadc.vxm vd, vs2, rs1, v0 # Vector-scalar # vd.mask[i] = carry_out(vs2[i] + imm + v0.mask[i]) vmadc.vim vd, vs2, imm, v0 # Vector-immediate # vd.mask[i] = carry_out(vs2[i] + vs1[i]) vmadc.vv vd, vs2, vs1 # Vector-vector, no carry-in # vd.mask[i] = carry_out(vs2[i] + x[rs1]) vmadc.vx vd, vs2, rs1 # Vector-scalar, no carry-in # vd.mask[i] = carry_out(vs2[i] + imm) vmadc.vi vd, vs2, imm # Vector-immediate, no carry-in\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VADC.VXM":
            return {
                "html": "<div><span class=\"opcode\"><b>VADC.VXM</b> vs2, rs1, vd</span><br><div>vadc and vsbc add or subtract the source operands and the carry-in or borrow-in, and write the result to vector register vd. These instructions are encoded as masked instructions (vm=0), but they operate on and write back all body elements. Encodings corresponding to the unmasked versions (vm=1) are reserved.<br>For vadc and vsbc, the instruction encoding is reserved if the destination vector register is v0.<br># Produce sum with carry. # vd[i] = vs2[i] + vs1[i] + v0.mask[i] vadc.vvm vd, vs2, vs1, v0 # Vector-vector # vd[i] = vs2[i] + x[rs1] + v0.mask[i] <b>vadc.vxm</b> vd, vs2, rs1, v0 # Vector-scalar # vd[i] = vs2[i] + imm + v0.mask[i] vadc.vim vd, vs2, imm, v0 # Vector-immediate # Produce carry out in mask register format # vd.mask[i] = carry_out(vs2[i] + vs1[i] + v0.mask[i]) vmadc.vvm vd, vs2, vs1, v0 # Vector-vector # vd.mask[i] = carry_out(vs2[i] + x[rs1] + v0.mask[i]) vmadc.vxm vd, vs2, rs1, v0 # Vector-scalar # vd.mask[i] = carry_out(vs2[i] + imm + v0.mask[i]) vmadc.vim vd, vs2, imm, v0 # Vector-immediate # vd.mask[i] = carry_out(vs2[i] + vs1[i]) vmadc.vv vd, vs2, vs1 # Vector-vector, no carry-in # vd.mask[i] = carry_out(vs2[i] + x[rs1]) vmadc.vx vd, vs2, rs1 # Vector-scalar, no carry-in # vd.mask[i] = carry_out(vs2[i] + imm) vmadc.vi vd, vs2, imm # Vector-immediate, no carry-in</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vadc and vsbc add or subtract the source operands and the carry-in or borrow-in, and write the result to vector register vd. These instructions are encoded as masked instructions (vm=0), but they operate on and write back all body elements. Encodings corresponding to the unmasked versions (vm=1) are reserved.\nFor vadc and vsbc, the instruction encoding is reserved if the destination vector register is v0.\n# Produce sum with carry. # vd[i] = vs2[i] + vs1[i] + v0.mask[i] vadc.vvm vd, vs2, vs1, v0 # Vector-vector # vd[i] = vs2[i] + x[rs1] + v0.mask[i] vadc.vxm vd, vs2, rs1, v0 # Vector-scalar # vd[i] = vs2[i] + imm + v0.mask[i] vadc.vim vd, vs2, imm, v0 # Vector-immediate # Produce carry out in mask register format # vd.mask[i] = carry_out(vs2[i] + vs1[i] + v0.mask[i]) vmadc.vvm vd, vs2, vs1, v0 # Vector-vector # vd.mask[i] = carry_out(vs2[i] + x[rs1] + v0.mask[i]) vmadc.vxm vd, vs2, rs1, v0 # Vector-scalar # vd.mask[i] = carry_out(vs2[i] + imm + v0.mask[i]) vmadc.vim vd, vs2, imm, v0 # Vector-immediate # vd.mask[i] = carry_out(vs2[i] + vs1[i]) vmadc.vv vd, vs2, vs1 # Vector-vector, no carry-in # vd.mask[i] = carry_out(vs2[i] + x[rs1]) vmadc.vx vd, vs2, rs1 # Vector-scalar, no carry-in # vd.mask[i] = carry_out(vs2[i] + imm) vmadc.vi vd, vs2, imm # Vector-immediate, no carry-in\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VADD.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VADD.VI</b> vs2, simm5, vd</span><br><div># Integer adds. vadd.vv vd, vs2, vs1, vm # Vector-vector vadd.vx vd, vs2, rs1, vm # vector-scalar <b>vadd.vi</b> vd, vs2, imm, vm # vector-immediate # Integer subtract vsub.vv vd, vs2, vs1, vm # Vector-vector vsub.vx vd, vs2, rs1, vm # vector-scalar # Integer reverse subtract vrsub.vx vd, vs2, rs1, vm # vd[i] = x[rs1] - vs2[i] vrsub.vi vd, vs2, imm, vm # vd[i] = imm - vs2[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Integer adds. vadd.vv vd, vs2, vs1, vm # Vector-vector vadd.vx vd, vs2, rs1, vm # vector-scalar vadd.vi vd, vs2, imm, vm # vector-immediate # Integer subtract vsub.vv vd, vs2, vs1, vm # Vector-vector vsub.vx vd, vs2, rs1, vm # vector-scalar # Integer reverse subtract vrsub.vx vd, vs2, rs1, vm # vd[i] = x[rs1] - vs2[i] vrsub.vi vd, vs2, imm, vm # vd[i] = imm - vs2[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_integer_add_and_subtract"
            };

        case "VADD.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VADD.VV</b> vs2, vs1, vd</span><br><div># Integer adds. <b>vadd.vv</b> vd, vs2, vs1, vm # Vector-vector vadd.vx vd, vs2, rs1, vm # vector-scalar vadd.vi vd, vs2, imm, vm # vector-immediate # Integer subtract vsub.vv vd, vs2, vs1, vm # Vector-vector vsub.vx vd, vs2, rs1, vm # vector-scalar # Integer reverse subtract vrsub.vx vd, vs2, rs1, vm # vd[i] = x[rs1] - vs2[i] vrsub.vi vd, vs2, imm, vm # vd[i] = imm - vs2[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Integer adds. vadd.vv vd, vs2, vs1, vm # Vector-vector vadd.vx vd, vs2, rs1, vm # vector-scalar vadd.vi vd, vs2, imm, vm # vector-immediate # Integer subtract vsub.vv vd, vs2, vs1, vm # Vector-vector vsub.vx vd, vs2, rs1, vm # vector-scalar # Integer reverse subtract vrsub.vx vd, vs2, rs1, vm # vd[i] = x[rs1] - vs2[i] vrsub.vi vd, vs2, imm, vm # vd[i] = imm - vs2[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_integer_add_and_subtract"
            };

        case "VADD.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VADD.VX</b> vs2, rs1, vd</span><br><div># Integer adds. vadd.vv vd, vs2, vs1, vm # Vector-vector <b>vadd.vx</b> vd, vs2, rs1, vm # vector-scalar vadd.vi vd, vs2, imm, vm # vector-immediate # Integer subtract vsub.vv vd, vs2, vs1, vm # Vector-vector vsub.vx vd, vs2, rs1, vm # vector-scalar # Integer reverse subtract vrsub.vx vd, vs2, rs1, vm # vd[i] = x[rs1] - vs2[i] vrsub.vi vd, vs2, imm, vm # vd[i] = imm - vs2[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Integer adds. vadd.vv vd, vs2, vs1, vm # Vector-vector vadd.vx vd, vs2, rs1, vm # vector-scalar vadd.vi vd, vs2, imm, vm # vector-immediate # Integer subtract vsub.vv vd, vs2, vs1, vm # Vector-vector vsub.vx vd, vs2, rs1, vm # vector-scalar # Integer reverse subtract vrsub.vx vd, vs2, rs1, vm # vd[i] = x[rs1] - vs2[i] vrsub.vi vd, vs2, imm, vm # vd[i] = imm - vs2[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_integer_add_and_subtract"
            };

        case "VAMOADDEI16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOADDEI16.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOADDEI32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOADDEI32.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOADDEI64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOADDEI64.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOADDEI8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOADDEI8.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOANDEI16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOANDEI16.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOANDEI32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOANDEI32.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOANDEI64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOANDEI64.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOANDEI8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOANDEI8.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOMAXEI16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOMAXEI16.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOMAXEI32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOMAXEI32.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOMAXEI64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOMAXEI64.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOMAXEI8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOMAXEI8.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOMAXUEI16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOMAXUEI16.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOMAXUEI32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOMAXUEI32.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOMAXUEI64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOMAXUEI64.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOMAXUEI8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOMAXUEI8.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOMINEI16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOMINEI16.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOMINEI32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOMINEI32.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOMINEI64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOMINEI64.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOMINEI8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOMINEI8.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOMINUEI16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOMINUEI16.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOMINUEI32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOMINUEI32.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOMINUEI64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOMINUEI64.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOMINUEI8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOMINUEI8.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOOREI16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOOREI16.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOOREI32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOOREI32.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOOREI64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOOREI64.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOOREI8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOOREI8.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOSWAPEI16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOSWAPEI16.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOSWAPEI32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOSWAPEI32.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOSWAPEI64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOSWAPEI64.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOSWAPEI8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOSWAPEI8.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOXOREI16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOXOREI16.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOXOREI32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOXOREI32.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOXOREI64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOXOREI64.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAMOXOREI8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VAMOXOREI8.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VAND.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VAND.VI</b> vs2, simm5, vd</span><br><div># Bitwise logical operations. vand.vv vd, vs2, vs1, vm # Vector-vector vand.vx vd, vs2, rs1, vm # vector-scalar <b>vand.vi</b> vd, vs2, imm, vm # vector-immediate vor.vv vd, vs2, vs1, vm # Vector-vector vor.vx vd, vs2, rs1, vm # vector-scalar vor.vi vd, vs2, imm, vm # vector-immediate vxor.vv vd, vs2, vs1, vm # Vector-vector vxor.vx vd, vs2, rs1, vm # vector-scalar vxor.vi vd, vs2, imm, vm # vector-immediate</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Bitwise logical operations. vand.vv vd, vs2, vs1, vm # Vector-vector vand.vx vd, vs2, rs1, vm # vector-scalar vand.vi vd, vs2, imm, vm # vector-immediate vor.vv vd, vs2, vs1, vm # Vector-vector vor.vx vd, vs2, rs1, vm # vector-scalar vor.vi vd, vs2, imm, vm # vector-immediate vxor.vv vd, vs2, vs1, vm # Vector-vector vxor.vx vd, vs2, rs1, vm # vector-scalar vxor.vi vd, vs2, imm, vm # vector-immediate\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_bitwise_logical_instructions"
            };

        case "VAND.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VAND.VV</b> vs2, vs1, vd</span><br><div># Bitwise logical operations. <b>vand.vv</b> vd, vs2, vs1, vm # Vector-vector vand.vx vd, vs2, rs1, vm # vector-scalar vand.vi vd, vs2, imm, vm # vector-immediate vor.vv vd, vs2, vs1, vm # Vector-vector vor.vx vd, vs2, rs1, vm # vector-scalar vor.vi vd, vs2, imm, vm # vector-immediate vxor.vv vd, vs2, vs1, vm # Vector-vector vxor.vx vd, vs2, rs1, vm # vector-scalar vxor.vi vd, vs2, imm, vm # vector-immediate</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Bitwise logical operations. vand.vv vd, vs2, vs1, vm # Vector-vector vand.vx vd, vs2, rs1, vm # vector-scalar vand.vi vd, vs2, imm, vm # vector-immediate vor.vv vd, vs2, vs1, vm # Vector-vector vor.vx vd, vs2, rs1, vm # vector-scalar vor.vi vd, vs2, imm, vm # vector-immediate vxor.vv vd, vs2, vs1, vm # Vector-vector vxor.vx vd, vs2, rs1, vm # vector-scalar vxor.vi vd, vs2, imm, vm # vector-immediate\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_bitwise_logical_instructions"
            };

        case "VAND.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VAND.VX</b> vs2, rs1, vd</span><br><div># Bitwise logical operations. vand.vv vd, vs2, vs1, vm # Vector-vector <b>vand.vx</b> vd, vs2, rs1, vm # vector-scalar vand.vi vd, vs2, imm, vm # vector-immediate vor.vv vd, vs2, vs1, vm # Vector-vector vor.vx vd, vs2, rs1, vm # vector-scalar vor.vi vd, vs2, imm, vm # vector-immediate vxor.vv vd, vs2, vs1, vm # Vector-vector vxor.vx vd, vs2, rs1, vm # vector-scalar vxor.vi vd, vs2, imm, vm # vector-immediate</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Bitwise logical operations. vand.vv vd, vs2, vs1, vm # Vector-vector vand.vx vd, vs2, rs1, vm # vector-scalar vand.vi vd, vs2, imm, vm # vector-immediate vor.vv vd, vs2, vs1, vm # Vector-vector vor.vx vd, vs2, rs1, vm # vector-scalar vor.vi vd, vs2, imm, vm # vector-immediate vxor.vv vd, vs2, vs1, vm # Vector-vector vxor.vx vd, vs2, rs1, vm # vector-scalar vxor.vi vd, vs2, imm, vm # vector-immediate\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_bitwise_logical_instructions"
            };

        case "VASUB.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VASUB.VV</b> vs2, vs1, vd</span><br><div>vasub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vasub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VASUB.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VASUB.VX</b> vs2, rs1, vd</span><br><div>vasub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vasub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VASUBU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VASUBU.VV</b> vs2, vs1, vd</span><br><div>vasubu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vasubu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VASUBU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VASUBU.VX</b> vs2, rs1, vd</span><br><div>vasubu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vasubu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VCOMPRESS.VM":
            return {
                "html": "<div><span class=\"opcode\"><b>VCOMPRESS.VM</b> vs2, vs1, vd</span><br><div>vcompress is encoded as an unmasked instruction (vm=1). The equivalent masked instruction (vm=0) is reserved.<br>A trap on a vcompress instruction is always reported with a vstart of 0. Executing a vcompress instruction with a non-zero vstart raises an illegal instruction exception.<br><b>vcompress.vm</b> vd, vs2, vs1 # Compress into vd elements of vs2 where vs1 is enabled<br>Example use of vcompress instruction 8 7 6 5 4 3 2 1 0 Element number 1 1 0 1 0 0 1 0 1 v0 8 7 6 5 4 3 2 1 0 v1 1 2 3 4 5 6 7 8 9 v2 <b>vcompress.vm</b> v2, v1, v0 1 2 3 4 8 7 5 2 0 v2</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vcompress is encoded as an unmasked instruction (vm=1). The equivalent masked instruction (vm=0) is reserved.\nA trap on a vcompress instruction is always reported with a vstart of 0. Executing a vcompress instruction with a non-zero vstart raises an illegal instruction exception.\nvcompress.vm vd, vs2, vs1 # Compress into vd elements of vs2 where vs1 is enabled\nExample use of vcompress instruction 8 7 6 5 4 3 2 1 0 Element number 1 1 0 1 0 0 1 0 1 v0 8 7 6 5 4 3 2 1 0 v1 1 2 3 4 5 6 7 8 9 v2 vcompress.vm v2, v1, v0 1 2 3 4 8 7 5 2 0 v2\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_compress_instruction"
            };

        case "VCPOP.M":
            return {
                "html": "<div><span class=\"opcode\"><b>VCPOP.M</b> vs2, rd</span><br><div>The <b>vcpop.m</b> instruction counts the number of mask elements of the active elements of the vector source mask register that have the value 1 and writes the result to a scalar x register.<br>The <b>vcpop.m</b> instruction writes x[rd] even if vl=0 (with the value 0, since no mask elements are active).<br>Traps on <b>vcpop.m</b> are always reported with a vstart of 0. The <b>vcpop.m</b> instruction will raise an illegal instruction exception if vstart is non-zero.<br><b>vcpop.m</b> rd, vs2, vm<br><b>vcpop.m</b> rd, vs2, v0.t # x[rd] = sum_i ( vs2.mask[i] && v0.mask[i] )</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vcpop.m instruction counts the number of mask elements of the active elements of the vector source mask register that have the value 1 and writes the result to a scalar x register.\nThe vcpop.m instruction writes x[rd] even if vl=0 (with the value 0, since no mask elements are active).\nTraps on vcpop.m are always reported with a vstart of 0. The vcpop.m instruction will raise an illegal instruction exception if vstart is non-zero.\nvcpop.m rd, vs2, vm\nvcpop.m rd, vs2, v0.t # x[rd] = sum_i ( vs2.mask[i] && v0.mask[i] )\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_count_population_in_mask_vcpop_m"
            };

        case "VDIV.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VDIV.VV</b> vs2, vs1, vd</span><br><div>vdiv</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vdiv\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VDIV.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VDIV.VX</b> vs2, rs1, vd</span><br><div>vdiv</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vdiv\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VDIVU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VDIVU.VV</b> vs2, vs1, vd</span><br><div># Unsigned divide. <b>vdivu.vv</b> vd, vs2, vs1, vm # Vector-vector vdivu.vx vd, vs2, rs1, vm # vector-scalar # Signed divide vdiv.vv vd, vs2, vs1, vm # Vector-vector vdiv.vx vd, vs2, rs1, vm # vector-scalar # Unsigned remainder vremu.vv vd, vs2, vs1, vm # Vector-vector vremu.vx vd, vs2, rs1, vm # vector-scalar # Signed remainder vrem.vv vd, vs2, vs1, vm # Vector-vector vrem.vx vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Unsigned divide. vdivu.vv vd, vs2, vs1, vm # Vector-vector vdivu.vx vd, vs2, rs1, vm # vector-scalar # Signed divide vdiv.vv vd, vs2, vs1, vm # Vector-vector vdiv.vx vd, vs2, rs1, vm # vector-scalar # Unsigned remainder vremu.vv vd, vs2, vs1, vm # Vector-vector vremu.vx vd, vs2, rs1, vm # vector-scalar # Signed remainder vrem.vv vd, vs2, vs1, vm # Vector-vector vrem.vx vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_divide_instructions"
            };

        case "VDIVU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VDIVU.VX</b> vs2, rs1, vd</span><br><div># Unsigned divide. vdivu.vv vd, vs2, vs1, vm # Vector-vector <b>vdivu.vx</b> vd, vs2, rs1, vm # vector-scalar # Signed divide vdiv.vv vd, vs2, vs1, vm # Vector-vector vdiv.vx vd, vs2, rs1, vm # vector-scalar # Unsigned remainder vremu.vv vd, vs2, vs1, vm # Vector-vector vremu.vx vd, vs2, rs1, vm # vector-scalar # Signed remainder vrem.vv vd, vs2, vs1, vm # Vector-vector vrem.vx vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Unsigned divide. vdivu.vv vd, vs2, vs1, vm # Vector-vector vdivu.vx vd, vs2, rs1, vm # vector-scalar # Signed divide vdiv.vv vd, vs2, vs1, vm # Vector-vector vdiv.vx vd, vs2, rs1, vm # vector-scalar # Unsigned remainder vremu.vv vd, vs2, vs1, vm # Vector-vector vremu.vx vd, vs2, rs1, vm # vector-scalar # Signed remainder vrem.vv vd, vs2, vs1, vm # Vector-vector vrem.vx vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_divide_instructions"
            };

        case "VFADD.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFADD.VF</b> vs2, rs1, vd</span><br><div># Floating-point add vfadd.vv vd, vs2, vs1, vm # Vector-vector <b>vfadd.vf</b> vd, vs2, rs1, vm # vector-scalar # Floating-point subtract vfsub.vv vd, vs2, vs1, vm # Vector-vector vfsub.vf vd, vs2, rs1, vm # Vector-scalar vd[i] = vs2[i] - f[rs1] vfrsub.vf vd, vs2, rs1, vm # Scalar-vector vd[i] = f[rs1] - vs2[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Floating-point add vfadd.vv vd, vs2, vs1, vm # Vector-vector vfadd.vf vd, vs2, rs1, vm # vector-scalar # Floating-point subtract vfsub.vv vd, vs2, vs1, vm # Vector-vector vfsub.vf vd, vs2, rs1, vm # Vector-scalar vd[i] = vs2[i] - f[rs1] vfrsub.vf vd, vs2, rs1, vm # Scalar-vector vd[i] = f[rs1] - vs2[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_floating_point_addsubtract_instructions"
            };

        case "VFADD.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFADD.VV</b> vs2, vs1, vd</span><br><div># Floating-point add <b>vfadd.vv</b> vd, vs2, vs1, vm # Vector-vector vfadd.vf vd, vs2, rs1, vm # vector-scalar # Floating-point subtract vfsub.vv vd, vs2, vs1, vm # Vector-vector vfsub.vf vd, vs2, rs1, vm # Vector-scalar vd[i] = vs2[i] - f[rs1] vfrsub.vf vd, vs2, rs1, vm # Scalar-vector vd[i] = f[rs1] - vs2[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Floating-point add vfadd.vv vd, vs2, vs1, vm # Vector-vector vfadd.vf vd, vs2, rs1, vm # vector-scalar # Floating-point subtract vfsub.vv vd, vs2, vs1, vm # Vector-vector vfsub.vf vd, vs2, rs1, vm # Vector-scalar vd[i] = vs2[i] - f[rs1] vfrsub.vf vd, vs2, rs1, vm # Scalar-vector vd[i] = f[rs1] - vs2[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_floating_point_addsubtract_instructions"
            };

        case "VFCLASS.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFCLASS.V</b> vs2, vd</span><br><div><b>vfclass.v</b> vd, vs2, vm # Vector-vector</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfclass.v vd, vs2, vm # Vector-vector\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_floating_point_classify_instruction"
            };

        case "VFCVT.F.X.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFCVT.F.X.V</b> vs2, vd</span><br><div>vfcvt.xu.f.v vd, vs2, vm # Convert float to unsigned integer. vfcvt.x.f.v vd, vs2, vm # Convert float to signed integer. vfcvt.rtz.xu.f.v vd, vs2, vm # Convert float to unsigned integer, truncating. vfcvt.rtz.x.f.v vd, vs2, vm # Convert float to signed integer, truncating. vfcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to float. <b>vfcvt.f.x.v</b> vd, vs2, vm # Convert signed integer to float.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfcvt.xu.f.v vd, vs2, vm # Convert float to unsigned integer. vfcvt.x.f.v vd, vs2, vm # Convert float to signed integer. vfcvt.rtz.xu.f.v vd, vs2, vm # Convert float to unsigned integer, truncating. vfcvt.rtz.x.f.v vd, vs2, vm # Convert float to signed integer, truncating. vfcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to float. vfcvt.f.x.v vd, vs2, vm # Convert signed integer to float.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_single_width_floating_pointinteger_type_convert_instructions"
            };

        case "VFCVT.F.XU.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFCVT.F.XU.V</b> vs2, vd</span><br><div>vfcvt.xu.f.v vd, vs2, vm # Convert float to unsigned integer. vfcvt.x.f.v vd, vs2, vm # Convert float to signed integer. vfcvt.rtz.xu.f.v vd, vs2, vm # Convert float to unsigned integer, truncating. vfcvt.rtz.x.f.v vd, vs2, vm # Convert float to signed integer, truncating. <b>vfcvt.f.xu.v</b> vd, vs2, vm # Convert unsigned integer to float. vfcvt.f.x.v vd, vs2, vm # Convert signed integer to float.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfcvt.xu.f.v vd, vs2, vm # Convert float to unsigned integer. vfcvt.x.f.v vd, vs2, vm # Convert float to signed integer. vfcvt.rtz.xu.f.v vd, vs2, vm # Convert float to unsigned integer, truncating. vfcvt.rtz.x.f.v vd, vs2, vm # Convert float to signed integer, truncating. vfcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to float. vfcvt.f.x.v vd, vs2, vm # Convert signed integer to float.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_single_width_floating_pointinteger_type_convert_instructions"
            };

        case "VFCVT.RTZ.X.F.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFCVT.RTZ.X.F.V</b> vs2, vd</span><br><div>vfcvt.xu.f.v vd, vs2, vm # Convert float to unsigned integer. vfcvt.x.f.v vd, vs2, vm # Convert float to signed integer. vfcvt.rtz.xu.f.v vd, vs2, vm # Convert float to unsigned integer, truncating. <b>vfcvt.rtz.x.f.v</b> vd, vs2, vm # Convert float to signed integer, truncating. vfcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to float. vfcvt.f.x.v vd, vs2, vm # Convert signed integer to float.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfcvt.xu.f.v vd, vs2, vm # Convert float to unsigned integer. vfcvt.x.f.v vd, vs2, vm # Convert float to signed integer. vfcvt.rtz.xu.f.v vd, vs2, vm # Convert float to unsigned integer, truncating. vfcvt.rtz.x.f.v vd, vs2, vm # Convert float to signed integer, truncating. vfcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to float. vfcvt.f.x.v vd, vs2, vm # Convert signed integer to float.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_single_width_floating_pointinteger_type_convert_instructions"
            };

        case "VFCVT.RTZ.XU.F.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFCVT.RTZ.XU.F.V</b> vs2, vd</span><br><div>vfcvt.xu.f.v vd, vs2, vm # Convert float to unsigned integer. vfcvt.x.f.v vd, vs2, vm # Convert float to signed integer. <b>vfcvt.rtz.xu.f.v</b> vd, vs2, vm # Convert float to unsigned integer, truncating. vfcvt.rtz.x.f.v vd, vs2, vm # Convert float to signed integer, truncating. vfcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to float. vfcvt.f.x.v vd, vs2, vm # Convert signed integer to float.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfcvt.xu.f.v vd, vs2, vm # Convert float to unsigned integer. vfcvt.x.f.v vd, vs2, vm # Convert float to signed integer. vfcvt.rtz.xu.f.v vd, vs2, vm # Convert float to unsigned integer, truncating. vfcvt.rtz.x.f.v vd, vs2, vm # Convert float to signed integer, truncating. vfcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to float. vfcvt.f.x.v vd, vs2, vm # Convert signed integer to float.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_single_width_floating_pointinteger_type_convert_instructions"
            };

        case "VFCVT.X.F.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFCVT.X.F.V</b> vs2, vd</span><br><div>vfcvt.xu.f.v vd, vs2, vm # Convert float to unsigned integer. <b>vfcvt.x.f.v</b> vd, vs2, vm # Convert float to signed integer. vfcvt.rtz.xu.f.v vd, vs2, vm # Convert float to unsigned integer, truncating. vfcvt.rtz.x.f.v vd, vs2, vm # Convert float to signed integer, truncating. vfcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to float. vfcvt.f.x.v vd, vs2, vm # Convert signed integer to float.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfcvt.xu.f.v vd, vs2, vm # Convert float to unsigned integer. vfcvt.x.f.v vd, vs2, vm # Convert float to signed integer. vfcvt.rtz.xu.f.v vd, vs2, vm # Convert float to unsigned integer, truncating. vfcvt.rtz.x.f.v vd, vs2, vm # Convert float to signed integer, truncating. vfcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to float. vfcvt.f.x.v vd, vs2, vm # Convert signed integer to float.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_single_width_floating_pointinteger_type_convert_instructions"
            };

        case "VFCVT.XU.F.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFCVT.XU.F.V</b> vs2, vd</span><br><div><b>vfcvt.xu.f.v</b> vd, vs2, vm # Convert float to unsigned integer. vfcvt.x.f.v vd, vs2, vm # Convert float to signed integer. vfcvt.rtz.xu.f.v vd, vs2, vm # Convert float to unsigned integer, truncating. vfcvt.rtz.x.f.v vd, vs2, vm # Convert float to signed integer, truncating. vfcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to float. vfcvt.f.x.v vd, vs2, vm # Convert signed integer to float.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfcvt.xu.f.v vd, vs2, vm # Convert float to unsigned integer. vfcvt.x.f.v vd, vs2, vm # Convert float to signed integer. vfcvt.rtz.xu.f.v vd, vs2, vm # Convert float to unsigned integer, truncating. vfcvt.rtz.x.f.v vd, vs2, vm # Convert float to signed integer, truncating. vfcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to float. vfcvt.f.x.v vd, vs2, vm # Convert signed integer to float.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_single_width_floating_pointinteger_type_convert_instructions"
            };

        case "VFDIV.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFDIV.VF</b> vs2, rs1, vd</span><br><div>vfdiv</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfdiv\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFDIV.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFDIV.VV</b> vs2, vs1, vd</span><br><div>vfdiv</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfdiv\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFIRST.M":
            return {
                "html": "<div><span class=\"opcode\"><b>VFIRST.M</b> vs2, rd</span><br><div>The vfirst instruction finds the lowest-numbered active element of the source mask vector that has the value 1 and writes that element's index to a GPR. If no active element has the value 1, -1 is written to the GPR.<br>The <b>vfirst.m</b> instruction writes x[rd] even if vl=0 (with the value -1, since no mask elements are active).<br>Traps on vfirst are always reported with a vstart of 0. The vfirst instruction will raise an illegal instruction exception if vstart is non-zero.<br><b>vfirst.m</b> rd, vs2, vm</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vfirst instruction finds the lowest-numbered active element of the source mask vector that has the value 1 and writes that element's index to a GPR. If no active element has the value 1, -1 is written to the GPR.\nThe vfirst.m instruction writes x[rd] even if vl=0 (with the value -1, since no mask elements are active).\nTraps on vfirst are always reported with a vstart of 0. The vfirst instruction will raise an illegal instruction exception if vstart is non-zero.\nvfirst.m rd, vs2, vm\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vfirst_find_first_set_mask_bit"
            };

        case "VFMACC.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMACC.VF</b> vs2, rs1, vd</span><br><div># FP multiply-accumulate, overwrites addend vfmacc.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] <b>vfmacc.vf</b> vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vs2[i]) + vd[i] # FP negate-(multiply-accumulate), overwrites subtrahend vfnmacc.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) - vd[i] vfnmacc.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vs2[i]) - vd[i] # FP multiply-subtract-accumulator, overwrites subtrahend vfmsac.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) - vd[i] vfmsac.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vs2[i]) - vd[i] # FP negate-(multiply-subtract-accumulator), overwrites minuend vfnmsac.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) + vd[i] vfnmsac.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vs2[i]) + vd[i] # FP multiply-add, overwrites multiplicand vfmadd.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vd[i]) + vs2[i] vfmadd.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vd[i]) + vs2[i] # FP negate-(multiply-add), overwrites multiplicand vfnmadd.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vd[i]) - vs2[i] vfnmadd.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vd[i]) - vs2[i] # FP multiply-sub, overwrites multiplicand vfmsub.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vd[i]) - vs2[i] vfmsub.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vd[i]) - vs2[i] # FP negate-(multiply-sub), overwrites multiplicand vfnmsub.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vd[i]) + vs2[i] vfnmsub.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vd[i]) + vs2[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# FP multiply-accumulate, overwrites addend vfmacc.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] vfmacc.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vs2[i]) + vd[i] # FP negate-(multiply-accumulate), overwrites subtrahend vfnmacc.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) - vd[i] vfnmacc.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vs2[i]) - vd[i] # FP multiply-subtract-accumulator, overwrites subtrahend vfmsac.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) - vd[i] vfmsac.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vs2[i]) - vd[i] # FP negate-(multiply-subtract-accumulator), overwrites minuend vfnmsac.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) + vd[i] vfnmsac.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vs2[i]) + vd[i] # FP multiply-add, overwrites multiplicand vfmadd.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vd[i]) + vs2[i] vfmadd.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vd[i]) + vs2[i] # FP negate-(multiply-add), overwrites multiplicand vfnmadd.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vd[i]) - vs2[i] vfnmadd.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vd[i]) - vs2[i] # FP multiply-sub, overwrites multiplicand vfmsub.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vd[i]) - vs2[i] vfmsub.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vd[i]) - vs2[i] # FP negate-(multiply-sub), overwrites multiplicand vfnmsub.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vd[i]) + vs2[i] vfnmsub.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vd[i]) + vs2[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFMACC.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMACC.VV</b> vs2, vs1, vd</span><br><div># FP multiply-accumulate, overwrites addend <b>vfmacc.vv</b> vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] vfmacc.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vs2[i]) + vd[i] # FP negate-(multiply-accumulate), overwrites subtrahend vfnmacc.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) - vd[i] vfnmacc.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vs2[i]) - vd[i] # FP multiply-subtract-accumulator, overwrites subtrahend vfmsac.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) - vd[i] vfmsac.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vs2[i]) - vd[i] # FP negate-(multiply-subtract-accumulator), overwrites minuend vfnmsac.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) + vd[i] vfnmsac.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vs2[i]) + vd[i] # FP multiply-add, overwrites multiplicand vfmadd.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vd[i]) + vs2[i] vfmadd.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vd[i]) + vs2[i] # FP negate-(multiply-add), overwrites multiplicand vfnmadd.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vd[i]) - vs2[i] vfnmadd.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vd[i]) - vs2[i] # FP multiply-sub, overwrites multiplicand vfmsub.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vd[i]) - vs2[i] vfmsub.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vd[i]) - vs2[i] # FP negate-(multiply-sub), overwrites multiplicand vfnmsub.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vd[i]) + vs2[i] vfnmsub.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vd[i]) + vs2[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# FP multiply-accumulate, overwrites addend vfmacc.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] vfmacc.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vs2[i]) + vd[i] # FP negate-(multiply-accumulate), overwrites subtrahend vfnmacc.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) - vd[i] vfnmacc.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vs2[i]) - vd[i] # FP multiply-subtract-accumulator, overwrites subtrahend vfmsac.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) - vd[i] vfmsac.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vs2[i]) - vd[i] # FP negate-(multiply-subtract-accumulator), overwrites minuend vfnmsac.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) + vd[i] vfnmsac.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vs2[i]) + vd[i] # FP multiply-add, overwrites multiplicand vfmadd.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vd[i]) + vs2[i] vfmadd.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vd[i]) + vs2[i] # FP negate-(multiply-add), overwrites multiplicand vfnmadd.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vd[i]) - vs2[i] vfnmadd.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vd[i]) - vs2[i] # FP multiply-sub, overwrites multiplicand vfmsub.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vd[i]) - vs2[i] vfmsub.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vd[i]) - vs2[i] # FP negate-(multiply-sub), overwrites multiplicand vfnmsub.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vd[i]) + vs2[i] vfnmsub.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vd[i]) + vs2[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_floating_point_fused_multiply_add_instructions"
            };

        case "VFMADD.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMADD.VF</b> vs2, rs1, vd</span><br><div>vfmadd</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfmadd\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFMADD.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMADD.VV</b> vs2, vs1, vd</span><br><div>vfmadd</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfmadd\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFMAX.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMAX.VF</b> vs2, rs1, vd</span><br><div>vfmax</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfmax\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFMAX.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMAX.VV</b> vs2, vs1, vd</span><br><div>vfmax</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfmax\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFMERGE.VFM":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMERGE.VFM</b> vs2, rs1, vd</span><br><div>The <b>vfmerge.vfm</b> instruction is encoded as a masked instruction (vm=0). At elements where the mask value is zero, the first vector operand is copied to the destination element, otherwise a scalar floating-point register value is copied to the destination element.<br><b>vfmerge.vfm</b> vd, vs2, rs1, v0 # vd[i] = v0.mask[i] ? f[rs1] : vs2[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vfmerge.vfm instruction is encoded as a masked instruction (vm=0). At elements where the mask value is zero, the first vector operand is copied to the destination element, otherwise a scalar floating-point register value is copied to the destination element.\nvfmerge.vfm vd, vs2, rs1, v0 # vd[i] = v0.mask[i] ? f[rs1] : vs2[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_floating_point_merge_instruction"
            };

        case "VFMIN.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMIN.VF</b> vs2, rs1, vd</span><br><div>The vector floating-point vfmin and vfmax instructions have the same behavior as the corresponding scalar floating-point instructions in version 2.2 of the RISC-V F/D/Q extension.<br># Floating-point minimum vfmin.vv vd, vs2, vs1, vm # Vector-vector <b>vfmin.vf</b> vd, vs2, rs1, vm # vector-scalar # Floating-point maximum vfmax.vv vd, vs2, vs1, vm # Vector-vector vfmax.vf vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vector floating-point vfmin and vfmax instructions have the same behavior as the corresponding scalar floating-point instructions in version 2.2 of the RISC-V F/D/Q extension.\n# Floating-point minimum vfmin.vv vd, vs2, vs1, vm # Vector-vector vfmin.vf vd, vs2, rs1, vm # vector-scalar # Floating-point maximum vfmax.vv vd, vs2, vs1, vm # Vector-vector vfmax.vf vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_floating_point_minmax_instructions"
            };

        case "VFMIN.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMIN.VV</b> vs2, vs1, vd</span><br><div>The vector floating-point vfmin and vfmax instructions have the same behavior as the corresponding scalar floating-point instructions in version 2.2 of the RISC-V F/D/Q extension.<br># Floating-point minimum <b>vfmin.vv</b> vd, vs2, vs1, vm # Vector-vector vfmin.vf vd, vs2, rs1, vm # vector-scalar # Floating-point maximum vfmax.vv vd, vs2, vs1, vm # Vector-vector vfmax.vf vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vector floating-point vfmin and vfmax instructions have the same behavior as the corresponding scalar floating-point instructions in version 2.2 of the RISC-V F/D/Q extension.\n# Floating-point minimum vfmin.vv vd, vs2, vs1, vm # Vector-vector vfmin.vf vd, vs2, rs1, vm # vector-scalar # Floating-point maximum vfmax.vv vd, vs2, vs1, vm # Vector-vector vfmax.vf vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_floating_point_minmax_instructions"
            };

        case "VFMSAC.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMSAC.VF</b> vs2, rs1, vd</span><br><div>vfmsac</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfmsac\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFMSAC.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMSAC.VV</b> vs2, vs1, vd</span><br><div>vfmsac</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfmsac\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFMSUB.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMSUB.VF</b> vs2, rs1, vd</span><br><div>vfmsub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfmsub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFMSUB.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMSUB.VV</b> vs2, vs1, vd</span><br><div>vfmsub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfmsub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFMUL.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMUL.VF</b> vs2, rs1, vd</span><br><div># Floating-point multiply vfmul.vv vd, vs2, vs1, vm # Vector-vector <b>vfmul.vf</b> vd, vs2, rs1, vm # vector-scalar # Floating-point divide vfdiv.vv vd, vs2, vs1, vm # Vector-vector vfdiv.vf vd, vs2, rs1, vm # vector-scalar # Reverse floating-point divide vector = scalar / vector vfrdiv.vf vd, vs2, rs1, vm # scalar-vector, vd[i] = f[rs1]/vs2[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Floating-point multiply vfmul.vv vd, vs2, vs1, vm # Vector-vector vfmul.vf vd, vs2, rs1, vm # vector-scalar # Floating-point divide vfdiv.vv vd, vs2, vs1, vm # Vector-vector vfdiv.vf vd, vs2, rs1, vm # vector-scalar # Reverse floating-point divide vector = scalar / vector vfrdiv.vf vd, vs2, rs1, vm # scalar-vector, vd[i] = f[rs1]/vs2[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_floating_point_multiplydivide_instructions"
            };

        case "VFMUL.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMUL.VV</b> vs2, vs1, vd</span><br><div># Floating-point multiply <b>vfmul.vv</b> vd, vs2, vs1, vm # Vector-vector vfmul.vf vd, vs2, rs1, vm # vector-scalar # Floating-point divide vfdiv.vv vd, vs2, vs1, vm # Vector-vector vfdiv.vf vd, vs2, rs1, vm # vector-scalar # Reverse floating-point divide vector = scalar / vector vfrdiv.vf vd, vs2, rs1, vm # scalar-vector, vd[i] = f[rs1]/vs2[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Floating-point multiply vfmul.vv vd, vs2, vs1, vm # Vector-vector vfmul.vf vd, vs2, rs1, vm # vector-scalar # Floating-point divide vfdiv.vv vd, vs2, vs1, vm # Vector-vector vfdiv.vf vd, vs2, rs1, vm # vector-scalar # Reverse floating-point divide vector = scalar / vector vfrdiv.vf vd, vs2, rs1, vm # scalar-vector, vd[i] = f[rs1]/vs2[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_floating_point_multiplydivide_instructions"
            };

        case "VFMV.F.S":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMV.F.S</b> vs2, rd</span><br><div>vfmv.v.f vd, rs1 # vd[i] = f[rs1]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfmv.v.f vd, rs1 # vd[i] = f[rs1]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_floating_point_move_instruction"
            };

        case "VFMV.S.F":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMV.S.F</b> rs1, vd</span><br><div>vfmv.v.f vd, rs1 # vd[i] = f[rs1]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfmv.v.f vd, rs1 # vd[i] = f[rs1]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_floating_point_move_instruction"
            };

        case "VFMV.V.F":
            return {
                "html": "<div><span class=\"opcode\"><b>VFMV.V.F</b> rs1, vd</span><br><div><b>vfmv.v.f</b> vd, rs1 # vd[i] = f[rs1]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfmv.v.f vd, rs1 # vd[i] = f[rs1]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_floating_point_move_instruction"
            };

        case "VFNCVT.F.F.W":
            return {
                "html": "<div><span class=\"opcode\"><b>VFNCVT.F.F.W</b> vs2, vd</span><br><div>vfncvt.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer. vfncvt.x.f.w vd, vs2, vm # Convert double-width float to signed integer. vfncvt.rtz.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer, truncating. vfncvt.rtz.x.f.w vd, vs2, vm # Convert double-width float to signed integer, truncating. vfncvt.f.xu.w vd, vs2, vm # Convert double-width unsigned integer to float. vfncvt.f.x.w vd, vs2, vm # Convert double-width signed integer to float. <b>vfncvt.f.f.w</b> vd, vs2, vm # Convert double-width float to single-width float. vfncvt.rod.f.f.w vd, vs2, vm # Convert double-width float to single-width float, # rounding towards odd.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfncvt.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer. vfncvt.x.f.w vd, vs2, vm # Convert double-width float to signed integer. vfncvt.rtz.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer, truncating. vfncvt.rtz.x.f.w vd, vs2, vm # Convert double-width float to signed integer, truncating. vfncvt.f.xu.w vd, vs2, vm # Convert double-width unsigned integer to float. vfncvt.f.x.w vd, vs2, vm # Convert double-width signed integer to float. vfncvt.f.f.w vd, vs2, vm # Convert double-width float to single-width float. vfncvt.rod.f.f.w vd, vs2, vm # Convert double-width float to single-width float, # rounding towards odd.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_narrowing_floating_pointinteger_type_convert_instructions"
            };

        case "VFNCVT.F.X.W":
            return {
                "html": "<div><span class=\"opcode\"><b>VFNCVT.F.X.W</b> vs2, vd</span><br><div>vfncvt.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer. vfncvt.x.f.w vd, vs2, vm # Convert double-width float to signed integer. vfncvt.rtz.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer, truncating. vfncvt.rtz.x.f.w vd, vs2, vm # Convert double-width float to signed integer, truncating. vfncvt.f.xu.w vd, vs2, vm # Convert double-width unsigned integer to float. <b>vfncvt.f.x.w</b> vd, vs2, vm # Convert double-width signed integer to float. vfncvt.f.f.w vd, vs2, vm # Convert double-width float to single-width float. vfncvt.rod.f.f.w vd, vs2, vm # Convert double-width float to single-width float, # rounding towards odd.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfncvt.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer. vfncvt.x.f.w vd, vs2, vm # Convert double-width float to signed integer. vfncvt.rtz.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer, truncating. vfncvt.rtz.x.f.w vd, vs2, vm # Convert double-width float to signed integer, truncating. vfncvt.f.xu.w vd, vs2, vm # Convert double-width unsigned integer to float. vfncvt.f.x.w vd, vs2, vm # Convert double-width signed integer to float. vfncvt.f.f.w vd, vs2, vm # Convert double-width float to single-width float. vfncvt.rod.f.f.w vd, vs2, vm # Convert double-width float to single-width float, # rounding towards odd.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_narrowing_floating_pointinteger_type_convert_instructions"
            };

        case "VFNCVT.F.XU.W":
            return {
                "html": "<div><span class=\"opcode\"><b>VFNCVT.F.XU.W</b> vs2, vd</span><br><div>vfncvt.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer. vfncvt.x.f.w vd, vs2, vm # Convert double-width float to signed integer. vfncvt.rtz.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer, truncating. vfncvt.rtz.x.f.w vd, vs2, vm # Convert double-width float to signed integer, truncating. <b>vfncvt.f.xu.w</b> vd, vs2, vm # Convert double-width unsigned integer to float. vfncvt.f.x.w vd, vs2, vm # Convert double-width signed integer to float. vfncvt.f.f.w vd, vs2, vm # Convert double-width float to single-width float. vfncvt.rod.f.f.w vd, vs2, vm # Convert double-width float to single-width float, # rounding towards odd.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfncvt.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer. vfncvt.x.f.w vd, vs2, vm # Convert double-width float to signed integer. vfncvt.rtz.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer, truncating. vfncvt.rtz.x.f.w vd, vs2, vm # Convert double-width float to signed integer, truncating. vfncvt.f.xu.w vd, vs2, vm # Convert double-width unsigned integer to float. vfncvt.f.x.w vd, vs2, vm # Convert double-width signed integer to float. vfncvt.f.f.w vd, vs2, vm # Convert double-width float to single-width float. vfncvt.rod.f.f.w vd, vs2, vm # Convert double-width float to single-width float, # rounding towards odd.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_narrowing_floating_pointinteger_type_convert_instructions"
            };

        case "VFNCVT.ROD.F.F.W":
            return {
                "html": "<div><span class=\"opcode\"><b>VFNCVT.ROD.F.F.W</b> vs2, vd</span><br><div>vfncvt.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer. vfncvt.x.f.w vd, vs2, vm # Convert double-width float to signed integer. vfncvt.rtz.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer, truncating. vfncvt.rtz.x.f.w vd, vs2, vm # Convert double-width float to signed integer, truncating. vfncvt.f.xu.w vd, vs2, vm # Convert double-width unsigned integer to float. vfncvt.f.x.w vd, vs2, vm # Convert double-width signed integer to float. vfncvt.f.f.w vd, vs2, vm # Convert double-width float to single-width float. <b>vfncvt.rod.f.f.w</b> vd, vs2, vm # Convert double-width float to single-width float, # rounding towards odd.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfncvt.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer. vfncvt.x.f.w vd, vs2, vm # Convert double-width float to signed integer. vfncvt.rtz.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer, truncating. vfncvt.rtz.x.f.w vd, vs2, vm # Convert double-width float to signed integer, truncating. vfncvt.f.xu.w vd, vs2, vm # Convert double-width unsigned integer to float. vfncvt.f.x.w vd, vs2, vm # Convert double-width signed integer to float. vfncvt.f.f.w vd, vs2, vm # Convert double-width float to single-width float. vfncvt.rod.f.f.w vd, vs2, vm # Convert double-width float to single-width float, # rounding towards odd.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_narrowing_floating_pointinteger_type_convert_instructions"
            };

        case "VFNCVT.RTZ.X.F.W":
            return {
                "html": "<div><span class=\"opcode\"><b>VFNCVT.RTZ.X.F.W</b> vs2, vd</span><br><div>vfncvt.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer. vfncvt.x.f.w vd, vs2, vm # Convert double-width float to signed integer. vfncvt.rtz.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer, truncating. <b>vfncvt.rtz.x.f.w</b> vd, vs2, vm # Convert double-width float to signed integer, truncating. vfncvt.f.xu.w vd, vs2, vm # Convert double-width unsigned integer to float. vfncvt.f.x.w vd, vs2, vm # Convert double-width signed integer to float. vfncvt.f.f.w vd, vs2, vm # Convert double-width float to single-width float. vfncvt.rod.f.f.w vd, vs2, vm # Convert double-width float to single-width float, # rounding towards odd.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfncvt.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer. vfncvt.x.f.w vd, vs2, vm # Convert double-width float to signed integer. vfncvt.rtz.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer, truncating. vfncvt.rtz.x.f.w vd, vs2, vm # Convert double-width float to signed integer, truncating. vfncvt.f.xu.w vd, vs2, vm # Convert double-width unsigned integer to float. vfncvt.f.x.w vd, vs2, vm # Convert double-width signed integer to float. vfncvt.f.f.w vd, vs2, vm # Convert double-width float to single-width float. vfncvt.rod.f.f.w vd, vs2, vm # Convert double-width float to single-width float, # rounding towards odd.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_narrowing_floating_pointinteger_type_convert_instructions"
            };

        case "VFNCVT.RTZ.XU.F.W":
            return {
                "html": "<div><span class=\"opcode\"><b>VFNCVT.RTZ.XU.F.W</b> vs2, vd</span><br><div>vfncvt.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer. vfncvt.x.f.w vd, vs2, vm # Convert double-width float to signed integer. <b>vfncvt.rtz.xu.f.w</b> vd, vs2, vm # Convert double-width float to unsigned integer, truncating. vfncvt.rtz.x.f.w vd, vs2, vm # Convert double-width float to signed integer, truncating. vfncvt.f.xu.w vd, vs2, vm # Convert double-width unsigned integer to float. vfncvt.f.x.w vd, vs2, vm # Convert double-width signed integer to float. vfncvt.f.f.w vd, vs2, vm # Convert double-width float to single-width float. vfncvt.rod.f.f.w vd, vs2, vm # Convert double-width float to single-width float, # rounding towards odd.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfncvt.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer. vfncvt.x.f.w vd, vs2, vm # Convert double-width float to signed integer. vfncvt.rtz.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer, truncating. vfncvt.rtz.x.f.w vd, vs2, vm # Convert double-width float to signed integer, truncating. vfncvt.f.xu.w vd, vs2, vm # Convert double-width unsigned integer to float. vfncvt.f.x.w vd, vs2, vm # Convert double-width signed integer to float. vfncvt.f.f.w vd, vs2, vm # Convert double-width float to single-width float. vfncvt.rod.f.f.w vd, vs2, vm # Convert double-width float to single-width float, # rounding towards odd.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_narrowing_floating_pointinteger_type_convert_instructions"
            };

        case "VFNCVT.X.F.W":
            return {
                "html": "<div><span class=\"opcode\"><b>VFNCVT.X.F.W</b> vs2, vd</span><br><div>vfncvt.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer. <b>vfncvt.x.f.w</b> vd, vs2, vm # Convert double-width float to signed integer. vfncvt.rtz.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer, truncating. vfncvt.rtz.x.f.w vd, vs2, vm # Convert double-width float to signed integer, truncating. vfncvt.f.xu.w vd, vs2, vm # Convert double-width unsigned integer to float. vfncvt.f.x.w vd, vs2, vm # Convert double-width signed integer to float. vfncvt.f.f.w vd, vs2, vm # Convert double-width float to single-width float. vfncvt.rod.f.f.w vd, vs2, vm # Convert double-width float to single-width float, # rounding towards odd.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfncvt.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer. vfncvt.x.f.w vd, vs2, vm # Convert double-width float to signed integer. vfncvt.rtz.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer, truncating. vfncvt.rtz.x.f.w vd, vs2, vm # Convert double-width float to signed integer, truncating. vfncvt.f.xu.w vd, vs2, vm # Convert double-width unsigned integer to float. vfncvt.f.x.w vd, vs2, vm # Convert double-width signed integer to float. vfncvt.f.f.w vd, vs2, vm # Convert double-width float to single-width float. vfncvt.rod.f.f.w vd, vs2, vm # Convert double-width float to single-width float, # rounding towards odd.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_narrowing_floating_pointinteger_type_convert_instructions"
            };

        case "VFNCVT.XU.F.W":
            return {
                "html": "<div><span class=\"opcode\"><b>VFNCVT.XU.F.W</b> vs2, vd</span><br><div><b>vfncvt.xu.f.w</b> vd, vs2, vm # Convert double-width float to unsigned integer. vfncvt.x.f.w vd, vs2, vm # Convert double-width float to signed integer. vfncvt.rtz.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer, truncating. vfncvt.rtz.x.f.w vd, vs2, vm # Convert double-width float to signed integer, truncating. vfncvt.f.xu.w vd, vs2, vm # Convert double-width unsigned integer to float. vfncvt.f.x.w vd, vs2, vm # Convert double-width signed integer to float. vfncvt.f.f.w vd, vs2, vm # Convert double-width float to single-width float. vfncvt.rod.f.f.w vd, vs2, vm # Convert double-width float to single-width float, # rounding towards odd.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfncvt.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer. vfncvt.x.f.w vd, vs2, vm # Convert double-width float to signed integer. vfncvt.rtz.xu.f.w vd, vs2, vm # Convert double-width float to unsigned integer, truncating. vfncvt.rtz.x.f.w vd, vs2, vm # Convert double-width float to signed integer, truncating. vfncvt.f.xu.w vd, vs2, vm # Convert double-width unsigned integer to float. vfncvt.f.x.w vd, vs2, vm # Convert double-width signed integer to float. vfncvt.f.f.w vd, vs2, vm # Convert double-width float to single-width float. vfncvt.rod.f.f.w vd, vs2, vm # Convert double-width float to single-width float, # rounding towards odd.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_narrowing_floating_pointinteger_type_convert_instructions"
            };

        case "VFNMACC.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFNMACC.VF</b> vs2, rs1, vd</span><br><div>vfnmacc</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfnmacc\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFNMACC.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFNMACC.VV</b> vs2, vs1, vd</span><br><div>vfnmacc</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfnmacc\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFNMADD.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFNMADD.VF</b> vs2, rs1, vd</span><br><div>vfnmadd</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfnmadd\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFNMADD.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFNMADD.VV</b> vs2, vs1, vd</span><br><div>vfnmadd</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfnmadd\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFNMSAC.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFNMSAC.VF</b> vs2, rs1, vd</span><br><div>vfnmsac</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfnmsac\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFNMSAC.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFNMSAC.VV</b> vs2, vs1, vd</span><br><div>vfnmsac</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfnmsac\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFNMSUB.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFNMSUB.VF</b> vs2, rs1, vd</span><br><div>vfnmsub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfnmsub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFNMSUB.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFNMSUB.VV</b> vs2, vs1, vd</span><br><div>vfnmsub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfnmsub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFRDIV.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFRDIV.VF</b> vs2, rs1, vd</span><br><div>vfrdiv</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfrdiv\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFREC7.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFREC7.V</b> vs2, vd</span><br><div>Table 17. <b>vfrec7.v</b> common-case lookup table contents<br># Floating-point reciprocal estimate to 7 bits. <b>vfrec7.v</b> vd, vs2, vm</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "Table 17. vfrec7.v common-case lookup table contents\n# Floating-point reciprocal estimate to 7 bits. vfrec7.v vd, vs2, vm\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_floating_point_reciprocal_estimate_instruction"
            };

        case "VFREDMAX.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VFREDMAX.VS</b> vs2, vs1, vd</span><br><div>vfredmax</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfredmax\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFREDMIN.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VFREDMIN.VS</b> vs2, vs1, vd</span><br><div>vfredmin</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfredmin\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFREDOSUM.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VFREDOSUM.VS</b> vs2, vs1, vd</span><br><div># Simple reductions. <b>vfredosum.vs</b> vd, vs2, vs1, vm # Ordered sum vfredusum.vs vd, vs2, vs1, vm # Unordered sum vfredmax.vs vd, vs2, vs1, vm # Maximum value vfredmin.vs vd, vs2, vs1, vm # Minimum value</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Simple reductions. vfredosum.vs vd, vs2, vs1, vm # Ordered sum vfredusum.vs vd, vs2, vs1, vm # Unordered sum vfredmax.vs vd, vs2, vs1, vm # Maximum value vfredmin.vs vd, vs2, vs1, vm # Minimum value\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-vector-float-reduce"
            };

        case "VFREDSUM.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VFREDSUM.VS</b> vs2, vs1, vd</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VFREDUSUM.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VFREDUSUM.VS</b> vs2, vs1, vd</span><br><div>The unordered sum reduction instruction, vfredusum, provides an implementation more freedom in performing the reduction.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The unordered sum reduction instruction, vfredusum, provides an implementation more freedom in performing the reduction.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_unordered_single_width_floating_point_sum_reduction"
            };

        case "VFRSQRT7.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFRSQRT7.V</b> vs2, vd</span><br><div>Table 16. <b>vfrsqrt7.v</b> common-case lookup table contents<br># Floating-point reciprocal square-root estimate to 7 bits. <b>vfrsqrt7.v</b> vd, vs2, vm</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "Table 16. vfrsqrt7.v common-case lookup table contents\n# Floating-point reciprocal square-root estimate to 7 bits. vfrsqrt7.v vd, vs2, vm\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_floating_point_reciprocal_square_root_estimate_instruction"
            };

        case "VFRSUB.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFRSUB.VF</b> vs2, rs1, vd</span><br><div>vfrsub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfrsub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFSGNJ.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFSGNJ.VF</b> vs2, rs1, vd</span><br><div>vfsgnj.vv vd, vs2, vs1, vm # Vector-vector <b>vfsgnj.vf</b> vd, vs2, rs1, vm # vector-scalar vfsgnjn.vv vd, vs2, vs1, vm # Vector-vector vfsgnjn.vf vd, vs2, rs1, vm # vector-scalar vfsgnjx.vv vd, vs2, vs1, vm # Vector-vector vfsgnjx.vf vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfsgnj.vv vd, vs2, vs1, vm # Vector-vector vfsgnj.vf vd, vs2, rs1, vm # vector-scalar vfsgnjn.vv vd, vs2, vs1, vm # Vector-vector vfsgnjn.vf vd, vs2, rs1, vm # vector-scalar vfsgnjx.vv vd, vs2, vs1, vm # Vector-vector vfsgnjx.vf vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_floating_point_sign_injection_instructions"
            };

        case "VFSGNJ.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFSGNJ.VV</b> vs2, vs1, vd</span><br><div><b>vfsgnj.vv</b> vd, vs2, vs1, vm # Vector-vector vfsgnj.vf vd, vs2, rs1, vm # vector-scalar vfsgnjn.vv vd, vs2, vs1, vm # Vector-vector vfsgnjn.vf vd, vs2, rs1, vm # vector-scalar vfsgnjx.vv vd, vs2, vs1, vm # Vector-vector vfsgnjx.vf vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfsgnj.vv vd, vs2, vs1, vm # Vector-vector vfsgnj.vf vd, vs2, rs1, vm # vector-scalar vfsgnjn.vv vd, vs2, vs1, vm # Vector-vector vfsgnjn.vf vd, vs2, rs1, vm # vector-scalar vfsgnjx.vv vd, vs2, vs1, vm # Vector-vector vfsgnjx.vf vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_floating_point_sign_injection_instructions"
            };

        case "VFSGNJN.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFSGNJN.VF</b> vs2, rs1, vd</span><br><div>vfsgnjn</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfsgnjn\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFSGNJN.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFSGNJN.VV</b> vs2, vs1, vd</span><br><div>vfsgnjn</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfsgnjn\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFSGNJX.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFSGNJX.VF</b> vs2, rs1, vd</span><br><div>vfsgnjx</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfsgnjx\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFSGNJX.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFSGNJX.VV</b> vs2, vs1, vd</span><br><div>vfsgnjx</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfsgnjx\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFSLIDE1DOWN.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFSLIDE1DOWN.VF</b> vs2, rs1, vd</span><br><div>The vfslide1down instruction is defined analogously, but sources its scalar argument from an f register.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vfslide1down instruction is defined analogously, but sources its scalar argument from an f register.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_slide1down_instruction"
            };

        case "VFSLIDE1UP.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFSLIDE1UP.VF</b> vs2, rs1, vd</span><br><div>The vfslide1up instruction is defined analogously, but sources its scalar argument from an f register.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vfslide1up instruction is defined analogously, but sources its scalar argument from an f register.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_slide1up"
            };

        case "VFSQRT.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFSQRT.V</b> vs2, vd</span><br><div># Floating-point square root <b>vfsqrt.v</b> vd, vs2, vm # Vector-vector square root</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Floating-point square root vfsqrt.v vd, vs2, vm # Vector-vector square root\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_floating_point_square_root_instruction"
            };

        case "VFSUB.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFSUB.VF</b> vs2, rs1, vd</span><br><div>vfsub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfsub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFSUB.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFSUB.VV</b> vs2, vs1, vd</span><br><div>vfsub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfsub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFWADD.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWADD.VF</b> vs2, rs1, vd</span><br><div># Widening FP add/subtract, 2*SEW = SEW +/- SEW vfwadd.vv vd, vs2, vs1, vm # vector-vector <b>vfwadd.vf</b> vd, vs2, rs1, vm # vector-scalar vfwsub.vv vd, vs2, vs1, vm # vector-vector vfwsub.vf vd, vs2, rs1, vm # vector-scalar # Widening FP add/subtract, 2*SEW = 2*SEW +/- SEW vfwadd.wv vd, vs2, vs1, vm # vector-vector vfwadd.wf vd, vs2, rs1, vm # vector-scalar vfwsub.wv vd, vs2, vs1, vm # vector-vector vfwsub.wf vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Widening FP add/subtract, 2*SEW = SEW +/- SEW vfwadd.vv vd, vs2, vs1, vm # vector-vector vfwadd.vf vd, vs2, rs1, vm # vector-scalar vfwsub.vv vd, vs2, vs1, vm # vector-vector vfwsub.vf vd, vs2, rs1, vm # vector-scalar # Widening FP add/subtract, 2*SEW = 2*SEW +/- SEW vfwadd.wv vd, vs2, vs1, vm # vector-vector vfwadd.wf vd, vs2, rs1, vm # vector-scalar vfwsub.wv vd, vs2, vs1, vm # vector-vector vfwsub.wf vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_widening_floating_point_addsubtract_instructions"
            };

        case "VFWADD.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWADD.VV</b> vs2, vs1, vd</span><br><div># Widening FP add/subtract, 2*SEW = SEW +/- SEW <b>vfwadd.vv</b> vd, vs2, vs1, vm # vector-vector vfwadd.vf vd, vs2, rs1, vm # vector-scalar vfwsub.vv vd, vs2, vs1, vm # vector-vector vfwsub.vf vd, vs2, rs1, vm # vector-scalar # Widening FP add/subtract, 2*SEW = 2*SEW +/- SEW vfwadd.wv vd, vs2, vs1, vm # vector-vector vfwadd.wf vd, vs2, rs1, vm # vector-scalar vfwsub.wv vd, vs2, vs1, vm # vector-vector vfwsub.wf vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Widening FP add/subtract, 2*SEW = SEW +/- SEW vfwadd.vv vd, vs2, vs1, vm # vector-vector vfwadd.vf vd, vs2, rs1, vm # vector-scalar vfwsub.vv vd, vs2, vs1, vm # vector-vector vfwsub.vf vd, vs2, rs1, vm # vector-scalar # Widening FP add/subtract, 2*SEW = 2*SEW +/- SEW vfwadd.wv vd, vs2, vs1, vm # vector-vector vfwadd.wf vd, vs2, rs1, vm # vector-scalar vfwsub.wv vd, vs2, vs1, vm # vector-vector vfwsub.wf vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_widening_floating_point_addsubtract_instructions"
            };

        case "VFWADD.WF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWADD.WF</b> vs2, rs1, vd</span><br><div># Widening FP add/subtract, 2*SEW = SEW +/- SEW vfwadd.vv vd, vs2, vs1, vm # vector-vector vfwadd.vf vd, vs2, rs1, vm # vector-scalar vfwsub.vv vd, vs2, vs1, vm # vector-vector vfwsub.vf vd, vs2, rs1, vm # vector-scalar # Widening FP add/subtract, 2*SEW = 2*SEW +/- SEW vfwadd.wv vd, vs2, vs1, vm # vector-vector <b>vfwadd.wf</b> vd, vs2, rs1, vm # vector-scalar vfwsub.wv vd, vs2, vs1, vm # vector-vector vfwsub.wf vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Widening FP add/subtract, 2*SEW = SEW +/- SEW vfwadd.vv vd, vs2, vs1, vm # vector-vector vfwadd.vf vd, vs2, rs1, vm # vector-scalar vfwsub.vv vd, vs2, vs1, vm # vector-vector vfwsub.vf vd, vs2, rs1, vm # vector-scalar # Widening FP add/subtract, 2*SEW = 2*SEW +/- SEW vfwadd.wv vd, vs2, vs1, vm # vector-vector vfwadd.wf vd, vs2, rs1, vm # vector-scalar vfwsub.wv vd, vs2, vs1, vm # vector-vector vfwsub.wf vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_widening_floating_point_addsubtract_instructions"
            };

        case "VFWADD.WV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWADD.WV</b> vs2, vs1, vd</span><br><div># Widening FP add/subtract, 2*SEW = SEW +/- SEW vfwadd.vv vd, vs2, vs1, vm # vector-vector vfwadd.vf vd, vs2, rs1, vm # vector-scalar vfwsub.vv vd, vs2, vs1, vm # vector-vector vfwsub.vf vd, vs2, rs1, vm # vector-scalar # Widening FP add/subtract, 2*SEW = 2*SEW +/- SEW <b>vfwadd.wv</b> vd, vs2, vs1, vm # vector-vector vfwadd.wf vd, vs2, rs1, vm # vector-scalar vfwsub.wv vd, vs2, vs1, vm # vector-vector vfwsub.wf vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Widening FP add/subtract, 2*SEW = SEW +/- SEW vfwadd.vv vd, vs2, vs1, vm # vector-vector vfwadd.vf vd, vs2, rs1, vm # vector-scalar vfwsub.vv vd, vs2, vs1, vm # vector-vector vfwsub.vf vd, vs2, rs1, vm # vector-scalar # Widening FP add/subtract, 2*SEW = 2*SEW +/- SEW vfwadd.wv vd, vs2, vs1, vm # vector-vector vfwadd.wf vd, vs2, rs1, vm # vector-scalar vfwsub.wv vd, vs2, vs1, vm # vector-vector vfwsub.wf vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_widening_floating_point_addsubtract_instructions"
            };

        case "VFWCVT.F.F.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWCVT.F.F.V</b> vs2, vd</span><br><div>vfwcvt.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer. vfwcvt.x.f.v vd, vs2, vm # Convert float to double-width signed integer. vfwcvt.rtz.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer, truncating. vfwcvt.rtz.x.f.v vd, vs2, vm # Convert float to double-width signed integer, truncating. vfwcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to double-width float. vfwcvt.f.x.v vd, vs2, vm # Convert signed integer to double-width float. <b>vfwcvt.f.f.v</b> vd, vs2, vm # Convert single-width float to double-width float.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwcvt.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer. vfwcvt.x.f.v vd, vs2, vm # Convert float to double-width signed integer. vfwcvt.rtz.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer, truncating. vfwcvt.rtz.x.f.v vd, vs2, vm # Convert float to double-width signed integer, truncating. vfwcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to double-width float. vfwcvt.f.x.v vd, vs2, vm # Convert signed integer to double-width float. vfwcvt.f.f.v vd, vs2, vm # Convert single-width float to double-width float.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_widening_floating_pointinteger_type_convert_instructions"
            };

        case "VFWCVT.F.X.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWCVT.F.X.V</b> vs2, vd</span><br><div>vfwcvt.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer. vfwcvt.x.f.v vd, vs2, vm # Convert float to double-width signed integer. vfwcvt.rtz.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer, truncating. vfwcvt.rtz.x.f.v vd, vs2, vm # Convert float to double-width signed integer, truncating. vfwcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to double-width float. <b>vfwcvt.f.x.v</b> vd, vs2, vm # Convert signed integer to double-width float. vfwcvt.f.f.v vd, vs2, vm # Convert single-width float to double-width float.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwcvt.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer. vfwcvt.x.f.v vd, vs2, vm # Convert float to double-width signed integer. vfwcvt.rtz.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer, truncating. vfwcvt.rtz.x.f.v vd, vs2, vm # Convert float to double-width signed integer, truncating. vfwcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to double-width float. vfwcvt.f.x.v vd, vs2, vm # Convert signed integer to double-width float. vfwcvt.f.f.v vd, vs2, vm # Convert single-width float to double-width float.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_widening_floating_pointinteger_type_convert_instructions"
            };

        case "VFWCVT.F.XU.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWCVT.F.XU.V</b> vs2, vd</span><br><div>vfwcvt.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer. vfwcvt.x.f.v vd, vs2, vm # Convert float to double-width signed integer. vfwcvt.rtz.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer, truncating. vfwcvt.rtz.x.f.v vd, vs2, vm # Convert float to double-width signed integer, truncating. <b>vfwcvt.f.xu.v</b> vd, vs2, vm # Convert unsigned integer to double-width float. vfwcvt.f.x.v vd, vs2, vm # Convert signed integer to double-width float. vfwcvt.f.f.v vd, vs2, vm # Convert single-width float to double-width float.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwcvt.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer. vfwcvt.x.f.v vd, vs2, vm # Convert float to double-width signed integer. vfwcvt.rtz.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer, truncating. vfwcvt.rtz.x.f.v vd, vs2, vm # Convert float to double-width signed integer, truncating. vfwcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to double-width float. vfwcvt.f.x.v vd, vs2, vm # Convert signed integer to double-width float. vfwcvt.f.f.v vd, vs2, vm # Convert single-width float to double-width float.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_widening_floating_pointinteger_type_convert_instructions"
            };

        case "VFWCVT.RTZ.X.F.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWCVT.RTZ.X.F.V</b> vs2, vd</span><br><div>vfwcvt.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer. vfwcvt.x.f.v vd, vs2, vm # Convert float to double-width signed integer. vfwcvt.rtz.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer, truncating. <b>vfwcvt.rtz.x.f.v</b> vd, vs2, vm # Convert float to double-width signed integer, truncating. vfwcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to double-width float. vfwcvt.f.x.v vd, vs2, vm # Convert signed integer to double-width float. vfwcvt.f.f.v vd, vs2, vm # Convert single-width float to double-width float.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwcvt.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer. vfwcvt.x.f.v vd, vs2, vm # Convert float to double-width signed integer. vfwcvt.rtz.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer, truncating. vfwcvt.rtz.x.f.v vd, vs2, vm # Convert float to double-width signed integer, truncating. vfwcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to double-width float. vfwcvt.f.x.v vd, vs2, vm # Convert signed integer to double-width float. vfwcvt.f.f.v vd, vs2, vm # Convert single-width float to double-width float.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_widening_floating_pointinteger_type_convert_instructions"
            };

        case "VFWCVT.RTZ.XU.F.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWCVT.RTZ.XU.F.V</b> vs2, vd</span><br><div>vfwcvt.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer. vfwcvt.x.f.v vd, vs2, vm # Convert float to double-width signed integer. <b>vfwcvt.rtz.xu.f.v</b> vd, vs2, vm # Convert float to double-width unsigned integer, truncating. vfwcvt.rtz.x.f.v vd, vs2, vm # Convert float to double-width signed integer, truncating. vfwcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to double-width float. vfwcvt.f.x.v vd, vs2, vm # Convert signed integer to double-width float. vfwcvt.f.f.v vd, vs2, vm # Convert single-width float to double-width float.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwcvt.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer. vfwcvt.x.f.v vd, vs2, vm # Convert float to double-width signed integer. vfwcvt.rtz.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer, truncating. vfwcvt.rtz.x.f.v vd, vs2, vm # Convert float to double-width signed integer, truncating. vfwcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to double-width float. vfwcvt.f.x.v vd, vs2, vm # Convert signed integer to double-width float. vfwcvt.f.f.v vd, vs2, vm # Convert single-width float to double-width float.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_widening_floating_pointinteger_type_convert_instructions"
            };

        case "VFWCVT.X.F.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWCVT.X.F.V</b> vs2, vd</span><br><div>vfwcvt.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer. <b>vfwcvt.x.f.v</b> vd, vs2, vm # Convert float to double-width signed integer. vfwcvt.rtz.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer, truncating. vfwcvt.rtz.x.f.v vd, vs2, vm # Convert float to double-width signed integer, truncating. vfwcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to double-width float. vfwcvt.f.x.v vd, vs2, vm # Convert signed integer to double-width float. vfwcvt.f.f.v vd, vs2, vm # Convert single-width float to double-width float.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwcvt.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer. vfwcvt.x.f.v vd, vs2, vm # Convert float to double-width signed integer. vfwcvt.rtz.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer, truncating. vfwcvt.rtz.x.f.v vd, vs2, vm # Convert float to double-width signed integer, truncating. vfwcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to double-width float. vfwcvt.f.x.v vd, vs2, vm # Convert signed integer to double-width float. vfwcvt.f.f.v vd, vs2, vm # Convert single-width float to double-width float.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_widening_floating_pointinteger_type_convert_instructions"
            };

        case "VFWCVT.XU.F.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWCVT.XU.F.V</b> vs2, vd</span><br><div><b>vfwcvt.xu.f.v</b> vd, vs2, vm # Convert float to double-width unsigned integer. vfwcvt.x.f.v vd, vs2, vm # Convert float to double-width signed integer. vfwcvt.rtz.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer, truncating. vfwcvt.rtz.x.f.v vd, vs2, vm # Convert float to double-width signed integer, truncating. vfwcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to double-width float. vfwcvt.f.x.v vd, vs2, vm # Convert signed integer to double-width float. vfwcvt.f.f.v vd, vs2, vm # Convert single-width float to double-width float.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwcvt.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer. vfwcvt.x.f.v vd, vs2, vm # Convert float to double-width signed integer. vfwcvt.rtz.xu.f.v vd, vs2, vm # Convert float to double-width unsigned integer, truncating. vfwcvt.rtz.x.f.v vd, vs2, vm # Convert float to double-width signed integer, truncating. vfwcvt.f.xu.v vd, vs2, vm # Convert unsigned integer to double-width float. vfwcvt.f.x.v vd, vs2, vm # Convert signed integer to double-width float. vfwcvt.f.f.v vd, vs2, vm # Convert single-width float to double-width float.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_widening_floating_pointinteger_type_convert_instructions"
            };

        case "VFWMACC.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWMACC.VF</b> vs2, rs1, vd</span><br><div># FP widening multiply-accumulate, overwrites addend vfwmacc.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] <b>vfwmacc.vf</b> vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vs2[i]) + vd[i] # FP widening negate-(multiply-accumulate), overwrites addend vfwnmacc.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) - vd[i] vfwnmacc.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vs2[i]) - vd[i] # FP widening multiply-subtract-accumulator, overwrites addend vfwmsac.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) - vd[i] vfwmsac.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vs2[i]) - vd[i] # FP widening negate-(multiply-subtract-accumulator), overwrites addend vfwnmsac.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) + vd[i] vfwnmsac.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vs2[i]) + vd[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# FP widening multiply-accumulate, overwrites addend vfwmacc.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] vfwmacc.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vs2[i]) + vd[i] # FP widening negate-(multiply-accumulate), overwrites addend vfwnmacc.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) - vd[i] vfwnmacc.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vs2[i]) - vd[i] # FP widening multiply-subtract-accumulator, overwrites addend vfwmsac.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) - vd[i] vfwmsac.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vs2[i]) - vd[i] # FP widening negate-(multiply-subtract-accumulator), overwrites addend vfwnmsac.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) + vd[i] vfwnmsac.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vs2[i]) + vd[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_widening_floating_point_fused_multiply_add_instructions"
            };

        case "VFWMACC.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWMACC.VV</b> vs2, vs1, vd</span><br><div># FP widening multiply-accumulate, overwrites addend <b>vfwmacc.vv</b> vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] vfwmacc.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vs2[i]) + vd[i] # FP widening negate-(multiply-accumulate), overwrites addend vfwnmacc.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) - vd[i] vfwnmacc.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vs2[i]) - vd[i] # FP widening multiply-subtract-accumulator, overwrites addend vfwmsac.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) - vd[i] vfwmsac.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vs2[i]) - vd[i] # FP widening negate-(multiply-subtract-accumulator), overwrites addend vfwnmsac.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) + vd[i] vfwnmsac.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vs2[i]) + vd[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# FP widening multiply-accumulate, overwrites addend vfwmacc.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] vfwmacc.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vs2[i]) + vd[i] # FP widening negate-(multiply-accumulate), overwrites addend vfwnmacc.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) - vd[i] vfwnmacc.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vs2[i]) - vd[i] # FP widening multiply-subtract-accumulator, overwrites addend vfwmsac.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) - vd[i] vfwmsac.vf vd, rs1, vs2, vm # vd[i] = +(f[rs1] * vs2[i]) - vd[i] # FP widening negate-(multiply-subtract-accumulator), overwrites addend vfwnmsac.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) + vd[i] vfwnmsac.vf vd, rs1, vs2, vm # vd[i] = -(f[rs1] * vs2[i]) + vd[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_widening_floating_point_fused_multiply_add_instructions"
            };

        case "VFWMSAC.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWMSAC.VF</b> vs2, rs1, vd</span><br><div>vfwmsac</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwmsac\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFWMSAC.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWMSAC.VV</b> vs2, vs1, vd</span><br><div>vfwmsac</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwmsac\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFWMUL.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWMUL.VF</b> vs2, rs1, vd</span><br><div># Widening floating-point multiply vfwmul.vv vd, vs2, vs1, vm # vector-vector <b>vfwmul.vf</b> vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Widening floating-point multiply vfwmul.vv vd, vs2, vs1, vm # vector-vector vfwmul.vf vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_widening_floating_point_multiply"
            };

        case "VFWMUL.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWMUL.VV</b> vs2, vs1, vd</span><br><div># Widening floating-point multiply <b>vfwmul.vv</b> vd, vs2, vs1, vm # vector-vector vfwmul.vf vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Widening floating-point multiply vfwmul.vv vd, vs2, vs1, vm # vector-vector vfwmul.vf vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_widening_floating_point_multiply"
            };

        case "VFWNMACC.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWNMACC.VF</b> vs2, rs1, vd</span><br><div>vfwnmacc</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwnmacc\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFWNMACC.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWNMACC.VV</b> vs2, vs1, vd</span><br><div>vfwnmacc</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwnmacc\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFWNMSAC.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWNMSAC.VF</b> vs2, rs1, vd</span><br><div>vfwnmsac</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwnmsac\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFWNMSAC.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWNMSAC.VV</b> vs2, vs1, vd</span><br><div>vfwnmsac</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwnmsac\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFWREDOSUM.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWREDOSUM.VS</b> vs2, vs1, vd</span><br><div># Simple reductions. <b>vfwredosum.vs</b> vd, vs2, vs1, vm # Ordered sum vfwredusum.vs vd, vs2, vs1, vm # Unordered sum</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Simple reductions. vfwredosum.vs vd, vs2, vs1, vm # Ordered sum vfwredusum.vs vd, vs2, vs1, vm # Unordered sum\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-vector-float-reduce-widen"
            };

        case "VFWREDSUM.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWREDSUM.VS</b> vs2, vs1, vd</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VFWREDUSUM.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWREDUSUM.VS</b> vs2, vs1, vd</span><br><div>vfwredusum</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwredusum\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFWSUB.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWSUB.VF</b> vs2, rs1, vd</span><br><div>vfwsub<br>vfwsub.w</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwsub\nvfwsub.w\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFWSUB.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWSUB.VV</b> vs2, vs1, vd</span><br><div>vfwsub<br>vfwsub.w</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwsub\nvfwsub.w\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFWSUB.WF":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWSUB.WF</b> vs2, rs1, vd</span><br><div>vfwsub<br>vfwsub.w</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwsub\nvfwsub.w\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VFWSUB.WV":
            return {
                "html": "<div><span class=\"opcode\"><b>VFWSUB.WV</b> vs2, vs1, vd</span><br><div>vfwsub<br>vfwsub.w</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vfwsub\nvfwsub.w\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VID.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VID.V</b> vd</span><br><div>The <b>vid.v</b> instruction writes each element's index to the destination vector register group, from 0 to vl-1.<br><b>vid.v</b> vd, vm # Write element ID to destination.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vid.v instruction writes each element's index to the destination vector register group, from 0 to vl-1.\nvid.v vd, vm # Write element ID to destination.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_element_index_instruction"
            };

        case "VIOTA.M":
            return {
                "html": "<div><span class=\"opcode\"><b>VIOTA.M</b> vs2, vd</span><br><div>The <b>viota.m</b> instruction reads a source vector mask register and writes to each element of the destination vector register group the sum of all the bits of elements in the mask register whose index is less than the element, e.g., a parallel prefix sum of the mask values.<br>Traps on <b>viota.m</b> are always reported with a vstart of 0, and execution is always restarted from the beginning when resuming after a trap handler. An illegal instruction exception is raised if vstart is non-zero.<br>The <b>viota.m</b> instruction can be combined with memory scatter instructions (indexed stores) to perform vector compress functions.<br><b>viota.m</b> vd, vs2, vm # Example 7 6 5 4 3 2 1 0 Element number 1 0 0 1 0 0 0 1 v2 contents <b>viota.m</b> v4, v2 # Unmasked 2 2 2 1 1 1 1 0 v4 result 1 1 1 0 1 0 1 1 v0 contents 1 0 0 1 0 0 0 1 v2 contents 2 3 4 5 6 7 8 9 v4 contents <b>viota.m</b> v4, v2, v0.t # Masked, vtype.vma=0 1 1 1 5 1 7 1 0 v4 results</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The viota.m instruction reads a source vector mask register and writes to each element of the destination vector register group the sum of all the bits of elements in the mask register whose index is less than the element, e.g., a parallel prefix sum of the mask values.\nTraps on viota.m are always reported with a vstart of 0, and execution is always restarted from the beginning when resuming after a trap handler. An illegal instruction exception is raised if vstart is non-zero.\nThe viota.m instruction can be combined with memory scatter instructions (indexed stores) to perform vector compress functions.\nviota.m vd, vs2, vm # Example 7 6 5 4 3 2 1 0 Element number 1 0 0 1 0 0 0 1 v2 contents viota.m v4, v2 # Unmasked 2 2 2 1 1 1 1 0 v4 result 1 1 1 0 1 0 1 1 v0 contents 1 0 0 1 0 0 0 1 v2 contents 2 3 4 5 6 7 8 9 v4 contents viota.m v4, v2, v0.t # Masked, vtype.vma=0 1 1 1 5 1 7 1 0 v4 results\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_iota_instruction"
            };

        case "VL1R.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL1R.V</b> rs1, vd</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL1RE16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL1RE16.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL1RE32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL1RE32.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL1RE64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL1RE64.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL1RE8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL1RE8.V</b> rs1, vd</span><br><div># Format of whole register load and store instructions. vl1r.v v3, (a0) # Pseudoinstruction equal to <b>vl1re8.v</b> <b>vl1re8.v</b> v3, (a0) # Load v3 with VLEN/8 bytes held at address in a0 vl1re16.v v3, (a0) # Load v3 with VLEN/16 halfwords held at address in a0 vl1re32.v v3, (a0) # Load v3 with VLEN/32 words held at address in a0 vl1re64.v v3, (a0) # Load v3 with VLEN/64 doublewords held at address in a0 vl2r.v v2, (a0) # Pseudoinstruction equal to vl2re8.v v2, (a0) vl2re8.v v2, (a0) # Load v2-v3 with 2*VLEN/8 bytes from address in a0 vl2re16.v v2, (a0) # Load v2-v3 with 2*VLEN/16 halfwords held at address in a0 vl2re32.v v2, (a0) # Load v2-v3 with 2*VLEN/32 words held at address in a0 vl2re64.v v2, (a0) # Load v2-v3 with 2*VLEN/64 doublewords held at address in a0 vl4r.v v4, (a0) # Pseudoinstruction equal to vl4re8.v vl4re8.v v4, (a0) # Load v4-v7 with 4*VLEN/8 bytes from address in a0 vl4re16.v v4, (a0) vl4re32.v v4, (a0) vl4re64.v v4, (a0) vl8r.v v8, (a0) # Pseudoinstruction equal to vl8re8.v vl8re8.v v8, (a0) # Load v8-v15 with 8*VLEN/8 bytes from address in a0 vl8re16.v v8, (a0) vl8re32.v v8, (a0) vl8re64.v v8, (a0) vs1r.v v3, (a1) # Store v3 to address in a1 vs2r.v v2, (a1) # Store v2-v3 to address in a1 vs4r.v v4, (a1) # Store v4-v7 to address in a1 vs8r.v v8, (a1) # Store v8-v15 to address in a1</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Format of whole register load and store instructions. vl1r.v v3, (a0) # Pseudoinstruction equal to vl1re8.v vl1re8.v v3, (a0) # Load v3 with VLEN/8 bytes held at address in a0 vl1re16.v v3, (a0) # Load v3 with VLEN/16 halfwords held at address in a0 vl1re32.v v3, (a0) # Load v3 with VLEN/32 words held at address in a0 vl1re64.v v3, (a0) # Load v3 with VLEN/64 doublewords held at address in a0 vl2r.v v2, (a0) # Pseudoinstruction equal to vl2re8.v v2, (a0) vl2re8.v v2, (a0) # Load v2-v3 with 2*VLEN/8 bytes from address in a0 vl2re16.v v2, (a0) # Load v2-v3 with 2*VLEN/16 halfwords held at address in a0 vl2re32.v v2, (a0) # Load v2-v3 with 2*VLEN/32 words held at address in a0 vl2re64.v v2, (a0) # Load v2-v3 with 2*VLEN/64 doublewords held at address in a0 vl4r.v v4, (a0) # Pseudoinstruction equal to vl4re8.v vl4re8.v v4, (a0) # Load v4-v7 with 4*VLEN/8 bytes from address in a0 vl4re16.v v4, (a0) vl4re32.v v4, (a0) vl4re64.v v4, (a0) vl8r.v v8, (a0) # Pseudoinstruction equal to vl8re8.v vl8re8.v v8, (a0) # Load v8-v15 with 8*VLEN/8 bytes from address in a0 vl8re16.v v8, (a0) vl8re32.v v8, (a0) vl8re64.v v8, (a0) vs1r.v v3, (a1) # Store v3 to address in a1 vs2r.v v2, (a1) # Store v2-v3 to address in a1 vs4r.v v4, (a1) # Store v4-v7 to address in a1 vs8r.v v8, (a1) # Store v8-v15 to address in a1\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_loadstore_whole_register_instructions"
            };

        case "VL2R.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL2R.V</b> rs1, vd</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL2RE16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL2RE16.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL2RE32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL2RE32.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL2RE64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL2RE64.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL2RE8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL2RE8.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL4R.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL4R.V</b> rs1, vd</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL4RE16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL4RE16.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL4RE32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL4RE32.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL4RE64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL4RE64.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL4RE8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL4RE8.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL8R.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL8R.V</b> rs1, vd</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL8RE16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL8RE16.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL8RE32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL8RE32.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL8RE64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL8RE64.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VL8RE8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VL8RE8.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLE1.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE1.V</b> rs1, vd</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLE1024.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE1024.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLE1024FF.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE1024FF.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLE128.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE128.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLE128FF.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE128FF.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLE16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE16.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLE16FF.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE16FF.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLE256.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE256.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLE256FF.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE256FF.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLE32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE32.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLE32FF.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE32FF.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLE512.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE512.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLE512FF.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE512FF.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLE64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE64.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLE64FF.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE64FF.V</b> rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLE8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE8.V</b> rs1, vd</span><br><div># Vector unit-stride loads and stores # vd destination, rs1 base address, vm is mask encoding (v0.t or <missing>) <b>vle8.v</b> vd, (rs1), vm # 8-bit unit-stride load vle16.v vd, (rs1), vm # 16-bit unit-stride load vle32.v vd, (rs1), vm # 32-bit unit-stride load vle64.v vd, (rs1), vm # 64-bit unit-stride load # vs3 store data, rs1 base address, vm is mask encoding (v0.t or <missing>) vse8.v vs3, (rs1), vm # 8-bit unit-stride store vse16.v vs3, (rs1), vm # 16-bit unit-stride store vse32.v vs3, (rs1), vm # 32-bit unit-stride store vse64.v vs3, (rs1), vm # 64-bit unit-stride store</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Vector unit-stride loads and stores # vd destination, rs1 base address, vm is mask encoding (v0.t or <missing>) vle8.v vd, (rs1), vm # 8-bit unit-stride load vle16.v vd, (rs1), vm # 16-bit unit-stride load vle32.v vd, (rs1), vm # 32-bit unit-stride load vle64.v vd, (rs1), vm # 64-bit unit-stride load # vs3 store data, rs1 base address, vm is mask encoding (v0.t or <missing>) vse8.v vs3, (rs1), vm # 8-bit unit-stride store vse16.v vs3, (rs1), vm # 16-bit unit-stride store vse32.v vs3, (rs1), vm # 32-bit unit-stride store vse64.v vs3, (rs1), vm # 64-bit unit-stride store\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_unit_stride_instructions"
            };

        case "VLE8FF.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLE8FF.V</b> rs1, vd</span><br><div># Vector unit-stride fault-only-first loads # vd destination, rs1 base address, vm is mask encoding (v0.t or <missing>) <b>vle8ff.v</b> vd, (rs1), vm # 8-bit unit-stride fault-only-first load vle16ff.v vd, (rs1), vm # 16-bit unit-stride fault-only-first load vle32ff.v vd, (rs1), vm # 32-bit unit-stride fault-only-first load vle64ff.v vd, (rs1), vm # 64-bit unit-stride fault-only-first load</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Vector unit-stride fault-only-first loads # vd destination, rs1 base address, vm is mask encoding (v0.t or <missing>) vle8ff.v vd, (rs1), vm # 8-bit unit-stride fault-only-first load vle16ff.v vd, (rs1), vm # 16-bit unit-stride fault-only-first load vle32ff.v vd, (rs1), vm # 32-bit unit-stride fault-only-first load vle64ff.v vd, (rs1), vm # 64-bit unit-stride fault-only-first load\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_unit_stride_fault_only_first_loads"
            };

        case "VLM.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLM.V</b> rs1, vd</span><br><div><b>vlm.v</b> and vsm.v are encoded with the same width[2:0]=0 encoding as vle8.v and vse8.v, but are distinguished by different lumop and sumop encodings. Since <b>vlm.v</b> and vsm.v operate as byte loads and stores, vstart is in units of bytes for these instructions.<br># Vector unit-stride mask load <b>vlm.v</b> vd, (rs1) # Load byte vector of length ceil(vl/8) # Vector unit-stride mask store vsm.v vs3, (rs1) # Store byte vector of length ceil(vl/8)</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vlm.v and vsm.v are encoded with the same width[2:0]=0 encoding as vle8.v and vse8.v, but are distinguished by different lumop and sumop encodings. Since vlm.v and vsm.v operate as byte loads and stores, vstart is in units of bytes for these instructions.\n# Vector unit-stride mask load vlm.v vd, (rs1) # Load byte vector of length ceil(vl/8) # Vector unit-stride mask store vsm.v vs3, (rs1) # Store byte vector of length ceil(vl/8)\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_unit_stride_instructions"
            };

        case "VLOXEI1024.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLOXEI1024.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLOXEI128.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLOXEI128.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLOXEI16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLOXEI16.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLOXEI256.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLOXEI256.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLOXEI32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLOXEI32.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLOXEI512.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLOXEI512.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLOXEI64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLOXEI64.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLOXEI8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLOXEI8.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLSE1024.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLSE1024.V</b> rs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLSE128.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLSE128.V</b> rs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLSE16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLSE16.V</b> rs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLSE256.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLSE256.V</b> rs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLSE32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLSE32.V</b> rs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLSE512.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLSE512.V</b> rs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLSE64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLSE64.V</b> rs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLSE8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLSE8.V</b> rs2, rs1, vd</span><br><div># Vector strided loads and stores # vd destination, rs1 base address, rs2 byte stride <b>vlse8.v</b> vd, (rs1), rs2, vm # 8-bit strided load vlse16.v vd, (rs1), rs2, vm # 16-bit strided load vlse32.v vd, (rs1), rs2, vm # 32-bit strided load vlse64.v vd, (rs1), rs2, vm # 64-bit strided load # vs3 store data, rs1 base address, rs2 byte stride vsse8.v vs3, (rs1), rs2, vm # 8-bit strided store vsse16.v vs3, (rs1), rs2, vm # 16-bit strided store vsse32.v vs3, (rs1), rs2, vm # 32-bit strided store vsse64.v vs3, (rs1), rs2, vm # 64-bit strided store</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Vector strided loads and stores # vd destination, rs1 base address, rs2 byte stride vlse8.v vd, (rs1), rs2, vm # 8-bit strided load vlse16.v vd, (rs1), rs2, vm # 16-bit strided load vlse32.v vd, (rs1), rs2, vm # 32-bit strided load vlse64.v vd, (rs1), rs2, vm # 64-bit strided load # vs3 store data, rs1 base address, rs2 byte stride vsse8.v vs3, (rs1), rs2, vm # 8-bit strided store vsse16.v vs3, (rs1), rs2, vm # 16-bit strided store vsse32.v vs3, (rs1), rs2, vm # 32-bit strided store vsse64.v vs3, (rs1), rs2, vm # 64-bit strided store\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_strided_instructions"
            };

        case "VLUXEI1024.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLUXEI1024.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLUXEI128.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLUXEI128.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLUXEI16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLUXEI16.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLUXEI256.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLUXEI256.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLUXEI32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLUXEI32.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLUXEI512.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLUXEI512.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLUXEI64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLUXEI64.V</b> vs2, rs1, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VLUXEI8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VLUXEI8.V</b> vs2, rs1, vd</span><br><div># Vector indexed loads and stores # Vector indexed-unordered load instructions # vd destination, rs1 base address, vs2 byte offsets <b>vluxei8.v</b> vd, (rs1), vs2, vm # unordered 8-bit indexed load of SEW data vluxei16.v vd, (rs1), vs2, vm # unordered 16-bit indexed load of SEW data vluxei32.v vd, (rs1), vs2, vm # unordered 32-bit indexed load of SEW data vluxei64.v vd, (rs1), vs2, vm # unordered 64-bit indexed load of SEW data # Vector indexed-ordered load instructions # vd destination, rs1 base address, vs2 byte offsets vloxei8.v vd, (rs1), vs2, vm # ordered 8-bit indexed load of SEW data vloxei16.v vd, (rs1), vs2, vm # ordered 16-bit indexed load of SEW data vloxei32.v vd, (rs1), vs2, vm # ordered 32-bit indexed load of SEW data vloxei64.v vd, (rs1), vs2, vm # ordered 64-bit indexed load of SEW data # Vector indexed-unordered store instructions # vs3 store data, rs1 base address, vs2 byte offsets vsuxei8.v vs3, (rs1), vs2, vm # unordered 8-bit indexed store of SEW data vsuxei16.v vs3, (rs1), vs2, vm # unordered 16-bit indexed store of SEW data vsuxei32.v vs3, (rs1), vs2, vm # unordered 32-bit indexed store of SEW data vsuxei64.v vs3, (rs1), vs2, vm # unordered 64-bit indexed store of SEW data # Vector indexed-ordered store instructions # vs3 store data, rs1 base address, vs2 byte offsets vsoxei8.v vs3, (rs1), vs2, vm # ordered 8-bit indexed store of SEW data vsoxei16.v vs3, (rs1), vs2, vm # ordered 16-bit indexed store of SEW data vsoxei32.v vs3, (rs1), vs2, vm # ordered 32-bit indexed store of SEW data vsoxei64.v vs3, (rs1), vs2, vm # ordered 64-bit indexed store of SEW data</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Vector indexed loads and stores # Vector indexed-unordered load instructions # vd destination, rs1 base address, vs2 byte offsets vluxei8.v vd, (rs1), vs2, vm # unordered 8-bit indexed load of SEW data vluxei16.v vd, (rs1), vs2, vm # unordered 16-bit indexed load of SEW data vluxei32.v vd, (rs1), vs2, vm # unordered 32-bit indexed load of SEW data vluxei64.v vd, (rs1), vs2, vm # unordered 64-bit indexed load of SEW data # Vector indexed-ordered load instructions # vd destination, rs1 base address, vs2 byte offsets vloxei8.v vd, (rs1), vs2, vm # ordered 8-bit indexed load of SEW data vloxei16.v vd, (rs1), vs2, vm # ordered 16-bit indexed load of SEW data vloxei32.v vd, (rs1), vs2, vm # ordered 32-bit indexed load of SEW data vloxei64.v vd, (rs1), vs2, vm # ordered 64-bit indexed load of SEW data # Vector indexed-unordered store instructions # vs3 store data, rs1 base address, vs2 byte offsets vsuxei8.v vs3, (rs1), vs2, vm # unordered 8-bit indexed store of SEW data vsuxei16.v vs3, (rs1), vs2, vm # unordered 16-bit indexed store of SEW data vsuxei32.v vs3, (rs1), vs2, vm # unordered 32-bit indexed store of SEW data vsuxei64.v vs3, (rs1), vs2, vm # unordered 64-bit indexed store of SEW data # Vector indexed-ordered store instructions # vs3 store data, rs1 base address, vs2 byte offsets vsoxei8.v vs3, (rs1), vs2, vm # ordered 8-bit indexed store of SEW data vsoxei16.v vs3, (rs1), vs2, vm # ordered 16-bit indexed store of SEW data vsoxei32.v vs3, (rs1), vs2, vm # ordered 32-bit indexed store of SEW data vsoxei64.v vs3, (rs1), vs2, vm # ordered 64-bit indexed store of SEW data\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_indexed_instructions"
            };

        case "VMACC.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMACC.VV</b> vs2, vs1, vd</span><br><div>The integer multiply-add instructions are destructive and are provided in two forms, one that overwrites the addend or minuend (vmacc, vnmsac) and one that overwrites the first multiplicand (vmadd, vnmsub).<br># Integer multiply-add, overwrite addend <b>vmacc.vv</b> vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] vmacc.vx vd, rs1, vs2, vm # vd[i] = +(x[rs1] * vs2[i]) + vd[i] # Integer multiply-sub, overwrite minuend vnmsac.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) + vd[i] vnmsac.vx vd, rs1, vs2, vm # vd[i] = -(x[rs1] * vs2[i]) + vd[i] # Integer multiply-add, overwrite multiplicand vmadd.vv vd, vs1, vs2, vm # vd[i] = (vs1[i] * vd[i]) + vs2[i] vmadd.vx vd, rs1, vs2, vm # vd[i] = (x[rs1] * vd[i]) + vs2[i] # Integer multiply-sub, overwrite multiplicand vnmsub.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vd[i]) + vs2[i] vnmsub.vx vd, rs1, vs2, vm # vd[i] = -(x[rs1] * vd[i]) + vs2[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The integer multiply-add instructions are destructive and are provided in two forms, one that overwrites the addend or minuend (vmacc, vnmsac) and one that overwrites the first multiplicand (vmadd, vnmsub).\n# Integer multiply-add, overwrite addend vmacc.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] vmacc.vx vd, rs1, vs2, vm # vd[i] = +(x[rs1] * vs2[i]) + vd[i] # Integer multiply-sub, overwrite minuend vnmsac.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) + vd[i] vnmsac.vx vd, rs1, vs2, vm # vd[i] = -(x[rs1] * vs2[i]) + vd[i] # Integer multiply-add, overwrite multiplicand vmadd.vv vd, vs1, vs2, vm # vd[i] = (vs1[i] * vd[i]) + vs2[i] vmadd.vx vd, rs1, vs2, vm # vd[i] = (x[rs1] * vd[i]) + vs2[i] # Integer multiply-sub, overwrite multiplicand vnmsub.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vd[i]) + vs2[i] vnmsub.vx vd, rs1, vs2, vm # vd[i] = -(x[rs1] * vd[i]) + vs2[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_integer_multiply_add_instructions"
            };

        case "VMACC.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMACC.VX</b> vs2, rs1, vd</span><br><div>The integer multiply-add instructions are destructive and are provided in two forms, one that overwrites the addend or minuend (vmacc, vnmsac) and one that overwrites the first multiplicand (vmadd, vnmsub).<br># Integer multiply-add, overwrite addend vmacc.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] <b>vmacc.vx</b> vd, rs1, vs2, vm # vd[i] = +(x[rs1] * vs2[i]) + vd[i] # Integer multiply-sub, overwrite minuend vnmsac.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) + vd[i] vnmsac.vx vd, rs1, vs2, vm # vd[i] = -(x[rs1] * vs2[i]) + vd[i] # Integer multiply-add, overwrite multiplicand vmadd.vv vd, vs1, vs2, vm # vd[i] = (vs1[i] * vd[i]) + vs2[i] vmadd.vx vd, rs1, vs2, vm # vd[i] = (x[rs1] * vd[i]) + vs2[i] # Integer multiply-sub, overwrite multiplicand vnmsub.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vd[i]) + vs2[i] vnmsub.vx vd, rs1, vs2, vm # vd[i] = -(x[rs1] * vd[i]) + vs2[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The integer multiply-add instructions are destructive and are provided in two forms, one that overwrites the addend or minuend (vmacc, vnmsac) and one that overwrites the first multiplicand (vmadd, vnmsub).\n# Integer multiply-add, overwrite addend vmacc.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] vmacc.vx vd, rs1, vs2, vm # vd[i] = +(x[rs1] * vs2[i]) + vd[i] # Integer multiply-sub, overwrite minuend vnmsac.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vs2[i]) + vd[i] vnmsac.vx vd, rs1, vs2, vm # vd[i] = -(x[rs1] * vs2[i]) + vd[i] # Integer multiply-add, overwrite multiplicand vmadd.vv vd, vs1, vs2, vm # vd[i] = (vs1[i] * vd[i]) + vs2[i] vmadd.vx vd, rs1, vs2, vm # vd[i] = (x[rs1] * vd[i]) + vs2[i] # Integer multiply-sub, overwrite multiplicand vnmsub.vv vd, vs1, vs2, vm # vd[i] = -(vs1[i] * vd[i]) + vs2[i] vnmsub.vx vd, rs1, vs2, vm # vd[i] = -(x[rs1] * vd[i]) + vs2[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_integer_multiply_add_instructions"
            };

        case "VMADC.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VMADC.VI</b> vs2, simm5, vd</span><br><div>vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked (vm=0), and write the result back to mask register vd. If unmasked (vm=1), there is no carry-in or borrow-in. These instructions operate on and write back all body elements, even if masked. Because these instructions produce a mask value, they always operate with a tail-agnostic policy.<br># Example multi-word arithmetic sequence, accumulating into v4 vmadc.vvm v1, v4, v8, v0 # Get carry into temp register v1 vadc.vvm v4, v4, v8, v0 # Calc new sum vmmv.m v0, v1 # Move temp carry into v0 for next word</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked (vm=0), and write the result back to mask register vd. If unmasked (vm=1), there is no carry-in or borrow-in. These instructions operate on and write back all body elements, even if masked. Because these instructions produce a mask value, they always operate with a tail-agnostic policy.\n# Example multi-word arithmetic sequence, accumulating into v4 vmadc.vvm v1, v4, v8, v0 # Get carry into temp register v1 vadc.vvm v4, v4, v8, v0 # Calc new sum vmmv.m v0, v1 # Move temp carry into v0 for next word\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VMADC.VIM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMADC.VIM</b> vs2, simm5, vd</span><br><div>vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked (vm=0), and write the result back to mask register vd. If unmasked (vm=1), there is no carry-in or borrow-in. These instructions operate on and write back all body elements, even if masked. Because these instructions produce a mask value, they always operate with a tail-agnostic policy.<br># Example multi-word arithmetic sequence, accumulating into v4 vmadc.vvm v1, v4, v8, v0 # Get carry into temp register v1 vadc.vvm v4, v4, v8, v0 # Calc new sum vmmv.m v0, v1 # Move temp carry into v0 for next word</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked (vm=0), and write the result back to mask register vd. If unmasked (vm=1), there is no carry-in or borrow-in. These instructions operate on and write back all body elements, even if masked. Because these instructions produce a mask value, they always operate with a tail-agnostic policy.\n# Example multi-word arithmetic sequence, accumulating into v4 vmadc.vvm v1, v4, v8, v0 # Get carry into temp register v1 vadc.vvm v4, v4, v8, v0 # Calc new sum vmmv.m v0, v1 # Move temp carry into v0 for next word\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VMADC.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMADC.VV</b> vs2, vs1, vd</span><br><div>vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked (vm=0), and write the result back to mask register vd. If unmasked (vm=1), there is no carry-in or borrow-in. These instructions operate on and write back all body elements, even if masked. Because these instructions produce a mask value, they always operate with a tail-agnostic policy.<br># Example multi-word arithmetic sequence, accumulating into v4 vmadc.vvm v1, v4, v8, v0 # Get carry into temp register v1 vadc.vvm v4, v4, v8, v0 # Calc new sum vmmv.m v0, v1 # Move temp carry into v0 for next word</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked (vm=0), and write the result back to mask register vd. If unmasked (vm=1), there is no carry-in or borrow-in. These instructions operate on and write back all body elements, even if masked. Because these instructions produce a mask value, they always operate with a tail-agnostic policy.\n# Example multi-word arithmetic sequence, accumulating into v4 vmadc.vvm v1, v4, v8, v0 # Get carry into temp register v1 vadc.vvm v4, v4, v8, v0 # Calc new sum vmmv.m v0, v1 # Move temp carry into v0 for next word\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VMADC.VVM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMADC.VVM</b> vs2, vs1, vd</span><br><div>vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked (vm=0), and write the result back to mask register vd. If unmasked (vm=1), there is no carry-in or borrow-in. These instructions operate on and write back all body elements, even if masked. Because these instructions produce a mask value, they always operate with a tail-agnostic policy.<br># Example multi-word arithmetic sequence, accumulating into v4 <b>vmadc.vvm</b> v1, v4, v8, v0 # Get carry into temp register v1 vadc.vvm v4, v4, v8, v0 # Calc new sum vmmv.m v0, v1 # Move temp carry into v0 for next word</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked (vm=0), and write the result back to mask register vd. If unmasked (vm=1), there is no carry-in or borrow-in. These instructions operate on and write back all body elements, even if masked. Because these instructions produce a mask value, they always operate with a tail-agnostic policy.\n# Example multi-word arithmetic sequence, accumulating into v4 vmadc.vvm v1, v4, v8, v0 # Get carry into temp register v1 vadc.vvm v4, v4, v8, v0 # Calc new sum vmmv.m v0, v1 # Move temp carry into v0 for next word\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VMADC.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMADC.VX</b> vs2, rs1, vd</span><br><div>vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked (vm=0), and write the result back to mask register vd. If unmasked (vm=1), there is no carry-in or borrow-in. These instructions operate on and write back all body elements, even if masked. Because these instructions produce a mask value, they always operate with a tail-agnostic policy.<br># Example multi-word arithmetic sequence, accumulating into v4 vmadc.vvm v1, v4, v8, v0 # Get carry into temp register v1 vadc.vvm v4, v4, v8, v0 # Calc new sum vmmv.m v0, v1 # Move temp carry into v0 for next word</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked (vm=0), and write the result back to mask register vd. If unmasked (vm=1), there is no carry-in or borrow-in. These instructions operate on and write back all body elements, even if masked. Because these instructions produce a mask value, they always operate with a tail-agnostic policy.\n# Example multi-word arithmetic sequence, accumulating into v4 vmadc.vvm v1, v4, v8, v0 # Get carry into temp register v1 vadc.vvm v4, v4, v8, v0 # Calc new sum vmmv.m v0, v1 # Move temp carry into v0 for next word\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VMADC.VXM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMADC.VXM</b> vs2, rs1, vd</span><br><div>vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked (vm=0), and write the result back to mask register vd. If unmasked (vm=1), there is no carry-in or borrow-in. These instructions operate on and write back all body elements, even if masked. Because these instructions produce a mask value, they always operate with a tail-agnostic policy.<br># Example multi-word arithmetic sequence, accumulating into v4 vmadc.vvm v1, v4, v8, v0 # Get carry into temp register v1 vadc.vvm v4, v4, v8, v0 # Calc new sum vmmv.m v0, v1 # Move temp carry into v0 for next word</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmadc and vmsbc add or subtract the source operands, optionally add the carry-in or subtract the borrow-in if masked (vm=0), and write the result back to mask register vd. If unmasked (vm=1), there is no carry-in or borrow-in. These instructions operate on and write back all body elements, even if masked. Because these instructions produce a mask value, they always operate with a tail-agnostic policy.\n# Example multi-word arithmetic sequence, accumulating into v4 vmadc.vvm v1, v4, v8, v0 # Get carry into temp register v1 vadc.vvm v4, v4, v8, v0 # Calc new sum vmmv.m v0, v1 # Move temp carry into v0 for next word\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VMADD.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMADD.VV</b> vs2, vs1, vd</span><br><div>vmadd</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmadd\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMADD.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMADD.VX</b> vs2, rs1, vd</span><br><div>vmadd</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmadd\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMAND.MM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMAND.MM</b> vs2, vs1, vd</span><br><div><b>vmand.mm</b> vd, src1, src2<br><b>vmand.mm</b> vd, src2, src2<br><b>vmand.mm</b> vd, src1, src1<br><b>vmand.mm</b> vd, vs2, vs1 # vd.mask[i] = vs2.mask[i] && vs1.mask[i] vmnand.mm vd, vs2, vs1 # vd.mask[i] = !(vs2.mask[i] && vs1.mask[i]) vmandn.mm vd, vs2, vs1 # vd.mask[i] = vs2.mask[i] && !vs1.mask[i] vmxor.mm vd, vs2, vs1 # vd.mask[i] = vs2.mask[i] ^^ vs1.mask[i] vmor.mm vd, vs2, vs1 # vd.mask[i] = vs2.mask[i] || vs1.mask[i] vmnor.mm vd, vs2, vs1 # vd.mask[i] = !(vs2.mask[i] || vs1.mask[i]) vmorn.mm vd, vs2, vs1 # vd.mask[i] = vs2.mask[i] || !vs1.mask[i] vmxnor.mm vd, vs2, vs1 # vd.mask[i] = !(vs2.mask[i] ^^ vs1.mask[i])<br>vmmv.m vd, vs => <b>vmand.mm</b> vd, vs, vs # Copy mask register vmclr.m vd => vmxor.mm vd, vd, vd # Clear mask register vmset.m vd => vmxnor.mm vd, vd, vd # Set mask register vmnot.m vd, vs => vmnand.mm vd, vs, vs # Invert bits</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmand.mm vd, src1, src2\nvmand.mm vd, src2, src2\nvmand.mm vd, src1, src1\nvmand.mm vd, vs2, vs1 # vd.mask[i] = vs2.mask[i] && vs1.mask[i] vmnand.mm vd, vs2, vs1 # vd.mask[i] = !(vs2.mask[i] && vs1.mask[i]) vmandn.mm vd, vs2, vs1 # vd.mask[i] = vs2.mask[i] && !vs1.mask[i] vmxor.mm vd, vs2, vs1 # vd.mask[i] = vs2.mask[i] ^^ vs1.mask[i] vmor.mm vd, vs2, vs1 # vd.mask[i] = vs2.mask[i] || vs1.mask[i] vmnor.mm vd, vs2, vs1 # vd.mask[i] = !(vs2.mask[i] || vs1.mask[i]) vmorn.mm vd, vs2, vs1 # vd.mask[i] = vs2.mask[i] || !vs1.mask[i] vmxnor.mm vd, vs2, vs1 # vd.mask[i] = !(vs2.mask[i] ^^ vs1.mask[i])\nvmmv.m vd, vs => vmand.mm vd, vs, vs # Copy mask register vmclr.m vd => vmxor.mm vd, vd, vd # Clear mask register vmset.m vd => vmxnor.mm vd, vd, vd # Set mask register vmnot.m vd, vs => vmnand.mm vd, vs, vs # Invert bits\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-mask-register-logical"
            };

        case "VMANDN.MM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMANDN.MM</b> vs2, vs1, vd</span><br><div><b>vmandn.mm</b> vd, src2, src1<br><b>vmandn.mm</b> vd, src1, src2</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmandn.mm vd, src2, src1\nvmandn.mm vd, src1, src2\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-mask-register-logical"
            };

        case "VMANDNOT.MM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMANDNOT.MM</b> vs2, vs1, vd</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VMAX.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMAX.VV</b> vs2, vs1, vd</span><br><div>vmax</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmax\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMAX.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMAX.VX</b> vs2, rs1, vd</span><br><div>vmax</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmax\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMAXU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMAXU.VV</b> vs2, vs1, vd</span><br><div>vmaxu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmaxu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMAXU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMAXU.VX</b> vs2, rs1, vd</span><br><div>vmaxu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmaxu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMERGE.VIM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMERGE.VIM</b> vs2, simm5, vd</span><br><div>The vmerge instructions are encoded as masked instructions (vm=0). The instructions combine two sources as follows. At elements where the mask value is zero, the first operand is copied to the destination element, otherwise the second operand is copied to the destination element. The first operand is always a vector register group specified by vs2. The second operand is a vector register group specified by vs1 or a scalar x register specified by rs1 or a 5-bit sign-extended immediate.<br>vmerge.vvm vd, vs2, vs1, v0 # vd[i] = v0.mask[i] ? vs1[i] : vs2[i] vmerge.vxm vd, vs2, rs1, v0 # vd[i] = v0.mask[i] ? x[rs1] : vs2[i] <b>vmerge.vim</b> vd, vs2, imm, v0 # vd[i] = v0.mask[i] ? imm : vs2[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vmerge instructions are encoded as masked instructions (vm=0). The instructions combine two sources as follows. At elements where the mask value is zero, the first operand is copied to the destination element, otherwise the second operand is copied to the destination element. The first operand is always a vector register group specified by vs2. The second operand is a vector register group specified by vs1 or a scalar x register specified by rs1 or a 5-bit sign-extended immediate.\nvmerge.vvm vd, vs2, vs1, v0 # vd[i] = v0.mask[i] ? vs1[i] : vs2[i] vmerge.vxm vd, vs2, rs1, v0 # vd[i] = v0.mask[i] ? x[rs1] : vs2[i] vmerge.vim vd, vs2, imm, v0 # vd[i] = v0.mask[i] ? imm : vs2[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_merge_instructions"
            };

        case "VMERGE.VVM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMERGE.VVM</b> vs2, vs1, vd</span><br><div>The vmerge instructions are encoded as masked instructions (vm=0). The instructions combine two sources as follows. At elements where the mask value is zero, the first operand is copied to the destination element, otherwise the second operand is copied to the destination element. The first operand is always a vector register group specified by vs2. The second operand is a vector register group specified by vs1 or a scalar x register specified by rs1 or a 5-bit sign-extended immediate.<br><b>vmerge.vvm</b> vd, vs2, vs1, v0 # vd[i] = v0.mask[i] ? vs1[i] : vs2[i] vmerge.vxm vd, vs2, rs1, v0 # vd[i] = v0.mask[i] ? x[rs1] : vs2[i] vmerge.vim vd, vs2, imm, v0 # vd[i] = v0.mask[i] ? imm : vs2[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vmerge instructions are encoded as masked instructions (vm=0). The instructions combine two sources as follows. At elements where the mask value is zero, the first operand is copied to the destination element, otherwise the second operand is copied to the destination element. The first operand is always a vector register group specified by vs2. The second operand is a vector register group specified by vs1 or a scalar x register specified by rs1 or a 5-bit sign-extended immediate.\nvmerge.vvm vd, vs2, vs1, v0 # vd[i] = v0.mask[i] ? vs1[i] : vs2[i] vmerge.vxm vd, vs2, rs1, v0 # vd[i] = v0.mask[i] ? x[rs1] : vs2[i] vmerge.vim vd, vs2, imm, v0 # vd[i] = v0.mask[i] ? imm : vs2[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_merge_instructions"
            };

        case "VMERGE.VXM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMERGE.VXM</b> vs2, rs1, vd</span><br><div>The vmerge instructions are encoded as masked instructions (vm=0). The instructions combine two sources as follows. At elements where the mask value is zero, the first operand is copied to the destination element, otherwise the second operand is copied to the destination element. The first operand is always a vector register group specified by vs2. The second operand is a vector register group specified by vs1 or a scalar x register specified by rs1 or a 5-bit sign-extended immediate.<br>vmerge.vvm vd, vs2, vs1, v0 # vd[i] = v0.mask[i] ? vs1[i] : vs2[i] <b>vmerge.vxm</b> vd, vs2, rs1, v0 # vd[i] = v0.mask[i] ? x[rs1] : vs2[i] vmerge.vim vd, vs2, imm, v0 # vd[i] = v0.mask[i] ? imm : vs2[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vmerge instructions are encoded as masked instructions (vm=0). The instructions combine two sources as follows. At elements where the mask value is zero, the first operand is copied to the destination element, otherwise the second operand is copied to the destination element. The first operand is always a vector register group specified by vs2. The second operand is a vector register group specified by vs1 or a scalar x register specified by rs1 or a 5-bit sign-extended immediate.\nvmerge.vvm vd, vs2, vs1, v0 # vd[i] = v0.mask[i] ? vs1[i] : vs2[i] vmerge.vxm vd, vs2, rs1, v0 # vd[i] = v0.mask[i] ? x[rs1] : vs2[i] vmerge.vim vd, vs2, imm, v0 # vd[i] = v0.mask[i] ? imm : vs2[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_merge_instructions"
            };

        case "VMFEQ.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VMFEQ.VF</b> vs2, rs1, vd</span><br><div>The compare instructions follow the semantics of the scalar floating-point compare instructions. vmfeq and vmfne raise the invalid operation exception only on signaling NaN inputs. vmflt, vmfle, vmfgt, and vmfge raise the invalid operation exception on both signaling and quiet NaN inputs. vmfne writes 1 to the destination element when either operand is NaN, whereas the other compares write 0 when either operand is NaN.<br># Compare equal vmfeq.vv vd, vs2, vs1, vm # Vector-vector <b>vmfeq.vf</b> vd, vs2, rs1, vm # vector-scalar # Compare not equal vmfne.vv vd, vs2, vs1, vm # Vector-vector vmfne.vf vd, vs2, rs1, vm # vector-scalar # Compare less than vmflt.vv vd, vs2, vs1, vm # Vector-vector vmflt.vf vd, vs2, rs1, vm # vector-scalar # Compare less than or equal vmfle.vv vd, vs2, vs1, vm # Vector-vector vmfle.vf vd, vs2, rs1, vm # vector-scalar # Compare greater than vmfgt.vf vd, vs2, rs1, vm # vector-scalar # Compare greater than or equal vmfge.vf vd, vs2, rs1, vm # vector-scalar<br># Example of implementing isgreater() vmfeq.vv v0, va, va # Only set where A is not NaN. vmfeq.vv v1, vb, vb # Only set where B is not NaN. vmand.mm v0, v0, v1 # Only set where A and B are ordered, vmfgt.vv v0, va, vb, v0.t # so only set flags on ordered values.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The compare instructions follow the semantics of the scalar floating-point compare instructions. vmfeq and vmfne raise the invalid operation exception only on signaling NaN inputs. vmflt, vmfle, vmfgt, and vmfge raise the invalid operation exception on both signaling and quiet NaN inputs. vmfne writes 1 to the destination element when either operand is NaN, whereas the other compares write 0 when either operand is NaN.\n# Compare equal vmfeq.vv vd, vs2, vs1, vm # Vector-vector vmfeq.vf vd, vs2, rs1, vm # vector-scalar # Compare not equal vmfne.vv vd, vs2, vs1, vm # Vector-vector vmfne.vf vd, vs2, rs1, vm # vector-scalar # Compare less than vmflt.vv vd, vs2, vs1, vm # Vector-vector vmflt.vf vd, vs2, rs1, vm # vector-scalar # Compare less than or equal vmfle.vv vd, vs2, vs1, vm # Vector-vector vmfle.vf vd, vs2, rs1, vm # vector-scalar # Compare greater than vmfgt.vf vd, vs2, rs1, vm # vector-scalar # Compare greater than or equal vmfge.vf vd, vs2, rs1, vm # vector-scalar\n# Example of implementing isgreater() vmfeq.vv v0, va, va # Only set where A is not NaN. vmfeq.vv v1, vb, vb # Only set where B is not NaN. vmand.mm v0, v0, v1 # Only set where A and B are ordered, vmfgt.vv v0, va, vb, v0.t # so only set flags on ordered values.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_floating_point_compare_instructions"
            };

        case "VMFEQ.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMFEQ.VV</b> vs2, vs1, vd</span><br><div>The compare instructions follow the semantics of the scalar floating-point compare instructions. vmfeq and vmfne raise the invalid operation exception only on signaling NaN inputs. vmflt, vmfle, vmfgt, and vmfge raise the invalid operation exception on both signaling and quiet NaN inputs. vmfne writes 1 to the destination element when either operand is NaN, whereas the other compares write 0 when either operand is NaN.<br># Compare equal <b>vmfeq.vv</b> vd, vs2, vs1, vm # Vector-vector vmfeq.vf vd, vs2, rs1, vm # vector-scalar # Compare not equal vmfne.vv vd, vs2, vs1, vm # Vector-vector vmfne.vf vd, vs2, rs1, vm # vector-scalar # Compare less than vmflt.vv vd, vs2, vs1, vm # Vector-vector vmflt.vf vd, vs2, rs1, vm # vector-scalar # Compare less than or equal vmfle.vv vd, vs2, vs1, vm # Vector-vector vmfle.vf vd, vs2, rs1, vm # vector-scalar # Compare greater than vmfgt.vf vd, vs2, rs1, vm # vector-scalar # Compare greater than or equal vmfge.vf vd, vs2, rs1, vm # vector-scalar<br># Example of implementing isgreater() <b>vmfeq.vv</b> v0, va, va # Only set where A is not NaN. <b>vmfeq.vv</b> v1, vb, vb # Only set where B is not NaN. vmand.mm v0, v0, v1 # Only set where A and B are ordered, vmfgt.vv v0, va, vb, v0.t # so only set flags on ordered values.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The compare instructions follow the semantics of the scalar floating-point compare instructions. vmfeq and vmfne raise the invalid operation exception only on signaling NaN inputs. vmflt, vmfle, vmfgt, and vmfge raise the invalid operation exception on both signaling and quiet NaN inputs. vmfne writes 1 to the destination element when either operand is NaN, whereas the other compares write 0 when either operand is NaN.\n# Compare equal vmfeq.vv vd, vs2, vs1, vm # Vector-vector vmfeq.vf vd, vs2, rs1, vm # vector-scalar # Compare not equal vmfne.vv vd, vs2, vs1, vm # Vector-vector vmfne.vf vd, vs2, rs1, vm # vector-scalar # Compare less than vmflt.vv vd, vs2, vs1, vm # Vector-vector vmflt.vf vd, vs2, rs1, vm # vector-scalar # Compare less than or equal vmfle.vv vd, vs2, vs1, vm # Vector-vector vmfle.vf vd, vs2, rs1, vm # vector-scalar # Compare greater than vmfgt.vf vd, vs2, rs1, vm # vector-scalar # Compare greater than or equal vmfge.vf vd, vs2, rs1, vm # vector-scalar\n# Example of implementing isgreater() vmfeq.vv v0, va, va # Only set where A is not NaN. vmfeq.vv v1, vb, vb # Only set where B is not NaN. vmand.mm v0, v0, v1 # Only set where A and B are ordered, vmfgt.vv v0, va, vb, v0.t # so only set flags on ordered values.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_floating_point_compare_instructions"
            };

        case "VMFGE.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VMFGE.VF</b> vs2, rs1, vd</span><br><div>vmfge</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmfge\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMFGT.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VMFGT.VF</b> vs2, rs1, vd</span><br><div>vmfgt</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmfgt\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMFLE.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VMFLE.VF</b> vs2, rs1, vd</span><br><div>vmfle</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmfle\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMFLE.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMFLE.VV</b> vs2, vs1, vd</span><br><div>vmfle</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmfle\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMFLT.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VMFLT.VF</b> vs2, rs1, vd</span><br><div>Comparison Assembler Mapping Assembler pseudoinstruction va < vb vmflt.vv vd, va, vb, vm va <= vb vmfle.vv vd, va, vb, vm va > vb vmflt.vv vd, vb, va, vm vmfgt.vv vd, va, vb, vm va >= vb vmfle.vv vd, vb, va, vm vmfge.vv vd, va, vb, vm va < f <b>vmflt.vf</b> vd, va, f, vm va <= f vmfle.vf vd, va, f, vm va > f vmfgt.vf vd, va, f, vm va >= f vmfge.vf vd, va, f, vm va, vb vector register groups f scalar floating-point register</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "Comparison Assembler Mapping Assembler pseudoinstruction va < vb vmflt.vv vd, va, vb, vm va <= vb vmfle.vv vd, va, vb, vm va > vb vmflt.vv vd, vb, va, vm vmfgt.vv vd, va, vb, vm va >= vb vmfle.vv vd, vb, va, vm vmfge.vv vd, va, vb, vm va < f vmflt.vf vd, va, f, vm va <= f vmfle.vf vd, va, f, vm va > f vmfgt.vf vd, va, f, vm va >= f vmfge.vf vd, va, f, vm va, vb vector register groups f scalar floating-point register\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_floating_point_compare_instructions"
            };

        case "VMFLT.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMFLT.VV</b> vs2, vs1, vd</span><br><div>Comparison Assembler Mapping Assembler pseudoinstruction va < vb <b>vmflt.vv</b> vd, va, vb, vm va <= vb vmfle.vv vd, va, vb, vm va > vb <b>vmflt.vv</b> vd, vb, va, vm vmfgt.vv vd, va, vb, vm va >= vb vmfle.vv vd, vb, va, vm vmfge.vv vd, va, vb, vm va < f vmflt.vf vd, va, f, vm va <= f vmfle.vf vd, va, f, vm va > f vmfgt.vf vd, va, f, vm va >= f vmfge.vf vd, va, f, vm va, vb vector register groups f scalar floating-point register</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "Comparison Assembler Mapping Assembler pseudoinstruction va < vb vmflt.vv vd, va, vb, vm va <= vb vmfle.vv vd, va, vb, vm va > vb vmflt.vv vd, vb, va, vm vmfgt.vv vd, va, vb, vm va >= vb vmfle.vv vd, vb, va, vm vmfge.vv vd, va, vb, vm va < f vmflt.vf vd, va, f, vm va <= f vmfle.vf vd, va, f, vm va > f vmfgt.vf vd, va, f, vm va >= f vmfge.vf vd, va, f, vm va, vb vector register groups f scalar floating-point register\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_floating_point_compare_instructions"
            };

        case "VMFNE.VF":
            return {
                "html": "<div><span class=\"opcode\"><b>VMFNE.VF</b> vs2, rs1, vd</span><br><div>vmfne</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmfne\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMFNE.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMFNE.VV</b> vs2, vs1, vd</span><br><div>vmfne</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmfne\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMIN.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMIN.VV</b> vs2, vs1, vd</span><br><div>vmin</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmin\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMIN.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMIN.VX</b> vs2, rs1, vd</span><br><div>vmin</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmin\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMINU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMINU.VV</b> vs2, vs1, vd</span><br><div># Unsigned minimum <b>vminu.vv</b> vd, vs2, vs1, vm # Vector-vector vminu.vx vd, vs2, rs1, vm # vector-scalar # Signed minimum vmin.vv vd, vs2, vs1, vm # Vector-vector vmin.vx vd, vs2, rs1, vm # vector-scalar # Unsigned maximum vmaxu.vv vd, vs2, vs1, vm # Vector-vector vmaxu.vx vd, vs2, rs1, vm # vector-scalar # Signed maximum vmax.vv vd, vs2, vs1, vm # Vector-vector vmax.vx vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Unsigned minimum vminu.vv vd, vs2, vs1, vm # Vector-vector vminu.vx vd, vs2, rs1, vm # vector-scalar # Signed minimum vmin.vv vd, vs2, vs1, vm # Vector-vector vmin.vx vd, vs2, rs1, vm # vector-scalar # Unsigned maximum vmaxu.vv vd, vs2, vs1, vm # Vector-vector vmaxu.vx vd, vs2, rs1, vm # vector-scalar # Signed maximum vmax.vv vd, vs2, vs1, vm # Vector-vector vmax.vx vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_minmax_instructions"
            };

        case "VMINU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMINU.VX</b> vs2, rs1, vd</span><br><div># Unsigned minimum vminu.vv vd, vs2, vs1, vm # Vector-vector <b>vminu.vx</b> vd, vs2, rs1, vm # vector-scalar # Signed minimum vmin.vv vd, vs2, vs1, vm # Vector-vector vmin.vx vd, vs2, rs1, vm # vector-scalar # Unsigned maximum vmaxu.vv vd, vs2, vs1, vm # Vector-vector vmaxu.vx vd, vs2, rs1, vm # vector-scalar # Signed maximum vmax.vv vd, vs2, vs1, vm # Vector-vector vmax.vx vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Unsigned minimum vminu.vv vd, vs2, vs1, vm # Vector-vector vminu.vx vd, vs2, rs1, vm # vector-scalar # Signed minimum vmin.vv vd, vs2, vs1, vm # Vector-vector vmin.vx vd, vs2, rs1, vm # vector-scalar # Unsigned maximum vmaxu.vv vd, vs2, vs1, vm # Vector-vector vmaxu.vx vd, vs2, rs1, vm # vector-scalar # Signed maximum vmax.vv vd, vs2, vs1, vm # Vector-vector vmax.vx vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_minmax_instructions"
            };

        case "VMNAND.MM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMNAND.MM</b> vs2, vs1, vd</span><br><div><b>vmnand.mm</b> vd, src1, src1<br><b>vmnand.mm</b> vd, src2, src2<br><b>vmnand.mm</b> vd, src1, src2</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmnand.mm vd, src1, src1\nvmnand.mm vd, src2, src2\nvmnand.mm vd, src1, src2\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-mask-register-logical"
            };

        case "VMNOR.MM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMNOR.MM</b> vs2, vs1, vd</span><br><div><b>vmnor.mm</b> vd, src1, src2</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmnor.mm vd, src1, src2\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-mask-register-logical"
            };

        case "VMOR.MM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMOR.MM</b> vs2, vs1, vd</span><br><div>vmor</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmor\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMORN.MM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMORN.MM</b> vs2, vs1, vd</span><br><div><b>vmorn.mm</b> vd, src2, src1<br><b>vmorn.mm</b> vd, src1, src2</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmorn.mm vd, src2, src1\nvmorn.mm vd, src1, src2\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-mask-register-logical"
            };

        case "VMORNOT.MM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMORNOT.MM</b> vs2, vs1, vd</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VMSBC.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSBC.VV</b> vs2, vs1, vd</span><br><div>For vmsbc, the borrow is defined to be 1 iff the difference, prior to truncation, is negative.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "For vmsbc, the borrow is defined to be 1 iff the difference, prior to truncation, is negative.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VMSBC.VVM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSBC.VVM</b> vs2, vs1, vd</span><br><div>For vmsbc, the borrow is defined to be 1 iff the difference, prior to truncation, is negative.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "For vmsbc, the borrow is defined to be 1 iff the difference, prior to truncation, is negative.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VMSBC.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSBC.VX</b> vs2, rs1, vd</span><br><div>For vmsbc, the borrow is defined to be 1 iff the difference, prior to truncation, is negative.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "For vmsbc, the borrow is defined to be 1 iff the difference, prior to truncation, is negative.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VMSBC.VXM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSBC.VXM</b> vs2, rs1, vd</span><br><div>For vmsbc, the borrow is defined to be 1 iff the difference, prior to truncation, is negative.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "For vmsbc, the borrow is defined to be 1 iff the difference, prior to truncation, is negative.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VMSBF.M":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSBF.M</b> vs2, vd</span><br><div>In addition, except for mask load instructions, any element in the tail of a mask result can also be written with the value the mask-producing operation would have calculated with vl=VLMAX. Furthermore, for mask-logical instructions and <b>vmsbf.m</b>, vmsif.m, vmsof.m mask-manipulation instructions, any element in the tail of the result can be written with the value the mask-producing operation would have calculated with vl=VLEN, SEW=8, and LMUL=8 (i.e., all bits of the mask register can be overwritten).</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "In addition, except for mask load instructions, any element in the tail of a mask result can also be written with the value the mask-producing operation would have calculated with vl=VLMAX. Furthermore, for mask-logical instructions and vmsbf.m, vmsif.m, vmsof.m mask-manipulation instructions, any element in the tail of the result can be written with the value the mask-producing operation would have calculated with vl=VLEN, SEW=8, and LMUL=8 (i.e., all bits of the mask register can be overwritten).\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-agnostic"
            };

        case "VMSEQ.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSEQ.VI</b> vs2, simm5, vd</span><br><div># Set if equal vmseq.vv vd, vs2, vs1, vm # Vector-vector vmseq.vx vd, vs2, rs1, vm # vector-scalar <b>vmseq.vi</b> vd, vs2, imm, vm # vector-immediate # Set if not equal vmsne.vv vd, vs2, vs1, vm # Vector-vector vmsne.vx vd, vs2, rs1, vm # vector-scalar vmsne.vi vd, vs2, imm, vm # vector-immediate # Set if less than, unsigned vmsltu.vv vd, vs2, vs1, vm # Vector-vector vmsltu.vx vd, vs2, rs1, vm # Vector-scalar # Set if less than, signed vmslt.vv vd, vs2, vs1, vm # Vector-vector vmslt.vx vd, vs2, rs1, vm # vector-scalar # Set if less than or equal, unsigned vmsleu.vv vd, vs2, vs1, vm # Vector-vector vmsleu.vx vd, vs2, rs1, vm # vector-scalar vmsleu.vi vd, vs2, imm, vm # Vector-immediate # Set if less than or equal, signed vmsle.vv vd, vs2, vs1, vm # Vector-vector vmsle.vx vd, vs2, rs1, vm # vector-scalar vmsle.vi vd, vs2, imm, vm # vector-immediate # Set if greater than, unsigned vmsgtu.vx vd, vs2, rs1, vm # Vector-scalar vmsgtu.vi vd, vs2, imm, vm # Vector-immediate # Set if greater than, signed vmsgt.vx vd, vs2, rs1, vm # Vector-scalar vmsgt.vi vd, vs2, imm, vm # Vector-immediate # Following two instructions are not provided directly # Set if greater than or equal, unsigned # vmsgeu.vx vd, vs2, rs1, vm # Vector-scalar # Set if greater than or equal, signed # vmsge.vx vd, vs2, rs1, vm # Vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Set if equal vmseq.vv vd, vs2, vs1, vm # Vector-vector vmseq.vx vd, vs2, rs1, vm # vector-scalar vmseq.vi vd, vs2, imm, vm # vector-immediate # Set if not equal vmsne.vv vd, vs2, vs1, vm # Vector-vector vmsne.vx vd, vs2, rs1, vm # vector-scalar vmsne.vi vd, vs2, imm, vm # vector-immediate # Set if less than, unsigned vmsltu.vv vd, vs2, vs1, vm # Vector-vector vmsltu.vx vd, vs2, rs1, vm # Vector-scalar # Set if less than, signed vmslt.vv vd, vs2, vs1, vm # Vector-vector vmslt.vx vd, vs2, rs1, vm # vector-scalar # Set if less than or equal, unsigned vmsleu.vv vd, vs2, vs1, vm # Vector-vector vmsleu.vx vd, vs2, rs1, vm # vector-scalar vmsleu.vi vd, vs2, imm, vm # Vector-immediate # Set if less than or equal, signed vmsle.vv vd, vs2, vs1, vm # Vector-vector vmsle.vx vd, vs2, rs1, vm # vector-scalar vmsle.vi vd, vs2, imm, vm # vector-immediate # Set if greater than, unsigned vmsgtu.vx vd, vs2, rs1, vm # Vector-scalar vmsgtu.vi vd, vs2, imm, vm # Vector-immediate # Set if greater than, signed vmsgt.vx vd, vs2, rs1, vm # Vector-scalar vmsgt.vi vd, vs2, imm, vm # Vector-immediate # Following two instructions are not provided directly # Set if greater than or equal, unsigned # vmsgeu.vx vd, vs2, rs1, vm # Vector-scalar # Set if greater than or equal, signed # vmsge.vx vd, vs2, rs1, vm # Vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_compare_instructions"
            };

        case "VMSEQ.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSEQ.VV</b> vs2, vs1, vd</span><br><div># Set if equal <b>vmseq.vv</b> vd, vs2, vs1, vm # Vector-vector vmseq.vx vd, vs2, rs1, vm # vector-scalar vmseq.vi vd, vs2, imm, vm # vector-immediate # Set if not equal vmsne.vv vd, vs2, vs1, vm # Vector-vector vmsne.vx vd, vs2, rs1, vm # vector-scalar vmsne.vi vd, vs2, imm, vm # vector-immediate # Set if less than, unsigned vmsltu.vv vd, vs2, vs1, vm # Vector-vector vmsltu.vx vd, vs2, rs1, vm # Vector-scalar # Set if less than, signed vmslt.vv vd, vs2, vs1, vm # Vector-vector vmslt.vx vd, vs2, rs1, vm # vector-scalar # Set if less than or equal, unsigned vmsleu.vv vd, vs2, vs1, vm # Vector-vector vmsleu.vx vd, vs2, rs1, vm # vector-scalar vmsleu.vi vd, vs2, imm, vm # Vector-immediate # Set if less than or equal, signed vmsle.vv vd, vs2, vs1, vm # Vector-vector vmsle.vx vd, vs2, rs1, vm # vector-scalar vmsle.vi vd, vs2, imm, vm # vector-immediate # Set if greater than, unsigned vmsgtu.vx vd, vs2, rs1, vm # Vector-scalar vmsgtu.vi vd, vs2, imm, vm # Vector-immediate # Set if greater than, signed vmsgt.vx vd, vs2, rs1, vm # Vector-scalar vmsgt.vi vd, vs2, imm, vm # Vector-immediate # Following two instructions are not provided directly # Set if greater than or equal, unsigned # vmsgeu.vx vd, vs2, rs1, vm # Vector-scalar # Set if greater than or equal, signed # vmsge.vx vd, vs2, rs1, vm # Vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Set if equal vmseq.vv vd, vs2, vs1, vm # Vector-vector vmseq.vx vd, vs2, rs1, vm # vector-scalar vmseq.vi vd, vs2, imm, vm # vector-immediate # Set if not equal vmsne.vv vd, vs2, vs1, vm # Vector-vector vmsne.vx vd, vs2, rs1, vm # vector-scalar vmsne.vi vd, vs2, imm, vm # vector-immediate # Set if less than, unsigned vmsltu.vv vd, vs2, vs1, vm # Vector-vector vmsltu.vx vd, vs2, rs1, vm # Vector-scalar # Set if less than, signed vmslt.vv vd, vs2, vs1, vm # Vector-vector vmslt.vx vd, vs2, rs1, vm # vector-scalar # Set if less than or equal, unsigned vmsleu.vv vd, vs2, vs1, vm # Vector-vector vmsleu.vx vd, vs2, rs1, vm # vector-scalar vmsleu.vi vd, vs2, imm, vm # Vector-immediate # Set if less than or equal, signed vmsle.vv vd, vs2, vs1, vm # Vector-vector vmsle.vx vd, vs2, rs1, vm # vector-scalar vmsle.vi vd, vs2, imm, vm # vector-immediate # Set if greater than, unsigned vmsgtu.vx vd, vs2, rs1, vm # Vector-scalar vmsgtu.vi vd, vs2, imm, vm # Vector-immediate # Set if greater than, signed vmsgt.vx vd, vs2, rs1, vm # Vector-scalar vmsgt.vi vd, vs2, imm, vm # Vector-immediate # Following two instructions are not provided directly # Set if greater than or equal, unsigned # vmsgeu.vx vd, vs2, rs1, vm # Vector-scalar # Set if greater than or equal, signed # vmsge.vx vd, vs2, rs1, vm # Vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_compare_instructions"
            };

        case "VMSEQ.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSEQ.VX</b> vs2, rs1, vd</span><br><div># Set if equal vmseq.vv vd, vs2, vs1, vm # Vector-vector <b>vmseq.vx</b> vd, vs2, rs1, vm # vector-scalar vmseq.vi vd, vs2, imm, vm # vector-immediate # Set if not equal vmsne.vv vd, vs2, vs1, vm # Vector-vector vmsne.vx vd, vs2, rs1, vm # vector-scalar vmsne.vi vd, vs2, imm, vm # vector-immediate # Set if less than, unsigned vmsltu.vv vd, vs2, vs1, vm # Vector-vector vmsltu.vx vd, vs2, rs1, vm # Vector-scalar # Set if less than, signed vmslt.vv vd, vs2, vs1, vm # Vector-vector vmslt.vx vd, vs2, rs1, vm # vector-scalar # Set if less than or equal, unsigned vmsleu.vv vd, vs2, vs1, vm # Vector-vector vmsleu.vx vd, vs2, rs1, vm # vector-scalar vmsleu.vi vd, vs2, imm, vm # Vector-immediate # Set if less than or equal, signed vmsle.vv vd, vs2, vs1, vm # Vector-vector vmsle.vx vd, vs2, rs1, vm # vector-scalar vmsle.vi vd, vs2, imm, vm # vector-immediate # Set if greater than, unsigned vmsgtu.vx vd, vs2, rs1, vm # Vector-scalar vmsgtu.vi vd, vs2, imm, vm # Vector-immediate # Set if greater than, signed vmsgt.vx vd, vs2, rs1, vm # Vector-scalar vmsgt.vi vd, vs2, imm, vm # Vector-immediate # Following two instructions are not provided directly # Set if greater than or equal, unsigned # vmsgeu.vx vd, vs2, rs1, vm # Vector-scalar # Set if greater than or equal, signed # vmsge.vx vd, vs2, rs1, vm # Vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Set if equal vmseq.vv vd, vs2, vs1, vm # Vector-vector vmseq.vx vd, vs2, rs1, vm # vector-scalar vmseq.vi vd, vs2, imm, vm # vector-immediate # Set if not equal vmsne.vv vd, vs2, vs1, vm # Vector-vector vmsne.vx vd, vs2, rs1, vm # vector-scalar vmsne.vi vd, vs2, imm, vm # vector-immediate # Set if less than, unsigned vmsltu.vv vd, vs2, vs1, vm # Vector-vector vmsltu.vx vd, vs2, rs1, vm # Vector-scalar # Set if less than, signed vmslt.vv vd, vs2, vs1, vm # Vector-vector vmslt.vx vd, vs2, rs1, vm # vector-scalar # Set if less than or equal, unsigned vmsleu.vv vd, vs2, vs1, vm # Vector-vector vmsleu.vx vd, vs2, rs1, vm # vector-scalar vmsleu.vi vd, vs2, imm, vm # Vector-immediate # Set if less than or equal, signed vmsle.vv vd, vs2, vs1, vm # Vector-vector vmsle.vx vd, vs2, rs1, vm # vector-scalar vmsle.vi vd, vs2, imm, vm # vector-immediate # Set if greater than, unsigned vmsgtu.vx vd, vs2, rs1, vm # Vector-scalar vmsgtu.vi vd, vs2, imm, vm # Vector-immediate # Set if greater than, signed vmsgt.vx vd, vs2, rs1, vm # Vector-scalar vmsgt.vi vd, vs2, imm, vm # Vector-immediate # Following two instructions are not provided directly # Set if greater than or equal, unsigned # vmsgeu.vx vd, vs2, rs1, vm # Vector-scalar # Set if greater than or equal, signed # vmsge.vx vd, vs2, rs1, vm # Vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_compare_instructions"
            };

        case "VMSGT.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSGT.VI</b> vs2, simm5, vd</span><br><div>Similarly, vmsge{u}.vi is not provided and the compare is implemented using vmsgt{u}.vi with the immediate decremented by one. The resulting effective vmsge.vi range is -15 to 16, and the resulting effective vmsgeu.vi range is 1 to 16 (Note, vmsgeu.vi with immediate 0 is not useful as it is always true).<br>The vmsge{u}.vx operation can be synthesized by reducing the value of x by 1 and using the vmsgt{u}.vx instruction, when it is known that this will not underflow the representation in x.<br>Sequences to synthesize `vmsge{u}.vx` instruction va >= x, x > minimum addi t0, x, -1; vmsgt{u}.vx vd, va, t0, vm</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "Similarly, vmsge{u}.vi is not provided and the compare is implemented using vmsgt{u}.vi with the immediate decremented by one. The resulting effective vmsge.vi range is -15 to 16, and the resulting effective vmsgeu.vi range is 1 to 16 (Note, vmsgeu.vi with immediate 0 is not useful as it is always true).\nThe vmsge{u}.vx operation can be synthesized by reducing the value of x by 1 and using the vmsgt{u}.vx instruction, when it is known that this will not underflow the representation in x.\nSequences to synthesize `vmsge{u}.vx` instruction va >= x, x > minimum addi t0, x, -1; vmsgt{u}.vx vd, va, t0, vm\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_compare_instructions"
            };

        case "VMSGT.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSGT.VX</b> vs2, rs1, vd</span><br><div>Similarly, vmsge{u}.vi is not provided and the compare is implemented using vmsgt{u}.vi with the immediate decremented by one. The resulting effective vmsge.vi range is -15 to 16, and the resulting effective vmsgeu.vi range is 1 to 16 (Note, vmsgeu.vi with immediate 0 is not useful as it is always true).<br>The vmsge{u}.vx operation can be synthesized by reducing the value of x by 1 and using the vmsgt{u}.vx instruction, when it is known that this will not underflow the representation in x.<br>Sequences to synthesize `vmsge{u}.vx` instruction va >= x, x > minimum addi t0, x, -1; vmsgt{u}.vx vd, va, t0, vm</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "Similarly, vmsge{u}.vi is not provided and the compare is implemented using vmsgt{u}.vi with the immediate decremented by one. The resulting effective vmsge.vi range is -15 to 16, and the resulting effective vmsgeu.vi range is 1 to 16 (Note, vmsgeu.vi with immediate 0 is not useful as it is always true).\nThe vmsge{u}.vx operation can be synthesized by reducing the value of x by 1 and using the vmsgt{u}.vx instruction, when it is known that this will not underflow the representation in x.\nSequences to synthesize `vmsge{u}.vx` instruction va >= x, x > minimum addi t0, x, -1; vmsgt{u}.vx vd, va, t0, vm\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_compare_instructions"
            };

        case "VMSGTU.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSGTU.VI</b> vs2, simm5, vd</span><br><div>vmsgtu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmsgtu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMSGTU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSGTU.VX</b> vs2, rs1, vd</span><br><div>vmsgtu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmsgtu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMSIF.M":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSIF.M</b> vs2, vd</span><br><div>Traps on <b>vmsif.m</b> are always reported with a vstart of 0. The vmsif instruction will raise an illegal instruction exception if vstart is non-zero.<br><b>vmsif.m</b> vd, vs2, vm # Example 7 6 5 4 3 2 1 0 Element number 1 0 0 1 0 1 0 0 v3 contents <b>vmsif.m</b> v2, v3 0 0 0 0 0 1 1 1 v2 contents 1 0 0 1 0 1 0 1 v3 contents <b>vmsif.m</b> v2, v3 0 0 0 0 0 0 0 1 v2 1 1 0 0 0 0 1 1 v0 vcontents 1 0 0 1 0 1 0 0 v3 contents <b>vmsif.m</b> v2, v3, v0.t 1 1 x x x x 1 1 v2 contents</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "Traps on vmsif.m are always reported with a vstart of 0. The vmsif instruction will raise an illegal instruction exception if vstart is non-zero.\nvmsif.m vd, vs2, vm # Example 7 6 5 4 3 2 1 0 Element number 1 0 0 1 0 1 0 0 v3 contents vmsif.m v2, v3 0 0 0 0 0 1 1 1 v2 contents 1 0 0 1 0 1 0 1 v3 contents vmsif.m v2, v3 0 0 0 0 0 0 0 1 v2 1 1 0 0 0 0 1 1 v0 vcontents 1 0 0 1 0 1 0 0 v3 contents vmsif.m v2, v3, v0.t 1 1 x x x x 1 1 v2 contents\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vmsif_m_set_including_first_mask_bit"
            };

        case "VMSLE.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSLE.VI</b> vs2, simm5, vd</span><br><div>vmsle</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmsle\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMSLE.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSLE.VV</b> vs2, vs1, vd</span><br><div>vmsle</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmsle\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMSLE.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSLE.VX</b> vs2, rs1, vd</span><br><div>vmsle</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmsle\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMSLEU.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSLEU.VI</b> vs2, simm5, vd</span><br><div>vmsleu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmsleu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMSLEU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSLEU.VV</b> vs2, vs1, vd</span><br><div>vmsleu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmsleu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMSLEU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSLEU.VX</b> vs2, rs1, vd</span><br><div>vmsleu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmsleu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMSLT.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSLT.VV</b> vs2, vs1, vd</span><br><div>Comparison Assembler Mapping Assembler Pseudoinstruction va < vb vmslt{u}.vv vd, va, vb, vm va <= vb vmsle{u}.vv vd, va, vb, vm va > vb vmslt{u}.vv vd, vb, va, vm vmsgt{u}.vv vd, va, vb, vm va >= vb vmsle{u}.vv vd, vb, va, vm vmsge{u}.vv vd, va, vb, vm va < x vmslt{u}.vx vd, va, x, vm va <= x vmsle{u}.vx vd, va, x, vm va > x vmsgt{u}.vx vd, va, x, vm va >= x see below va < i vmsle{u}.vi vd, va, i-1, vm vmslt{u}.vi vd, va, i, vm va <= i vmsle{u}.vi vd, va, i, vm va > i vmsgt{u}.vi vd, va, i, vm va >= i vmsgt{u}.vi vd, va, i-1, vm vmsge{u}.vi vd, va, i, vm va, vb vector register groups x scalar integer register i immediate<br>unmasked va >= x pseudoinstruction: vmsge{u}.vx vd, va, x expansion: vmslt{u}.vx vd, va, x; vmnand.mm vd, vd, vd masked va >= x, vd != v0 pseudoinstruction: vmsge{u}.vx vd, va, x, v0.t expansion: vmslt{u}.vx vd, va, x, v0.t; vmxor.mm vd, vd, v0 masked va >= x, vd == v0 pseudoinstruction: vmsge{u}.vx vd, va, x, v0.t, vt expansion: vmslt{u}.vx vt, va, x; vmandn.mm vd, vd, vt masked va >= x, any vd pseudoinstruction: vmsge{u}.vx vd, va, x, v0.t, vt expansion: vmslt{u}.vx vt, va, x; vmandn.mm vt, v0, vt; vmandn.mm vd, vd, v0; vmor.mm vd, vt, vd The vt argument to the pseudoinstruction must name a temporary vector register that is not same as vd and which will be clobbered by the pseudoinstruction<br># (a < b) && (b < c) in two instructions when mask-undisturbed <b>vmslt.vv</b> v0, va, vb # All body elements written <b>vmslt.vv</b> v0, vb, vc, v0.t # Only update at set mask</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "Comparison Assembler Mapping Assembler Pseudoinstruction va < vb vmslt{u}.vv vd, va, vb, vm va <= vb vmsle{u}.vv vd, va, vb, vm va > vb vmslt{u}.vv vd, vb, va, vm vmsgt{u}.vv vd, va, vb, vm va >= vb vmsle{u}.vv vd, vb, va, vm vmsge{u}.vv vd, va, vb, vm va < x vmslt{u}.vx vd, va, x, vm va <= x vmsle{u}.vx vd, va, x, vm va > x vmsgt{u}.vx vd, va, x, vm va >= x see below va < i vmsle{u}.vi vd, va, i-1, vm vmslt{u}.vi vd, va, i, vm va <= i vmsle{u}.vi vd, va, i, vm va > i vmsgt{u}.vi vd, va, i, vm va >= i vmsgt{u}.vi vd, va, i-1, vm vmsge{u}.vi vd, va, i, vm va, vb vector register groups x scalar integer register i immediate\nunmasked va >= x pseudoinstruction: vmsge{u}.vx vd, va, x expansion: vmslt{u}.vx vd, va, x; vmnand.mm vd, vd, vd masked va >= x, vd != v0 pseudoinstruction: vmsge{u}.vx vd, va, x, v0.t expansion: vmslt{u}.vx vd, va, x, v0.t; vmxor.mm vd, vd, v0 masked va >= x, vd == v0 pseudoinstruction: vmsge{u}.vx vd, va, x, v0.t, vt expansion: vmslt{u}.vx vt, va, x; vmandn.mm vd, vd, vt masked va >= x, any vd pseudoinstruction: vmsge{u}.vx vd, va, x, v0.t, vt expansion: vmslt{u}.vx vt, va, x; vmandn.mm vt, v0, vt; vmandn.mm vd, vd, v0; vmor.mm vd, vt, vd The vt argument to the pseudoinstruction must name a temporary vector register that is not same as vd and which will be clobbered by the pseudoinstruction\n# (a < b) && (b < c) in two instructions when mask-undisturbed vmslt.vv v0, va, vb # All body elements written vmslt.vv v0, vb, vc, v0.t # Only update at set mask\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_compare_instructions"
            };

        case "VMSLT.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSLT.VX</b> vs2, rs1, vd</span><br><div>Comparison Assembler Mapping Assembler Pseudoinstruction va < vb vmslt{u}.vv vd, va, vb, vm va <= vb vmsle{u}.vv vd, va, vb, vm va > vb vmslt{u}.vv vd, vb, va, vm vmsgt{u}.vv vd, va, vb, vm va >= vb vmsle{u}.vv vd, vb, va, vm vmsge{u}.vv vd, va, vb, vm va < x vmslt{u}.vx vd, va, x, vm va <= x vmsle{u}.vx vd, va, x, vm va > x vmsgt{u}.vx vd, va, x, vm va >= x see below va < i vmsle{u}.vi vd, va, i-1, vm vmslt{u}.vi vd, va, i, vm va <= i vmsle{u}.vi vd, va, i, vm va > i vmsgt{u}.vi vd, va, i, vm va >= i vmsgt{u}.vi vd, va, i-1, vm vmsge{u}.vi vd, va, i, vm va, vb vector register groups x scalar integer register i immediate<br>unmasked va >= x pseudoinstruction: vmsge{u}.vx vd, va, x expansion: vmslt{u}.vx vd, va, x; vmnand.mm vd, vd, vd masked va >= x, vd != v0 pseudoinstruction: vmsge{u}.vx vd, va, x, v0.t expansion: vmslt{u}.vx vd, va, x, v0.t; vmxor.mm vd, vd, v0 masked va >= x, vd == v0 pseudoinstruction: vmsge{u}.vx vd, va, x, v0.t, vt expansion: vmslt{u}.vx vt, va, x; vmandn.mm vd, vd, vt masked va >= x, any vd pseudoinstruction: vmsge{u}.vx vd, va, x, v0.t, vt expansion: vmslt{u}.vx vt, va, x; vmandn.mm vt, v0, vt; vmandn.mm vd, vd, v0; vmor.mm vd, vt, vd The vt argument to the pseudoinstruction must name a temporary vector register that is not same as vd and which will be clobbered by the pseudoinstruction<br># (a < b) && (b < c) in two instructions when mask-undisturbed vmslt.vv v0, va, vb # All body elements written vmslt.vv v0, vb, vc, v0.t # Only update at set mask</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "Comparison Assembler Mapping Assembler Pseudoinstruction va < vb vmslt{u}.vv vd, va, vb, vm va <= vb vmsle{u}.vv vd, va, vb, vm va > vb vmslt{u}.vv vd, vb, va, vm vmsgt{u}.vv vd, va, vb, vm va >= vb vmsle{u}.vv vd, vb, va, vm vmsge{u}.vv vd, va, vb, vm va < x vmslt{u}.vx vd, va, x, vm va <= x vmsle{u}.vx vd, va, x, vm va > x vmsgt{u}.vx vd, va, x, vm va >= x see below va < i vmsle{u}.vi vd, va, i-1, vm vmslt{u}.vi vd, va, i, vm va <= i vmsle{u}.vi vd, va, i, vm va > i vmsgt{u}.vi vd, va, i, vm va >= i vmsgt{u}.vi vd, va, i-1, vm vmsge{u}.vi vd, va, i, vm va, vb vector register groups x scalar integer register i immediate\nunmasked va >= x pseudoinstruction: vmsge{u}.vx vd, va, x expansion: vmslt{u}.vx vd, va, x; vmnand.mm vd, vd, vd masked va >= x, vd != v0 pseudoinstruction: vmsge{u}.vx vd, va, x, v0.t expansion: vmslt{u}.vx vd, va, x, v0.t; vmxor.mm vd, vd, v0 masked va >= x, vd == v0 pseudoinstruction: vmsge{u}.vx vd, va, x, v0.t, vt expansion: vmslt{u}.vx vt, va, x; vmandn.mm vd, vd, vt masked va >= x, any vd pseudoinstruction: vmsge{u}.vx vd, va, x, v0.t, vt expansion: vmslt{u}.vx vt, va, x; vmandn.mm vt, v0, vt; vmandn.mm vd, vd, v0; vmor.mm vd, vt, vd The vt argument to the pseudoinstruction must name a temporary vector register that is not same as vd and which will be clobbered by the pseudoinstruction\n# (a < b) && (b < c) in two instructions when mask-undisturbed vmslt.vv v0, va, vb # All body elements written vmslt.vv v0, vb, vc, v0.t # Only update at set mask\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_compare_instructions"
            };

        case "VMSLTU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSLTU.VV</b> vs2, vs1, vd</span><br><div>vmsltu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmsltu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMSLTU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSLTU.VX</b> vs2, rs1, vd</span><br><div>vmsltu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmsltu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMSNE.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSNE.VI</b> vs2, simm5, vd</span><br><div>vmsne</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmsne\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMSNE.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSNE.VV</b> vs2, vs1, vd</span><br><div>vmsne</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmsne\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMSNE.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSNE.VX</b> vs2, rs1, vd</span><br><div>vmsne</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmsne\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMSOF.M":
            return {
                "html": "<div><span class=\"opcode\"><b>VMSOF.M</b> vs2, vd</span><br><div>Traps on <b>vmsof.m</b> are always reported with a vstart of 0. The vmsof instruction will raise an illegal instruction exception if vstart is non-zero.<br><b>vmsof.m</b> vd, vs2, vm # Example 7 6 5 4 3 2 1 0 Element number 1 0 0 1 0 1 0 0 v3 contents <b>vmsof.m</b> v2, v3 0 0 0 0 0 1 0 0 v2 contents 1 0 0 1 0 1 0 1 v3 contents <b>vmsof.m</b> v2, v3 0 0 0 0 0 0 0 1 v2 1 1 0 0 0 0 1 1 v0 vcontents 1 1 0 1 0 1 0 0 v3 contents <b>vmsof.m</b> v2, v3, v0.t 0 1 x x x x 0 0 v2 contents</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "Traps on vmsof.m are always reported with a vstart of 0. The vmsof instruction will raise an illegal instruction exception if vstart is non-zero.\nvmsof.m vd, vs2, vm # Example 7 6 5 4 3 2 1 0 Element number 1 0 0 1 0 1 0 0 v3 contents vmsof.m v2, v3 0 0 0 0 0 1 0 0 v2 contents 1 0 0 1 0 1 0 1 v3 contents vmsof.m v2, v3 0 0 0 0 0 0 0 1 v2 1 1 0 0 0 0 1 1 v0 vcontents 1 1 0 1 0 1 0 0 v3 contents vmsof.m v2, v3, v0.t 0 1 x x x x 0 0 v2 contents\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vmsof_m_set_only_first_mask_bit"
            };

        case "VMUL.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMUL.VV</b> vs2, vs1, vd</span><br><div># Signed multiply, returning low bits of product <b>vmul.vv</b> vd, vs2, vs1, vm # Vector-vector vmul.vx vd, vs2, rs1, vm # vector-scalar # Signed multiply, returning high bits of product vmulh.vv vd, vs2, vs1, vm # Vector-vector vmulh.vx vd, vs2, rs1, vm # vector-scalar # Unsigned multiply, returning high bits of product vmulhu.vv vd, vs2, vs1, vm # Vector-vector vmulhu.vx vd, vs2, rs1, vm # vector-scalar # Signed(vs2)-Unsigned multiply, returning high bits of product vmulhsu.vv vd, vs2, vs1, vm # Vector-vector vmulhsu.vx vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Signed multiply, returning low bits of product vmul.vv vd, vs2, vs1, vm # Vector-vector vmul.vx vd, vs2, rs1, vm # vector-scalar # Signed multiply, returning high bits of product vmulh.vv vd, vs2, vs1, vm # Vector-vector vmulh.vx vd, vs2, rs1, vm # vector-scalar # Unsigned multiply, returning high bits of product vmulhu.vv vd, vs2, vs1, vm # Vector-vector vmulhu.vx vd, vs2, rs1, vm # vector-scalar # Signed(vs2)-Unsigned multiply, returning high bits of product vmulhsu.vv vd, vs2, vs1, vm # Vector-vector vmulhsu.vx vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_integer_multiply_instructions"
            };

        case "VMUL.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMUL.VX</b> vs2, rs1, vd</span><br><div># Signed multiply, returning low bits of product vmul.vv vd, vs2, vs1, vm # Vector-vector <b>vmul.vx</b> vd, vs2, rs1, vm # vector-scalar # Signed multiply, returning high bits of product vmulh.vv vd, vs2, vs1, vm # Vector-vector vmulh.vx vd, vs2, rs1, vm # vector-scalar # Unsigned multiply, returning high bits of product vmulhu.vv vd, vs2, vs1, vm # Vector-vector vmulhu.vx vd, vs2, rs1, vm # vector-scalar # Signed(vs2)-Unsigned multiply, returning high bits of product vmulhsu.vv vd, vs2, vs1, vm # Vector-vector vmulhsu.vx vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Signed multiply, returning low bits of product vmul.vv vd, vs2, vs1, vm # Vector-vector vmul.vx vd, vs2, rs1, vm # vector-scalar # Signed multiply, returning high bits of product vmulh.vv vd, vs2, vs1, vm # Vector-vector vmulh.vx vd, vs2, rs1, vm # vector-scalar # Unsigned multiply, returning high bits of product vmulhu.vv vd, vs2, vs1, vm # Vector-vector vmulhu.vx vd, vs2, rs1, vm # vector-scalar # Signed(vs2)-Unsigned multiply, returning high bits of product vmulhsu.vv vd, vs2, vs1, vm # Vector-vector vmulhsu.vx vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_integer_multiply_instructions"
            };

        case "VMULH.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMULH.VV</b> vs2, vs1, vd</span><br><div>All Zve* extensions support all vector integer instructions (Section Vector Integer Arithmetic Instructions ), except that the vmulh integer multiply variants that return the high word of the product (<b>vmulh.vv</b>, vmulh.vx, vmulhu.vv, vmulhu.vx, vmulhsu.vv, vmulhsu.vx) are not included for EEW=64 in Zve64*.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "All Zve* extensions support all vector integer instructions (Section Vector Integer Arithmetic Instructions ), except that the vmulh integer multiply variants that return the high word of the product (vmulh.vv, vmulh.vx, vmulhu.vv, vmulhu.vx, vmulhsu.vv, vmulhsu.vx) are not included for EEW=64 in Zve64*.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_zve_vector_extensions_for_embedded_processors"
            };

        case "VMULH.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMULH.VX</b> vs2, rs1, vd</span><br><div>All Zve* extensions support all vector integer instructions (Section Vector Integer Arithmetic Instructions ), except that the vmulh integer multiply variants that return the high word of the product (vmulh.vv, <b>vmulh.vx</b>, vmulhu.vv, vmulhu.vx, vmulhsu.vv, vmulhsu.vx) are not included for EEW=64 in Zve64*.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "All Zve* extensions support all vector integer instructions (Section Vector Integer Arithmetic Instructions ), except that the vmulh integer multiply variants that return the high word of the product (vmulh.vv, vmulh.vx, vmulhu.vv, vmulhu.vx, vmulhsu.vv, vmulhsu.vx) are not included for EEW=64 in Zve64*.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_zve_vector_extensions_for_embedded_processors"
            };

        case "VMULHSU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMULHSU.VV</b> vs2, vs1, vd</span><br><div>vmulhsu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmulhsu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMULHSU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMULHSU.VX</b> vs2, rs1, vd</span><br><div>vmulhsu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmulhsu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMULHU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VMULHU.VV</b> vs2, vs1, vd</span><br><div>vmulhu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmulhu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMULHU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VMULHU.VX</b> vs2, rs1, vd</span><br><div>vmulhu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmulhu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VMV.S.X":
            return {
                "html": "<div><span class=\"opcode\"><b>VMV.S.X</b> rs1, vd</span><br><div>The vector integer move instructions copy a source operand to a vector register group. The vmv.v.v variant copies a vector register group, whereas the vmv.v.x and vmv.v.i variants splat a scalar register or immediate to all active elements of the destination vector register group. These instructions are encoded as unmasked instructions (vm=1). The first operand specifier (vs2) must contain v0, and any other vector register number in vs2 is reserved.<br>The form vmv.v.v vd, vd, which leaves body elements unchanged, can be used to indicate that the register will next be used with an EEW equal to SEW.<br>vmv.v.v vd, vs1 # vd[i] = vs1[i] vmv.v.x vd, rs1 # vd[i] = x[rs1] vmv.v.i vd, imm # vd[i] = imm</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vector integer move instructions copy a source operand to a vector register group. The vmv.v.v variant copies a vector register group, whereas the vmv.v.x and vmv.v.i variants splat a scalar register or immediate to all active elements of the destination vector register group. These instructions are encoded as unmasked instructions (vm=1). The first operand specifier (vs2) must contain v0, and any other vector register number in vs2 is reserved.\nThe form vmv.v.v vd, vd, which leaves body elements unchanged, can be used to indicate that the register will next be used with an EEW equal to SEW.\nvmv.v.v vd, vs1 # vd[i] = vs1[i] vmv.v.x vd, rs1 # vd[i] = x[rs1] vmv.v.i vd, imm # vd[i] = imm\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_move_instructions"
            };

        case "VMV.V.I":
            return {
                "html": "<div><span class=\"opcode\"><b>VMV.V.I</b> simm5, vd</span><br><div>The vector integer move instructions copy a source operand to a vector register group. The vmv.v.v variant copies a vector register group, whereas the vmv.v.x and <b>vmv.v.i</b> variants splat a scalar register or immediate to all active elements of the destination vector register group. These instructions are encoded as unmasked instructions (vm=1). The first operand specifier (vs2) must contain v0, and any other vector register number in vs2 is reserved.<br>The form vmv.v.v vd, vd, which leaves body elements unchanged, can be used to indicate that the register will next be used with an EEW equal to SEW.<br>vmv.v.v vd, vs1 # vd[i] = vs1[i] vmv.v.x vd, rs1 # vd[i] = x[rs1] <b>vmv.v.i</b> vd, imm # vd[i] = imm</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vector integer move instructions copy a source operand to a vector register group. The vmv.v.v variant copies a vector register group, whereas the vmv.v.x and vmv.v.i variants splat a scalar register or immediate to all active elements of the destination vector register group. These instructions are encoded as unmasked instructions (vm=1). The first operand specifier (vs2) must contain v0, and any other vector register number in vs2 is reserved.\nThe form vmv.v.v vd, vd, which leaves body elements unchanged, can be used to indicate that the register will next be used with an EEW equal to SEW.\nvmv.v.v vd, vs1 # vd[i] = vs1[i] vmv.v.x vd, rs1 # vd[i] = x[rs1] vmv.v.i vd, imm # vd[i] = imm\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_move_instructions"
            };

        case "VMV.V.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VMV.V.V</b> vs1, vd</span><br><div>The vector integer move instructions copy a source operand to a vector register group. The <b>vmv.v.v</b> variant copies a vector register group, whereas the vmv.v.x and vmv.v.i variants splat a scalar register or immediate to all active elements of the destination vector register group. These instructions are encoded as unmasked instructions (vm=1). The first operand specifier (vs2) must contain v0, and any other vector register number in vs2 is reserved.<br>The form <b>vmv.v.v</b> vd, vd, which leaves body elements unchanged, can be used to indicate that the register will next be used with an EEW equal to SEW.<br><b>vmv.v.v</b> vd, vs1 # vd[i] = vs1[i] vmv.v.x vd, rs1 # vd[i] = x[rs1] vmv.v.i vd, imm # vd[i] = imm</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vector integer move instructions copy a source operand to a vector register group. The vmv.v.v variant copies a vector register group, whereas the vmv.v.x and vmv.v.i variants splat a scalar register or immediate to all active elements of the destination vector register group. These instructions are encoded as unmasked instructions (vm=1). The first operand specifier (vs2) must contain v0, and any other vector register number in vs2 is reserved.\nThe form vmv.v.v vd, vd, which leaves body elements unchanged, can be used to indicate that the register will next be used with an EEW equal to SEW.\nvmv.v.v vd, vs1 # vd[i] = vs1[i] vmv.v.x vd, rs1 # vd[i] = x[rs1] vmv.v.i vd, imm # vd[i] = imm\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_move_instructions"
            };

        case "VMV.V.X":
            return {
                "html": "<div><span class=\"opcode\"><b>VMV.V.X</b> rs1, vd</span><br><div>The vector integer move instructions copy a source operand to a vector register group. The vmv.v.v variant copies a vector register group, whereas the <b>vmv.v.x</b> and vmv.v.i variants splat a scalar register or immediate to all active elements of the destination vector register group. These instructions are encoded as unmasked instructions (vm=1). The first operand specifier (vs2) must contain v0, and any other vector register number in vs2 is reserved.<br>The form vmv.v.v vd, vd, which leaves body elements unchanged, can be used to indicate that the register will next be used with an EEW equal to SEW.<br>vmv.v.v vd, vs1 # vd[i] = vs1[i] <b>vmv.v.x</b> vd, rs1 # vd[i] = x[rs1] vmv.v.i vd, imm # vd[i] = imm</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vector integer move instructions copy a source operand to a vector register group. The vmv.v.v variant copies a vector register group, whereas the vmv.v.x and vmv.v.i variants splat a scalar register or immediate to all active elements of the destination vector register group. These instructions are encoded as unmasked instructions (vm=1). The first operand specifier (vs2) must contain v0, and any other vector register number in vs2 is reserved.\nThe form vmv.v.v vd, vd, which leaves body elements unchanged, can be used to indicate that the register will next be used with an EEW equal to SEW.\nvmv.v.v vd, vs1 # vd[i] = vs1[i] vmv.v.x vd, rs1 # vd[i] = x[rs1] vmv.v.i vd, imm # vd[i] = imm\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_move_instructions"
            };

        case "VMV.X.S":
            return {
                "html": "<div><span class=\"opcode\"><b>VMV.X.S</b> vs2, rd</span><br><div>The vector integer move instructions copy a source operand to a vector register group. The vmv.v.v variant copies a vector register group, whereas the vmv.v.x and vmv.v.i variants splat a scalar register or immediate to all active elements of the destination vector register group. These instructions are encoded as unmasked instructions (vm=1). The first operand specifier (vs2) must contain v0, and any other vector register number in vs2 is reserved.<br>The form vmv.v.v vd, vd, which leaves body elements unchanged, can be used to indicate that the register will next be used with an EEW equal to SEW.<br>vmv.v.v vd, vs1 # vd[i] = vs1[i] vmv.v.x vd, rs1 # vd[i] = x[rs1] vmv.v.i vd, imm # vd[i] = imm</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vector integer move instructions copy a source operand to a vector register group. The vmv.v.v variant copies a vector register group, whereas the vmv.v.x and vmv.v.i variants splat a scalar register or immediate to all active elements of the destination vector register group. These instructions are encoded as unmasked instructions (vm=1). The first operand specifier (vs2) must contain v0, and any other vector register number in vs2 is reserved.\nThe form vmv.v.v vd, vd, which leaves body elements unchanged, can be used to indicate that the register will next be used with an EEW equal to SEW.\nvmv.v.v vd, vs1 # vd[i] = vs1[i] vmv.v.x vd, rs1 # vd[i] = x[rs1] vmv.v.i vd, imm # vd[i] = imm\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_move_instructions"
            };

        case "VMV1R.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VMV1R.V</b> vs2, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VMV2R.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VMV2R.V</b> vs2, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VMV4R.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VMV4R.V</b> vs2, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VMV8R.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VMV8R.V</b> vs2, vd</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VMXNOR.MM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMXNOR.MM</b> vs2, vs1, vd</span><br><div><b>vmxnor.mm</b> vd, src1, src2<br><b>vmxnor.mm</b> vd, vd, vd</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmxnor.mm vd, src1, src2\nvmxnor.mm vd, vd, vd\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-mask-register-logical"
            };

        case "VMXOR.MM":
            return {
                "html": "<div><span class=\"opcode\"><b>VMXOR.MM</b> vs2, vs1, vd</span><br><div><b>vmxor.mm</b> vd, vd, vd<br><b>vmxor.mm</b> vd, src1, src2</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vmxor.mm vd, vd, vd\nvmxor.mm vd, src1, src2\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-mask-register-logical"
            };

        case "VNCLIP.WI":
            return {
                "html": "<div><span class=\"opcode\"><b>VNCLIP.WI</b> vs2, simm5, vd</span><br><div>The vnclip instructions are used to pack a fixed-point value into a narrower destination. The instructions support rounding, scaling, and saturation into the final destination format. The source data is in the vector register group specified by vs2. The scaling shift amount value can come from a vector register group vs1, a scalar integer register rs1, or a zero-extended 5-bit immediate. The low lg2(2*SEW) bits of the vector or scalar shift-amount value (e.g., the low 6 bits for a SEW=64-bit to SEW=32-bit narrowing operation) are used to control the right shift amount, which provides the scaling.<br>For vnclip, the shifted rounded source value is treated as a signed integer and saturates if the result would overflow the destination viewed as a signed integer.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vnclip instructions are used to pack a fixed-point value into a narrower destination. The instructions support rounding, scaling, and saturation into the final destination format. The source data is in the vector register group specified by vs2. The scaling shift amount value can come from a vector register group vs1, a scalar integer register rs1, or a zero-extended 5-bit immediate. The low lg2(2*SEW) bits of the vector or scalar shift-amount value (e.g., the low 6 bits for a SEW=64-bit to SEW=32-bit narrowing operation) are used to control the right shift amount, which provides the scaling.\nFor vnclip, the shifted rounded source value is treated as a signed integer and saturates if the result would overflow the destination viewed as a signed integer.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_narrowing_fixed_point_clip_instructions"
            };

        case "VNCLIP.WV":
            return {
                "html": "<div><span class=\"opcode\"><b>VNCLIP.WV</b> vs2, vs1, vd</span><br><div>The vnclip instructions are used to pack a fixed-point value into a narrower destination. The instructions support rounding, scaling, and saturation into the final destination format. The source data is in the vector register group specified by vs2. The scaling shift amount value can come from a vector register group vs1, a scalar integer register rs1, or a zero-extended 5-bit immediate. The low lg2(2*SEW) bits of the vector or scalar shift-amount value (e.g., the low 6 bits for a SEW=64-bit to SEW=32-bit narrowing operation) are used to control the right shift amount, which provides the scaling.<br>For vnclip, the shifted rounded source value is treated as a signed integer and saturates if the result would overflow the destination viewed as a signed integer.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vnclip instructions are used to pack a fixed-point value into a narrower destination. The instructions support rounding, scaling, and saturation into the final destination format. The source data is in the vector register group specified by vs2. The scaling shift amount value can come from a vector register group vs1, a scalar integer register rs1, or a zero-extended 5-bit immediate. The low lg2(2*SEW) bits of the vector or scalar shift-amount value (e.g., the low 6 bits for a SEW=64-bit to SEW=32-bit narrowing operation) are used to control the right shift amount, which provides the scaling.\nFor vnclip, the shifted rounded source value is treated as a signed integer and saturates if the result would overflow the destination viewed as a signed integer.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_narrowing_fixed_point_clip_instructions"
            };

        case "VNCLIP.WX":
            return {
                "html": "<div><span class=\"opcode\"><b>VNCLIP.WX</b> vs2, rs1, vd</span><br><div>The vnclip instructions are used to pack a fixed-point value into a narrower destination. The instructions support rounding, scaling, and saturation into the final destination format. The source data is in the vector register group specified by vs2. The scaling shift amount value can come from a vector register group vs1, a scalar integer register rs1, or a zero-extended 5-bit immediate. The low lg2(2*SEW) bits of the vector or scalar shift-amount value (e.g., the low 6 bits for a SEW=64-bit to SEW=32-bit narrowing operation) are used to control the right shift amount, which provides the scaling.<br>For vnclip, the shifted rounded source value is treated as a signed integer and saturates if the result would overflow the destination viewed as a signed integer.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vnclip instructions are used to pack a fixed-point value into a narrower destination. The instructions support rounding, scaling, and saturation into the final destination format. The source data is in the vector register group specified by vs2. The scaling shift amount value can come from a vector register group vs1, a scalar integer register rs1, or a zero-extended 5-bit immediate. The low lg2(2*SEW) bits of the vector or scalar shift-amount value (e.g., the low 6 bits for a SEW=64-bit to SEW=32-bit narrowing operation) are used to control the right shift amount, which provides the scaling.\nFor vnclip, the shifted rounded source value is treated as a signed integer and saturates if the result would overflow the destination viewed as a signed integer.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_narrowing_fixed_point_clip_instructions"
            };

        case "VNCLIPU.WI":
            return {
                "html": "<div><span class=\"opcode\"><b>VNCLIPU.WI</b> vs2, simm5, vd</span><br><div>For vnclipu/vnclip, the rounding mode is specified in the vxrm CSR. Rounding occurs around the least-significant bit of the destination and before saturation.<br>For vnclipu, the shifted rounded source value is treated as an unsigned integer and saturates if the result would overflow the destination viewed as an unsigned integer.<br># Narrowing unsigned clip # SEW 2*SEW SEW vnclipu.wv vd, vs2, vs1, vm # vd[i] = clip(roundoff_unsigned(vs2[i], vs1[i])) vnclipu.wx vd, vs2, rs1, vm # vd[i] = clip(roundoff_unsigned(vs2[i], x[rs1])) <b>vnclipu.wi</b> vd, vs2, uimm, vm # vd[i] = clip(roundoff_unsigned(vs2[i], uimm)) # Narrowing signed clip vnclip.wv vd, vs2, vs1, vm # vd[i] = clip(roundoff_signed(vs2[i], vs1[i])) vnclip.wx vd, vs2, rs1, vm # vd[i] = clip(roundoff_signed(vs2[i], x[rs1])) vnclip.wi vd, vs2, uimm, vm # vd[i] = clip(roundoff_signed(vs2[i], uimm))</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "For vnclipu/vnclip, the rounding mode is specified in the vxrm CSR. Rounding occurs around the least-significant bit of the destination and before saturation.\nFor vnclipu, the shifted rounded source value is treated as an unsigned integer and saturates if the result would overflow the destination viewed as an unsigned integer.\n# Narrowing unsigned clip # SEW 2*SEW SEW vnclipu.wv vd, vs2, vs1, vm # vd[i] = clip(roundoff_unsigned(vs2[i], vs1[i])) vnclipu.wx vd, vs2, rs1, vm # vd[i] = clip(roundoff_unsigned(vs2[i], x[rs1])) vnclipu.wi vd, vs2, uimm, vm # vd[i] = clip(roundoff_unsigned(vs2[i], uimm)) # Narrowing signed clip vnclip.wv vd, vs2, vs1, vm # vd[i] = clip(roundoff_signed(vs2[i], vs1[i])) vnclip.wx vd, vs2, rs1, vm # vd[i] = clip(roundoff_signed(vs2[i], x[rs1])) vnclip.wi vd, vs2, uimm, vm # vd[i] = clip(roundoff_signed(vs2[i], uimm))\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_narrowing_fixed_point_clip_instructions"
            };

        case "VNCLIPU.WV":
            return {
                "html": "<div><span class=\"opcode\"><b>VNCLIPU.WV</b> vs2, vs1, vd</span><br><div>For vnclipu/vnclip, the rounding mode is specified in the vxrm CSR. Rounding occurs around the least-significant bit of the destination and before saturation.<br>For vnclipu, the shifted rounded source value is treated as an unsigned integer and saturates if the result would overflow the destination viewed as an unsigned integer.<br># Narrowing unsigned clip # SEW 2*SEW SEW <b>vnclipu.wv</b> vd, vs2, vs1, vm # vd[i] = clip(roundoff_unsigned(vs2[i], vs1[i])) vnclipu.wx vd, vs2, rs1, vm # vd[i] = clip(roundoff_unsigned(vs2[i], x[rs1])) vnclipu.wi vd, vs2, uimm, vm # vd[i] = clip(roundoff_unsigned(vs2[i], uimm)) # Narrowing signed clip vnclip.wv vd, vs2, vs1, vm # vd[i] = clip(roundoff_signed(vs2[i], vs1[i])) vnclip.wx vd, vs2, rs1, vm # vd[i] = clip(roundoff_signed(vs2[i], x[rs1])) vnclip.wi vd, vs2, uimm, vm # vd[i] = clip(roundoff_signed(vs2[i], uimm))</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "For vnclipu/vnclip, the rounding mode is specified in the vxrm CSR. Rounding occurs around the least-significant bit of the destination and before saturation.\nFor vnclipu, the shifted rounded source value is treated as an unsigned integer and saturates if the result would overflow the destination viewed as an unsigned integer.\n# Narrowing unsigned clip # SEW 2*SEW SEW vnclipu.wv vd, vs2, vs1, vm # vd[i] = clip(roundoff_unsigned(vs2[i], vs1[i])) vnclipu.wx vd, vs2, rs1, vm # vd[i] = clip(roundoff_unsigned(vs2[i], x[rs1])) vnclipu.wi vd, vs2, uimm, vm # vd[i] = clip(roundoff_unsigned(vs2[i], uimm)) # Narrowing signed clip vnclip.wv vd, vs2, vs1, vm # vd[i] = clip(roundoff_signed(vs2[i], vs1[i])) vnclip.wx vd, vs2, rs1, vm # vd[i] = clip(roundoff_signed(vs2[i], x[rs1])) vnclip.wi vd, vs2, uimm, vm # vd[i] = clip(roundoff_signed(vs2[i], uimm))\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_narrowing_fixed_point_clip_instructions"
            };

        case "VNCLIPU.WX":
            return {
                "html": "<div><span class=\"opcode\"><b>VNCLIPU.WX</b> vs2, rs1, vd</span><br><div>For vnclipu/vnclip, the rounding mode is specified in the vxrm CSR. Rounding occurs around the least-significant bit of the destination and before saturation.<br>For vnclipu, the shifted rounded source value is treated as an unsigned integer and saturates if the result would overflow the destination viewed as an unsigned integer.<br># Narrowing unsigned clip # SEW 2*SEW SEW vnclipu.wv vd, vs2, vs1, vm # vd[i] = clip(roundoff_unsigned(vs2[i], vs1[i])) <b>vnclipu.wx</b> vd, vs2, rs1, vm # vd[i] = clip(roundoff_unsigned(vs2[i], x[rs1])) vnclipu.wi vd, vs2, uimm, vm # vd[i] = clip(roundoff_unsigned(vs2[i], uimm)) # Narrowing signed clip vnclip.wv vd, vs2, vs1, vm # vd[i] = clip(roundoff_signed(vs2[i], vs1[i])) vnclip.wx vd, vs2, rs1, vm # vd[i] = clip(roundoff_signed(vs2[i], x[rs1])) vnclip.wi vd, vs2, uimm, vm # vd[i] = clip(roundoff_signed(vs2[i], uimm))</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "For vnclipu/vnclip, the rounding mode is specified in the vxrm CSR. Rounding occurs around the least-significant bit of the destination and before saturation.\nFor vnclipu, the shifted rounded source value is treated as an unsigned integer and saturates if the result would overflow the destination viewed as an unsigned integer.\n# Narrowing unsigned clip # SEW 2*SEW SEW vnclipu.wv vd, vs2, vs1, vm # vd[i] = clip(roundoff_unsigned(vs2[i], vs1[i])) vnclipu.wx vd, vs2, rs1, vm # vd[i] = clip(roundoff_unsigned(vs2[i], x[rs1])) vnclipu.wi vd, vs2, uimm, vm # vd[i] = clip(roundoff_unsigned(vs2[i], uimm)) # Narrowing signed clip vnclip.wv vd, vs2, vs1, vm # vd[i] = clip(roundoff_signed(vs2[i], vs1[i])) vnclip.wx vd, vs2, rs1, vm # vd[i] = clip(roundoff_signed(vs2[i], x[rs1])) vnclip.wi vd, vs2, uimm, vm # vd[i] = clip(roundoff_signed(vs2[i], uimm))\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_narrowing_fixed_point_clip_instructions"
            };

        case "VNMSAC.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VNMSAC.VV</b> vs2, vs1, vd</span><br><div>vnmsac</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vnmsac\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VNMSAC.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VNMSAC.VX</b> vs2, rs1, vd</span><br><div>vnmsac</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vnmsac\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VNMSUB.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VNMSUB.VV</b> vs2, vs1, vd</span><br><div>vnmsub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vnmsub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VNMSUB.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VNMSUB.VX</b> vs2, rs1, vd</span><br><div>vnmsub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vnmsub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VNSRA.WI":
            return {
                "html": "<div><span class=\"opcode\"><b>VNSRA.WI</b> vs2, simm5, vd</span><br><div>A vn* prefix on the opcode is used to distinguish these instructions in the assembler, or a vfn* prefix for narrowing floating-point opcodes. The double-width source vector register group is signified by a w in the source operand suffix (e.g., vnsra.wv)</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "A vn* prefix on the opcode is used to distinguish these instructions in the assembler, or a vfn* prefix for narrowing floating-point opcodes. The double-width source vector register group is signified by a w in the source operand suffix (e.g., vnsra.wv)\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-narrowing"
            };

        case "VNSRA.WV":
            return {
                "html": "<div><span class=\"opcode\"><b>VNSRA.WV</b> vs2, vs1, vd</span><br><div>A vn* prefix on the opcode is used to distinguish these instructions in the assembler, or a vfn* prefix for narrowing floating-point opcodes. The double-width source vector register group is signified by a w in the source operand suffix (e.g., <b>vnsra.wv</b>)</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "A vn* prefix on the opcode is used to distinguish these instructions in the assembler, or a vfn* prefix for narrowing floating-point opcodes. The double-width source vector register group is signified by a w in the source operand suffix (e.g., vnsra.wv)\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-narrowing"
            };

        case "VNSRA.WX":
            return {
                "html": "<div><span class=\"opcode\"><b>VNSRA.WX</b> vs2, rs1, vd</span><br><div>A vn* prefix on the opcode is used to distinguish these instructions in the assembler, or a vfn* prefix for narrowing floating-point opcodes. The double-width source vector register group is signified by a w in the source operand suffix (e.g., vnsra.wv)</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "A vn* prefix on the opcode is used to distinguish these instructions in the assembler, or a vfn* prefix for narrowing floating-point opcodes. The double-width source vector register group is signified by a w in the source operand suffix (e.g., vnsra.wv)\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-narrowing"
            };

        case "VNSRL.WI":
            return {
                "html": "<div><span class=\"opcode\"><b>VNSRL.WI</b> vs2, simm5, vd</span><br><div>The destination EEW is smaller than the source EEW and the overlap is in the lowest-numbered part of the source register group (e.g., when LMUL=1, <b>vnsrl.wi</b> v0, v0, 3 is legal, but a destination of v1 is not).</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The destination EEW is smaller than the source EEW and the overlap is in the lowest-numbered part of the source register group (e.g., when LMUL=1, vnsrl.wi v0, v0, 3 is legal, but a destination of v1 is not).\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-vec-operands"
            };

        case "VNSRL.WV":
            return {
                "html": "<div><span class=\"opcode\"><b>VNSRL.WV</b> vs2, vs1, vd</span><br><div>The destination EEW is smaller than the source EEW and the overlap is in the lowest-numbered part of the source register group (e.g., when LMUL=1, vnsrl.wi v0, v0, 3 is legal, but a destination of v1 is not).</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The destination EEW is smaller than the source EEW and the overlap is in the lowest-numbered part of the source register group (e.g., when LMUL=1, vnsrl.wi v0, v0, 3 is legal, but a destination of v1 is not).\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-vec-operands"
            };

        case "VNSRL.WX":
            return {
                "html": "<div><span class=\"opcode\"><b>VNSRL.WX</b> vs2, rs1, vd</span><br><div>The destination EEW is smaller than the source EEW and the overlap is in the lowest-numbered part of the source register group (e.g., when LMUL=1, vnsrl.wi v0, v0, 3 is legal, but a destination of v1 is not).</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The destination EEW is smaller than the source EEW and the overlap is in the lowest-numbered part of the source register group (e.g., when LMUL=1, vnsrl.wi v0, v0, 3 is legal, but a destination of v1 is not).\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-vec-operands"
            };

        case "VOR.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VOR.VI</b> vs2, simm5, vd</span><br><div>vor</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vor\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VOR.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VOR.VV</b> vs2, vs1, vd</span><br><div>vor</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vor\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VOR.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VOR.VX</b> vs2, rs1, vd</span><br><div>vor</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vor\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VPOPC.M":
            return {
                "html": "<div><span class=\"opcode\"><b>VPOPC.M</b> vs2, rd</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VREDAND.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VREDAND.VS</b> vs2, vs1, vd</span><br><div>vredand</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vredand\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VREDMAX.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VREDMAX.VS</b> vs2, vs1, vd</span><br><div>vredmax</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vredmax\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VREDMAXU.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VREDMAXU.VS</b> vs2, vs1, vd</span><br><div>vredmaxu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vredmaxu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VREDMIN.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VREDMIN.VS</b> vs2, vs1, vd</span><br><div>vredmin</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vredmin\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VREDMINU.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VREDMINU.VS</b> vs2, vs1, vd</span><br><div>vredminu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vredminu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VREDOR.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VREDOR.VS</b> vs2, vs1, vd</span><br><div>vredor</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vredor\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VREDSUM.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VREDSUM.VS</b> vs2, vs1, vd</span><br><div># Simple reductions, where [*] denotes all active elements: <b>vredsum.vs</b> vd, vs2, vs1, vm # vd[0] = sum( vs1[0] , vs2[*] ) vredmaxu.vs vd, vs2, vs1, vm # vd[0] = maxu( vs1[0] , vs2[*] ) vredmax.vs vd, vs2, vs1, vm # vd[0] = max( vs1[0] , vs2[*] ) vredminu.vs vd, vs2, vs1, vm # vd[0] = minu( vs1[0] , vs2[*] ) vredmin.vs vd, vs2, vs1, vm # vd[0] = min( vs1[0] , vs2[*] ) vredand.vs vd, vs2, vs1, vm # vd[0] = and( vs1[0] , vs2[*] ) vredor.vs vd, vs2, vs1, vm # vd[0] = or( vs1[0] , vs2[*] ) vredxor.vs vd, vs2, vs1, vm # vd[0] = xor( vs1[0] , vs2[*] )</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Simple reductions, where [*] denotes all active elements: vredsum.vs vd, vs2, vs1, vm # vd[0] = sum( vs1[0] , vs2[*] ) vredmaxu.vs vd, vs2, vs1, vm # vd[0] = maxu( vs1[0] , vs2[*] ) vredmax.vs vd, vs2, vs1, vm # vd[0] = max( vs1[0] , vs2[*] ) vredminu.vs vd, vs2, vs1, vm # vd[0] = minu( vs1[0] , vs2[*] ) vredmin.vs vd, vs2, vs1, vm # vd[0] = min( vs1[0] , vs2[*] ) vredand.vs vd, vs2, vs1, vm # vd[0] = and( vs1[0] , vs2[*] ) vredor.vs vd, vs2, vs1, vm # vd[0] = or( vs1[0] , vs2[*] ) vredxor.vs vd, vs2, vs1, vm # vd[0] = xor( vs1[0] , vs2[*] )\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-vector-integer-reduce"
            };

        case "VREDXOR.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VREDXOR.VS</b> vs2, vs1, vd</span><br><div>vredxor</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vredxor\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VREM.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VREM.VV</b> vs2, vs1, vd</span><br><div>vrem</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vrem\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VREM.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VREM.VX</b> vs2, rs1, vd</span><br><div>vrem</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vrem\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VREMU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VREMU.VV</b> vs2, vs1, vd</span><br><div>vremu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vremu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VREMU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VREMU.VX</b> vs2, rs1, vd</span><br><div>vremu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vremu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VRGATHER.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VRGATHER.VI</b> vs2, simm5, vd</span><br><div>The vrgather.vv form uses SEW/LMUL for both the data and indices. The vrgatherei16.vv form uses SEW/LMUL for the data in vs2 but EEW=16 and EMUL = (16/SEW)*LMUL for the indices in vs1.<br>For any vrgather instruction, the destination vector register group cannot overlap with the source vector register groups, otherwise the instruction encoding is reserved.<br>vrgather.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]]; vrgatherei16.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]];<br>vrgather.vx vd, vs2, rs1, vm # vd[i] = (x[rs1] >= VLMAX) ? 0 : vs2[x[rs1]] <b>vrgather.vi</b> vd, vs2, uimm, vm # vd[i] = (uimm >= VLMAX) ? 0 : vs2[uimm]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vrgather.vv form uses SEW/LMUL for both the data and indices. The vrgatherei16.vv form uses SEW/LMUL for the data in vs2 but EEW=16 and EMUL = (16/SEW)*LMUL for the indices in vs1.\nFor any vrgather instruction, the destination vector register group cannot overlap with the source vector register groups, otherwise the instruction encoding is reserved.\nvrgather.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]]; vrgatherei16.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]];\nvrgather.vx vd, vs2, rs1, vm # vd[i] = (x[rs1] >= VLMAX) ? 0 : vs2[x[rs1]] vrgather.vi vd, vs2, uimm, vm # vd[i] = (uimm >= VLMAX) ? 0 : vs2[uimm]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_register_gather_instructions"
            };

        case "VRGATHER.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VRGATHER.VV</b> vs2, vs1, vd</span><br><div>The <b>vrgather.vv</b> form uses SEW/LMUL for both the data and indices. The vrgatherei16.vv form uses SEW/LMUL for the data in vs2 but EEW=16 and EMUL = (16/SEW)*LMUL for the indices in vs1.<br>For any vrgather instruction, the destination vector register group cannot overlap with the source vector register groups, otherwise the instruction encoding is reserved.<br><b>vrgather.vv</b> vd, vs2, vs1, vm # vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]]; vrgatherei16.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]];<br>vrgather.vx vd, vs2, rs1, vm # vd[i] = (x[rs1] >= VLMAX) ? 0 : vs2[x[rs1]] vrgather.vi vd, vs2, uimm, vm # vd[i] = (uimm >= VLMAX) ? 0 : vs2[uimm]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vrgather.vv form uses SEW/LMUL for both the data and indices. The vrgatherei16.vv form uses SEW/LMUL for the data in vs2 but EEW=16 and EMUL = (16/SEW)*LMUL for the indices in vs1.\nFor any vrgather instruction, the destination vector register group cannot overlap with the source vector register groups, otherwise the instruction encoding is reserved.\nvrgather.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]]; vrgatherei16.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]];\nvrgather.vx vd, vs2, rs1, vm # vd[i] = (x[rs1] >= VLMAX) ? 0 : vs2[x[rs1]] vrgather.vi vd, vs2, uimm, vm # vd[i] = (uimm >= VLMAX) ? 0 : vs2[uimm]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_register_gather_instructions"
            };

        case "VRGATHER.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VRGATHER.VX</b> vs2, rs1, vd</span><br><div>The vrgather.vv form uses SEW/LMUL for both the data and indices. The vrgatherei16.vv form uses SEW/LMUL for the data in vs2 but EEW=16 and EMUL = (16/SEW)*LMUL for the indices in vs1.<br>For any vrgather instruction, the destination vector register group cannot overlap with the source vector register groups, otherwise the instruction encoding is reserved.<br>vrgather.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]]; vrgatherei16.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]];<br><b>vrgather.vx</b> vd, vs2, rs1, vm # vd[i] = (x[rs1] >= VLMAX) ? 0 : vs2[x[rs1]] vrgather.vi vd, vs2, uimm, vm # vd[i] = (uimm >= VLMAX) ? 0 : vs2[uimm]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vrgather.vv form uses SEW/LMUL for both the data and indices. The vrgatherei16.vv form uses SEW/LMUL for the data in vs2 but EEW=16 and EMUL = (16/SEW)*LMUL for the indices in vs1.\nFor any vrgather instruction, the destination vector register group cannot overlap with the source vector register groups, otherwise the instruction encoding is reserved.\nvrgather.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]]; vrgatherei16.vv vd, vs2, vs1, vm # vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]];\nvrgather.vx vd, vs2, rs1, vm # vd[i] = (x[rs1] >= VLMAX) ? 0 : vs2[x[rs1]] vrgather.vi vd, vs2, uimm, vm # vd[i] = (uimm >= VLMAX) ? 0 : vs2[uimm]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_register_gather_instructions"
            };

        case "VRGATHEREI16.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VRGATHEREI16.VV</b> vs2, vs1, vd</span><br><div>vrgatherei16</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vrgatherei16\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VRSUB.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VRSUB.VI</b> vs2, simm5, vd</span><br><div>vrsub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vrsub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VRSUB.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VRSUB.VX</b> vs2, rs1, vd</span><br><div>vrsub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vrsub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VS1R.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VS1R.V</b> rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VS2R.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VS2R.V</b> rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VS4R.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VS4R.V</b> rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VS8R.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VS8R.V</b> rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSADD.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VSADD.VI</b> vs2, simm5, vd</span><br><div>vsadd</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vsadd\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSADD.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VSADD.VV</b> vs2, vs1, vd</span><br><div>vsadd</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vsadd\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSADD.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VSADD.VX</b> vs2, rs1, vd</span><br><div>vsadd</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vsadd\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSADDU.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VSADDU.VI</b> vs2, simm5, vd</span><br><div># Saturating adds of unsigned integers. vsaddu.vv vd, vs2, vs1, vm # Vector-vector vsaddu.vx vd, vs2, rs1, vm # vector-scalar <b>vsaddu.vi</b> vd, vs2, imm, vm # vector-immediate # Saturating adds of signed integers. vsadd.vv vd, vs2, vs1, vm # Vector-vector vsadd.vx vd, vs2, rs1, vm # vector-scalar vsadd.vi vd, vs2, imm, vm # vector-immediate # Saturating subtract of unsigned integers. vssubu.vv vd, vs2, vs1, vm # Vector-vector vssubu.vx vd, vs2, rs1, vm # vector-scalar # Saturating subtract of signed integers. vssub.vv vd, vs2, vs1, vm # Vector-vector vssub.vx vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Saturating adds of unsigned integers. vsaddu.vv vd, vs2, vs1, vm # Vector-vector vsaddu.vx vd, vs2, rs1, vm # vector-scalar vsaddu.vi vd, vs2, imm, vm # vector-immediate # Saturating adds of signed integers. vsadd.vv vd, vs2, vs1, vm # Vector-vector vsadd.vx vd, vs2, rs1, vm # vector-scalar vsadd.vi vd, vs2, imm, vm # vector-immediate # Saturating subtract of unsigned integers. vssubu.vv vd, vs2, vs1, vm # Vector-vector vssubu.vx vd, vs2, rs1, vm # vector-scalar # Saturating subtract of signed integers. vssub.vv vd, vs2, vs1, vm # Vector-vector vssub.vx vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_saturating_add_and_subtract"
            };

        case "VSADDU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VSADDU.VV</b> vs2, vs1, vd</span><br><div># Saturating adds of unsigned integers. <b>vsaddu.vv</b> vd, vs2, vs1, vm # Vector-vector vsaddu.vx vd, vs2, rs1, vm # vector-scalar vsaddu.vi vd, vs2, imm, vm # vector-immediate # Saturating adds of signed integers. vsadd.vv vd, vs2, vs1, vm # Vector-vector vsadd.vx vd, vs2, rs1, vm # vector-scalar vsadd.vi vd, vs2, imm, vm # vector-immediate # Saturating subtract of unsigned integers. vssubu.vv vd, vs2, vs1, vm # Vector-vector vssubu.vx vd, vs2, rs1, vm # vector-scalar # Saturating subtract of signed integers. vssub.vv vd, vs2, vs1, vm # Vector-vector vssub.vx vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Saturating adds of unsigned integers. vsaddu.vv vd, vs2, vs1, vm # Vector-vector vsaddu.vx vd, vs2, rs1, vm # vector-scalar vsaddu.vi vd, vs2, imm, vm # vector-immediate # Saturating adds of signed integers. vsadd.vv vd, vs2, vs1, vm # Vector-vector vsadd.vx vd, vs2, rs1, vm # vector-scalar vsadd.vi vd, vs2, imm, vm # vector-immediate # Saturating subtract of unsigned integers. vssubu.vv vd, vs2, vs1, vm # Vector-vector vssubu.vx vd, vs2, rs1, vm # vector-scalar # Saturating subtract of signed integers. vssub.vv vd, vs2, vs1, vm # Vector-vector vssub.vx vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_saturating_add_and_subtract"
            };

        case "VSADDU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VSADDU.VX</b> vs2, rs1, vd</span><br><div># Saturating adds of unsigned integers. vsaddu.vv vd, vs2, vs1, vm # Vector-vector <b>vsaddu.vx</b> vd, vs2, rs1, vm # vector-scalar vsaddu.vi vd, vs2, imm, vm # vector-immediate # Saturating adds of signed integers. vsadd.vv vd, vs2, vs1, vm # Vector-vector vsadd.vx vd, vs2, rs1, vm # vector-scalar vsadd.vi vd, vs2, imm, vm # vector-immediate # Saturating subtract of unsigned integers. vssubu.vv vd, vs2, vs1, vm # Vector-vector vssubu.vx vd, vs2, rs1, vm # vector-scalar # Saturating subtract of signed integers. vssub.vv vd, vs2, vs1, vm # Vector-vector vssub.vx vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Saturating adds of unsigned integers. vsaddu.vv vd, vs2, vs1, vm # Vector-vector vsaddu.vx vd, vs2, rs1, vm # vector-scalar vsaddu.vi vd, vs2, imm, vm # vector-immediate # Saturating adds of signed integers. vsadd.vv vd, vs2, vs1, vm # Vector-vector vsadd.vx vd, vs2, rs1, vm # vector-scalar vsadd.vi vd, vs2, imm, vm # vector-immediate # Saturating subtract of unsigned integers. vssubu.vv vd, vs2, vs1, vm # Vector-vector vssubu.vx vd, vs2, rs1, vm # vector-scalar # Saturating subtract of signed integers. vssub.vv vd, vs2, vs1, vm # Vector-vector vssub.vx vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_saturating_add_and_subtract"
            };

        case "VSBC.VVM":
            return {
                "html": "<div><span class=\"opcode\"><b>VSBC.VVM</b> vs2, vs1, vd</span><br><div>The subtract with borrow instruction vsbc performs the equivalent function to support long word arithmetic for subtraction. There are no subtract with immediate instructions.<br># Produce difference with borrow. # vd[i] = vs2[i] - vs1[i] - v0.mask[i] <b>vsbc.vvm</b> vd, vs2, vs1, v0 # Vector-vector # vd[i] = vs2[i] - x[rs1] - v0.mask[i] vsbc.vxm vd, vs2, rs1, v0 # Vector-scalar # Produce borrow out in mask register format # vd.mask[i] = borrow_out(vs2[i] - vs1[i] - v0.mask[i]) vmsbc.vvm vd, vs2, vs1, v0 # Vector-vector # vd.mask[i] = borrow_out(vs2[i] - x[rs1] - v0.mask[i]) vmsbc.vxm vd, vs2, rs1, v0 # Vector-scalar # vd.mask[i] = borrow_out(vs2[i] - vs1[i]) vmsbc.vv vd, vs2, vs1 # Vector-vector, no borrow-in # vd.mask[i] = borrow_out(vs2[i] - x[rs1]) vmsbc.vx vd, vs2, rs1 # Vector-scalar, no borrow-in</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The subtract with borrow instruction vsbc performs the equivalent function to support long word arithmetic for subtraction. There are no subtract with immediate instructions.\n# Produce difference with borrow. # vd[i] = vs2[i] - vs1[i] - v0.mask[i] vsbc.vvm vd, vs2, vs1, v0 # Vector-vector # vd[i] = vs2[i] - x[rs1] - v0.mask[i] vsbc.vxm vd, vs2, rs1, v0 # Vector-scalar # Produce borrow out in mask register format # vd.mask[i] = borrow_out(vs2[i] - vs1[i] - v0.mask[i]) vmsbc.vvm vd, vs2, vs1, v0 # Vector-vector # vd.mask[i] = borrow_out(vs2[i] - x[rs1] - v0.mask[i]) vmsbc.vxm vd, vs2, rs1, v0 # Vector-scalar # vd.mask[i] = borrow_out(vs2[i] - vs1[i]) vmsbc.vv vd, vs2, vs1 # Vector-vector, no borrow-in # vd.mask[i] = borrow_out(vs2[i] - x[rs1]) vmsbc.vx vd, vs2, rs1 # Vector-scalar, no borrow-in\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VSBC.VXM":
            return {
                "html": "<div><span class=\"opcode\"><b>VSBC.VXM</b> vs2, rs1, vd</span><br><div>The subtract with borrow instruction vsbc performs the equivalent function to support long word arithmetic for subtraction. There are no subtract with immediate instructions.<br># Produce difference with borrow. # vd[i] = vs2[i] - vs1[i] - v0.mask[i] vsbc.vvm vd, vs2, vs1, v0 # Vector-vector # vd[i] = vs2[i] - x[rs1] - v0.mask[i] <b>vsbc.vxm</b> vd, vs2, rs1, v0 # Vector-scalar # Produce borrow out in mask register format # vd.mask[i] = borrow_out(vs2[i] - vs1[i] - v0.mask[i]) vmsbc.vvm vd, vs2, vs1, v0 # Vector-vector # vd.mask[i] = borrow_out(vs2[i] - x[rs1] - v0.mask[i]) vmsbc.vxm vd, vs2, rs1, v0 # Vector-scalar # vd.mask[i] = borrow_out(vs2[i] - vs1[i]) vmsbc.vv vd, vs2, vs1 # Vector-vector, no borrow-in # vd.mask[i] = borrow_out(vs2[i] - x[rs1]) vmsbc.vx vd, vs2, rs1 # Vector-scalar, no borrow-in</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The subtract with borrow instruction vsbc performs the equivalent function to support long word arithmetic for subtraction. There are no subtract with immediate instructions.\n# Produce difference with borrow. # vd[i] = vs2[i] - vs1[i] - v0.mask[i] vsbc.vvm vd, vs2, vs1, v0 # Vector-vector # vd[i] = vs2[i] - x[rs1] - v0.mask[i] vsbc.vxm vd, vs2, rs1, v0 # Vector-scalar # Produce borrow out in mask register format # vd.mask[i] = borrow_out(vs2[i] - vs1[i] - v0.mask[i]) vmsbc.vvm vd, vs2, vs1, v0 # Vector-vector # vd.mask[i] = borrow_out(vs2[i] - x[rs1] - v0.mask[i]) vmsbc.vxm vd, vs2, rs1, v0 # Vector-scalar # vd.mask[i] = borrow_out(vs2[i] - vs1[i]) vmsbc.vv vd, vs2, vs1 # Vector-vector, no borrow-in # vd.mask[i] = borrow_out(vs2[i] - x[rs1]) vmsbc.vx vd, vs2, rs1 # Vector-scalar, no borrow-in\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_integer_add_with_carry_subtract_with_borrow_instructions"
            };

        case "VSE1.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSE1.V</b> rs1, vs3</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSE1024.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSE1024.V</b> rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSE128.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSE128.V</b> rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSE16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSE16.V</b> rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSE256.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSE256.V</b> rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSE32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSE32.V</b> rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSE512.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSE512.V</b> rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSE64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSE64.V</b> rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSE8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSE8.V</b> rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSETIVLI":
            return {
                "html": "<div><span class=\"opcode\"><b>VSETIVLI</b> zimm10, zimm, rd</span><br><div>{reg: [ {bits: 7, name: 0x57, attr: '<b>vsetivli</b>'}, {bits: 5, name: 'rd', type: 4}, {bits: 3, name: 7}, {bits: 5, name: 'uimm[4:0]', type: 5}, {bits: 10, name: 'zimm[9:0]', type: 5}, {bits: 1, name: '1'}, {bits: 1, name: '1'}, ]}</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "{reg: [ {bits: 7, name: 0x57, attr: 'vsetivli'}, {bits: 5, name: 'rd', type: 4}, {bits: 3, name: 7}, {bits: 5, name: 'uimm[4:0]', type: 5}, {bits: 10, name: 'zimm[9:0]', type: 5}, {bits: 1, name: '1'}, {bits: 1, name: '1'}, ]}\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_formats"
            };

        case "VSETVL":
            return {
                "html": "<div><span class=\"opcode\"><b>VSETVL</b> rs2, rs1, rd</span><br><div>The vector extension must have a consistent state at reset. In particular, vtype and vl must have values that can be read and then restored with a single <b>vsetvl</b> instruction.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vector extension must have a consistent state at reset. In particular, vtype and vl must have values that can be read and then restored with a single vsetvl instruction.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_state_of_vector_extension_at_reset"
            };

        case "VSETVLI":
            return {
                "html": "<div><span class=\"opcode\"><b>VSETVLI</b> zimm11, rs1, rd</span><br><div>The assembly syntax adds two mandatory flags to the <b>vsetvli</b> instruction:<br>ta # Tail agnostic tu # Tail undisturbed ma # Mask agnostic mu # Mask undisturbed <b>vsetvli</b> t0, a0, e32, m4, ta, ma # Tail agnostic, mask agnostic <b>vsetvli</b> t0, a0, e32, m4, tu, ma # Tail undisturbed, mask agnostic <b>vsetvli</b> t0, a0, e32, m4, ta, mu # Tail agnostic, mask undisturbed <b>vsetvli</b> t0, a0, e32, m4, tu, mu # Tail undisturbed, mask undisturbed</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The assembly syntax adds two mandatory flags to the vsetvli instruction:\nta # Tail agnostic tu # Tail undisturbed ma # Mask agnostic mu # Mask undisturbed vsetvli t0, a0, e32, m4, ta, ma # Tail agnostic, mask agnostic vsetvli t0, a0, e32, m4, tu, ma # Tail undisturbed, mask agnostic vsetvli t0, a0, e32, m4, ta, mu # Tail agnostic, mask undisturbed vsetvli t0, a0, e32, m4, tu, mu # Tail undisturbed, mask undisturbed\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-agnostic"
            };

        case "VSEXT.VF2":
            return {
                "html": "<div><span class=\"opcode\"><b>VSEXT.VF2</b> vs2, vd</span><br><div>vsext.vf8<br>vsext.vf4<br><b>vsext.vf2</b></div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vsext.vf8\nvsext.vf4\nvsext.vf2\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSEXT.VF4":
            return {
                "html": "<div><span class=\"opcode\"><b>VSEXT.VF4</b> vs2, vd</span><br><div>vsext.vf8<br><b>vsext.vf4</b><br>vsext.vf2</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vsext.vf8\nvsext.vf4\nvsext.vf2\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSEXT.VF8":
            return {
                "html": "<div><span class=\"opcode\"><b>VSEXT.VF8</b> vs2, vd</span><br><div><b>vsext.vf8</b><br>vsext.vf4<br>vsext.vf2</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vsext.vf8\nvsext.vf4\nvsext.vf2\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSLIDE1DOWN.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VSLIDE1DOWN.VX</b> vs2, rs1, vd</span><br><div>The vslide1down instruction copies the first vl-1 active elements values from index i+1 in the source vector register group to index i in the destination vector register group.<br>The vslide1down instruction places the x register argument at location vl-1 in the destination vector register, provided that element vl-1 is active, otherwise the destination element is unchanged. If XLEN < SEW, the value is sign-extended to SEW bits. If XLEN > SEW, the least-significant bits are copied over and the high SEW-XLEN bits are ignored.<br><b>vslide1down.vx</b> vd, vs2, rs1, vm # vd[i] = vs2[i+1], vd[vl-1]=x[rs1] vfslide1down.vf vd, vs2, rs1, vm # vd[i] = vs2[i+1], vd[vl-1]=f[rs1]<br>vslide1down behavior i < vstart unchanged vstart <= i < vl-1 vd[i] = vs2[i+1] if v0.mask[i] enabled vstart <= i = vl-1 vd[vl-1] = x[rs1] if v0.mask[i] enabled vl <= i < VLMAX Follow tail policy</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vslide1down instruction copies the first vl-1 active elements values from index i+1 in the source vector register group to index i in the destination vector register group.\nThe vslide1down instruction places the x register argument at location vl-1 in the destination vector register, provided that element vl-1 is active, otherwise the destination element is unchanged. If XLEN < SEW, the value is sign-extended to SEW bits. If XLEN > SEW, the least-significant bits are copied over and the high SEW-XLEN bits are ignored.\nvslide1down.vx vd, vs2, rs1, vm # vd[i] = vs2[i+1], vd[vl-1]=x[rs1] vfslide1down.vf vd, vs2, rs1, vm # vd[i] = vs2[i+1], vd[vl-1]=f[rs1]\nvslide1down behavior i < vstart unchanged vstart <= i < vl-1 vd[i] = vs2[i+1] if v0.mask[i] enabled vstart <= i = vl-1 vd[vl-1] = x[rs1] if v0.mask[i] enabled vl <= i < VLMAX Follow tail policy\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_slide1down_instruction"
            };

        case "VSLIDE1UP.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VSLIDE1UP.VX</b> vs2, rs1, vd</span><br><div>The vslide1up instruction places the x register argument at location 0 of the destination vector register group, provided that element 0 is active, otherwise the destination element update follows the current mask agnostic/undisturbed policy. If XLEN < SEW, the value is sign-extended to SEW bits. If XLEN > SEW, the least-significant bits are copied over and the high SEW-XLEN bits are ignored.<br>The vslide1up instruction requires that the destination vector register group does not overlap the source vector register group. Otherwise, the instruction encoding is reserved.<br><b>vslide1up.vx</b> vd, vs2, rs1, vm # vd[0]=x[rs1], vd[i+1] = vs2[i] vfslide1up.vf vd, vs2, rs1, vm # vd[0]=f[rs1], vd[i+1] = vs2[i]<br>vslide1up behavior i < vstart unchanged 0 = i = vstart vd[i] = x[rs1] if v0.mask[i] enabled max(vstart, 1) <= i < vl vd[i] = vs2[i-1] if v0.mask[i] enabled vl <= i < VLMAX Follow tail policy</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vslide1up instruction places the x register argument at location 0 of the destination vector register group, provided that element 0 is active, otherwise the destination element update follows the current mask agnostic/undisturbed policy. If XLEN < SEW, the value is sign-extended to SEW bits. If XLEN > SEW, the least-significant bits are copied over and the high SEW-XLEN bits are ignored.\nThe vslide1up instruction requires that the destination vector register group does not overlap the source vector register group. Otherwise, the instruction encoding is reserved.\nvslide1up.vx vd, vs2, rs1, vm # vd[0]=x[rs1], vd[i+1] = vs2[i] vfslide1up.vf vd, vs2, rs1, vm # vd[0]=f[rs1], vd[i+1] = vs2[i]\nvslide1up behavior i < vstart unchanged 0 = i = vstart vd[i] = x[rs1] if v0.mask[i] enabled max(vstart, 1) <= i < vl vd[i] = vs2[i-1] if v0.mask[i] enabled vl <= i < VLMAX Follow tail policy\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_slide1up"
            };

        case "VSLIDEDOWN.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VSLIDEDOWN.VI</b> vs2, simm5, vd</span><br><div>For vslidedown, the value in vl specifies the maximum number of destination elements that are written. The remaining elements past vl are handled according to the current tail policy (Section Vector Tail Agnostic and Vector Mask Agnostic vta and vma ).<br>vslidedown.vx vd, vs2, rs1, vm # vd[i] = vs2[i+rs1] <b>vslidedown.vi</b> vd, vs2, uimm, vm # vd[i] = vs2[i+uimm]<br>vslidedown behavior for source elements for element i in slide 0 <= i+OFFSET < VLMAX src[i] = vs2[i+OFFSET] VLMAX <= i+OFFSET src[i] = 0 vslidedown behavior for destination element i in slide 0 < i < vstart Unchanged vstart <= i < vl vd[i] = src[i] if v0.mask[i] enabled vl <= i < VLMAX Follow tail policy</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "For vslidedown, the value in vl specifies the maximum number of destination elements that are written. The remaining elements past vl are handled according to the current tail policy (Section Vector Tail Agnostic and Vector Mask Agnostic vta and vma ).\nvslidedown.vx vd, vs2, rs1, vm # vd[i] = vs2[i+rs1] vslidedown.vi vd, vs2, uimm, vm # vd[i] = vs2[i+uimm]\nvslidedown behavior for source elements for element i in slide 0 <= i+OFFSET < VLMAX src[i] = vs2[i+OFFSET] VLMAX <= i+OFFSET src[i] = 0 vslidedown behavior for destination element i in slide 0 < i < vstart Unchanged vstart <= i < vl vd[i] = src[i] if v0.mask[i] enabled vl <= i < VLMAX Follow tail policy\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_slidedown_instructions"
            };

        case "VSLIDEDOWN.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VSLIDEDOWN.VX</b> vs2, rs1, vd</span><br><div>For vslidedown, the value in vl specifies the maximum number of destination elements that are written. The remaining elements past vl are handled according to the current tail policy (Section Vector Tail Agnostic and Vector Mask Agnostic vta and vma ).<br><b>vslidedown.vx</b> vd, vs2, rs1, vm # vd[i] = vs2[i+rs1] vslidedown.vi vd, vs2, uimm, vm # vd[i] = vs2[i+uimm]<br>vslidedown behavior for source elements for element i in slide 0 <= i+OFFSET < VLMAX src[i] = vs2[i+OFFSET] VLMAX <= i+OFFSET src[i] = 0 vslidedown behavior for destination element i in slide 0 < i < vstart Unchanged vstart <= i < vl vd[i] = src[i] if v0.mask[i] enabled vl <= i < VLMAX Follow tail policy</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "For vslidedown, the value in vl specifies the maximum number of destination elements that are written. The remaining elements past vl are handled according to the current tail policy (Section Vector Tail Agnostic and Vector Mask Agnostic vta and vma ).\nvslidedown.vx vd, vs2, rs1, vm # vd[i] = vs2[i+rs1] vslidedown.vi vd, vs2, uimm, vm # vd[i] = vs2[i+uimm]\nvslidedown behavior for source elements for element i in slide 0 <= i+OFFSET < VLMAX src[i] = vs2[i+OFFSET] VLMAX <= i+OFFSET src[i] = 0 vslidedown behavior for destination element i in slide 0 < i < vstart Unchanged vstart <= i < vl vd[i] = src[i] if v0.mask[i] enabled vl <= i < VLMAX Follow tail policy\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_slidedown_instructions"
            };

        case "VSLIDEUP.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VSLIDEUP.VI</b> vs2, simm5, vd</span><br><div>For all of the vslideup, vslidedown, v[f]slide1up, and v[f]slide1down instructions, if vstart >= vl, the instruction performs no operation and leaves the destination vector register unchanged.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "For all of the vslideup, vslidedown, v[f]slide1up, and v[f]slide1down instructions, if vstart >= vl, the instruction performs no operation and leaves the destination vector register unchanged.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_slide_instructions"
            };

        case "VSLIDEUP.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VSLIDEUP.VX</b> vs2, rs1, vd</span><br><div>For all of the vslideup, vslidedown, v[f]slide1up, and v[f]slide1down instructions, if vstart >= vl, the instruction performs no operation and leaves the destination vector register unchanged.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "For all of the vslideup, vslidedown, v[f]slide1up, and v[f]slide1down instructions, if vstart >= vl, the instruction performs no operation and leaves the destination vector register unchanged.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_slide_instructions"
            };

        case "VSLL.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VSLL.VI</b> vs2, simm5, vd</span><br><div># Bit shift operations vsll.vv vd, vs2, vs1, vm # Vector-vector vsll.vx vd, vs2, rs1, vm # vector-scalar <b>vsll.vi</b> vd, vs2, uimm, vm # vector-immediate vsrl.vv vd, vs2, vs1, vm # Vector-vector vsrl.vx vd, vs2, rs1, vm # vector-scalar vsrl.vi vd, vs2, uimm, vm # vector-immediate vsra.vv vd, vs2, vs1, vm # Vector-vector vsra.vx vd, vs2, rs1, vm # vector-scalar vsra.vi vd, vs2, uimm, vm # vector-immediate</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Bit shift operations vsll.vv vd, vs2, vs1, vm # Vector-vector vsll.vx vd, vs2, rs1, vm # vector-scalar vsll.vi vd, vs2, uimm, vm # vector-immediate vsrl.vv vd, vs2, vs1, vm # Vector-vector vsrl.vx vd, vs2, rs1, vm # vector-scalar vsrl.vi vd, vs2, uimm, vm # vector-immediate vsra.vv vd, vs2, vs1, vm # Vector-vector vsra.vx vd, vs2, rs1, vm # vector-scalar vsra.vi vd, vs2, uimm, vm # vector-immediate\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_shift_instructions"
            };

        case "VSLL.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VSLL.VV</b> vs2, vs1, vd</span><br><div># Bit shift operations <b>vsll.vv</b> vd, vs2, vs1, vm # Vector-vector vsll.vx vd, vs2, rs1, vm # vector-scalar vsll.vi vd, vs2, uimm, vm # vector-immediate vsrl.vv vd, vs2, vs1, vm # Vector-vector vsrl.vx vd, vs2, rs1, vm # vector-scalar vsrl.vi vd, vs2, uimm, vm # vector-immediate vsra.vv vd, vs2, vs1, vm # Vector-vector vsra.vx vd, vs2, rs1, vm # vector-scalar vsra.vi vd, vs2, uimm, vm # vector-immediate</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Bit shift operations vsll.vv vd, vs2, vs1, vm # Vector-vector vsll.vx vd, vs2, rs1, vm # vector-scalar vsll.vi vd, vs2, uimm, vm # vector-immediate vsrl.vv vd, vs2, vs1, vm # Vector-vector vsrl.vx vd, vs2, rs1, vm # vector-scalar vsrl.vi vd, vs2, uimm, vm # vector-immediate vsra.vv vd, vs2, vs1, vm # Vector-vector vsra.vx vd, vs2, rs1, vm # vector-scalar vsra.vi vd, vs2, uimm, vm # vector-immediate\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_shift_instructions"
            };

        case "VSLL.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VSLL.VX</b> vs2, rs1, vd</span><br><div># Bit shift operations vsll.vv vd, vs2, vs1, vm # Vector-vector <b>vsll.vx</b> vd, vs2, rs1, vm # vector-scalar vsll.vi vd, vs2, uimm, vm # vector-immediate vsrl.vv vd, vs2, vs1, vm # Vector-vector vsrl.vx vd, vs2, rs1, vm # vector-scalar vsrl.vi vd, vs2, uimm, vm # vector-immediate vsra.vv vd, vs2, vs1, vm # Vector-vector vsra.vx vd, vs2, rs1, vm # vector-scalar vsra.vi vd, vs2, uimm, vm # vector-immediate</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Bit shift operations vsll.vv vd, vs2, vs1, vm # Vector-vector vsll.vx vd, vs2, rs1, vm # vector-scalar vsll.vi vd, vs2, uimm, vm # vector-immediate vsrl.vv vd, vs2, vs1, vm # Vector-vector vsrl.vx vd, vs2, rs1, vm # vector-scalar vsrl.vi vd, vs2, uimm, vm # vector-immediate vsra.vv vd, vs2, vs1, vm # Vector-vector vsra.vx vd, vs2, rs1, vm # vector-scalar vsra.vi vd, vs2, uimm, vm # vector-immediate\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_shift_instructions"
            };

        case "VSM.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSM.V</b> rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSMUL.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VSMUL.VV</b> vs2, vs1, vd</span><br><div># Signed saturating and rounding fractional multiply # See vxrm description for rounding calculation <b>vsmul.vv</b> vd, vs2, vs1, vm # vd[i] = clip(roundoff_signed(vs2[i]*vs1[i], SEW-1)) vsmul.vx vd, vs2, rs1, vm # vd[i] = clip(roundoff_signed(vs2[i]*x[rs1], SEW-1))</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Signed saturating and rounding fractional multiply # See vxrm description for rounding calculation vsmul.vv vd, vs2, vs1, vm # vd[i] = clip(roundoff_signed(vs2[i]*vs1[i], SEW-1)) vsmul.vx vd, vs2, rs1, vm # vd[i] = clip(roundoff_signed(vs2[i]*x[rs1], SEW-1))\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_fractional_multiply_with_rounding_and_saturation"
            };

        case "VSMUL.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VSMUL.VX</b> vs2, rs1, vd</span><br><div># Signed saturating and rounding fractional multiply # See vxrm description for rounding calculation vsmul.vv vd, vs2, vs1, vm # vd[i] = clip(roundoff_signed(vs2[i]*vs1[i], SEW-1)) <b>vsmul.vx</b> vd, vs2, rs1, vm # vd[i] = clip(roundoff_signed(vs2[i]*x[rs1], SEW-1))</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Signed saturating and rounding fractional multiply # See vxrm description for rounding calculation vsmul.vv vd, vs2, vs1, vm # vd[i] = clip(roundoff_signed(vs2[i]*vs1[i], SEW-1)) vsmul.vx vd, vs2, rs1, vm # vd[i] = clip(roundoff_signed(vs2[i]*x[rs1], SEW-1))\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_fractional_multiply_with_rounding_and_saturation"
            };

        case "VSOXEI1024.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSOXEI1024.V</b> vs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSOXEI128.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSOXEI128.V</b> vs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSOXEI16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSOXEI16.V</b> vs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSOXEI256.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSOXEI256.V</b> vs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSOXEI32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSOXEI32.V</b> vs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSOXEI512.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSOXEI512.V</b> vs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSOXEI64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSOXEI64.V</b> vs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSOXEI8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSOXEI8.V</b> vs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSRA.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VSRA.VI</b> vs2, simm5, vd</span><br><div>vsra</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vsra\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSRA.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VSRA.VV</b> vs2, vs1, vd</span><br><div>vsra</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vsra\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSRA.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VSRA.VX</b> vs2, rs1, vd</span><br><div>vsra</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vsra\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSRL.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VSRL.VI</b> vs2, simm5, vd</span><br><div>vsrl</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vsrl\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSRL.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VSRL.VV</b> vs2, vs1, vd</span><br><div>vsrl</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vsrl\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSRL.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VSRL.VX</b> vs2, rs1, vd</span><br><div>vsrl</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vsrl\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSSE1024.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSE1024.V</b> rs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSSE128.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSE128.V</b> rs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSSE16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSE16.V</b> rs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSSE256.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSE256.V</b> rs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSSE32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSE32.V</b> rs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSSE512.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSE512.V</b> rs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSSE64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSE64.V</b> rs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSSE8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSE8.V</b> rs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSSRA.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSRA.VI</b> vs2, simm5, vd</span><br><div>vssra</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vssra\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSSRA.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSRA.VV</b> vs2, vs1, vd</span><br><div>vssra</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vssra\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSSRA.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSRA.VX</b> vs2, rs1, vd</span><br><div>vssra</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vssra\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSSRL.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSRL.VI</b> vs2, simm5, vd</span><br><div>These instructions shift the input value right, and round off the shifted out bits according to vxrm. The scaling right shifts have both zero-extending (vssrl) and sign-extending (vssra) forms. The data to be shifted is in the vector register group specified by vs2 and the shift amount value can come from a vector register group vs1, a scalar integer register rs1, or a zero-extended 5-bit immediate. Only the low lg2(SEW) bits of the shift-amount value are used to control the shift amount.<br># Scaling shift right logical vssrl.vv vd, vs2, vs1, vm # vd[i] = roundoff_unsigned(vs2[i], vs1[i]) vssrl.vx vd, vs2, rs1, vm # vd[i] = roundoff_unsigned(vs2[i], x[rs1]) <b>vssrl.vi</b> vd, vs2, uimm, vm # vd[i] = roundoff_unsigned(vs2[i], uimm) # Scaling shift right arithmetic vssra.vv vd, vs2, vs1, vm # vd[i] = roundoff_signed(vs2[i],vs1[i]) vssra.vx vd, vs2, rs1, vm # vd[i] = roundoff_signed(vs2[i], x[rs1]) vssra.vi vd, vs2, uimm, vm # vd[i] = roundoff_signed(vs2[i], uimm)</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "These instructions shift the input value right, and round off the shifted out bits according to vxrm. The scaling right shifts have both zero-extending (vssrl) and sign-extending (vssra) forms. The data to be shifted is in the vector register group specified by vs2 and the shift amount value can come from a vector register group vs1, a scalar integer register rs1, or a zero-extended 5-bit immediate. Only the low lg2(SEW) bits of the shift-amount value are used to control the shift amount.\n# Scaling shift right logical vssrl.vv vd, vs2, vs1, vm # vd[i] = roundoff_unsigned(vs2[i], vs1[i]) vssrl.vx vd, vs2, rs1, vm # vd[i] = roundoff_unsigned(vs2[i], x[rs1]) vssrl.vi vd, vs2, uimm, vm # vd[i] = roundoff_unsigned(vs2[i], uimm) # Scaling shift right arithmetic vssra.vv vd, vs2, vs1, vm # vd[i] = roundoff_signed(vs2[i],vs1[i]) vssra.vx vd, vs2, rs1, vm # vd[i] = roundoff_signed(vs2[i], x[rs1]) vssra.vi vd, vs2, uimm, vm # vd[i] = roundoff_signed(vs2[i], uimm)\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_scaling_shift_instructions"
            };

        case "VSSRL.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSRL.VV</b> vs2, vs1, vd</span><br><div>These instructions shift the input value right, and round off the shifted out bits according to vxrm. The scaling right shifts have both zero-extending (vssrl) and sign-extending (vssra) forms. The data to be shifted is in the vector register group specified by vs2 and the shift amount value can come from a vector register group vs1, a scalar integer register rs1, or a zero-extended 5-bit immediate. Only the low lg2(SEW) bits of the shift-amount value are used to control the shift amount.<br># Scaling shift right logical <b>vssrl.vv</b> vd, vs2, vs1, vm # vd[i] = roundoff_unsigned(vs2[i], vs1[i]) vssrl.vx vd, vs2, rs1, vm # vd[i] = roundoff_unsigned(vs2[i], x[rs1]) vssrl.vi vd, vs2, uimm, vm # vd[i] = roundoff_unsigned(vs2[i], uimm) # Scaling shift right arithmetic vssra.vv vd, vs2, vs1, vm # vd[i] = roundoff_signed(vs2[i],vs1[i]) vssra.vx vd, vs2, rs1, vm # vd[i] = roundoff_signed(vs2[i], x[rs1]) vssra.vi vd, vs2, uimm, vm # vd[i] = roundoff_signed(vs2[i], uimm)</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "These instructions shift the input value right, and round off the shifted out bits according to vxrm. The scaling right shifts have both zero-extending (vssrl) and sign-extending (vssra) forms. The data to be shifted is in the vector register group specified by vs2 and the shift amount value can come from a vector register group vs1, a scalar integer register rs1, or a zero-extended 5-bit immediate. Only the low lg2(SEW) bits of the shift-amount value are used to control the shift amount.\n# Scaling shift right logical vssrl.vv vd, vs2, vs1, vm # vd[i] = roundoff_unsigned(vs2[i], vs1[i]) vssrl.vx vd, vs2, rs1, vm # vd[i] = roundoff_unsigned(vs2[i], x[rs1]) vssrl.vi vd, vs2, uimm, vm # vd[i] = roundoff_unsigned(vs2[i], uimm) # Scaling shift right arithmetic vssra.vv vd, vs2, vs1, vm # vd[i] = roundoff_signed(vs2[i],vs1[i]) vssra.vx vd, vs2, rs1, vm # vd[i] = roundoff_signed(vs2[i], x[rs1]) vssra.vi vd, vs2, uimm, vm # vd[i] = roundoff_signed(vs2[i], uimm)\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_scaling_shift_instructions"
            };

        case "VSSRL.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSRL.VX</b> vs2, rs1, vd</span><br><div>These instructions shift the input value right, and round off the shifted out bits according to vxrm. The scaling right shifts have both zero-extending (vssrl) and sign-extending (vssra) forms. The data to be shifted is in the vector register group specified by vs2 and the shift amount value can come from a vector register group vs1, a scalar integer register rs1, or a zero-extended 5-bit immediate. Only the low lg2(SEW) bits of the shift-amount value are used to control the shift amount.<br># Scaling shift right logical vssrl.vv vd, vs2, vs1, vm # vd[i] = roundoff_unsigned(vs2[i], vs1[i]) <b>vssrl.vx</b> vd, vs2, rs1, vm # vd[i] = roundoff_unsigned(vs2[i], x[rs1]) vssrl.vi vd, vs2, uimm, vm # vd[i] = roundoff_unsigned(vs2[i], uimm) # Scaling shift right arithmetic vssra.vv vd, vs2, vs1, vm # vd[i] = roundoff_signed(vs2[i],vs1[i]) vssra.vx vd, vs2, rs1, vm # vd[i] = roundoff_signed(vs2[i], x[rs1]) vssra.vi vd, vs2, uimm, vm # vd[i] = roundoff_signed(vs2[i], uimm)</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "These instructions shift the input value right, and round off the shifted out bits according to vxrm. The scaling right shifts have both zero-extending (vssrl) and sign-extending (vssra) forms. The data to be shifted is in the vector register group specified by vs2 and the shift amount value can come from a vector register group vs1, a scalar integer register rs1, or a zero-extended 5-bit immediate. Only the low lg2(SEW) bits of the shift-amount value are used to control the shift amount.\n# Scaling shift right logical vssrl.vv vd, vs2, vs1, vm # vd[i] = roundoff_unsigned(vs2[i], vs1[i]) vssrl.vx vd, vs2, rs1, vm # vd[i] = roundoff_unsigned(vs2[i], x[rs1]) vssrl.vi vd, vs2, uimm, vm # vd[i] = roundoff_unsigned(vs2[i], uimm) # Scaling shift right arithmetic vssra.vv vd, vs2, vs1, vm # vd[i] = roundoff_signed(vs2[i],vs1[i]) vssra.vx vd, vs2, rs1, vm # vd[i] = roundoff_signed(vs2[i], x[rs1]) vssra.vi vd, vs2, uimm, vm # vd[i] = roundoff_signed(vs2[i], uimm)\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_single_width_scaling_shift_instructions"
            };

        case "VSSUB.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSUB.VV</b> vs2, vs1, vd</span><br><div>vssub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vssub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSSUB.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSUB.VX</b> vs2, rs1, vd</span><br><div>vssub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vssub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSSUBU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSUBU.VV</b> vs2, vs1, vd</span><br><div>vssubu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vssubu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSSUBU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VSSUBU.VX</b> vs2, rs1, vd</span><br><div>vssubu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vssubu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSUB.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VSUB.VV</b> vs2, vs1, vd</span><br><div>vsub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vsub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSUB.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VSUB.VX</b> vs2, rs1, vd</span><br><div>vsub</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vsub\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VSUXEI1024.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSUXEI1024.V</b> vs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSUXEI128.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSUXEI128.V</b> vs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSUXEI16.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSUXEI16.V</b> vs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSUXEI256.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSUXEI256.V</b> vs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSUXEI32.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSUXEI32.V</b> vs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSUXEI512.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSUXEI512.V</b> vs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSUXEI64.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSUXEI64.V</b> vs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VSUXEI8.V":
            return {
                "html": "<div><span class=\"opcode\"><b>VSUXEI8.V</b> vs2, rs1, vs3</span><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "VWADD.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VWADD.VV</b> vs2, vs1, vd</span><br><div>vwadd<br>vwadd.w</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwadd\nvwadd.w\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWADD.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VWADD.VX</b> vs2, rs1, vd</span><br><div>vwadd<br>vwadd.w</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwadd\nvwadd.w\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWADD.WV":
            return {
                "html": "<div><span class=\"opcode\"><b>VWADD.WV</b> vs2, vs1, vd</span><br><div>vwadd<br>vwadd.w</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwadd\nvwadd.w\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWADD.WX":
            return {
                "html": "<div><span class=\"opcode\"><b>VWADD.WX</b> vs2, rs1, vd</span><br><div>vwadd<br>vwadd.w</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwadd\nvwadd.w\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWADDU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VWADDU.VV</b> vs2, vs1, vd</span><br><div># Widening unsigned integer add/subtract, 2*SEW = SEW +/- SEW <b>vwaddu.vv</b> vd, vs2, vs1, vm # vector-vector vwaddu.vx vd, vs2, rs1, vm # vector-scalar vwsubu.vv vd, vs2, vs1, vm # vector-vector vwsubu.vx vd, vs2, rs1, vm # vector-scalar # Widening signed integer add/subtract, 2*SEW = SEW +/- SEW vwadd.vv vd, vs2, vs1, vm # vector-vector vwadd.vx vd, vs2, rs1, vm # vector-scalar vwsub.vv vd, vs2, vs1, vm # vector-vector vwsub.vx vd, vs2, rs1, vm # vector-scalar # Widening unsigned integer add/subtract, 2*SEW = 2*SEW +/- SEW vwaddu.wv vd, vs2, vs1, vm # vector-vector vwaddu.wx vd, vs2, rs1, vm # vector-scalar vwsubu.wv vd, vs2, vs1, vm # vector-vector vwsubu.wx vd, vs2, rs1, vm # vector-scalar # Widening signed integer add/subtract, 2*SEW = 2*SEW +/- SEW vwadd.wv vd, vs2, vs1, vm # vector-vector vwadd.wx vd, vs2, rs1, vm # vector-scalar vwsub.wv vd, vs2, vs1, vm # vector-vector vwsub.wx vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Widening unsigned integer add/subtract, 2*SEW = SEW +/- SEW vwaddu.vv vd, vs2, vs1, vm # vector-vector vwaddu.vx vd, vs2, rs1, vm # vector-scalar vwsubu.vv vd, vs2, vs1, vm # vector-vector vwsubu.vx vd, vs2, rs1, vm # vector-scalar # Widening signed integer add/subtract, 2*SEW = SEW +/- SEW vwadd.vv vd, vs2, vs1, vm # vector-vector vwadd.vx vd, vs2, rs1, vm # vector-scalar vwsub.vv vd, vs2, vs1, vm # vector-vector vwsub.vx vd, vs2, rs1, vm # vector-scalar # Widening unsigned integer add/subtract, 2*SEW = 2*SEW +/- SEW vwaddu.wv vd, vs2, vs1, vm # vector-vector vwaddu.wx vd, vs2, rs1, vm # vector-scalar vwsubu.wv vd, vs2, vs1, vm # vector-vector vwsubu.wx vd, vs2, rs1, vm # vector-scalar # Widening signed integer add/subtract, 2*SEW = 2*SEW +/- SEW vwadd.wv vd, vs2, vs1, vm # vector-vector vwadd.wx vd, vs2, rs1, vm # vector-scalar vwsub.wv vd, vs2, vs1, vm # vector-vector vwsub.wx vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_widening_integer_addsubtract"
            };

        case "VWADDU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VWADDU.VX</b> vs2, rs1, vd</span><br><div># Widening unsigned integer add/subtract, 2*SEW = SEW +/- SEW vwaddu.vv vd, vs2, vs1, vm # vector-vector <b>vwaddu.vx</b> vd, vs2, rs1, vm # vector-scalar vwsubu.vv vd, vs2, vs1, vm # vector-vector vwsubu.vx vd, vs2, rs1, vm # vector-scalar # Widening signed integer add/subtract, 2*SEW = SEW +/- SEW vwadd.vv vd, vs2, vs1, vm # vector-vector vwadd.vx vd, vs2, rs1, vm # vector-scalar vwsub.vv vd, vs2, vs1, vm # vector-vector vwsub.vx vd, vs2, rs1, vm # vector-scalar # Widening unsigned integer add/subtract, 2*SEW = 2*SEW +/- SEW vwaddu.wv vd, vs2, vs1, vm # vector-vector vwaddu.wx vd, vs2, rs1, vm # vector-scalar vwsubu.wv vd, vs2, vs1, vm # vector-vector vwsubu.wx vd, vs2, rs1, vm # vector-scalar # Widening signed integer add/subtract, 2*SEW = 2*SEW +/- SEW vwadd.wv vd, vs2, vs1, vm # vector-vector vwadd.wx vd, vs2, rs1, vm # vector-scalar vwsub.wv vd, vs2, vs1, vm # vector-vector vwsub.wx vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Widening unsigned integer add/subtract, 2*SEW = SEW +/- SEW vwaddu.vv vd, vs2, vs1, vm # vector-vector vwaddu.vx vd, vs2, rs1, vm # vector-scalar vwsubu.vv vd, vs2, vs1, vm # vector-vector vwsubu.vx vd, vs2, rs1, vm # vector-scalar # Widening signed integer add/subtract, 2*SEW = SEW +/- SEW vwadd.vv vd, vs2, vs1, vm # vector-vector vwadd.vx vd, vs2, rs1, vm # vector-scalar vwsub.vv vd, vs2, vs1, vm # vector-vector vwsub.vx vd, vs2, rs1, vm # vector-scalar # Widening unsigned integer add/subtract, 2*SEW = 2*SEW +/- SEW vwaddu.wv vd, vs2, vs1, vm # vector-vector vwaddu.wx vd, vs2, rs1, vm # vector-scalar vwsubu.wv vd, vs2, vs1, vm # vector-vector vwsubu.wx vd, vs2, rs1, vm # vector-scalar # Widening signed integer add/subtract, 2*SEW = 2*SEW +/- SEW vwadd.wv vd, vs2, vs1, vm # vector-vector vwadd.wx vd, vs2, rs1, vm # vector-scalar vwsub.wv vd, vs2, vs1, vm # vector-vector vwsub.wx vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_widening_integer_addsubtract"
            };

        case "VWADDU.WV":
            return {
                "html": "<div><span class=\"opcode\"><b>VWADDU.WV</b> vs2, vs1, vd</span><br><div># Widening unsigned integer add/subtract, 2*SEW = SEW +/- SEW vwaddu.vv vd, vs2, vs1, vm # vector-vector vwaddu.vx vd, vs2, rs1, vm # vector-scalar vwsubu.vv vd, vs2, vs1, vm # vector-vector vwsubu.vx vd, vs2, rs1, vm # vector-scalar # Widening signed integer add/subtract, 2*SEW = SEW +/- SEW vwadd.vv vd, vs2, vs1, vm # vector-vector vwadd.vx vd, vs2, rs1, vm # vector-scalar vwsub.vv vd, vs2, vs1, vm # vector-vector vwsub.vx vd, vs2, rs1, vm # vector-scalar # Widening unsigned integer add/subtract, 2*SEW = 2*SEW +/- SEW <b>vwaddu.wv</b> vd, vs2, vs1, vm # vector-vector vwaddu.wx vd, vs2, rs1, vm # vector-scalar vwsubu.wv vd, vs2, vs1, vm # vector-vector vwsubu.wx vd, vs2, rs1, vm # vector-scalar # Widening signed integer add/subtract, 2*SEW = 2*SEW +/- SEW vwadd.wv vd, vs2, vs1, vm # vector-vector vwadd.wx vd, vs2, rs1, vm # vector-scalar vwsub.wv vd, vs2, vs1, vm # vector-vector vwsub.wx vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Widening unsigned integer add/subtract, 2*SEW = SEW +/- SEW vwaddu.vv vd, vs2, vs1, vm # vector-vector vwaddu.vx vd, vs2, rs1, vm # vector-scalar vwsubu.vv vd, vs2, vs1, vm # vector-vector vwsubu.vx vd, vs2, rs1, vm # vector-scalar # Widening signed integer add/subtract, 2*SEW = SEW +/- SEW vwadd.vv vd, vs2, vs1, vm # vector-vector vwadd.vx vd, vs2, rs1, vm # vector-scalar vwsub.vv vd, vs2, vs1, vm # vector-vector vwsub.vx vd, vs2, rs1, vm # vector-scalar # Widening unsigned integer add/subtract, 2*SEW = 2*SEW +/- SEW vwaddu.wv vd, vs2, vs1, vm # vector-vector vwaddu.wx vd, vs2, rs1, vm # vector-scalar vwsubu.wv vd, vs2, vs1, vm # vector-vector vwsubu.wx vd, vs2, rs1, vm # vector-scalar # Widening signed integer add/subtract, 2*SEW = 2*SEW +/- SEW vwadd.wv vd, vs2, vs1, vm # vector-vector vwadd.wx vd, vs2, rs1, vm # vector-scalar vwsub.wv vd, vs2, vs1, vm # vector-vector vwsub.wx vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_widening_integer_addsubtract"
            };

        case "VWADDU.WX":
            return {
                "html": "<div><span class=\"opcode\"><b>VWADDU.WX</b> vs2, rs1, vd</span><br><div># Widening unsigned integer add/subtract, 2*SEW = SEW +/- SEW vwaddu.vv vd, vs2, vs1, vm # vector-vector vwaddu.vx vd, vs2, rs1, vm # vector-scalar vwsubu.vv vd, vs2, vs1, vm # vector-vector vwsubu.vx vd, vs2, rs1, vm # vector-scalar # Widening signed integer add/subtract, 2*SEW = SEW +/- SEW vwadd.vv vd, vs2, vs1, vm # vector-vector vwadd.vx vd, vs2, rs1, vm # vector-scalar vwsub.vv vd, vs2, vs1, vm # vector-vector vwsub.vx vd, vs2, rs1, vm # vector-scalar # Widening unsigned integer add/subtract, 2*SEW = 2*SEW +/- SEW vwaddu.wv vd, vs2, vs1, vm # vector-vector <b>vwaddu.wx</b> vd, vs2, rs1, vm # vector-scalar vwsubu.wv vd, vs2, vs1, vm # vector-vector vwsubu.wx vd, vs2, rs1, vm # vector-scalar # Widening signed integer add/subtract, 2*SEW = 2*SEW +/- SEW vwadd.wv vd, vs2, vs1, vm # vector-vector vwadd.wx vd, vs2, rs1, vm # vector-scalar vwsub.wv vd, vs2, vs1, vm # vector-vector vwsub.wx vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Widening unsigned integer add/subtract, 2*SEW = SEW +/- SEW vwaddu.vv vd, vs2, vs1, vm # vector-vector vwaddu.vx vd, vs2, rs1, vm # vector-scalar vwsubu.vv vd, vs2, vs1, vm # vector-vector vwsubu.vx vd, vs2, rs1, vm # vector-scalar # Widening signed integer add/subtract, 2*SEW = SEW +/- SEW vwadd.vv vd, vs2, vs1, vm # vector-vector vwadd.vx vd, vs2, rs1, vm # vector-scalar vwsub.vv vd, vs2, vs1, vm # vector-vector vwsub.vx vd, vs2, rs1, vm # vector-scalar # Widening unsigned integer add/subtract, 2*SEW = 2*SEW +/- SEW vwaddu.wv vd, vs2, vs1, vm # vector-vector vwaddu.wx vd, vs2, rs1, vm # vector-scalar vwsubu.wv vd, vs2, vs1, vm # vector-vector vwsubu.wx vd, vs2, rs1, vm # vector-scalar # Widening signed integer add/subtract, 2*SEW = 2*SEW +/- SEW vwadd.wv vd, vs2, vs1, vm # vector-vector vwadd.wx vd, vs2, rs1, vm # vector-scalar vwsub.wv vd, vs2, vs1, vm # vector-vector vwsub.wx vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_widening_integer_addsubtract"
            };

        case "VWMACC.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VWMACC.VV</b> vs2, vs1, vd</span><br><div>vwmacc</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwmacc\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWMACC.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VWMACC.VX</b> vs2, rs1, vd</span><br><div>vwmacc</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwmacc\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWMACCSU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VWMACCSU.VV</b> vs2, vs1, vd</span><br><div>vwmaccsu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwmaccsu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWMACCSU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VWMACCSU.VX</b> vs2, rs1, vd</span><br><div>vwmaccsu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwmaccsu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWMACCU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VWMACCU.VV</b> vs2, vs1, vd</span><br><div># Widening unsigned-integer multiply-add, overwrite addend <b>vwmaccu.vv</b> vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] vwmaccu.vx vd, rs1, vs2, vm # vd[i] = +(x[rs1] * vs2[i]) + vd[i] # Widening signed-integer multiply-add, overwrite addend vwmacc.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] vwmacc.vx vd, rs1, vs2, vm # vd[i] = +(x[rs1] * vs2[i]) + vd[i] # Widening signed-unsigned-integer multiply-add, overwrite addend vwmaccsu.vv vd, vs1, vs2, vm # vd[i] = +(signed(vs1[i]) * unsigned(vs2[i])) + vd[i] vwmaccsu.vx vd, rs1, vs2, vm # vd[i] = +(signed(x[rs1]) * unsigned(vs2[i])) + vd[i] # Widening unsigned-signed-integer multiply-add, overwrite addend vwmaccus.vx vd, rs1, vs2, vm # vd[i] = +(unsigned(x[rs1]) * signed(vs2[i])) + vd[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Widening unsigned-integer multiply-add, overwrite addend vwmaccu.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] vwmaccu.vx vd, rs1, vs2, vm # vd[i] = +(x[rs1] * vs2[i]) + vd[i] # Widening signed-integer multiply-add, overwrite addend vwmacc.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] vwmacc.vx vd, rs1, vs2, vm # vd[i] = +(x[rs1] * vs2[i]) + vd[i] # Widening signed-unsigned-integer multiply-add, overwrite addend vwmaccsu.vv vd, vs1, vs2, vm # vd[i] = +(signed(vs1[i]) * unsigned(vs2[i])) + vd[i] vwmaccsu.vx vd, rs1, vs2, vm # vd[i] = +(signed(x[rs1]) * unsigned(vs2[i])) + vd[i] # Widening unsigned-signed-integer multiply-add, overwrite addend vwmaccus.vx vd, rs1, vs2, vm # vd[i] = +(unsigned(x[rs1]) * signed(vs2[i])) + vd[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_widening_integer_multiply_add_instructions"
            };

        case "VWMACCU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VWMACCU.VX</b> vs2, rs1, vd</span><br><div># Widening unsigned-integer multiply-add, overwrite addend vwmaccu.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] <b>vwmaccu.vx</b> vd, rs1, vs2, vm # vd[i] = +(x[rs1] * vs2[i]) + vd[i] # Widening signed-integer multiply-add, overwrite addend vwmacc.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] vwmacc.vx vd, rs1, vs2, vm # vd[i] = +(x[rs1] * vs2[i]) + vd[i] # Widening signed-unsigned-integer multiply-add, overwrite addend vwmaccsu.vv vd, vs1, vs2, vm # vd[i] = +(signed(vs1[i]) * unsigned(vs2[i])) + vd[i] vwmaccsu.vx vd, rs1, vs2, vm # vd[i] = +(signed(x[rs1]) * unsigned(vs2[i])) + vd[i] # Widening unsigned-signed-integer multiply-add, overwrite addend vwmaccus.vx vd, rs1, vs2, vm # vd[i] = +(unsigned(x[rs1]) * signed(vs2[i])) + vd[i]</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Widening unsigned-integer multiply-add, overwrite addend vwmaccu.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] vwmaccu.vx vd, rs1, vs2, vm # vd[i] = +(x[rs1] * vs2[i]) + vd[i] # Widening signed-integer multiply-add, overwrite addend vwmacc.vv vd, vs1, vs2, vm # vd[i] = +(vs1[i] * vs2[i]) + vd[i] vwmacc.vx vd, rs1, vs2, vm # vd[i] = +(x[rs1] * vs2[i]) + vd[i] # Widening signed-unsigned-integer multiply-add, overwrite addend vwmaccsu.vv vd, vs1, vs2, vm # vd[i] = +(signed(vs1[i]) * unsigned(vs2[i])) + vd[i] vwmaccsu.vx vd, rs1, vs2, vm # vd[i] = +(signed(x[rs1]) * unsigned(vs2[i])) + vd[i] # Widening unsigned-signed-integer multiply-add, overwrite addend vwmaccus.vx vd, rs1, vs2, vm # vd[i] = +(unsigned(x[rs1]) * signed(vs2[i])) + vd[i]\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_widening_integer_multiply_add_instructions"
            };

        case "VWMACCUS.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VWMACCUS.VX</b> vs2, rs1, vd</span><br><div>vwmaccus</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwmaccus\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWMUL.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VWMUL.VV</b> vs2, vs1, vd</span><br><div># Widening signed-integer multiply <b>vwmul.vv</b> vd, vs2, vs1, vm # vector-vector vwmul.vx vd, vs2, rs1, vm # vector-scalar # Widening unsigned-integer multiply vwmulu.vv vd, vs2, vs1, vm # vector-vector vwmulu.vx vd, vs2, rs1, vm # vector-scalar # Widening signed(vs2)-unsigned integer multiply vwmulsu.vv vd, vs2, vs1, vm # vector-vector vwmulsu.vx vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Widening signed-integer multiply vwmul.vv vd, vs2, vs1, vm # vector-vector vwmul.vx vd, vs2, rs1, vm # vector-scalar # Widening unsigned-integer multiply vwmulu.vv vd, vs2, vs1, vm # vector-vector vwmulu.vx vd, vs2, rs1, vm # vector-scalar # Widening signed(vs2)-unsigned integer multiply vwmulsu.vv vd, vs2, vs1, vm # vector-vector vwmulsu.vx vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_widening_integer_multiply_instructions"
            };

        case "VWMUL.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VWMUL.VX</b> vs2, rs1, vd</span><br><div># Widening signed-integer multiply vwmul.vv vd, vs2, vs1, vm # vector-vector <b>vwmul.vx</b> vd, vs2, rs1, vm # vector-scalar # Widening unsigned-integer multiply vwmulu.vv vd, vs2, vs1, vm # vector-vector vwmulu.vx vd, vs2, rs1, vm # vector-scalar # Widening signed(vs2)-unsigned integer multiply vwmulsu.vv vd, vs2, vs1, vm # vector-vector vwmulsu.vx vd, vs2, rs1, vm # vector-scalar</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "# Widening signed-integer multiply vwmul.vv vd, vs2, vs1, vm # vector-vector vwmul.vx vd, vs2, rs1, vm # vector-scalar # Widening unsigned-integer multiply vwmulu.vv vd, vs2, vs1, vm # vector-vector vwmulu.vx vd, vs2, rs1, vm # vector-scalar # Widening signed(vs2)-unsigned integer multiply vwmulsu.vv vd, vs2, vs1, vm # vector-vector vwmulsu.vx vd, vs2, rs1, vm # vector-scalar\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_widening_integer_multiply_instructions"
            };

        case "VWMULSU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VWMULSU.VV</b> vs2, vs1, vd</span><br><div>vwmulsu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwmulsu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWMULSU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VWMULSU.VX</b> vs2, rs1, vd</span><br><div>vwmulsu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwmulsu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWMULU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VWMULU.VV</b> vs2, vs1, vd</span><br><div>vwmulu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwmulu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWMULU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VWMULU.VX</b> vs2, rs1, vd</span><br><div>vwmulu</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwmulu\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWREDSUM.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VWREDSUM.VS</b> vs2, vs1, vd</span><br><div>The <b>vwredsum.vs</b> instruction sign-extends the SEW-wide vector elements before summing them.</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The vwredsum.vs instruction sign-extends the SEW-wide vector elements before summing them.\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-vector-integer-reduce-widen"
            };

        case "VWREDSUMU.VS":
            return {
                "html": "<div><span class=\"opcode\"><b>VWREDSUMU.VS</b> vs2, vs1, vd</span><br><div>The unsigned <b>vwredsumu.vs</b> instruction zero-extends the SEW-wide vector elements before summing them, then adds the 2*SEW-width scalar element, and stores the result in a 2*SEW-width scalar element.<br>For both <b>vwredsumu.vs</b> and vwredsum.vs, overflows wrap around.<br># Unsigned sum reduction into double-width accumulator <b>vwredsumu.vs</b> vd, vs2, vs1, vm # 2*SEW = 2*SEW + sum(zero-extend(SEW)) # Signed sum reduction into double-width accumulator vwredsum.vs vd, vs2, vs1, vm # 2*SEW = 2*SEW + sum(sign-extend(SEW))</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The unsigned vwredsumu.vs instruction zero-extends the SEW-wide vector elements before summing them, then adds the 2*SEW-width scalar element, and stores the result in a 2*SEW-width scalar element.\nFor both vwredsumu.vs and vwredsum.vs, overflows wrap around.\n# Unsigned sum reduction into double-width accumulator vwredsumu.vs vd, vs2, vs1, vm # 2*SEW = 2*SEW + sum(zero-extend(SEW)) # Signed sum reduction into double-width accumulator vwredsum.vs vd, vs2, vs1, vm # 2*SEW = 2*SEW + sum(sign-extend(SEW))\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-vector-integer-reduce-widen"
            };

        case "VWSUB.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VWSUB.VV</b> vs2, vs1, vd</span><br><div>vwsub<br>vwsub.w</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwsub\nvwsub.w\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWSUB.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VWSUB.VX</b> vs2, rs1, vd</span><br><div>vwsub<br>vwsub.w</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwsub\nvwsub.w\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWSUB.WV":
            return {
                "html": "<div><span class=\"opcode\"><b>VWSUB.WV</b> vs2, vs1, vd</span><br><div>vwsub<br>vwsub.w</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwsub\nvwsub.w\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWSUB.WX":
            return {
                "html": "<div><span class=\"opcode\"><b>VWSUB.WX</b> vs2, rs1, vd</span><br><div>vwsub<br>vwsub.w</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwsub\nvwsub.w\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWSUBU.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VWSUBU.VV</b> vs2, vs1, vd</span><br><div>vwsubu<br>vwsubu.w</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwsubu\nvwsubu.w\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWSUBU.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VWSUBU.VX</b> vs2, rs1, vd</span><br><div>vwsubu<br>vwsubu.w</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwsubu\nvwsubu.w\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWSUBU.WV":
            return {
                "html": "<div><span class=\"opcode\"><b>VWSUBU.WV</b> vs2, vs1, vd</span><br><div>vwsubu<br>vwsubu.w</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwsubu\nvwsubu.w\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VWSUBU.WX":
            return {
                "html": "<div><span class=\"opcode\"><b>VWSUBU.WX</b> vs2, rs1, vd</span><br><div>vwsubu<br>vwsubu.w</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vwsubu\nvwsubu.w\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VXOR.VI":
            return {
                "html": "<div><span class=\"opcode\"><b>VXOR.VI</b> vs2, simm5, vd</span><br><div>vxor</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vxor\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VXOR.VV":
            return {
                "html": "<div><span class=\"opcode\"><b>VXOR.VV</b> vs2, vs1, vd</span><br><div>vxor</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vxor\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VXOR.VX":
            return {
                "html": "<div><span class=\"opcode\"><b>VXOR.VX</b> vs2, rs1, vd</span><br><div>vxor</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "vxor\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#_vector_instruction_listing"
            };

        case "VZEXT.VF2":
            return {
                "html": "<div><span class=\"opcode\"><b>VZEXT.VF2</b> vs2, vd</span><br><div>The destination EEW is greater than the source EEW, the source EMUL is at least 1, and the overlap is in the highest-numbered part of the destination register group (e.g., when LMUL=8, vzext.vf4 v0, v6 is legal, but a source of v0, v2, or v4 is not).</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The destination EEW is greater than the source EEW, the source EMUL is at least 1, and the overlap is in the highest-numbered part of the destination register group (e.g., when LMUL=8, vzext.vf4 v0, v6 is legal, but a source of v0, v2, or v4 is not).\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-vec-operands"
            };

        case "VZEXT.VF4":
            return {
                "html": "<div><span class=\"opcode\"><b>VZEXT.VF4</b> vs2, vd</span><br><div>The destination EEW is greater than the source EEW, the source EMUL is at least 1, and the overlap is in the highest-numbered part of the destination register group (e.g., when LMUL=8, <b>vzext.vf4</b> v0, v6 is legal, but a source of v0, v2, or v4 is not).</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The destination EEW is greater than the source EEW, the source EMUL is at least 1, and the overlap is in the highest-numbered part of the destination register group (e.g., when LMUL=8, vzext.vf4 v0, v6 is legal, but a source of v0, v2, or v4 is not).\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-vec-operands"
            };

        case "VZEXT.VF8":
            return {
                "html": "<div><span class=\"opcode\"><b>VZEXT.VF8</b> vs2, vd</span><br><div>The destination EEW is greater than the source EEW, the source EMUL is at least 1, and the overlap is in the highest-numbered part of the destination register group (e.g., when LMUL=8, vzext.vf4 v0, v6 is legal, but a source of v0, v2, or v4 is not).</div><br><div><b>ISA</b>: v</div></div>",
                "tooltip": "The destination EEW is greater than the source EEW, the source EMUL is at least 1, and the overlap is in the highest-numbered part of the destination register group (e.g., when LMUL=8, vzext.vf4 v0, v6 is legal, but a source of v0, v2, or v4 is not).\n\n\n\n(ISA: v)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-v-spec/v1.0//v-spec.html#sec-vec-operands"
            };

        case "WFI":
            return {
                "html": "<div><span class=\"opcode\"><b>WFI</b> </span><br><div>field that supports intercepting the <b>WFI</b> instruction (see Section 1.3.3 ). When TW=0, the <b>WFI</b> instruction may execute in lower privilege modes when not prevented for some other reason. When TW=1, then if <b>WFI</b> is executed in any less-privileged mode, and it does not complete within an implementation-specific, bounded time limit, the <b>WFI</b> instruction causes an illegal instruction exception. The time limit may always be 0, in which case <b>WFI</b> always causes an illegal instruction exception in less-privileged modes when TW=1. TW is read-only 0 when there are no modes less privileged than M.<br>Trapping the <b>WFI</b> instruction can trigger a world switch to another guest OS, rather than wastefully idling in the current guest.<br>When S-mode is implemented, then executing <b>WFI</b> in U-mode causes an illegal instruction exception, unless it completes within an implementation-specific, bounded time limit. A future revision of this specification might add a feature that allows S-mode to selectively permit <b>WFI</b> in U-mode. Such a feature would only be active when TW=0.</div><br><div><b>ISA</b>: machine</div></div>",
                "tooltip": "field that supports intercepting the WFI instruction (see Section 1.3.3 ). When TW=0, the WFI instruction may execute in lower privilege modes when not prevented for some other reason. When TW=1, then if WFI is executed in any less-privileged mode, and it does not complete within an implementation-specific, bounded time limit, the WFI instruction causes an illegal instruction exception. The time limit may always be 0, in which case WFI always causes an illegal instruction exception in less-privileged modes when TW=1. TW is read-only 0 when there are no modes less privileged than M.\nTrapping the WFI instruction can trigger a world switch to another guest OS, rather than wastefully idling in the current guest.\nWhen S-mode is implemented, then executing WFI in U-mode causes an illegal instruction exception, unless it completes within an implementation-specific, bounded time limit. A future revision of this specification might add a feature that allows S-mode to selectively permit WFI in U-mode. Such a feature would only be active when TW=0.\n\n\n\n(ISA: machine)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-priv-isa-manual/Priv-v1.12/machine.html#virt-control"
            };

        case "WRS.NTO":
            return {
                "html": "<div><span class=\"opcode\"><b>WRS.NTO</b> </span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "WRS.STO":
            return {
                "html": "<div><span class=\"opcode\"><b>WRS.STO</b> </span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "XNOR":
            return {
                "html": "<div><span class=\"opcode\"><b>XNOR</b> rd, rs1, rs2</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "XOR":
            return {
                "html": "<div><span class=\"opcode\"><b>XOR</b> rd, rs1, rs2</span><br><div>ADD performs the addition of rs1 and rs2. SUB performs the subtraction of rs2 from rs1. Overflows are ignored and the low XLEN bits of results are written to the destination rd. SLT and SLTU perform signed and unsigned compares respectively, writing 1 to rd if rs1 < rs2, 0 otherwise. Note, SLTU rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs). AND, OR, and <b>XOR</b> perform bitwise logical operations.</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "ADD performs the addition of rs1 and rs2. SUB performs the subtraction of rs2 from rs1. Overflows are ignored and the low XLEN bits of results are written to the destination rd. SLT and SLTU perform signed and unsigned compares respectively, writing 1 to rd if rs1 < rs2, 0 otherwise. Note, SLTU rd, x0, rs2 sets rd to 1 if rs2 is not equal to zero, otherwise sets rd to zero (assembler pseudoinstruction SNEZ rd, rs). AND, OR, and XOR perform bitwise logical operations.\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-register-operations"
            };

        case "XORI":
            return {
                "html": "<div><span class=\"opcode\"><b>XORI</b> rd, rs1, imm12</span><br><div>ANDI, ORI, <b>XORI</b> are logical operations that perform bitwise AND, OR, and XOR on register rs1 and the sign-extended 12-bit immediate and place the result in rd. Note, <b>XORI</b> rd, rs1, -1 performs a bitwise logical inversion of register rs1 (assembler pseudoinstruction NOT rd, rs).</div><br><div><b>ISA</b>: rv32</div></div>",
                "tooltip": "ANDI, ORI, XORI are logical operations that perform bitwise AND, OR, and XOR on register rs1 and the sign-extended 12-bit immediate and place the result in rd. Note, XORI rd, rs1, -1 performs a bitwise logical inversion of register rs1 (assembler pseudoinstruction NOT rd, rs).\n\n\n\n(ISA: rv32)",
                "url": "https://five-embeddev.github.io/riscv-docs-html//riscv-user-isa-manual/Priv-v1.12/rv32.html#integer-register-immediate-instructions"
            };

        case "ZEXT.B":
            return {
                "html": "<div><span class=\"opcode\"><b>ZEXT.B</b> rd, rs</span><br><div><b>Equivalent ASM:</b><pre>andi rd, rs, 255</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nandi rd, rs, 255\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "ZEXT.H":
            return {
                "html": "<div><span class=\"opcode\"><b>ZEXT.H</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "ZEXT.W":
            return {
                "html": "<div><span class=\"opcode\"><b>ZEXT.W</b> rd, rs</span><br><div><b>Equivalent ASM:</b><pre>slli rd, rs, XLEN - 32\nsrli rd, rd, XLEN - 32</pre></div><br><div><b>ISA</b>: (pseudo)</div></div>",
                "tooltip": "Psuedo Instruction.\n\nEquivalent ASM:\n\nslli rd, rs, XLEN - 32\nsrli rd, rd, XLEN - 32\n\n",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };

        case "ZIP":
            return {
                "html": "<div><span class=\"opcode\"><b>ZIP</b> rd, rs1</span><br><div><b>ISA</b>: </div></div>",
                "tooltip": "\n\n(ISA: )",
                "url": "https://five-embeddev.github.io/riscv-docs-html/"
            };


    }
}
