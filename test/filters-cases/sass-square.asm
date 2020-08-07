        .headerflags    @"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM30 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM30)"
        .elftype        @"ET_EXEC"


//--------------------- .text._Z6squarePii        --------------------------
        .section        .text._Z6squarePii,"ax",@progbits
        .sectioninfo    @"SHI_REGISTERS=5"
        .align  64
        .global         _Z6squarePii
        .type           _Z6squarePii,@function
        .size           _Z6squarePii,(.L_18 - _Z6squarePii)
        .other          _Z6squarePii,@"STO_CUDA_ENTRY STV_DEFAULT"
_Z6squarePii:
.text._Z6squarePii:
        //## File "/home/partouf/./example.cu", line 3
                                                                                /* 0x2282c042e2c28307 */
        /*0008*/                   MOV R1, c[0x0][0x44];                        /* 0x2800400110005de4 */
        /*0010*/                   S2R R0, SR_CTAID.X;                          /* 0x2c00000094001c04 */
        //## File "/home/partouf/./example.cu", line 4
        /*0018*/                   ISETP.GE.AND P0, PT, R0, c[0x0][0x148], PT;  /* 0x1b0e40052001dc23 */
        /*0020*/               @P0 EXIT;                                        /* 0x80000000000001e7 */
        //## File "/home/partouf/./example.cu", line 5
        /*0028*/                   ISCADD R2.CC, R0, c[0x0][0x140], 0x2;        /* 0x4001400500009c43 */
        /*0030*/                   MOV32I R3, 0x4;                              /* 0x180000001000dde2 */
        /*0038*/                   IMAD.HI.X R3, R0, R3, c[0x0][0x144];         /* 0x208680051000dce3 */
                                                                                /* 0x20000002f04283f7 */
        /*0048*/                   LD.E R0, [R2];                               /* 0x8400000000201c85 */
        /*0050*/                   IMUL R4, R0, R0;                             /* 0x5000000000011ca3 */
        /*0058*/                   ST.E [R2], R4;                               /* 0x9400000000211c85 */
        //## File "/home/partouf/./example.cu", line 6
        /*0060*/                   EXIT;                                        /* 0x8000000000001de7 */
.L_1:
        /*0068*/                   BRA `(.L_1);                                 /* 0x4003ffffe0001de7 */
.L_18:
