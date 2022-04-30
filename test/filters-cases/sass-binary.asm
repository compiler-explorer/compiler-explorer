	.headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM52 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM52)"
	.elftype	@"ET_EXEC"


//--------------------- .text._Z3p10Pdi           --------------------------
	.section	.text._Z3p10Pdi,"ax",@progbits
	.sectioninfo	@"SHI_REGISTERS=24"
	.align	32
        .global         _Z3p10Pdi
        .type           _Z3p10Pdi,@function
        .size           _Z3p10Pdi,(.L_x_3 - _Z3p10Pdi)
        .other          _Z3p10Pdi,@"STO_CUDA_ENTRY STV_DEFAULT"
_Z3p10Pdi:
.text._Z3p10Pdi:
	//## File "/tmp/compiler-explorer-compiler2022128-339335-18rolti.c8qc/example.cu", line 1
                                                                                    /* 0x003ff400e24007f6 */
        /*0008*/                   MOV R1, c[0x0][0x20] ;                           /* 0x4c98078000870001 */
	//## File "/opt/compiler-explorer/gcc-10.2.0/include/c++/10.2.0/cmath", line 418
        /*0010*/                   I2F.F64.S32 R2, c[0x0] [0x148] ;                 /* 0x4cb8000005272b02 */
        /*0018*/                   CAL `($_Z3p10Pdi$__internal_accurate_pow) ;      /* 0xe260000011800040 */
                                                                                    /* 0x001fd800fe200711 */
        /*0028*/                   DADD R4, R2, 10 ;                                /* 0x3870004024070204 */
        /*0030*/                   MOV R6, R8 ;                                     /* 0x5c98078000870006 */
        /*0038*/                   MOV R7, R9 ;                                     /* 0x5c98078000970007 */
                                                                                    /* 0x001fc000fda00ff6 */
        /*0048*/                   LOP32I.AND R0, R5, 0x7ff00000 ;                  /* 0x0407ff0000070500 */
        /*0050*/                   ISETP.NE.AND P0, PT, R0, c[0x2][0x0], PT ;       /* 0x4b6b038800070007 */
        /*0058*/         {         MOV R0, c[0x0][0x148] ;                          /* 0x4c98078005270000 */
                                                                                    /* 0x001c7400fe0007fd */
        /*0068*/               @P0 BRA `(.L_x_0)         }
                                                                                    /* 0xe24000000880000f */
        /*0070*/         {         MOV R6, R4 ;                                     /* 0x5c98078000470006 */
        /*0078*/                   DSETP.GTU.AND P0, PT, |R2|, +INF , PT         }
                                                                                    /* 0x368c03fff0070287 */
                                                                                    /* 0x001c7c01ffa007f0 */
        /*0088*/         {         MOV R7, R5 ;                                     /* 0x5c98078000570007 */
        /*0090*/               @P0 BRA `(.L_x_0)         }
                                                                                    /* 0xe24000000600000f */
                                                                                    /* 0x003fc401e3c00f1e */
        /*0528*/                   DFMA R14, R12, R14, c[0x2][0xb0] ;               /* 0x5370070802c70c0e */
        /*0530*/                   DFMA R14, R12, R14, c[0x2][0xb0] ;               /* 0x5370070802c70c0e */
        /*0538*/                   ISCADD R13, R8, R15, 0x14 ;                      /* 0x5c180a0000f7080d */
                                                                                    /* 0x001fc000ffa007f0 */
        /*0548*/         {         MOV R12, R14 ;                                   /* 0x5c98078000e7000c */
        /*0550*/              @!P0 BRA `(.L_x_1)         }
                                                                                    /* 0xe24000000988000f */
        /*0558*/         {         FSETP.GEU.AND P1, PT, |R11|, c[0x2][0xb8], PT ;  /* 0x4bbe038802e70b8f */
                                                                                    /* 0x003fd000e3000711 */
        /*0568*/                   DSETP.GEU.AND P0, PT, R10, RZ, PT         }
                                                                                    /* 0x5b8e03800ff70a07 */
        /*0570*/                   DADD R12, R10, +INF  ;                           /* 0x3870007ff0070a0c */
        /*0578*/                   SEL R12, R12, RZ, P0 ;                           /* 0x5ca000000ff70c0c */
                                                                                    /* 0x081fc400ffa007f0 */
        /*0588*/         {         SEL R13, R13, RZ, P0 ;                           /* 0x5ca000000ff70d0d */
        /*0590*/               @P1 BRA `(.L_x_1)         }
                                                                                    /* 0xe24000000581000f */
        /*0598*/                   LEA.HI R0, R8.reuse, R8, RZ, 0x1 ;               /* 0x5bdf7f8010870800 */
                                                                                    /* 0x001fd800fe8007f1 */
        /*0618*/                   ISETP.EQ.AND P0, PT, R12, RZ, !P0 ;              /* 0x5b6504000ff70c07 */
                                                                                    /* 0x001fc001fc201f18 */
        /*0628*/              @!P0 DFMA R12, R4, R12, R12 ;                         /* 0x5b70060000c8040c */
        /*0630*/                   MOV R8, R12 ;                                    /* 0x5c98078000c70008 */
        /*0638*/         {         MOV R9, R13 ;                                    /* 0x5c98078000d70009 */
                                                                                    /* 0x001f8000ffe007ff */
        /*0648*/                   RET         }
.L_x_2:
                                                                                    /* 0xe32000000007000f */
        /*0650*/                   BRA `(.L_x_2) ;                                  /* 0xe2400fffff87000f */
        /*0658*/                   NOP;                                             /* 0x50b0000000070f00 */
                                                                                    /* 0x001f8000fc0007e0 */
        /*0668*/                   NOP;                                             /* 0x50b0000000070f00 */
        /*0670*/                   NOP;                                             /* 0x50b0000000070f00 */
        /*0678*/                   NOP;                                             /* 0x50b0000000070f00 */
.L_x_3:
