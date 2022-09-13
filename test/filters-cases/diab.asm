#$$eb
#$$sz 0
#$$ss 0
#$$sg 0
#$$fp 0
#$$m2         - PowerPC mnemonics
#$$pPPC - PowerPC instructions
#$$oPPCE200Z1
#$$ko 1   - Reorder info
        .file         "example.cpp"
#$$dg 1
        .section        .PPC.EMB.apuinfo,,@note
        .4byte      8
        .4byte      0+4+4
        .4byte      2
        .byte         "APUinfo"
        .byte         0
        .align      2
        .4byte      0x01000001
        .4byte      0x00400001
        .text
        .section        .debug_line,,n
.L8:
        .text
#$$ld
.L4:
        .0byte      .L2
        .d2_line_start  .debug_line
        .text
        .align      2
        .text
        .d2file   "/tmp/compiler-explorer-compiler202107-8023-z5iran.8cqm/example.cpp"
        .d2line         4,8
#$$ld
.L84:

#$$bf   testFunction(double*, double),interprocedural,rasave,nostackparams
        .globl      testFunction(double*, double)
        .d2_cfa_start.r __cie
testFunction(double*, double):
#$$dr 0 0 0
.Llo1:
        stwu            r1,-48(r1)          
        .d2_cfa_def_cfa_offset  48
        mfspr         r0,lr
        stmw            r26,24(r1)          # offset r1+24  0x18
        .d2_cfa_offset_list     26,31,1,1
        stw       r0,52(r1)
        .d2_cfa_offset    108,-1
        mr          r31,r3      # input=r31 input=r3
        mr          r30,r5
        mr          r29,r6
        .d2prologue_end
        .d2line   5
        diab.li   r28,0
        diab.li   r27,0
        .text
.L92:
        .d2line   6
.Llo2:
        diab.li   r26,0             # i=r26
.L42:
.Llo3:
        mr          r3,r26      # i=r3 i=r26
        bl          _d_itod
        mr          r6,r29
        mr          r5,r30
        bl          _d_fge
        extsb.      r0,r3         # i=r3
        bc          4,2,.L44        # ne cr0
        .d2line   7
        rlwinm      r0,r26,3,0,28         # i=r26
        add       r3,r31,r0         # i=r3 input=r31
        lwzx            r5,r31,r0             # input=r31
        lwz       r6,4(r3)            # i=r3
        mr          r4,r27
        mr          r3,r28      # i=r3
        bl          _d_add
        mr          r27,r4
        mr          r28,r3      # i=r28
        .d2line   8
        addi            r26,r26,1             # i=r26 i=r26
        b             .L42
.L44:
        .text
.L93:
        .d2line   9
.Llo4:
        mr          r4,r27
        mr          r3,r28      # i=r3
        .d2line   10
        .d2epilogue_begin
        lmw       r26,24(r1)      # offset r1+24  0x18
        .d2_cfa_restore_list    2,10
        lwz       r0,52(r1)
        mtspr         lr,r0
        addi            r1,r1,48                
        .d2_cfa_def_cfa_offset  0
        blr
#$$ef
        .d2_cfa_end 2
        .text
#$$ld
.L85:
        .type         testFunction(double*, double),@function
        .size         testFunction(double*, double),.-testFunction(double*, double)
# Number of nodes = 24

# Allocations for testFunction(double*, double)
#       ?a4         input
#       ?a5         length
#       ?a7         sum
#       ?a9         i
        .align      2
        .text
        .d2line         12,5
#$$ld
.L101:

#$$bf   fibo(int),interprocedural,rasave,nostackparams
        .globl      fibo(int)
        .d2_cfa_start.r __cie
fibo(int):
#$$dr 0 0 0
.Llo5:
        stwu            r1,-32(r1)          
        .d2_cfa_def_cfa_offset  32
        mfspr         r0,lr
        stmw            r30,24(r1)          # offset r1+24  0x18
        .d2_cfa_offset_list     30,31,1,1
        stw       r0,36(r1)
        .d2_cfa_offset    108,-1
        .frame_info.r   r30,,,1,0,0,0,.L105
        mr          r31,r3      # n=r31 n=r3
        .d2prologue_end
        .d2line   14
        cmpi            0,0,r3,1                # n=r3
        bc          12,1,.L46       # gt cr0
        .d2line   15
.Llo6:
        b             .L45
.L46:
        .d2line   16
.Llo7:
        addi            r3,r3,-1                # n=r3 n=r3
        bl          fibo(int)
        mr          r30,r3      # n=r30
        addi            r3,r31,-2             # n=r3 n=r31
        bl          fibo(int)
        add       r3,r3,r30         # n=r3 n=r3 n=r30
.L45:
        .d2line   17
        .d2epilogue_begin
.L105:
.Llo8:
        lmw       r30,24(r1)      # offset r1+24  0x18
        .d2_cfa_restore_list    3,10
.Llo9:
        lwz       r0,36(r1)
        mtspr         lr,r0
        addi            r1,r1,32                
        .d2_cfa_def_cfa_offset  0
        blr
.Llo10:
#$$ef
        .d2_cfa_end 2
        .text
#$$ld
.L102:
        .type         fibo(int),@function
        .size         fibo(int),.-fibo(int)
# Number of nodes = 27

# Allocations for fibo(int)
#       ?a4         n
#       ?a5         $$2
#       ?a6         $$1
        .align      2
        .text
        .d2line         19,5
#$$ld
.L108:

#$$bf   fizz_buzz(),interprocedural,rasave,nostackparams
        .globl      fizz_buzz()
        .d2_cfa_start.r __cie
fizz_buzz():
#$$dr 0 0 0
.Llo11:
        stwu            r1,-32(r1)          
        .d2_cfa_def_cfa_offset  32
        mfspr         r0,lr
        stw       r31,28(r1)      # offset r1+28  0x1c
        .d2_cfa_offset_list     31,31,1,1
        stw       r0,36(r1)
        .d2_cfa_offset    108,-1
        .frame_info.r   r31,,,1,0,0,0,.L113
        .d2prologue_end
        .d2line   22
        diab.li   r31,1             # i=r31
.L48:
        cmpi            0,0,r31,100       # i=r31
        bc          12,1,.L50       # gt cr0
        .d2line   24
        lis       r0,21845
        ori       r0,r0,21846
        mulhw         r0,r0,r31         # i=r31
        srawi         r3,r31,31         # i=r31
        subf            r0,r3,r0
        add       r3,r0,r0
        add       r0,r0,r3
        subf.         r0,r0,r31         # i=?a4
        bc          4,2,.L51        # ne cr0
        .d2line   25
        addis         r3,0,.L114@ha
        addi            r3,r3,.L114@l
        bl          printf
.L51:
        .d2line   26
        lis       r0,26214
        ori       r0,r0,26215
        mulhw         r0,r0,r31         # i=r31
        srawi         r0,r0,1
        srawi         r3,r31,31         # i=r31
        subf            r0,r3,r0
        rlwinm      r3,r0,2,0,29
        add       r0,r0,r3
        subf.         r0,r0,r31         # i=?a4
        bc          4,2,.L52        # ne cr0
        .d2line   27
        addis         r3,0,.L115@ha
        addi            r3,r3,.L115@l
        bl          printf
.L52:
        .d2line   28
        lis       r0,21845
        ori       r0,r0,21846
        mulhw         r0,r0,r31         # i=r31
        srawi         r3,r31,31         # i=r31
        subf            r0,r3,r0
        add       r3,r0,r0
        add       r0,r0,r3
        subf.         r0,r0,r31         # i=?a4
        bc          12,2,.L53       # eq cr0
        lis       r0,26214
        ori       r0,r0,26215
        mulhw         r0,r0,r31         # i=r31
        srawi         r0,r0,1
        srawi         r3,r31,31         # i=r31
        subf            r0,r3,r0
        rlwinm      r3,r0,2,0,29
        add       r0,r0,r3
        subf.         r0,r0,r31         # i=?a4
        bc          12,2,.L53       # eq cr0
        .d2line   29
        addis         r3,0,.L116@ha
        addi            r3,r3,.L116@l
        mr          r4,r31      # i=r4 i=r31
        bl          printf
.L53:
        .d2line   30
        addis         r3,0,.L117@ha
        addi            r3,r3,.L117@l
        bl          printf
        .d2line   31
        addi            r31,r31,1             # i=r31 i=r31
        b             .L48
.L50:
        .d2line   33
        diab.li   r3,0
        .d2line   34
        .d2epilogue_begin
.L113:
.Llo12:
        lwz       r31,28(r1)      # offset r1+28  0x1c
        .d2_cfa_restore_list    3,10
        lwz       r0,36(r1)
        mtspr         lr,r0
        addi            r1,r1,32                
        .d2_cfa_def_cfa_offset  0
        blr
#$$ef
        .d2_cfa_end 2
        .text
#$$ld
.L109:
        .type         fizz_buzz(),@function
        .size         fizz_buzz(),.-fizz_buzz()
# Number of nodes = 50

# Allocations for fizz_buzz()
#       ?a4         i
        .align      2
        .text
        .d2line         37,6
#$$ld
.L120:

#$$bf   printEvenNumbers(int),interprocedural,rasave,nostackparams
        .globl      printEvenNumbers(int)
        .d2_cfa_start.r __cie
printEvenNumbers(int):
#$$dr 0 0 0
.Llo13:
        stwu            r1,-32(r1)          
        .d2_cfa_def_cfa_offset  32
        mfspr         r0,lr
        stmw            r30,24(r1)          # offset r1+24  0x18
        .d2_cfa_offset_list     30,31,1,1
        stw       r0,36(r1)
        .d2_cfa_offset    108,-1
        .frame_info.r   r30,,,1,0,0,0,.L124
        mr          r31,r3      # N=r31 N=r3
        .d2prologue_end
        .text
.L125:
        .d2line   39
.Llo14:
        diab.li   r30,1             # i=r30
.L55:
.Llo15:
        rlwinm      r0,r31,1,0,30         # N=r31
        cmp       0,0,r0,r30      # i=r30
        bc          12,0,.L54       # lt cr0
        .d2line   42
        srawi         r0,r30,1            # i=r30
        addze         r0,r0
        add       r0,r0,r0
        subf.         r0,r0,r30         # i=?a5
        bc          4,2,.L58        # ne cr0
        .d2line   43
        addis         r3,0,.L131@ha
        addi            r3,r3,.L131@l
        mr          r4,r30      # i=r4 i=r30
        bl          printf
.L58:
        .d2line   44
        addi            r30,r30,1             # i=r30 i=r30
        b             .L55
        .text
.L126:
.L54:
        .d2line   45
        .d2epilogue_begin
.L124:
.Llo16:
        lmw       r30,24(r1)      # offset r1+24  0x18
        .d2_cfa_restore_list    3,10
.Llo17:
        lwz       r0,36(r1)
        mtspr         lr,r0
        addi            r1,r1,32                
        .d2_cfa_def_cfa_offset  0
        blr
.Llo18:
#$$ef
        .d2_cfa_end 2
        .text
#$$ld
.L121:
        .type         printEvenNumbers(int),@function
        .size         printEvenNumbers(int),.-printEvenNumbers(int)
# Number of nodes = 23

# Allocations for printEvenNumbers(int)
#       ?a4         N
#       ?a5         i
        .align      2
        .text
        .d2line         48,6
#$$ld
.L134:

#$$bf   printOddNumbers(int),interprocedural,rasave,nostackparams
        .globl      printOddNumbers(int)
        .d2_cfa_start.r __cie
printOddNumbers(int):
#$$dr 0 0 0
.Llo19:
        stwu            r1,-32(r1)          
        .d2_cfa_def_cfa_offset  32
        mfspr         r0,lr
        stmw            r30,24(r1)          # offset r1+24  0x18
        .d2_cfa_offset_list     30,31,1,1
        stw       r0,36(r1)
        .d2_cfa_offset    108,-1
        .frame_info.r   r30,,,1,0,0,0,.L138
        mr          r31,r3      # N=r31 N=r3
        .d2prologue_end
        .text
.L139:
        .d2line   50
.Llo20:
        diab.li   r30,1             # i=r30
.L60:
.Llo21:
        rlwinm      r0,r31,1,0,30         # N=r31
        cmp       0,0,r0,r30      # i=r30
        bc          12,0,.L59       # lt cr0
        .d2line   53
        srawi         r0,r30,1            # i=r30
        addze         r0,r0
        add       r0,r0,r0
        subf.         r0,r0,r30         # i=?a5
        bc          12,2,.L63       # eq cr0
        .d2line   54
        addis         r3,0,.L131@ha
        addi            r3,r3,.L131@l
        mr          r4,r30      # i=r4 i=r30
        bl          printf
.L63:
        .d2line   55
        addi            r30,r30,1             # i=r30 i=r30
        b             .L60
        .text
.L140:
.L59:
        .d2line   56
        .d2epilogue_begin
.L138:
.Llo22:
        lmw       r30,24(r1)      # offset r1+24  0x18
        .d2_cfa_restore_list    3,10
.Llo23:
        lwz       r0,36(r1)
        mtspr         lr,r0
        addi            r1,r1,32                
        .d2_cfa_def_cfa_offset  0
        blr
.Llo24:
#$$ef
        .d2_cfa_end 2
        .text
#$$ld
.L135:
        .type         printOddNumbers(int),@function
        .size         printOddNumbers(int),.-printOddNumbers(int)
# Number of nodes = 23

# Allocations for printOddNumbers(int)
#       ?a4         N
#       ?a5         i
        .align      2
        .text
        .d2line         58,6
#$$ld
.L147:

#$$bf   tokenizeString(),interprocedural,rasave,nostackparams
        .globl      tokenizeString()
        .d2_cfa_start.r __cie
tokenizeString():
#$$dr 0 0 0
        stwu            r1,-32(r1)          
        .d2_cfa_def_cfa_offset  32
        mfspr         r0,lr
        stw       r0,36(r1)
        .d2_cfa_offset    108,-1
        .frame_info.r   ,,,1,0,0,0,.L157
        .d2prologue_end
        .d2line   60
        addis         r12,0,.L65@ha
        addi            r12,r12,.L65@l
        addi            r12,r12,-1
        addi            r11,r1,7
        diab.li   r10,8
        mtspr         ctr,r10
.L158:
        lbzu            r9,1(r12)
        lbzu            r10,1(r12)
        stbu            r9,1(r11)
        stbu            r10,1(r11)
        bc          16,0,.L158
        .d2line   63
        addi            r3,r1,8
        addis         r4,0,.L159@ha
        addi            r4,r4,.L159@l
        bl          strtok
.Llo25:
        mr          r4,r3         # token=r4 token=r3
.L66:
        .d2line   67
        cmpi            0,0,r3,0                # token=r3
        bc          12,2,.L64       # eq cr0
        .d2line   69
.Llo26:
        addis         r3,0,.L160@ha             # token=r3
.Llo27:
        addi            r3,r3,.L160@l         # token=r3 token=r3
.Llo28:
        bl          printf
        .d2line   70
        addis         r4,0,.L159@ha             # token=r4
.Llo29:
        addi            r4,r4,.L159@l         # token=r4 token=r4
.Llo30:
        diab.li   r3,0                # token=r3
        bl          strtok
.Llo31:
        mr          r4,r3         # token=r4 token=r3
        b             .L66
.L64:
        .d2line   72
        .d2epilogue_begin
.L157:
.Llo32:
        lwz       r0,36(r1)
        mtspr         lr,r0
        addi            r1,r1,32                
        .d2_cfa_def_cfa_offset  0
        blr
#$$ef
        .d2_cfa_end 2
        .text
#$$ld
.L148:
        .type         tokenizeString(),@function
        .size         tokenizeString(),.-tokenizeString()
# Number of nodes = 37

# Allocations for tokenizeString()
#       ?a4         $$4
#       ?a5         $$3
#       SP,8      str
#       ?a6         token
        .align      2
        .text
        .d2line         74,5
#$$ld
.L168:

#$$bf   main,interprocedural,rasave,nostackparams
        .globl      main
        .d2_cfa_start.r __cie
main:
#$$dr 0 0 0
        stwu            r1,-64(r1)          
        .d2_cfa_def_cfa_offset  64
        mfspr         r0,lr
        stw       r31,60(r1)      # offset r1+60  0x3c
        .d2_cfa_offset_list     31,31,1,1
        stw       r0,68(r1)
        .d2_cfa_offset    108,-1
        .frame_info.r   r31,,,1,0,0,0,.L177
        .d2prologue_end
        .d2line   76
        diab.li   r3,10
        bl          fibo(int)
        mr          r31,r3
        .d2line   77
        bl          fizz_buzz()
        .d2line   78
        mr          r3,r31
        bl          printOddNumbers(int)
        .d2line   79
        lis       r0,16368
        diab.li   r6,0
        stw       r0,8(r1)
        stw       r6,12(r1)
        lis       r0,16384
        stw       r0,16(r1)
        stw       r6,20(r1)
        lis       r0,16392
        stw       r0,24(r1)
        stw       r6,28(r1)
        lis       r0,16400
        stw       r0,32(r1)
        stw       r6,36(r1)
        lis       r5,16404
        stw       r5,40(r1)
        stw       r6,44(r1)
        .d2line   80
        addi            r3,r1,8
        bl          testFunction(double*, double)
        .d2line   81
        bl          tokenizeString()
        .d2line   82
        diab.li   r3,0
        .d2line   83
        .d2epilogue_begin
.L177:
        lwz       r31,60(r1)      # offset r1+60  0x3c
        .d2_cfa_restore_list    3,10
        lwz       r0,68(r1)
        mtspr         lr,r0
        addi            r1,r1,64                
        .d2_cfa_def_cfa_offset  0
        blr
#$$ef
        .d2_cfa_end 2
        .text
#$$ld
.L169:
        .type         main,@function
        .size         main,.-main
# Number of nodes = 64

# Allocations for main
#       ?a4         $$5
#       ?a5         n
#       SP,8      a
#       ?a6         sum

# Allocations for module
        .text
        .align      2
#       Begin local data area
#       LDA + 0
        .type         .L65,@object
        .size         .L65,16
        .align      2
#       static                __static_init1
.L65:
        .byte         71,101,101,107,115,45,102,111,114,45,71,101,101,107,115
        .byte         0
        .sdata2
        .type         FLOAT_TEMP.161,@object
        .size         FLOAT_TEMP.161,8
        .align      3
FLOAT_TEMP.161:
        .double   +1.0000000000000000000
        .type         FLOAT_TEMP.162,@object
        .size         FLOAT_TEMP.162,8
        .align      3
FLOAT_TEMP.162:
        .double   +2.0000000000000000000
        .type         FLOAT_TEMP.163,@object
        .size         FLOAT_TEMP.163,8
        .align      3
FLOAT_TEMP.163:
        .double   +3.0000000000000000000
        .type         FLOAT_TEMP.164,@object
        .size         FLOAT_TEMP.164,8
        .align      3
FLOAT_TEMP.164:
        .double   +4.0000000000000000000
        .type         FLOAT_TEMP.165,@object
        .size         FLOAT_TEMP.165,8
        .align      3
FLOAT_TEMP.165:
        .double   +5.0000000000000000000
        .text
        .align      2
.L114:
        .byte         70,105,122,122
        .byte         0
        .text
        .align      2
.L115:
        .byte         66,117,122,122
        .byte         0
        .text
        .align      2
.L116:
        .byte         110,117,109,98,101,114,61,37,100
        .byte         0
        .text
        .align      2
.L117:
        .byte         10
        .byte         0
        .text
        .align      2
.L131:
        .byte         37,100
        .byte         0
        .text
        .align      2
.L159:
        .byte         45
        .byte         0
        .text
        .align      2
.L160:
        .byte         37,115,10
        .byte         0
        .text
#$$ld
.L5:
.L86:   .d2filenum "/tmp/compiler-explorer-compiler202107-8023-z5iran.8cqm/example.cpp"
        .d2_line_end

        .section        .debug_abbrev,,n
.L9:
        .section        .debug_abbrev,,n
        .uleb128        1
        .uleb128        17
        .byte         0x1
        .uleb128        1
        .uleb128        19
        .uleb128        3
        .uleb128        8
        .uleb128        37
        .uleb128        8
        .uleb128        27
        .uleb128        8
        .uleb128        19
        .uleb128        15
        .uleb128        17
        .uleb128        1
        .uleb128        18
        .uleb128        1
        .uleb128        16
        .uleb128        6
        .uleb128        0
        .uleb128        0
        .section        .debug_abbrev,,n
        .uleb128        2
        .uleb128        46
        .byte         0x1
        .uleb128        1
        .uleb128        19
        .uleb128        3
        .uleb128        8
        .uleb128        58
        .uleb128        6
        .uleb128        59
        .uleb128        15
        .uleb128        57
        .uleb128        15
        .uleb128        73
        .uleb128        16
        .uleb128        63
        .uleb128        12
        .uleb128        39
        .uleb128        12
        .uleb128        17
        .uleb128        1
        .uleb128        18
        .uleb128        1
        .uleb128        0
        .uleb128        0
        .section        .debug_abbrev,,n
        .uleb128        3
        .uleb128        5
        .byte         0x0
        .uleb128        58
        .uleb128        6
        .uleb128        59
        .uleb128        15
        .uleb128        57
        .uleb128        15
        .uleb128        3
        .uleb128        8
        .uleb128        73
        .uleb128        16
        .uleb128        2
        .uleb128        6
        .uleb128        0
        .uleb128        0
        .section        .debug_abbrev,,n
        .uleb128        4
        .uleb128        5
        .byte         0x0
        .uleb128        58
        .uleb128        6
        .uleb128        59
        .uleb128        15
        .uleb128        57
        .uleb128        15
        .uleb128        3
        .uleb128        8
        .uleb128        73
        .uleb128        16
        .uleb128        0
        .uleb128        0
        .section        .debug_abbrev,,n
        .uleb128        5
        .uleb128        52
        .byte         0x0
        .uleb128        58
        .uleb128        6
        .uleb128        59
        .uleb128        15
        .uleb128        57
        .uleb128        15
        .uleb128        3
        .uleb128        8
        .uleb128        73
        .uleb128        16
        .uleb128        0
        .uleb128        0
        .section        .debug_abbrev,,n
        .uleb128        6
        .uleb128        11
        .byte         0x1
        .uleb128        1
        .uleb128        16
        .uleb128        17
        .uleb128        1
        .uleb128        18
        .uleb128        1
        .uleb128        0
        .uleb128        0
        .section        .debug_abbrev,,n
        .uleb128        7
        .uleb128        52
        .byte         0x0
        .uleb128        58
        .uleb128        6
        .uleb128        59
        .uleb128        15
        .uleb128        57
        .uleb128        15
        .uleb128        3
        .uleb128        8
        .uleb128        73
        .uleb128        16
        .uleb128        2
        .uleb128        6
        .uleb128        0
        .uleb128        0
        .section        .debug_abbrev,,n
        .uleb128        8
        .uleb128        46
        .byte         0x1
        .uleb128        1
        .uleb128        19
        .uleb128        3
        .uleb128        8
        .uleb128        58
        .uleb128        6
        .uleb128        59
        .uleb128        15
        .uleb128        57
        .uleb128        15
        .uleb128        63
        .uleb128        12
        .uleb128        39
        .uleb128        12
        .uleb128        17
        .uleb128        1
        .uleb128        18
        .uleb128        1
        .uleb128        0
        .uleb128        0
        .section        .debug_abbrev,,n
        .uleb128        9
        .uleb128        52
        .byte         0x0
        .uleb128        58
        .uleb128        6
        .uleb128        59
        .uleb128        15
        .uleb128        57
        .uleb128        15
        .uleb128        3
        .uleb128        8
        .uleb128        73
        .uleb128        16
        .uleb128        2
        .uleb128        9
        .uleb128        0
        .uleb128        0
        .section        .debug_abbrev,,n
        .uleb128        10
        .uleb128        36
        .byte         0x0
        .uleb128        3
        .uleb128        8
        .uleb128        62
        .uleb128        11
        .uleb128        11
        .uleb128        11
        .uleb128        0
        .uleb128        0
        .section        .debug_abbrev,,n
        .uleb128        11
        .uleb128        15
        .byte         0x0
        .uleb128        73
        .uleb128        16
        .uleb128        0
        .uleb128        0
        .section        .debug_abbrev,,n
        .uleb128        12
        .uleb128        1
        .byte         0x1
        .uleb128        1
        .uleb128        19
        .uleb128        73
        .uleb128        16
        .uleb128        0
        .uleb128        0
        .section        .debug_abbrev,,n
        .uleb128        13
        .uleb128        33
        .byte         0x0
        .uleb128        47
        .uleb128        15
        .uleb128        0
        .uleb128        0
        .section        .debug_abbrev,,n
        .sleb128        0

        .section        .debug_info,,n
.L2:
        .4byte      .L3-.L1
.L1:
        .2byte      0x2
        .4byte      .L9
        .byte         0x4
        .section        .debug_info,,n
        .sleb128        1
        .4byte      .L7-.L2
        .byte         "/tmp/compiler-explorer-compiler202107-8023-z5iran.8cqm/example.cpp"
        .byte         0
        .byte         "Diab Data, Inc:dplus Rel 5.9.7.1-a_1:PPCE200Z1N"
        .byte         0
        .byte         "/tmp/compiler-explorer-compiler202107-8023-z5iran.8cqm"
        .byte         0
        .uleb128        4
        .4byte      .L4
        .4byte      .L5
        .4byte      .L8
        .section        .debug_info,,n
.L88:
        .sleb128        2
        .4byte      .L83-.L2
        .byte         "testFunction"
        .byte         0
        .4byte      .L86
        .uleb128        4
        .uleb128        8
        .4byte      .L87
        .byte         0x1
        .byte         0x1
        .4byte      .L84
        .4byte      .L85
        .section        .debug_info,,n
        .sleb128        3
        .4byte      .L86
        .uleb128        4
        .uleb128        8
        .byte         "input"
        .byte         0
        .4byte      .L89
        .4byte      .L90
        .section        .debug_info,,n
        .sleb128        4
        .4byte      .L86
        .uleb128        4
        .uleb128        8
        .byte         "length"
        .byte         0
        .4byte      .L87
        .section        .debug_info,,n
.L91:
        .sleb128        5
        .4byte      .L86
        .uleb128        5
        .uleb128        10
        .byte         "sum"
        .byte         0
        .4byte      .L87
        .section        .debug_info,,n
        .sleb128        6
        .4byte      .L95
        .4byte      .L92
        .4byte      .L93
        .section        .debug_info,,n
.L96:
        .sleb128        7
        .4byte      .L86
        .uleb128        6
        .uleb128        12
        .byte         "i"
        .byte         0
        .4byte      .L97
        .4byte      .L98
        .section        .debug_info,,n
        .sleb128        0
.L95:
        .section        .debug_info,,n
        .sleb128        0
.L83:
        .section        .debug_info,,n
.L103:
        .sleb128        2
        .4byte      .L100-.L2
        .byte         "fibo"
        .byte         0
        .4byte      .L86
        .uleb128        12
        .uleb128        5
        .4byte      .L97
        .byte         0x1
        .byte         0x1
        .4byte      .L101
        .4byte      .L102
        .sleb128        3
        .4byte      .L86
        .uleb128        12
        .uleb128        5
        .byte         "n"
        .byte         0
        .4byte      .L97
        .4byte      .L104
        .section        .debug_info,,n
        .sleb128        0
.L100:
        .section        .debug_info,,n
.L110:
        .sleb128        2
        .4byte      .L107-.L2
        .byte         "fizz_buzz"
        .byte         0
        .4byte      .L86
        .uleb128        19
        .uleb128        5
        .4byte      .L97
        .byte         0x1
        .byte         0x1
        .4byte      .L108
        .4byte      .L109
.L111:
        .sleb128        7
        .4byte      .L86
        .uleb128        21
        .uleb128        9
        .byte         "i"
        .byte         0
        .4byte      .L97
        .4byte      .L112
        .section        .debug_info,,n
        .sleb128        0
.L107:
        .section        .debug_info,,n
.L122:
        .sleb128        8
        .4byte      .L119-.L2
        .byte         "printEvenNumbers"
        .byte         0
        .4byte      .L86
        .uleb128        37
        .uleb128        6
        .byte         0x1
        .byte         0x1
        .4byte      .L120
        .4byte      .L121
        .sleb128        3
        .4byte      .L86
        .uleb128        37
        .uleb128        6
        .byte         "N"
        .byte         0
        .4byte      .L97
        .4byte      .L123
        .section        .debug_info,,n
        .sleb128        6
        .4byte      .L128
        .4byte      .L125
        .4byte      .L126
.L129:
        .sleb128        7
        .4byte      .L86
        .uleb128        39
        .uleb128        14
        .byte         "i"
        .byte         0
        .4byte      .L97
        .4byte      .L130
        .section        .debug_info,,n
        .sleb128        0
.L128:
        .section        .debug_info,,n
        .sleb128        0
.L119:
        .section        .debug_info,,n
.L136:
        .sleb128        8
        .4byte      .L133-.L2
        .byte         "printOddNumbers"
        .byte         0
        .4byte      .L86
        .uleb128        48
        .uleb128        6
        .byte         0x1
        .byte         0x1
        .4byte      .L134
        .4byte      .L135
        .sleb128        3
        .4byte      .L86
        .uleb128        48
        .uleb128        6
        .byte         "N"
        .byte         0
        .4byte      .L97
        .4byte      .L137
        .section        .debug_info,,n
        .sleb128        6
        .4byte      .L142
        .4byte      .L139
        .4byte      .L140
.L143:
        .sleb128        7
        .4byte      .L86
        .uleb128        50
        .uleb128        14
        .byte         "i"
        .byte         0
        .4byte      .L97
        .4byte      .L144
        .section        .debug_info,,n
        .sleb128        0
.L142:
        .section        .debug_info,,n
        .sleb128        0
.L133:
        .section        .debug_info,,n
.L149:
        .sleb128        8
        .4byte      .L146-.L2
        .byte         "tokenizeString"
        .byte         0
        .4byte      .L86
        .uleb128        58
        .uleb128        6
        .byte         0x1
        .byte         0x1
        .4byte      .L147
        .4byte      .L148
        .section        .debug_info,,n
.L150:
        .sleb128        9
        .4byte      .L86
        .uleb128        60
        .uleb128        10
        .byte         "str"
        .byte         0
        .4byte      .L151
        .sleb128        3
        .byte         0x92
        .uleb128        1
        .sleb128        8
.L154:
        .sleb128        7
        .4byte      .L86
        .uleb128        63
        .uleb128        11
        .byte         "token"
        .byte         0
        .4byte      .L155
        .4byte      .L156
        .section        .debug_info,,n
        .sleb128        0
.L146:
        .section        .debug_info,,n
.L170:
        .sleb128        2
        .4byte      .L167-.L2
        .byte         "main"
        .byte         0
        .4byte      .L86
        .uleb128        74
        .uleb128        5
        .4byte      .L97
        .byte         0x1
        .byte         0x1
        .4byte      .L168
        .4byte      .L169
.L171:
        .sleb128        7
        .4byte      .L86
        .uleb128        76
        .uleb128        9
        .byte         "n"
        .byte         0
        .4byte      .L97
        .4byte      .L172
.L173:
        .sleb128        9
        .4byte      .L86
        .uleb128        79
        .uleb128        12
        .byte         "a"
        .byte         0
        .4byte      .L174
        .sleb128        3
        .byte         0x92
        .uleb128        1
        .sleb128        8
.L176:
        .sleb128        5
        .4byte      .L86
        .uleb128        80
        .uleb128        12
        .byte         "sum"
        .byte         0
        .4byte      .L87
        .section        .debug_info,,n
        .sleb128        0
.L167:
        .section        .debug_info,,n
.L87:
        .sleb128        10
        .byte         "double"
        .byte         0
        .byte         0x4
        .byte         0x8
        .section        .debug_info,,n
.L89:
        .sleb128        11
        .4byte      .L87
.L97:
        .sleb128        10
        .byte         "int"
        .byte         0
        .byte         0x5
        .byte         0x4
.L153:
        .sleb128        10
        .byte         "char"
        .byte         0
        .byte         0x8
        .byte         0x1
        .section        .debug_info,,n
.L151:
        .sleb128        12
        .4byte      .L152-.L2
        .4byte      .L153
        .section        .debug_info,,n
        .sleb128        13
        .uleb128        15
        .sleb128        0
.L152:
.L155:
        .sleb128        11
        .4byte      .L153
.L174:
        .sleb128        12
        .4byte      .L175-.L2
        .4byte      .L87
        .sleb128        13
        .uleb128        4
        .sleb128        0
.L175:
.L7:
        .sleb128        0
.L3:

        .section        .debug_loc,,n
        .align  0
.L90:
        .d2locreg       %offsetof(.Llo1), %offsetof(.Llo2),3
        .d2locreg       %offsetof(.Llo3), %offsetof(.Llo4),31
        .d2locend
.L98:
        .d2locreg       %offsetof(.Llo3), %offsetof(.Llo4),26
        .d2locend
.L104:
        .d2locreg       %offsetof(.Llo5), %offsetof(.Llo6),3
        .d2locreg       %offsetof(.Llo7), %offsetof(.Llo8),31
        .d2locreg       %offsetof(.Llo9), %offsetof(.Llo10),3
        .d2locend
.L112:
        .d2locreg       %offsetof(.Llo11), %offsetof(.Llo12),31
        .d2locend
.L123:
        .d2locreg       %offsetof(.Llo13), %offsetof(.Llo14),3
        .d2locreg       %offsetof(.Llo15), %offsetof(.Llo16),31
        .d2locreg       %offsetof(.Llo17), %offsetof(.Llo18),3
        .d2locend
.L130:
        .d2locreg       %offsetof(.Llo15), %offsetof(.Llo16),30
        .d2locend
.L137:
        .d2locreg       %offsetof(.Llo19), %offsetof(.Llo20),3
        .d2locreg       %offsetof(.Llo21), %offsetof(.Llo22),31
        .d2locreg       %offsetof(.Llo23), %offsetof(.Llo24),3
        .d2locend
.L144:
        .d2locreg       %offsetof(.Llo21), %offsetof(.Llo22),30
        .d2locend
.L156:
        .d2locreg       %offsetof(.Llo25), %offsetof(.Llo26),3
        .d2locreg       %offsetof(.Llo27), %offsetof(.Llo28),3
        .d2locreg       %offsetof(.Llo29), %offsetof(.Llo30),4
        .d2locreg       %offsetof(.Llo31), %offsetof(.Llo32),3
        .d2locend
.L172:
        .d2locend
        .wrcm.doc
        .wrcm.elem
        .wrcm.nelem "code"
        .wrcm.nelem "functions"
        .wrcm.nelem "main"
        .wrcm.nint32 "frameSize", 64
        .wrcm.nstrlist "calls", "testFunction(double*, double)","tokenizeString()","printOddNumbers(int)","fibo(int)","fizz_buzz()"
        .wrcm.end
        .wrcm.nelem "tokenizeString()"
        .wrcm.nint32 "frameSize", 32
        .wrcm.nstrlist "calls", "printf","strtok"
        .wrcm.end
        .wrcm.nelem "printOddNumbers(int)"
        .wrcm.nint32 "frameSize", 32
        .wrcm.nstrlist "calls", "printf"
        .wrcm.end
        .wrcm.nelem "printEvenNumbers(int)"
        .wrcm.nint32 "frameSize", 32
        .wrcm.nstrlist "calls", "printf"
        .wrcm.end
        .wrcm.nelem "fizz_buzz()"
        .wrcm.nint32 "frameSize", 32
        .wrcm.nstrlist "calls", "printf"
        .wrcm.end
        .wrcm.nelem "fibo(int)"
        .wrcm.nint32 "frameSize", 32
        .wrcm.nstrlist "calls", "fibo(int)"
        .wrcm.end
        .wrcm.nelem "testFunction(double*, double)"
        .wrcm.nint32 "frameSize", 48
        .wrcm.nstrlist "calls", "_d_add","_d_fge","_d_itod"
        .wrcm.end
        .wrcm.end
        .wrcm.end
        .wrcm.nelem "tools"
        .wrcm.nelem "llopt"
        .wrcm.nelem "version"
        .wrcm.nstr "string", "5.9.7.1"
        .wrcm.nint32 "number", 5971
        .wrcm.nstr "path", "/home/ubuntu/WindRiver/compilers/diab-5.9.7.1/LINUX386/bin/llopt"
        .wrcm.nstr "label", "DIAB_5_9_7_1-FCS_20191207_153706"
        .wrcm.end
        .wrcm.end
        .wrcm.nelem "etoa"
        .wrcm.nelem "version"
        .wrcm.nstr "string", "5.9.7.1-a_1"
        .wrcm.nint32 "number", 5971
        .wrcm.nstr "path", "/home/ubuntu/WindRiver/compilers/diab-5.9.7.1/LINUX386/lib/etoa"
        .wrcm.nstr "label", "DIAB_5_9_7_1-a_1-FCS_20200605_220353"
        .wrcm.end
        .wrcm.nstr "options", "-Xmake-opt-key=/tmp/compiler-explorer-compiler202107-8023-z5iran.8cqm/output.s -Xuser-specified-proc=PPCE200Z1N -XPPCE200Z1 -Xstsw-slow -Xintrinsic-mask=0xc00041 -Xconventions-eabi -Xsoft-float -Xcoloring -Xtarget-family=2 -Xlicense-proxy-use -Xlicense-proxy-path=/home/ubuntu/WindRiver/compilers/diab-5.9.7.1/LINUX386 -Xmake-opt-key=/tmp/compiler-explorer-compiler202107-8023-z5iran.8cqm/output.s -M/home/ubuntu/WindRiver/compilers/diab-5.9.7.1/PPC/PPC.cd -Z/home/ubuntu/WindRiver/compilers/diab-5.9.7.1/LINUX386/lib/cderror.cat -Xuser-specified-proc=PPCE200Z1N -XPPCE200Z1 -Xstsw-slow -Xintrinsic-mask=0xc00041 -Xconventions-eabi -Xsoft-float -Xcoloring -Xtarget-family=2 -Xlicense-proxy-use -Xlicense-proxy-path=/home/ubuntu/WindRiver/compilers/diab-5.9.7.1/LINUX386 -g"
        .wrcm.end
        .wrcm.end
        .wrcm.nelem "file"
        .wrcm.nstr "input", "/tmp/compiler-explorer-compiler202107-8023-z5iran.8cqm/example.cpp"
        .wrcm.end
        .wrcm.end
        .wrcm.end
