.text
        .file   "/tmp/compiler-explorer-compiler1151011-11958-1r6gk9o/example.cpp"
        .section        .debug_info,"",@progbits
.Lsection_info:
        .section        .debug_abbrev,"",@progbits
.Lsection_abbrev:
        .section        .debug_line,"",@progbits
.Lsection_line:
        .section        .debug_pubnames,"",@progbits
        .section        .debug_pubtypes,"",@progbits
        .section        .debug_str,"MS",@progbits,1
.Linfo_string:
        .section        .debug_loc,"",@progbits
.Lsection_debug_loc:
        .section        .debug_ranges,"",@progbits
.Ldebug_range:
        .text
        .globl  maxArray(double*, double*)
        .align  16, 0x90
        .type   maxArray(double*, double*),@function
maxArray(double*, double*):                        # @maxArray(double*, double*)
.Lfunc_begin0:
        .file   1 "/tmp/compiler-explorer-compiler1151011-11958-1r6gk9o" "example.cpp"
        .loc    1 1 0                   # /tmp/compiler-explorer-compiler1151011-11958-1r6gk9o/example.cpp:1:0
        .cfi_startproc
# BB#0:
        #DEBUG_VALUE: maxArray:x <- RDI
        #DEBUG_VALUE: maxArray:y <- RSI
        xor     eax, eax
.Ltmp0:
        #DEBUG_VALUE: i <- 0
        .align  16, 0x90
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
        #DEBUG_VALUE: maxArray:x <- RDI
        #DEBUG_VALUE: maxArray:y <- RSI
        #DEBUG_VALUE: i <- 0
        .loc    1 3 13 prologue_end     # /tmp/compiler-explorer-compiler1151011-11958-1r6gk9o/example.cpp:3:13
        movsd   xmm0, qword ptr [rsi + 8*rax]
        ucomisd xmm0, qword ptr [rdi + 8*rax]
        jbe     .LBB0_3
# BB#2:                                 #   in Loop: Header=BB0_1 Depth=1
        #DEBUG_VALUE: maxArray:x <- RDI
        #DEBUG_VALUE: maxArray:y <- RSI
        #DEBUG_VALUE: i <- 0
        .loc    1 3 26 discriminator 1  # /tmp/compiler-explorer-compiler1151011-11958-1r6gk9o/example.cpp:3:26
.Ltmp1:
        movsd   qword ptr [rdi + 8*rax], xmm0
.Ltmp2:
.LBB0_3:                                #   in Loop: Header=BB0_1 Depth=1
        #DEBUG_VALUE: maxArray:x <- RDI
        #DEBUG_VALUE: maxArray:y <- RSI
        #DEBUG_VALUE: i <- 0
        .loc    1 2 10 discriminator 2  # /tmp/compiler-explorer-compiler1151011-11958-1r6gk9o/example.cpp:2:10
        lea     rcx, qword ptr [rax + 1]
.Ltmp3:
        .loc    1 3 13                  # /tmp/compiler-explorer-compiler1151011-11958-1r6gk9o/example.cpp:3:13
        movsd   xmm0, qword ptr [rsi + 8*rax + 8]
        ucomisd xmm0, qword ptr [rdi + 8*rax + 8]
        jbe     .LBB0_5
# BB#4:                                 #   in Loop: Header=BB0_1 Depth=1
        #DEBUG_VALUE: maxArray:x <- RDI
        #DEBUG_VALUE: maxArray:y <- RSI
        #DEBUG_VALUE: i <- 0
        .loc    1 3 26 discriminator 1  # /tmp/compiler-explorer-compiler1151011-11958-1r6gk9o/example.cpp:3:26
.Ltmp4:
        movsd   qword ptr [rdi + 8*rax + 8], xmm0
.Ltmp5:
.LBB0_5:                                #   in Loop: Header=BB0_1 Depth=1
        #DEBUG_VALUE: maxArray:x <- RDI
        #DEBUG_VALUE: maxArray:y <- RSI
        #DEBUG_VALUE: i <- 0
        .loc    1 2 10 discriminator 2  # /tmp/compiler-explorer-compiler1151011-11958-1r6gk9o/example.cpp:2:10
        inc     rcx
        cmp     rcx, 65536
        mov     rax, rcx
        jne     .LBB0_1
.Ltmp6:
# BB#6:
        .loc    1 5 1                   # /tmp/compiler-explorer-compiler1151011-11958-1r6gk9o/example.cpp:5:1
        ret
.Ltmp7:
.Ltmp8:
        .size   maxArray(double*, double*), .Ltmp8-maxArray(double*, double*)
.Lfunc_end0:
        .cfi_endproc
