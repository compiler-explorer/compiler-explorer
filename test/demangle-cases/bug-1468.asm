        .text
        .intel_syntax noprefix
        .file   "example.3a1fbbbh-cgu.0"
        .section        .text._ZN4core3fmt10ArgumentV13new17h58834fbfa8735787E,"ax",@progbits
        .globl  _ZN4core3fmt10ArgumentV13new17h58834fbfa8735787E
        .p2align        4, 0x90
        .type   _ZN4core3fmt10ArgumentV13new17h58834fbfa8735787E,@function
_ZN4core3fmt10ArgumentV13new17h58834fbfa8735787E:
.Lfunc_begin0:
        .file   1 "/rustc/3c235d5600393dfe6c36eeed34042efad8d4f26e/src/libcore/fmt/mod.rs"
        .loc    1 278 0
        .cfi_startproc
        sub     rsp, 56
        .cfi_def_cfa_offset 64
.Ltmp0:
        .loc    1 282 27 prologue_end
        mov     qword ptr [rsp + 40], rsi
        mov     rsi, qword ptr [rsp + 40]
        mov     qword ptr [rsp + 16], rdi
        mov     qword ptr [rsp + 8], rsi
        .loc    1 0 27 is_stmt 0
        mov     rax, qword ptr [rsp + 16]
        .loc    1 283 23 is_stmt 1
        mov     qword ptr [rsp + 48], rax
        mov     rcx, qword ptr [rsp + 48]
        mov     qword ptr [rsp], rcx
        .loc    1 0 23 is_stmt 0
        mov     rax, qword ptr [rsp]
        .loc    1 281 12 is_stmt 1
        mov     qword ptr [rsp + 24], rax
        mov     rcx, qword ptr [rsp + 8]
        mov     qword ptr [rsp + 32], rcx
.Ltmp1:
        .loc    1 286 5
        mov     rax, qword ptr [rsp + 24]
        mov     rdx, qword ptr [rsp + 32]
        add     rsp, 56
        .cfi_def_cfa_offset 8
        ret
.Ltmp2:
.Lfunc_end0:
        .size   _ZN4core3fmt10ArgumentV13new17h58834fbfa8735787E, .Lfunc_end0-_ZN4core3fmt10ArgumentV13new17h58834fbfa8735787E
        .cfi_endproc

        .section        .text._ZN4core3fmt9Arguments6new_v117h225d1328f6c7a9acE,"ax",@progbits
        .p2align        4, 0x90
        .type   _ZN4core3fmt9Arguments6new_v117h225d1328f6c7a9acE,@function
_ZN4core3fmt9Arguments6new_v117h225d1328f6c7a9acE:
.Lfunc_begin1:
        .loc    1 314 0
        .cfi_startproc
        sub     rsp, 16
        .cfi_def_cfa_offset 24
        mov     rax, rdi
.Ltmp3:
        .loc    1 318 17 prologue_end
        mov     qword ptr [rsp], 0
        .loc    1 316 8
        mov     qword ptr [rdi], rsi
        mov     qword ptr [rdi + 8], rdx
        mov     rdx, qword ptr [rsp]
        mov     rsi, qword ptr [rsp + 8]
        mov     qword ptr [rdi + 16], rdx
        mov     qword ptr [rdi + 24], rsi
        mov     qword ptr [rdi + 32], rcx
        mov     qword ptr [rdi + 40], r8
        .loc    1 321 5
        add     rsp, 16
        .cfi_def_cfa_offset 8
        ret
.Ltmp4:
.Lfunc_end1:
        .size   _ZN4core3fmt9Arguments6new_v117h225d1328f6c7a9acE, .Lfunc_end1-_ZN4core3fmt9Arguments6new_v117h225d1328f6c7a9acE
        .cfi_endproc

        .section        .text._ZN7example6square17h888700303ddfdc38E,"ax",@progbits
        .globl  _ZN7example6square17h888700303ddfdc38E
        .p2align        4, 0x90
        .type   _ZN7example6square17h888700303ddfdc38E,@function
_ZN7example6square17h888700303ddfdc38E:
.Lfunc_begin2:
        .file   2 "/home/ubuntu/./example.rs"
        .loc    2 1 0
        .cfi_startproc
        push    rax
        .cfi_def_cfa_offset 16
.Ltmp5:
        .loc    2 2 4 prologue_end
        imul    edi, edi
        seto    al
        test    al, 1
        mov     dword ptr [rsp + 4], edi
        jne     .LBB2_2
        .loc    2 0 4 is_stmt 0
        mov     eax, dword ptr [rsp + 4]
        .loc    2 3 1 is_stmt 1
        pop     rcx
        .cfi_def_cfa_offset 8
        ret
.LBB2_2:
        .cfi_def_cfa_offset 16
        .loc    2 2 4
        lea     rdi, [rip + .L__unnamed_1]
        mov     rax, qword ptr [rip + _ZN4core9panicking5panic17h5137ce59069236b2E@GOTPCREL]
        call    rax
        ud2
.Ltmp6:
.Lfunc_end2:
        .size   _ZN7example6square17h888700303ddfdc38E, .Lfunc_end2-_ZN7example6square17h888700303ddfdc38E
        .cfi_endproc

        .section        .text._ZN7example4main17he964733f9437e39dE,"ax",@progbits
        .globl  _ZN7example4main17he964733f9437e39dE
        .p2align        4, 0x90
        .type   _ZN7example4main17he964733f9437e39dE,@function
_ZN7example4main17he964733f9437e39dE:
.Lfunc_begin3:
        .loc    2 5 0
        .cfi_startproc
        sub     rsp, 104
        .cfi_def_cfa_offset 112
.Ltmp7:
        .loc    2 6 19 prologue_end
        mov     edi, 2
        call    qword ptr [rip + _ZN7example6square17h888700303ddfdc38E@GOTPCREL]
        mov     dword ptr [rsp + 100], eax
        .loc    2 0 19 is_stmt 0
        mov     rsi, qword ptr [rip + _ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$i32$GT$3fmt17h0626a48fe3cb5792E@GOTPCREL]
        .loc    2 6 4
        lea     rax, [rsp + 100]
        mov     qword ptr [rsp + 88], rax
        mov     rdi, qword ptr [rsp + 88]
.Ltmp8:
        .loc    2 6 4
        call    qword ptr [rip + _ZN4core3fmt10ArgumentV13new17h58834fbfa8735787E@GOTPCREL]
        mov     qword ptr [rsp + 16], rax
        mov     qword ptr [rsp + 8], rdx
        .loc    2 0 4
        lea     rax, [rip + .L__unnamed_2]
        mov     rcx, qword ptr [rsp + 16]
        .loc    2 6 4
        mov     qword ptr [rsp + 72], rcx
        mov     rdx, qword ptr [rsp + 8]
        mov     qword ptr [rsp + 80], rdx
.Ltmp9:
        .loc    2 6 4
        lea     rsi, [rsp + 72]
        lea     rdi, [rsp + 24]
        mov     qword ptr [rsp], rsi
        mov     rsi, rax
        mov     edx, 2
        mov     rcx, qword ptr [rsp]
        mov     r8d, 1
        call    _ZN4core3fmt9Arguments6new_v117h225d1328f6c7a9acE
        lea     rdi, [rsp + 24]
        call    qword ptr [rip + _ZN3std2io5stdio6_print17ha4c0b9f4da5c9e13E@GOTPCREL]
        .loc    2 7 1 is_stmt 1
        add     rsp, 104
        .cfi_def_cfa_offset 8
        ret
.Ltmp10:
.Lfunc_end3:
        .size   _ZN7example4main17he964733f9437e39dE, .Lfunc_end3-_ZN7example4main17he964733f9437e39dE
        .cfi_endproc

        .type   str.0,@object
        .section        .rodata.str.0,"a",@progbits
str.0:
        .ascii  "./example.rs"
        .size   str.0, 12

        .type   str.1,@object
        .section        .rodata.str.1,"a",@progbits
        .p2align        4
str.1:
        .ascii  "attempt to multiply with overflow"
        .size   str.1, 33

        .type   .L__unnamed_1,@object
        .section        .data.rel.ro..L__unnamed_1,"aw",@progbits
        .p2align        3
.L__unnamed_1:
        .quad   str.1
        .quad   33
        .quad   str.0
        .quad   12
        .long   2
        .long   5
        .size   .L__unnamed_1, 40

        .type   .L__unnamed_3,@object
        .section        .rodata..L__unnamed_3,"a",@progbits
.L__unnamed_3:
        .size   .L__unnamed_3, 0

        .type   .L__unnamed_4,@object
        .section        .rodata..L__unnamed_4,"a",@progbits
.L__unnamed_4:
        .byte   10
        .size   .L__unnamed_4, 1

        .type   .L__unnamed_2,@object
        .section        .data.rel.ro..L__unnamed_2,"aw",@progbits
        .p2align        3
.L__unnamed_2:
        .quad   .L__unnamed_3
        .zero   8
        .quad   .L__unnamed_4
        .asciz  "\001\000\000\000\000\000\000"
        .size   .L__unnamed_2, 32

        .type   __rustc_debug_gdb_scripts_section__,@object
        .section        .debug_gdb_scripts,"aMS",@progbits,1
        .weak   __rustc_debug_gdb_scripts_section__
__rustc_debug_gdb_scripts_section__:
        .asciz  "\001gdb_load_rust_pretty_printers.py"
        .size   __rustc_debug_gdb_scripts_section__, 34

        .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "clang LLVM (rustc version 1.35.0 (3c235d560 2019-05-20))"
.Linfo_string1:
        .asciz  "./example.rs"
.Linfo_string2:
        .asciz  "/home/ubuntu"
.Linfo_string3:
        .asciz  "core"
.Linfo_string4:
        .asciz  "result"
.Linfo_string5:
        .asciz  "u8"
.Linfo_string6:
        .asciz  "Ok"
.Linfo_string7:
        .asciz  "Err"
.Linfo_string8:
        .asciz  "Result"
.Linfo_string9:
        .asciz  "fmt"
.Linfo_string10:
        .asciz  "rt"
.Linfo_string11:
        .asciz  "v1"
.Linfo_string12:
        .asciz  "Left"
.Linfo_string13:
        .asciz  "Right"
.Linfo_string14:
        .asciz  "Center"
.Linfo_string15:
        .asciz  "Unknown"
.Linfo_string16:
        .asciz  "Alignment"
        .section        .debug_abbrev,"",@progbits
        .byte   1
        .byte   17
        .byte   1
        .byte   37
        .byte   14
        .byte   19
        .byte   5
        .byte   3
        .byte   14
        .byte   16
        .byte   23
        .byte   27
        .byte   14
        .byte   17
        .byte   1
        .byte   85
        .byte   23
        .byte   0
        .byte   0
        .byte   2
        .byte   57
        .byte   1
        .byte   3
        .byte   14
        .byte   0
        .byte   0
        .byte   3
        .byte   4
        .byte   1
        .byte   73
        .byte   19
        .byte   109
        .byte   25
        .byte   3
        .byte   14
        .byte   11
        .byte   11
        .ascii  "\210\001"
        .byte   15
        .byte   0
        .byte   0
        .byte   4
        .byte   40
        .byte   0
        .byte   3
        .byte   14
        .byte   28
        .byte   15
        .byte   0
        .byte   0
        .byte   5
        .byte   36
        .byte   0
        .byte   3
        .byte   14
        .byte   62
        .byte   11
        .byte   11
        .byte   11
        .byte   0
        .byte   0
        .byte   0
        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0
.Ldebug_info_start0:
        .short  4
        .long   .debug_abbrev
        .byte   8
        .byte   1
        .long   .Linfo_string0
        .short  28
        .long   .Linfo_string1
        .long   .Lline_table_start0
        .long   .Linfo_string2
        .quad   0
        .long   .Ldebug_ranges0
        .byte   2
        .long   .Linfo_string3
        .byte   2
        .long   .Linfo_string4
        .byte   3
        .long   132

        .long   .Linfo_string8
        .byte   1
        .byte   1
        .byte   4
        .long   .Linfo_string6
        .byte   0
        .byte   4
        .long   .Linfo_string7
        .byte   1
        .byte   0
        .byte   0
        .byte   2
        .long   .Linfo_string9
        .byte   2
        .long   .Linfo_string10
        .byte   2
        .long   .Linfo_string11
        .byte   3
        .long   132

        .long   .Linfo_string16
        .byte   1
        .byte   1
        .byte   4
        .long   .Linfo_string12
        .byte   0
        .byte   4
        .long   .Linfo_string13
        .byte   1
        .byte   4
        .long   .Linfo_string14
        .byte   2
        .byte   4
        .long   .Linfo_string15
        .byte   3
        .byte   0
        .byte   0
        .byte   0
        .byte   0
        .byte   0
        .byte   5
        .long   .Linfo_string5
        .byte   7
        .byte   1
        .byte   0
.Ldebug_info_end0:
        .section        .debug_ranges,"",@progbits
.Ldebug_ranges0:
        .quad   .Lfunc_begin0
        .quad   .Lfunc_end0
        .quad   .Lfunc_begin1
        .quad   .Lfunc_end1
        .quad   .Lfunc_begin2
        .quad   .Lfunc_end2
        .quad   .Lfunc_begin3
        .quad   .Lfunc_end3
        .quad   0
        .quad   0
        .section        .debug_macinfo,"",@progbits
        .byte   0

        .section        ".note.GNU-stack","",@progbits
        .section        .debug_line,"",@progbits
.Lline_table_start0: