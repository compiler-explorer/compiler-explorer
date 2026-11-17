; https://godbolt.org/z/hsMaPocc8
        .intel_syntax noprefix
        .file   "example.32c4274de85e6574-cgu.0"
        .section        .text._RNvCs4me6dhhIFVM_7example7add_i32,"ax",@progbits
        .globl  _RNvCs4me6dhhIFVM_7example7add_i32
        .prefalign      4, .Lfunc_end0, nop
        .type   _RNvCs4me6dhhIFVM_7example7add_i32,@function
_RNvCs4me6dhhIFVM_7example7add_i32:
.Lfunc_begin0:
        .cfi_startproc
        .file   1 "/app" "example.rs"
        .loc    1 13 5 prologue_end
        lea     eax, [rdi + rsi]
        .loc    1 14 2
        ret
.Ltmp0:
.Lfunc_end0:
        .size   _RNvCs4me6dhhIFVM_7example7add_i32, .Lfunc_end0-_RNvCs4me6dhhIFVM_7example7add_i32
        .cfi_endproc

        .section        .text._RNvCs4me6dhhIFVM_7example8call_i32,"ax",@progbits
        .globl  _RNvCs4me6dhhIFVM_7example8call_i32
        .prefalign      4, .Lfunc_end1, nop
        .type   _RNvCs4me6dhhIFVM_7example8call_i32,@function
_RNvCs4me6dhhIFVM_7example8call_i32:
.Lfunc_begin1:
        .cfi_startproc
        .loc    1 18 13 prologue_end
        mov     eax, dword ptr [rdi + 4]
        .loc    1 18 20 is_stmt 0
        mov     esi, dword ptr [rdi + 8]
        .loc    1 18 5
        mov     edi, eax
.Ltmp1:
        jmp     qword ptr [rip + _RNvCs4me6dhhIFVM_7example7add_i32@GOTPCREL]
.Ltmp2:
.Lfunc_end1:
        .size   _RNvCs4me6dhhIFVM_7example8call_i32, .Lfunc_end1-_RNvCs4me6dhhIFVM_7example8call_i32
        .cfi_endproc

        .section        .text._RNvCs4me6dhhIFVM_7example8call_u32,"ax",@progbits
        .globl  _RNvCs4me6dhhIFVM_7example8call_u32
        .prefalign      4, .Lfunc_end2, nop
        .type   _RNvCs4me6dhhIFVM_7example8call_u32,@function
_RNvCs4me6dhhIFVM_7example8call_u32:
.Lfunc_begin2:
        .cfi_startproc
        .loc    1 3 13 prologue_end is_stmt 1
        mov     eax, dword ptr [rdi]
        .loc    1 3 20 is_stmt 0
        mov     esi, dword ptr [rdi + 4]
        .loc    1 3 5
        mov     edi, eax
.Ltmp3:
        jmp     qword ptr [rip + _RNvCs4me6dhhIFVM_7example7add_i32@GOTPCREL]
.Ltmp4:
.Lfunc_end2:
        .size   _RNvCs4me6dhhIFVM_7example8call_u32, .Lfunc_end2-_RNvCs4me6dhhIFVM_7example8call_u32
        .cfi_endproc

        .globl  _RNvCs4me6dhhIFVM_7example7add_u32
        .type   _RNvCs4me6dhhIFVM_7example7add_u32,@function
_RNvCs4me6dhhIFVM_7example7add_u32 = _RNvCs4me6dhhIFVM_7example7add_i32
        .section        .debug_loc,"",@progbits
.Ldebug_loc0:
        .quad   -1
        .quad   .Lfunc_begin1
        .quad   .Lfunc_begin1-.Lfunc_begin1
        .quad   .Ltmp1-.Lfunc_begin1
        .short  1
        .byte   85
        .quad   .Ltmp1-.Lfunc_begin1
        .quad   .Lfunc_end1-.Lfunc_begin1
        .short  4
        .byte   243
        .byte   1
        .byte   85
        .byte   159
        .quad   0
        .quad   0
.Ldebug_loc1:
        .quad   -1
        .quad   .Lfunc_begin2
        .quad   .Lfunc_begin2-.Lfunc_begin2
        .quad   .Ltmp3-.Lfunc_begin2
        .short  1
        .byte   85
        .quad   .Ltmp3-.Lfunc_begin2
        .quad   .Lfunc_end2-.Lfunc_begin2
        .short  4
        .byte   243
        .byte   1
        .byte   85
        .byte   159
        .quad   0
        .quad   0
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
        .byte   46
        .byte   1
        .byte   17
        .byte   1
        .byte   18
        .byte   6
        .byte   64
        .byte   24
        .byte   110
        .byte   14
        .byte   3
        .byte   14
        .byte   58
        .byte   11
        .byte   59
        .byte   11
        .byte   73
        .byte   19
        .byte   63
        .byte   25
        .byte   0
        .byte   0
        .byte   4
        .byte   5
        .byte   0
        .byte   2
        .byte   24
        .byte   3
        .byte   14
        .byte   58
        .byte   11
        .byte   59
        .byte   11
        .byte   73
        .byte   19
        .byte   0
        .byte   0
        .byte   5
        .byte   5
        .byte   0
        .byte   2
        .byte   23
        .byte   3
        .byte   14
        .byte   58
        .byte   11
        .byte   59
        .byte   11
        .byte   73
        .byte   19
        .byte   0
        .byte   0
        .byte   6
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
        .byte   7
        .byte   15
        .byte   0
        .byte   73
        .byte   19
        .byte   3
        .byte   14
        .byte   51
        .byte   6
        .byte   0
        .byte   0
        .byte   8
        .byte   1
        .byte   1
        .byte   73
        .byte   19
        .byte   0
        .byte   0
        .byte   9
        .byte   33
        .byte   0
        .byte   73
        .byte   19
        .byte   34
        .byte   13
        .byte   55
        .byte   11
        .byte   0
        .byte   0
        .byte   10
        .byte   36
        .byte   0
        .byte   3
        .byte   14
        .byte   11
        .byte   11
        .byte   62
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
        .byte   3
        .quad   .Lfunc_begin0
        .long   .Lfunc_end0-.Lfunc_begin0
        .byte   1
        .byte   87
        .long   .Linfo_string4
        .long   .Linfo_string5
        .byte   1
        .byte   12
        .long   194

        .byte   4
        .byte   1
        .byte   85
        .long   .Linfo_string12
        .byte   1
        .byte   12
        .long   194
        .byte   4
        .byte   1
        .byte   84
        .long   .Linfo_string13
        .byte   1
        .byte   12
        .long   194
        .byte   0
        .byte   3
        .quad   .Lfunc_begin1
        .long   .Lfunc_end1-.Lfunc_begin1
        .byte   1
        .byte   87
        .long   .Linfo_string7
        .long   .Linfo_string8
        .byte   1
        .byte   17
        .long   194

        .byte   5
        .long   .Ldebug_loc0
        .long   .Linfo_string14
        .byte   1
        .byte   17
        .long   208
        .byte   0
        .byte   3
        .quad   .Lfunc_begin2
        .long   .Lfunc_end2-.Lfunc_begin2
        .byte   1
        .byte   87
        .long   .Linfo_string9
        .long   .Linfo_string10
        .byte   1
        .byte   2
        .long   201

        .byte   5
        .long   .Ldebug_loc1
        .long   .Linfo_string14
        .byte   1
        .byte   2
        .long   241
        .byte   0
        .byte   0
        .byte   6
        .long   .Linfo_string6
        .byte   5
        .byte   4
        .byte   6
        .long   .Linfo_string11
        .byte   7
        .byte   4
        .byte   7
        .long   221
        .long   .Linfo_string16
        .long   0
        .byte   8
        .long   194
        .byte   9
        .long   234
        .byte   0
        .byte   3
        .byte   0
        .byte   10
        .long   .Linfo_string15
        .byte   8
        .byte   7
        .byte   7
        .long   254
        .long   .Linfo_string17
        .long   0
        .byte   8
        .long   201
        .byte   9
        .long   234
        .byte   0
        .byte   2
        .byte   0
        .byte   0
.Ldebug_info_end0:
        .section        .text._RNvCs4me6dhhIFVM_7example7add_i32,"ax",@progbits
.Lsec_end0:
        .section        .text._RNvCs4me6dhhIFVM_7example8call_i32,"ax",@progbits
.Lsec_end1:
        .section        .text._RNvCs4me6dhhIFVM_7example8call_u32,"ax",@progbits
.Lsec_end2:
        .section        .debug_aranges,"",@progbits
        .long   76
        .short  2
        .long   .Lcu_begin0
        .byte   8
        .byte   0
        .zero   4,255
        .quad   .Lfunc_begin0
        .quad   .Lsec_end0-.Lfunc_begin0
        .quad   .Lfunc_begin1
        .quad   .Lsec_end1-.Lfunc_begin1
        .quad   .Lfunc_begin2
        .quad   .Lsec_end2-.Lfunc_begin2
        .quad   0
        .quad   0
        .section        .debug_ranges,"",@progbits
.Ldebug_ranges0:
        .quad   .Lfunc_begin0
        .quad   .Lfunc_end0
        .quad   .Lfunc_begin1
        .quad   .Lfunc_end1
        .quad   .Lfunc_begin2
        .quad   .Lfunc_end2
        .quad   0
        .quad   0
        .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "clang LLVM (rustc version 1.99.0 (b940084d7 2026-09-28))"
.Linfo_string1:
        .asciz  "/app/example.rs/@/example.32c4274de85e6574-cgu.0"
.Linfo_string2:
        .asciz  "/app"
.Linfo_string3:
        .asciz  "example"
.Linfo_string4:
        .asciz  "_RNvCs4me6dhhIFVM_7example7add_i32"
.Linfo_string5:
        .asciz  "add_i32"
.Linfo_string6:
        .asciz  "i32"
.Linfo_string7:
        .asciz  "_RNvCs4me6dhhIFVM_7example8call_i32"
.Linfo_string8:
        .asciz  "call_i32"
.Linfo_string9:
        .asciz  "_RNvCs4me6dhhIFVM_7example8call_u32"
.Linfo_string10:
        .asciz  "call_u32"
.Linfo_string11:
        .asciz  "u32"
.Linfo_string12:
        .asciz  "x"
.Linfo_string13:
        .asciz  "y"
.Linfo_string14:
        .asciz  "xs"
.Linfo_string15:
        .asciz  "__ARRAY_SIZE_TYPE__"
.Linfo_string16:
        .asciz  "&[i32; 3]"
.Linfo_string17:
        .asciz  "&[u32; 2]"
        .ident  "rustc version 1.99.0 (b940084d7 2026-09-28)"
        .section        ".note.GNU-stack","",@progbits
        .section        .debug_line,"",@progbits
.Lline_table_start0:
