        .text
        .syntax unified
        .eabi_attribute 67, "2.09"
        .eabi_attribute 6, 12
        .eabi_attribute 7, 77
        .eabi_attribute 8, 0
        .eabi_attribute 9, 1
        .eabi_attribute 34, 0
        .eabi_attribute 17, 1
        .eabi_attribute 20, 1
        .eabi_attribute 21, 0
        .eabi_attribute 23, 3
        .eabi_attribute 24, 1
        .eabi_attribute 25, 1
        .eabi_attribute 38, 1
        .eabi_attribute 14, 0
        .file   "example.3a1fbbbh-cgu.0"
        .section        .text.example::abort,"ax",%progbits
        .globl  example::abort
        .p2align        1
        .type   example::abort,%function
        .code   16
        .thumb_func
example::abort:
.Lfunc_begin0:
        .file   1 "/home/ce/./example.rs"
        .loc    1 4 0
        .fnstart
        .cfi_sections .debug_frame
        .cfi_startproc
        .loc    1 5 5 prologue_end
        .inst.n 0xdefe
        .inst.n 0xdefe
.Lfunc_end0:
        .size   example::abort, .Lfunc_end0-example::abort
        .cfi_endproc
        .cantunwind
        .fnend

        .section        .debug_abbrev,"",%progbits
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
        .ascii  "\264B"
        .byte   25
        .byte   17
        .byte   1
        .byte   18
        .byte   6
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
        .byte   0
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
        .byte   63
        .byte   25
        .ascii  "\207\001"
        .byte   25
        .byte   0
        .byte   0
        .byte   0
        .section        .debug_info,"",%progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0
.Ldebug_info_start0:
        .short  4
        .long   .debug_abbrev
        .byte   4
        .byte   1
        .long   .Linfo_string0
        .short  28
        .long   .Linfo_string1
        .long   .Lline_table_start0
        .long   .Linfo_string2

        .long   .Lfunc_begin0
        .long   .Lfunc_end0-.Lfunc_begin0
        .byte   2
        .long   .Linfo_string3
        .byte   3
        .long   .Lfunc_begin0
        .long   .Lfunc_end0-.Lfunc_begin0
        .byte   1
        .byte   87
        .long   .Linfo_string4
        .long   .Linfo_string5
        .byte   1
        .byte   4

        .byte   0
        .byte   0
.Ldebug_info_end0:
        .section        .text.example::abort,"ax",%progbits
.Lsec_end0:
        .section        .debug_aranges,"",%progbits
        .long   28
        .short  2
        .long   .Lcu_begin0
        .byte   4
        .byte   0
        .zero   4,255
        .long   .Lfunc_begin0
        .long   .Lsec_end0-.Lfunc_begin0
        .long   0
        .long   0
        .section        .debug_str,"MS",%progbits,1
.Linfo_string0:
        .asciz  "clang LLVM (rustc version 1.48.0-nightly (c59199efc 2020-09-04))"
.Linfo_string1:
        .asciz  "./example.rs"
.Linfo_string2:
        .asciz  "/home/ce"
.Linfo_string3:
        .asciz  "example"
.Linfo_string4:
        .asciz  "example::abort"
.Linfo_string5:
        .asciz  "abort"
        .section        .debug_pubnames,"",%progbits
        .long   .LpubNames_end0-.LpubNames_begin0
.LpubNames_begin0:
        .short  2
        .long   .Lcu_begin0
        .long   66
        .long   43
        .asciz  "abort"
        .long   38
        .asciz  "example"
        .long   0
.LpubNames_end0:
        .section        .debug_pubtypes,"",%progbits
        .long   .LpubTypes_end0-.LpubTypes_begin0
.LpubTypes_begin0:
        .short  2
        .long   .Lcu_begin0
        .long   66
        .long   0
.LpubTypes_end0:
        .section        ".note.GNU-stack","",%progbits
        .eabi_attribute 30, 5
        .section        .debug_line,"",%progbits
.Lline_table_start0: