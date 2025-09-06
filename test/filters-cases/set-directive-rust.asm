; https://rust.godbolt.org/z/nc9nd35W3
        .intel_syntax noprefix
        .file   "example.79c311dc4783170-cgu.0"
        .section        .text._ZN7example14first_function17h65b92e135846244bE,"ax",@progbits
        .globl  _ZN7example14first_function17h65b92e135846244bE
        .p2align        4
        .type   _ZN7example14first_function17h65b92e135846244bE,@function
_ZN7example14first_function17h65b92e135846244bE:
.Lfunc_begin0:
        .cfi_startproc
        .file   1 "/app" "example.rs"
        .loc    1 3 5 prologue_end
        lea     rax, [rdi + 1]
        .loc    1 4 2
        ret
.Lfunc_end0:
        .size   _ZN7example14first_function17h65b92e135846244bE, .Lfunc_end0-_ZN7example14first_function17h65b92e135846244bE
        .cfi_endproc

        .globl  _ZN7example15second_function17h4a2f85ffbcd4ec12E
        .type   _ZN7example15second_function17h4a2f85ffbcd4ec12E,@function
.set _ZN7example15second_function17h4a2f85ffbcd4ec12E, _ZN7example14first_function17h65b92e135846244bE
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
        .quad   .Lfunc_begin0
        .long   .Lfunc_end0-.Lfunc_begin0
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
        .byte   2
        .long   91

        .byte   4
        .byte   1
        .byte   85
        .long   .Linfo_string7
        .byte   1
        .byte   2
        .long   91
        .byte   0
        .byte   0
        .byte   5
        .long   .Linfo_string6
        .byte   7
        .byte   8
        .byte   0
.Ldebug_info_end0:
        .section        .text._ZN7example14first_function17h65b92e135846244bE,"ax",@progbits
.Lsec_end0:
        .section        .debug_aranges,"",@progbits
        .long   44
        .short  2
        .long   .Lcu_begin0
        .byte   8
        .byte   0
        .zero   4,255
        .quad   .Lfunc_begin0
        .quad   .Lsec_end0-.Lfunc_begin0
        .quad   0
        .quad   0
        .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "clang LLVM (rustc version 1.89.0 (29483883e 2025-08-04))"
.Linfo_string1:
        .asciz  "/app/example.rs/@/example.79c311dc4783170-cgu.0"
.Linfo_string2:
        .asciz  "/app"
.Linfo_string3:
        .asciz  "example"
.Linfo_string4:
        .asciz  "_ZN7example14first_function17h65b92e135846244bE"
.Linfo_string5:
        .asciz  "first_function"
.Linfo_string6:
        .asciz  "u64"
.Linfo_string7:
        .asciz  "n"
        .ident  "rustc version 1.89.0 (29483883e 2025-08-04)"
        .section        ".note.GNU-stack","",@progbits
        .section        .debug_line,"",@progbits
.Lline_table_start0:
