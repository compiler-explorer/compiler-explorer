; https://godbolt.org/z/v6ojxxsGj
        .intel_syntax noprefix
        .file   "example.c"
        .file   1 "/app" "example.c"
        .text
        .globl  main                            # -- Begin function main
        .p2align        4
        .type   main,@function
main:                                   # @main
.Lfunc_begin0:
        .loc    1 3 0                           # example.c:3:0
        .cfi_startproc
# %bb.0:
        push    rbp
        .cfi_def_cfa_offset 16
        .cfi_offset rbp, -16
        mov     rbp, rsp
        .cfi_def_cfa_register rbp
.Ltmp0:
        .loc    1 4 5 prologue_end              # example.c:4:5
        lea     rdi, [rip + .L.str]
        call    puts@PLT
        .loc    1 5 1                           # example.c:5:1
        xor     eax, eax
        .loc    1 5 1 epilogue_begin is_stmt 0  # example.c:5:1
        pop     rbp
        .cfi_def_cfa rsp, 8
        ret
.Ltmp1:
.Lfunc_end0:
        .size   main, .Lfunc_end0-main
        .cfi_endproc
                                        # -- End function
        .type   .L.str,@object                  # @.str
        .section        .rodata.str1.1,"aMS",@progbits,1
.L.str:
        .asciz  "hello world"
        .size   .L.str, 12

        .section        .debug_abbrev,"",@progbits
        .byte   1                               # Abbreviation Code
        .byte   17                              # DW_TAG_compile_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   37                              # DW_AT_producer
        .byte   14                              # DW_FORM_strp
        .byte   19                              # DW_AT_language
        .byte   5                               # DW_FORM_data2
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   16                              # DW_AT_stmt_list
        .byte   23                              # DW_FORM_sec_offset
        .byte   27                              # DW_AT_comp_dir
        .byte   14                              # DW_FORM_strp
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   6                               # DW_FORM_data4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   2                               # Abbreviation Code
        .byte   52                              # DW_TAG_variable
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   2                               # DW_AT_location
        .byte   24                              # DW_FORM_exprloc
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   3                               # Abbreviation Code
        .byte   1                               # DW_TAG_array_type
        .byte   1                               # DW_CHILDREN_yes
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   4                               # Abbreviation Code
        .byte   33                              # DW_TAG_subrange_type
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   55                              # DW_AT_count
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   5                               # Abbreviation Code
        .byte   36                              # DW_TAG_base_type
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   62                              # DW_AT_encoding
        .byte   11                              # DW_FORM_data1
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   6                               # Abbreviation Code
        .byte   36                              # DW_TAG_base_type
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   62                              # DW_AT_encoding
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   7                               # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   0                               # DW_CHILDREN_no
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   6                               # DW_FORM_data4
        .byte   64                              # DW_AT_frame_base
        .byte   24                              # DW_FORM_exprloc
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   39                              # DW_AT_prototyped
        .byte   25                              # DW_FORM_flag_present
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   0                               # EOM(3)
        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  4                               # DWARF version number
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .byte   8                               # Address Size (in bytes)
        .byte   1                               # Abbrev [1] 0xb:0x6b DW_TAG_compile_unit
        .long   .Linfo_string0                  # DW_AT_producer
        .short  29                              # DW_AT_language
        .long   .Linfo_string1                  # DW_AT_name
        .long   .Lline_table_start0             # DW_AT_stmt_list
        .long   .Linfo_string2                  # DW_AT_comp_dir
        .quad   .Lfunc_begin0                   # DW_AT_low_pc
        .long   .Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
        .byte   2                               # Abbrev [2] 0x2a:0x11 DW_TAG_variable
        .long   59                              # DW_AT_type
        .byte   1                               # DW_AT_decl_file
        .byte   4                               # DW_AT_decl_line
        .byte   9                               # DW_AT_location
        .byte   3
        .quad   .L.str
        .byte   3                               # Abbrev [3] 0x3b:0xc DW_TAG_array_type
        .long   71                              # DW_AT_type
        .byte   4                               # Abbrev [4] 0x40:0x6 DW_TAG_subrange_type
        .long   78                              # DW_AT_type
        .byte   12                              # DW_AT_count
        .byte   0                               # End Of Children Mark
        .byte   5                               # Abbrev [5] 0x47:0x7 DW_TAG_base_type
        .long   .Linfo_string3                  # DW_AT_name
        .byte   6                               # DW_AT_encoding
        .byte   1                               # DW_AT_byte_size
        .byte   6                               # Abbrev [6] 0x4e:0x7 DW_TAG_base_type
        .long   .Linfo_string4                  # DW_AT_name
        .byte   8                               # DW_AT_byte_size
        .byte   7                               # DW_AT_encoding
        .byte   7                               # Abbrev [7] 0x55:0x19 DW_TAG_subprogram
        .quad   .Lfunc_begin0                   # DW_AT_low_pc
        .long   .Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
        .byte   1                               # DW_AT_frame_base
        .byte   86
        .long   .Linfo_string5                  # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .byte   3                               # DW_AT_decl_line
                                        # DW_AT_prototyped
        .long   110                             # DW_AT_type
                                        # DW_AT_external
        .byte   5                               # Abbrev [5] 0x6e:0x7 DW_TAG_base_type
        .long   .Linfo_string6                  # DW_AT_name
        .byte   5                               # DW_AT_encoding
        .byte   4                               # DW_AT_byte_size
        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:
        .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "clang version 20.1.0 (https://github.com/llvm/llvm-project.git 24a30daaa559829ad079f2ff7f73eb4e18095f88)" # string offset=0
.Linfo_string1:
        .asciz  "/app/example.c"                # string offset=105
.Linfo_string2:
        .asciz  "/app"                          # string offset=120
.Linfo_string3:
        .asciz  "char"                          # string offset=125
.Linfo_string4:
        .asciz  "__ARRAY_SIZE_TYPE__"           # string offset=130
.Linfo_string5:
        .asciz  "main"                          # string offset=150
.Linfo_string6:
        .asciz  "int"                           # string offset=155
        .ident  "clang version 20.1.0 (https://github.com/llvm/llvm-project.git 24a30daaa559829ad079f2ff7f73eb4e18095f88)"
        .section        ".note.GNU-stack","",@progbits
        .addrsig
        .addrsig_sym puts
        .section        .debug_line,"",@progbits
.Lline_table_start0:
