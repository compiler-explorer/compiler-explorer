        .file   "-"
        .file   1 "/home/mgodbolt/dev/compiler-explorer/-"
        .file   2 "/home/mgodbolt/dev/compiler-explorer/<stdin>"
        .section        .debug_info,"",@progbits
.Lsection_info:
        .section        .debug_abbrev,"",@progbits
.Lsection_abbrev:
        .section        .debug_aranges,"",@progbits
        .section        .debug_macinfo,"",@progbits
        .section        .debug_line,"",@progbits
.Lsection_line:
        .section        .debug_loc,"",@progbits
        .section        .debug_pubnames,"",@progbits
        .section        .debug_pubtypes,"",@progbits
        .section        .debug_str,"",@progbits
.Lsection_str:
        .section        .debug_ranges,"",@progbits
.Ldebug_range:
        .section        .debug_loc,"",@progbits
.Lsection_debug_loc:
        .text
.Ltext_begin:
        .data
        .text
        .globl  main
        .align  16, 0x90
        .type   main,@function
main:                                   # @main
.Ltmp2:
        .cfi_startproc
.Lfunc_begin0:
        .loc    2 2 0                   # <stdin>:2:0
# BB#0:
        pushq   %rbp
.Ltmp3:
        .cfi_def_cfa_offset 16
.Ltmp4:
        .cfi_offset %rbp, -16
        movq    %rsp, %rbp
.Ltmp5:
        .cfi_def_cfa_register %rbp
        .loc    2 3 1 prologue_end      # <stdin>:3:1
.Ltmp6:
        movl    $.L.str, %edi
        xorb    %al, %al
        callq   printf
        .loc    2 4 3                   # <stdin>:4:3
        movl    $str, %edi
        callq   puts
        xorl    %eax, %eax
        .loc    2 5 1                   # <stdin>:5:1
        popq    %rbp
        ret
.Ltmp7:
.Ltmp8:
        .size   main, .Ltmp8-main
.Lfunc_end0:
.Ltmp9:
        .cfi_endproc
.Leh_func_end0:
        .type   .L.str,@object          # @.str
        .section        .rodata.str1.1,"aMS",@progbits,1
.L.str:
        .asciz   "Hello world"
        .size   .L.str, 12
        .type   str,@object             # @str
        .section        .rodata,"a",@progbits
str:
        .asciz   "moo"
        .size   str, 4
        .text
.Ltext_end:
        .data
.Ldata_end:
        .text
.Lsection_end1:
        .section        .debug_info,"",@progbits
.Linfo_begin1:
        .long   175                     # Length of Compilation Unit Info
        .short  2                       # DWARF version number
        .long   .Labbrev_begin          # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0xa8 DW_TAG_compile_unit
        .ascii   "Ubuntu clang version 3.0-6ubuntu3 (tags/RELEASE_30/final) (based on LLVM 3.0)" # DW_AT_producer
        .byte   0
        .short  4                       # DW_AT_language
        .byte   45                      # DW_AT_name
        .byte   0
        .quad   0                       # DW_AT_entry_pc
        .long   .Lsection_line          # DW_AT_stmt_list
        .ascii   "/home/mgodbolt/dev/compiler-explorer" # DW_AT_comp_dir
        .byte   0
        .byte   1                       # DW_AT_APPLE_optimized
        .byte   2                       # Abbrev [2] 0x8b:0x20 DW_TAG_subprogram
        .ascii   "main"                 # DW_AT_name
        .byte   0
        .byte   2                       # DW_AT_decl_file
        .byte   2                       # DW_AT_decl_line
        .byte   1                       # DW_AT_prototyped
        .long   171                     # DW_AT_type
        .byte   1                       # DW_AT_external
        .quad   .Lfunc_begin0           # DW_AT_low_pc
        .quad   .Lfunc_end0             # DW_AT_high_pc
        .byte   1                       # DW_AT_frame_base
        .byte   86
        .byte   3                       # Abbrev [3] 0xab:0x7 DW_TAG_base_type
        .ascii   "int"                  # DW_AT_name
        .byte   0
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # End Of Children Mark
.Linfo_end1:
        .section        .debug_abbrev,"",@progbits
.Labbrev_begin:
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   82                      # DW_AT_entry_pc
        .byte   1                       # DW_FORM_addr
        .byte   16                      # DW_AT_stmt_list
        .byte   6                       # DW_FORM_data4
        .byte   27                      # DW_AT_comp_dir
        .byte   8                       # DW_FORM_string
        .ascii   "\341\177"             # DW_AT_APPLE_optimized
        .byte   12                      # DW_FORM_flag
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   58                      # DW_AT_decl_file
        .byte   11                      # DW_FORM_data1
        .byte   59                      # DW_AT_decl_line
        .byte   11                      # DW_FORM_data1
        .byte   39                      # DW_AT_prototyped
        .byte   12                      # DW_FORM_flag
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   63                      # DW_AT_external
        .byte   12                      # DW_FORM_flag
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   1                       # DW_FORM_addr
        .byte   64                      # DW_AT_frame_base
        .byte   10                      # DW_FORM_block1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   36                      # DW_TAG_base_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   62                      # DW_AT_encoding
        .byte   11                      # DW_FORM_data1
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)
.Labbrev_end:
        .section        .debug_pubnames,"",@progbits
.Lset0 = .Lpubnames_end1-.Lpubnames_begin1 # Length of Public Names Info
        .long   .Lset0
.Lpubnames_begin1:
        .short  2                       # DWARF Version
        .long   .Linfo_begin1           # Offset of Compilation Unit Info
.Lset1 = .Linfo_end1-.Linfo_begin1      # Compilation Unit Length
        .long   .Lset1
        .long   139                     # DIE offset
        .asciz   "main"                 # External Name
        .long   0                       # End Mark
.Lpubnames_end1:
        .section        .debug_pubtypes,"",@progbits
.Lset2 = .Lpubtypes_end1-.Lpubtypes_begin1 # Length of Public Types Info
        .long   .Lset2
.Lpubtypes_begin1:
        .short  2                       # DWARF Version
        .long   .Linfo_begin1           # Offset of Compilation Unit Info
.Lset3 = .Linfo_end1-.Linfo_begin1      # Compilation Unit Length
        .long   .Lset3
        .long   0                       # End Mark
.Lpubtypes_end1:
        .section        .debug_aranges,"",@progbits
        .section        .debug_ranges,"",@progbits
        .section        .debug_macinfo,"",@progbits
        .section        ".note.GNU-stack","",@progbits
