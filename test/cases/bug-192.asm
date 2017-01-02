        .text
        .intel_syntax noprefix
        .file   "/tmp/compiler-explorer-compiler1161023-8026-16e0svr/example.cpp"
        .file   1 "/tmp/compiler-explorer-compiler1161023-8026-16e0svr" "example.cpp"
        .globl  main
        .p2align        4, 0x90
        .type   main,@function
main:                                   # @main
.Lfunc_begin0:
        .loc    1 14 0                  # /tmp/compiler-explorer-compiler1161023-8026-16e0svr/example.cpp:14:0
        .cfi_startproc
        .loc    1 15 3 prologue_end     # /tmp/compiler-explorer-compiler1161023-8026-16e0svr/example.cpp:15:3
        mov     eax, 3
        ret
.Ltmp0:
.Lfunc_end0:
        .size   main, .Lfunc_end0-main
        .cfi_endproc

        .section        .text.startup,"axG",@progbits,asdf<float>,comdat
        .p2align        4, 0x90
        .type   __cxx_global_var_init,@function
__cxx_global_var_init:                  # @__cxx_global_var_init
.Lfunc_begin1:
        .loc    1 5 0                   # /tmp/compiler-explorer-compiler1161023-8026-16e0svr/example.cpp:5:0
        .cfi_startproc
        .loc    1 5 15 prologue_end     # /tmp/compiler-explorer-compiler1161023-8026-16e0svr/example.cpp:5:15
        cmp     byte ptr [rip + guard variable for asdf<float>], 0
        jne     .LBB1_2
        .loc    1 5 15 is_stmt 0 discriminator 1 # /tmp/compiler-explorer-compiler1161023-8026-16e0svr/example.cpp:5:15
        mov     qword ptr [rip + guard variable for asdf<float>], 1
.LBB1_2:
        .loc    1 5 15 discriminator 2  # /tmp/compiler-explorer-compiler1161023-8026-16e0svr/example.cpp:5:15
        ret
.Ltmp1:
.Lfunc_end1:
        .size   __cxx_global_var_init, .Lfunc_end1-__cxx_global_var_init
        .cfi_endproc

        .type   asdf<float>,@object      # @asdf<float>
        .section        .bss._Z4asdfIfE,"aGw",@nobits,asdf<float>,comdat
        .weak   asdf<float>
asdf<float>:
        .zero   1
        .size   asdf<float>, 1

        .type   guard variable for asdf<float>,@object    # @guard variable for asdf<float>
        .section        .bss._ZGV4asdfIfE,"aGw",@nobits,asdf<float>,comdat
        .weak   guard variable for asdf<float>
        .p2align        3
guard variable for asdf<float>:
        .quad   0                       # 0x0
        .size   guard variable for asdf<float>, 8

        .section        .init_array,"aGw",@init_array,asdf<float>,comdat
        .p2align        3
        .quad   __cxx_global_var_init
        .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "clang version 3.9.0-1ubuntu1 (tags/RELEASE_390/final)" # string offset=0
.Linfo_string1:
        .asciz  "/tmp/compiler-explorer-compiler1161023-8026-16e0svr/example.cpp" # string offset=54
.Linfo_string2:
        .asciz  "/home/mgodbolt/dev/compiler-explorer" # string offset=113
.Linfo_string3:
        .asciz  "asdf"                  # string offset=145
.Linfo_string4:
        .asciz  "asdf<float>"            # string offset=150
.Linfo_string5:
        .asciz  "xyz"                   # string offset=161
.Linfo_string6:
        .asciz  "float"                 # string offset=165
.Linfo_string7:
        .asciz  "foo"                   # string offset=171
.Linfo_string8:
        .asciz  "main"                  # string offset=175
.Linfo_string9:
        .asciz  "int"                   # string offset=180
.Linfo_string10:
        .asciz  "__cxx_global_var_init" # string offset=184
        .section        .debug_loc,"",@progbits
        .section        .debug_abbrev,"",@progbits
.Lsection_abbrev:
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   14                      # DW_FORM_strp
        .byte   19                      # DW_AT_language
        .byte   5                       # DW_FORM_data2
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   16                      # DW_AT_stmt_list
        .byte   23                      # DW_FORM_sec_offset
        .byte   27                      # DW_AT_comp_dir
        .byte   14                      # DW_FORM_strp
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   85                      # DW_AT_ranges
        .byte   23                      # DW_FORM_sec_offset
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   52                      # DW_TAG_variable
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   63                      # DW_AT_external
        .byte   25                      # DW_FORM_flag_present
        .byte   58                      # DW_AT_decl_file
        .byte   11                      # DW_FORM_data1
        .byte   59                      # DW_AT_decl_line
        .byte   11                      # DW_FORM_data1
        .byte   2                       # DW_AT_location
        .byte   24                      # DW_FORM_exprloc
        .byte   110                     # DW_AT_linkage_name
        .byte   14                      # DW_FORM_strp
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   3                       # Abbreviation Code
        .byte   2                       # DW_TAG_class_type
        .byte   0                       # DW_CHILDREN_no
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   58                      # DW_AT_decl_file
        .byte   11                      # DW_FORM_data1
        .byte   59                      # DW_AT_decl_line
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   4                       # Abbreviation Code
        .byte   19                      # DW_TAG_structure_type
        .byte   1                       # DW_CHILDREN_yes
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   58                      # DW_AT_decl_file
        .byte   11                      # DW_FORM_data1
        .byte   59                      # DW_AT_decl_line
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   5                       # Abbreviation Code
        .byte   13                      # DW_TAG_member
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   58                      # DW_AT_decl_file
        .byte   11                      # DW_FORM_data1
        .byte   59                      # DW_AT_decl_line
        .byte   11                      # DW_FORM_data1
        .byte   56                      # DW_AT_data_member_location
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   6                       # Abbreviation Code
        .byte   36                      # DW_TAG_base_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   62                      # DW_AT_encoding
        .byte   11                      # DW_FORM_data1
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   7                       # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   0                       # DW_CHILDREN_no
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   64                      # DW_AT_frame_base
        .byte   24                      # DW_FORM_exprloc
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   58                      # DW_AT_decl_file
        .byte   11                      # DW_FORM_data1
        .byte   59                      # DW_AT_decl_line
        .byte   11                      # DW_FORM_data1
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   63                      # DW_AT_external
        .byte   25                      # DW_FORM_flag_present
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   8                       # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   0                       # DW_CHILDREN_no
        .byte   17                      # DW_AT_low_pc
        .byte   1                       # DW_FORM_addr
        .byte   18                      # DW_AT_high_pc
        .byte   6                       # DW_FORM_data4
        .byte   64                      # DW_AT_frame_base
        .byte   24                      # DW_FORM_exprloc
        .byte   3                       # DW_AT_name
        .byte   14                      # DW_FORM_strp
        .byte   58                      # DW_AT_decl_file
        .byte   11                      # DW_FORM_data1
        .byte   59                      # DW_AT_decl_line
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)
        .section        .debug_info,"",@progbits
.Lsection_info:
.Lcu_begin0:
        .long   149                     # Length of Unit
        .short  4                       # DWARF version number
        .long   .Lsection_abbrev        # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0x8e DW_TAG_compile_unit
        .long   .Linfo_string0          # DW_AT_producer
        .short  4                       # DW_AT_language
        .long   .Linfo_string1          # DW_AT_name
        .long   .Lline_table_start0     # DW_AT_stmt_list
        .long   .Linfo_string2          # DW_AT_comp_dir
        .quad   0                       # DW_AT_low_pc
        .long   .Ldebug_ranges0         # DW_AT_ranges
        .byte   2                       # Abbrev [2] 0x2a:0x19 DW_TAG_variable
        .long   .Linfo_string3          # DW_AT_name
        .long   67                      # DW_AT_type
        .byte   1                       # DW_AT_decl_file
        .byte   5                       # DW_AT_decl_line
        .byte   9                       # DW_AT_location
        .byte   3
        .quad   asdf<float>
        .long   .Linfo_string4          # DW_AT_linkage_name
        .byte   3                       # Abbrev [3] 0x43:0x4 DW_TAG_class_type
        .byte   1                       # DW_AT_byte_size
        .byte   1                       # DW_AT_decl_file
        .byte   5                       # DW_AT_decl_line
        .byte   4                       # Abbrev [4] 0x47:0x15 DW_TAG_structure_type
        .long   .Linfo_string7          # DW_AT_name
        .byte   4                       # DW_AT_byte_size
        .byte   1                       # DW_AT_decl_file
        .byte   6                       # DW_AT_decl_line
        .byte   5                       # Abbrev [5] 0x4f:0xc DW_TAG_member
        .long   .Linfo_string5          # DW_AT_name
        .long   92                      # DW_AT_type
        .byte   1                       # DW_AT_decl_file
        .byte   7                       # DW_AT_decl_line
        .byte   0                       # DW_AT_data_member_location
        .byte   0                       # End Of Children Mark
        .byte   6                       # Abbrev [6] 0x5c:0x7 DW_TAG_base_type
        .long   .Linfo_string6          # DW_AT_name
        .byte   4                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   7                       # Abbrev [7] 0x63:0x19 DW_TAG_subprogram
        .quad   .Lfunc_begin0           # DW_AT_low_pc
        .long   .Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
        .byte   1                       # DW_AT_frame_base
        .byte   87
        .long   .Linfo_string8          # DW_AT_name
        .byte   1                       # DW_AT_decl_file
        .byte   14                      # DW_AT_decl_line
        .long   145                     # DW_AT_type
        .byte   8                       # Abbrev [8] 0x7c:0x15 DW_TAG_subprogram
        .quad   .Lfunc_begin1           # DW_AT_low_pc
        .long   .Lfunc_end1-.Lfunc_begin1 # DW_AT_high_pc
        .byte   1                       # DW_AT_frame_base
        .byte   87
        .long   .Linfo_string10         # DW_AT_name
        .byte   1                       # DW_AT_decl_file
        .byte   5                       # DW_AT_decl_line
        .byte   6                       # Abbrev [6] 0x91:0x7 DW_TAG_base_type
        .long   .Linfo_string9          # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size
        .byte   0                       # End Of Children Mark
        .section        .debug_ranges,"",@progbits
.Ldebug_range:
.Ldebug_ranges0:
        .quad   .Lfunc_begin0
        .quad   .Lfunc_end0
        .quad   .Lfunc_begin1
        .quad   .Lfunc_end1
        .quad   0
        .quad   0
        .section        .debug_macinfo,"",@progbits
.Ldebug_macinfo:
.Lcu_macro_begin0:
        .byte   0                       # End Of Macro List Mark
        .section        .debug_pubnames,"",@progbits
        .long   .LpubNames_end0-.LpubNames_begin0 # Length of Public Names Info
.LpubNames_begin0:
        .short  2                       # DWARF Version
        .long   .Lcu_begin0             # Offset of Compilation Unit Info
        .long   153                     # Compilation Unit Length
        .long   99                      # DIE offset
        .asciz  "main"                  # External Name
        .long   124                     # DIE offset
        .asciz  "__cxx_global_var_init" # External Name
        .long   42                      # DIE offset
        .asciz  "asdf"                  # External Name
        .long   0                       # End Mark
.LpubNames_end0:
        .section        .debug_pubtypes,"",@progbits
        .long   .LpubTypes_end0-.LpubTypes_begin0 # Length of Public Types Info
.LpubTypes_begin0:
        .short  2                       # DWARF Version
        .long   .Lcu_begin0             # Offset of Compilation Unit Info
        .long   153                     # Compilation Unit Length
        .long   71                      # DIE offset
        .asciz  "foo"                   # External Name
        .long   92                      # DIE offset
        .asciz  "float"                 # External Name
        .long   145                     # DIE offset
        .asciz  "int"                   # External Name
        .long   0                       # End Mark
.LpubTypes_end0:

        .ident  "clang version 3.9.0-1ubuntu1 (tags/RELEASE_390/final)"
        .section        ".note.GNU-stack","",@progbits
        .section        .debug_line,"",@progbits
.Lline_table_start0:
