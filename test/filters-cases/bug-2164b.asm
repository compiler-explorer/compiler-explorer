        .text
        .syntax unified
        .eabi_attribute 67, "2.09"        @ Tag_conformance
        .cpu    arm7tdmi
        .eabi_attribute 6, 2      @ Tag_CPU_arch
        .eabi_attribute 8, 1      @ Tag_ARM_ISA_use
        .eabi_attribute 9, 1      @ Tag_THUMB_ISA_use
        .eabi_attribute 34, 0     @ Tag_CPU_unaligned_access
        .eabi_attribute 15, 1     @ Tag_ABI_PCS_RW_data
        .eabi_attribute 16, 1     @ Tag_ABI_PCS_RO_data
        .eabi_attribute 17, 2     @ Tag_ABI_PCS_GOT_use
        .eabi_attribute 20, 1     @ Tag_ABI_FP_denormal
        .eabi_attribute 21, 0     @ Tag_ABI_FP_exceptions
        .eabi_attribute 23, 3     @ Tag_ABI_FP_number_model
        .eabi_attribute 24, 1     @ Tag_ABI_align_needed
        .eabi_attribute 25, 1     @ Tag_ABI_align_preserved
        .eabi_attribute 38, 1     @ Tag_ABI_FP_16bit_format
        .eabi_attribute 18, 4     @ Tag_ABI_PCS_wchar_t
        .eabi_attribute 26, 2     @ Tag_ABI_enum_size
        .eabi_attribute 14, 0     @ Tag_ABI_PCS_R9_use
        .file   "example.c"
        .globl  trap_arm                @ -- Begin function trap_arm
        .p2align        2
        .type   trap_arm,%function
        .code   32                      @ @trap_arm
trap_arm:
.Lfunc_begin0:
        .file   1 "/home/ce" "./example.c"
        .loc    1 2 0                   @ ./example.c:2:0
        .fnstart
        .cfi_sections .debug_frame
        .cfi_startproc
        .loc    1 3 5 prologue_end      @ ./example.c:3:5
        .inst   0xe7ffdefe
.Lfunc_end0:
        .size   trap_arm, .Lfunc_end0-trap_arm
        .cfi_endproc
        .cantunwind
        .fnend
        .globl  trap_thumb              @ -- Begin function trap_thumb
        .p2align        1
        .type   trap_thumb,%function
        .code   16                      @ @trap_thumb
        .thumb_func
trap_thumb:
.Lfunc_begin1:
        .loc    1 7 0                   @ ./example.c:7:0
        .fnstart
        .cfi_startproc
        .loc    1 8 5 prologue_end      @ ./example.c:8:5
        .inst.n 0xdefe
.Lfunc_end1:
        .size   trap_thumb, .Lfunc_end1-trap_thumb
        .cfi_endproc
        .cantunwind
        .fnend
        .section        .debug_str,"MS",%progbits,1
.Linfo_string0:
        .asciz  "clang version 10.0.0 (https://github.com/llvm/llvm-project.git d32170dbd5b0d54436537b6b75beaf44324e0c28)" @ string offset=0
.Linfo_string1:
        .asciz  "example.c"             @ string offset=105
.Linfo_string2:
        .asciz  "/home/ce"              @ string offset=115
.Linfo_string3:
        .asciz  "trap_arm"              @ string offset=124
.Linfo_string4:
        .asciz  "trap_thumb"            @ string offset=133
        .section        .debug_abbrev,"",%progbits
        .byte   1                       @ Abbreviation Code
        .byte   17                      @ DW_TAG_compile_unit
        .byte   1                       @ DW_CHILDREN_yes
        .byte   37                      @ DW_AT_producer
        .byte   14                      @ DW_FORM_strp
        .byte   19                      @ DW_AT_language
        .byte   5                       @ DW_FORM_data2
        .byte   3                       @ DW_AT_name
        .byte   14                      @ DW_FORM_strp
        .byte   16                      @ DW_AT_stmt_list
        .byte   23                      @ DW_FORM_sec_offset
        .byte   27                      @ DW_AT_comp_dir
        .byte   14                      @ DW_FORM_strp
        .byte   17                      @ DW_AT_low_pc
        .byte   1                       @ DW_FORM_addr
        .byte   18                      @ DW_AT_high_pc
        .byte   6                       @ DW_FORM_data4
        .byte   0                       @ EOM(1)
        .byte   0                       @ EOM(2)
        .byte   2                       @ Abbreviation Code
        .byte   46                      @ DW_TAG_subprogram
        .byte   0                       @ DW_CHILDREN_no
        .byte   17                      @ DW_AT_low_pc
        .byte   1                       @ DW_FORM_addr
        .byte   18                      @ DW_AT_high_pc
        .byte   6                       @ DW_FORM_data4
        .byte   64                      @ DW_AT_frame_base
        .byte   24                      @ DW_FORM_exprloc
        .ascii  "\227B"                 @ DW_AT_GNU_all_call_sites
        .byte   25                      @ DW_FORM_flag_present
        .byte   3                       @ DW_AT_name
        .byte   14                      @ DW_FORM_strp
        .byte   58                      @ DW_AT_decl_file
        .byte   11                      @ DW_FORM_data1
        .byte   59                      @ DW_AT_decl_line
        .byte   11                      @ DW_FORM_data1
        .byte   63                      @ DW_AT_external
        .byte   25                      @ DW_FORM_flag_present
        .byte   0                       @ EOM(1)
        .byte   0                       @ EOM(2)
        .byte   0                       @ EOM(3)
        .section        .debug_info,"",%progbits
        .long   .Ldebug_info_end0-.Ldebug_info_start0 @ Length of Unit
.Ldebug_info_start0:
        .short  4                       @ DWARF version number
        .long   .debug_abbrev           @ Offset Into Abbrev. Section
        .byte   4                       @ Address Size (in bytes)
        .byte   1                       @ Abbrev [1] 0xb:0x3e DW_TAG_compile_unit
        .long   .Linfo_string0          @ DW_AT_producer
        .short  12                      @ DW_AT_language
        .long   .Linfo_string1          @ DW_AT_name
        .long   .Lline_table_start0     @ DW_AT_stmt_list
        .long   .Linfo_string2          @ DW_AT_comp_dir
        .long   .Lfunc_begin0           @ DW_AT_low_pc
        .long   .Lfunc_end1-.Lfunc_begin0 @ DW_AT_high_pc
        .byte   2                       @ Abbrev [2] 0x26:0x11 DW_TAG_subprogram
        .long   .Lfunc_begin0           @ DW_AT_low_pc
        .long   .Lfunc_end0-.Lfunc_begin0 @ DW_AT_high_pc
        .byte   1                       @ DW_AT_frame_base
        .byte   91
        .long   .Linfo_string3          @ DW_AT_name
        .byte   1                       @ DW_AT_decl_file
        .byte   2                       @ DW_AT_decl_line
        .byte   2                       @ Abbrev [2] 0x37:0x11 DW_TAG_subprogram
        .long   .Lfunc_begin1           @ DW_AT_low_pc
        .long   .Lfunc_end1-.Lfunc_begin1 @ DW_AT_high_pc
        .byte   1                       @ DW_AT_frame_base
        .byte   87
        .long   .Linfo_string4          @ DW_AT_name
        .byte   1                       @ DW_AT_decl_file
        .byte   7                       @ DW_AT_decl_line
        .byte   0                       @ End Of Children Mark
.Ldebug_info_end0:
        .ident  "clang version 10.0.0 (https://github.com/llvm/llvm-project.git d32170dbd5b0d54436537b6b75beaf44324e0c28)"
        .section        ".note.GNU-stack","",%progbits
        .section        .debug_line,"",%progbits
.Lline_table_start0: