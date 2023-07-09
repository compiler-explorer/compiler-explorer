        .text
        .intel_syntax noprefix
        .file   "example.cpp"
        .file   1 "/opt/compiler-explorer/gcc-snapshot/lib/gcc/x86_64-linux-gnu/14.0.0/../../../../include/c++/14.0.0" "variant"
        .file   2 "/opt/compiler-explorer/gcc-snapshot/lib/gcc/x86_64-linux-gnu/14.0.0/../../../../include/c++/14.0.0" "type_traits"
        .file   3 "/opt/compiler-explorer/gcc-snapshot/lib/gcc/x86_64-linux-gnu/14.0.0/../../../../include/c++/14.0.0/bits" "invoke.h"
        .file   4 "/opt/compiler-explorer/gcc-snapshot/lib/gcc/x86_64-linux-gnu/14.0.0/../../../../include/c++/14.0.0/bits" "enable_special_members.h"
        .file   5 "/opt/compiler-explorer/gcc-snapshot/lib/gcc/x86_64-linux-gnu/14.0.0/../../../../include/c++/14.0.0/x86_64-linux-gnu/bits" "c++config.h"
        .globl  _Z1fSt7variantIJPiPlPcEE        # -- Begin function _Z1fSt7variantIJPiPlPcEE
        .p2align        4, 0x90
        .type   _Z1fSt7variantIJPiPlPcEE,@function
_Z1fSt7variantIJPiPlPcEE:               # @_Z1fSt7variantIJPiPlPcEE
.Lfunc_begin0:
        .cfi_startproc
# %bb.0:
        #DEBUG_VALUE: f:v <- [DW_OP_LLVM_fragment 0 64] $rdi
        #DEBUG_VALUE: f:v <- [DW_OP_LLVM_fragment 64 8] $esi
        #DEBUG_VALUE: __visit_rettypes_match <- 1
        #DEBUG_VALUE: __max <- 11
        #DEBUG_VALUE: __n <- 3
        .loc    1 1789 8 prologue_end           # /opt/compiler-explorer/gcc-snapshot/lib/gcc/x86_64-linux-gnu/14.0.0/../../../../include/c++/14.0.0/variant:1789:8
        cmp     sil, 2
        je      .LBB0_2
.Ltmp0:
# %bb.1:
        #DEBUG_VALUE: f:v <- [DW_OP_LLVM_fragment 0 64] $rdi
        #DEBUG_VALUE: f:v <- [DW_OP_LLVM_fragment 64 8] $esi
        #DEBUG_VALUE: __visit_rettypes_match <- 1
        #DEBUG_VALUE: __max <- 11
        #DEBUG_VALUE: __n <- 3
        movzx   eax, sil
        cmp     eax, 1
.Ltmp1:
.LBB0_2:
        #DEBUG_VALUE: f:v <- [DW_OP_LLVM_fragment 0 64] $rdi
        #DEBUG_VALUE: f:v <- [DW_OP_LLVM_fragment 64 8] $esi
        #DEBUG_VALUE: __visit_rettypes_match <- 1
        #DEBUG_VALUE: __max <- 11
        #DEBUG_VALUE: __n <- 3
        #DEBUG_VALUE: __visit_invoke:__visitor <- undef
        #DEBUG_VALUE: __visit_invoke:__vars <- undef
        #DEBUG_VALUE: __invoke<(lambda at /app/example.cpp:3:16), char *&>:__fn <- undef
        #DEBUG_VALUE: __invoke<(lambda at /app/example.cpp:3:16), char *&>:__args <- undef
        #DEBUG_VALUE: __invoke_impl<void, (lambda at /app/example.cpp:3:16), char *&>:__f <- undef
        #DEBUG_VALUE: __invoke_impl<void, (lambda at /app/example.cpp:3:16), char *&>:__args <- undef
        #DEBUG_VALUE: operator()<char *&>:this <- undef
        #DEBUG_VALUE: operator()<char *&>:p <- undef
        .loc    1 0 0 is_stmt 0                 # /opt/compiler-explorer/gcc-snapshot/lib/gcc/x86_64-linux-gnu/14.0.0/../../../../include/c++/14.0.0/variant:0:0
        test    rdi, rdi
.Ltmp2:
        jne     _ZdlPv@PLT                      # TAILCALL
.Ltmp3:
# %bb.3:
        .file   6 "/app" "example.cpp"
        .loc    6 4 1 is_stmt 1                 # example.cpp:4:1
        ret
.Ltmp4:
.Lfunc_end0:
        .size   _Z1fSt7variantIJPiPlPcEE, .Lfunc_end0-_Z1fSt7variantIJPiPlPcEE
        .cfi_endproc
                                        # -- End function
        .section        .debug_loc,"",@progbits
.Ldebug_loc0:
        .quad   .Lfunc_begin0-.Lfunc_begin0
        .quad   .Ltmp3-.Lfunc_begin0
        .short  6                               # Loc expr size
        .byte   85                              # DW_OP_reg5
        .byte   147                             # DW_OP_piece
        .byte   8                               # 8
        .byte   84                              # super-register DW_OP_reg4
        .byte   147                             # DW_OP_piece
        .byte   1                               # 1
        .quad   0
        .quad   0
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
        .byte   5                               # DW_FORM_data2
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
        .byte   38                              # DW_TAG_const_type
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   6                               # Abbreviation Code
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
        .byte   7                               # Abbreviation Code
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
        .byte   8                               # Abbreviation Code
        .byte   57                              # DW_TAG_namespace
        .byte   1                               # DW_CHILDREN_yes
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   9                               # Abbreviation Code
        .byte   2                               # DW_TAG_class_type
        .byte   1                               # DW_CHILDREN_yes
        .byte   29                              # DW_AT_containing_type
        .byte   19                              # DW_FORM_ref4
        .byte   54                              # DW_AT_calling_convention
        .byte   11                              # DW_FORM_data1
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   10                              # Abbreviation Code
        .byte   28                              # DW_TAG_inheritance
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   56                              # DW_AT_data_member_location
        .byte   11                              # DW_FORM_data1
        .byte   50                              # DW_AT_accessibility
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   11                              # Abbreviation Code
        .byte   13                              # DW_TAG_member
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   56                              # DW_AT_data_member_location
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   12                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   50                              # DW_AT_accessibility
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   13                              # Abbreviation Code
        .byte   5                               # DW_TAG_formal_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   52                              # DW_AT_artificial
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   14                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   110                             # DW_AT_linkage_name
        .byte   14                              # DW_FORM_strp
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   76                              # DW_AT_virtuality
        .byte   11                              # DW_FORM_data1
        .byte   77                              # DW_AT_vtable_elem_location
        .byte   24                              # DW_FORM_exprloc
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   50                              # DW_AT_accessibility
        .byte   11                              # DW_FORM_data1
        .byte   29                              # DW_AT_containing_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   15                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   16                              # Abbreviation Code
        .byte   5                               # DW_TAG_formal_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   17                              # Abbreviation Code
        .byte   2                               # DW_TAG_class_type
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   18                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   19                              # Abbreviation Code
        .byte   22                              # DW_TAG_typedef
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   20                              # Abbreviation Code
        .byte   19                              # DW_TAG_structure_type
        .byte   1                               # DW_CHILDREN_yes
        .byte   54                              # DW_AT_calling_convention
        .byte   11                              # DW_FORM_data1
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   21                              # Abbreviation Code
        .byte   47                              # DW_TAG_template_type_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   22                              # Abbreviation Code
        .byte   47                              # DW_TAG_template_type_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   23                              # Abbreviation Code
        .byte   28                              # DW_TAG_inheritance
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   56                              # DW_AT_data_member_location
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   24                              # Abbreviation Code
        .byte   22                              # DW_TAG_typedef
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   25                              # Abbreviation Code
        .byte   19                              # DW_TAG_structure_type
        .byte   0                               # DW_CHILDREN_no
        .byte   54                              # DW_AT_calling_convention
        .byte   11                              # DW_FORM_data1
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   26                              # Abbreviation Code
        .byte   22                              # DW_TAG_typedef
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   27                              # Abbreviation Code
        .byte   23                              # DW_TAG_union_type
        .byte   1                               # DW_CHILDREN_yes
        .byte   54                              # DW_AT_calling_convention
        .byte   11                              # DW_FORM_data1
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   28                              # Abbreviation Code
        .ascii  "\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   29                              # Abbreviation Code
        .ascii  "\207\202\001"                  # DW_TAG_GNU_template_parameter_pack
        .byte   1                               # DW_CHILDREN_yes
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   30                              # Abbreviation Code
        .byte   47                              # DW_TAG_template_type_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   31                              # Abbreviation Code
        .byte   19                              # DW_TAG_structure_type
        .byte   1                               # DW_CHILDREN_yes
        .byte   54                              # DW_AT_calling_convention
        .byte   11                              # DW_FORM_data1
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   32                              # Abbreviation Code
        .byte   48                              # DW_TAG_template_value_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   30                              # DW_AT_default_value
        .byte   25                              # DW_FORM_flag_present
        .byte   28                              # DW_AT_const_value
        .byte   15                              # DW_FORM_udata
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   33                              # Abbreviation Code
        .byte   13                              # DW_TAG_member
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   56                              # DW_AT_data_member_location
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   34                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   110                             # DW_AT_linkage_name
        .byte   14                              # DW_FORM_strp
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   119                             # DW_AT_reference
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   35                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   110                             # DW_AT_linkage_name
        .byte   14                              # DW_FORM_strp
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   120                             # DW_AT_rvalue_reference
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   36                              # Abbreviation Code
        .byte   48                              # DW_TAG_template_value_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   28                              # DW_AT_const_value
        .byte   15                              # DW_FORM_udata
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   37                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   110                             # DW_AT_linkage_name
        .byte   14                              # DW_FORM_strp
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   38                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   110                             # DW_AT_linkage_name
        .byte   14                              # DW_FORM_strp
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   39                              # Abbreviation Code
        .byte   48                              # DW_TAG_template_value_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   28                              # DW_AT_const_value
        .byte   15                              # DW_FORM_udata
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   40                              # Abbreviation Code
        .byte   19                              # DW_TAG_structure_type
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   41                              # Abbreviation Code
        .byte   19                              # DW_TAG_structure_type
        .byte   0                               # DW_CHILDREN_no
        .byte   54                              # DW_AT_calling_convention
        .byte   11                              # DW_FORM_data1
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   42                              # Abbreviation Code
        .byte   13                              # DW_TAG_member
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   43                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   110                             # DW_AT_linkage_name
        .byte   14                              # DW_FORM_strp
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   44                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   45                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   99                              # DW_AT_explicit
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   46                              # Abbreviation Code
        .byte   2                               # DW_TAG_class_type
        .byte   1                               # DW_CHILDREN_yes
        .byte   54                              # DW_AT_calling_convention
        .byte   11                              # DW_FORM_data1
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   47                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   110                             # DW_AT_linkage_name
        .byte   14                              # DW_FORM_strp
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   50                              # DW_AT_accessibility
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   48                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   110                             # DW_AT_linkage_name
        .byte   14                              # DW_FORM_strp
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   60                              # DW_AT_declaration
        .byte   25                              # DW_FORM_flag_present
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   50                              # DW_AT_accessibility
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   49                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   110                             # DW_AT_linkage_name
        .byte   14                              # DW_FORM_strp
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   32                              # DW_AT_inline
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   50                              # Abbreviation Code
        .byte   5                               # DW_TAG_formal_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   51                              # Abbreviation Code
        .byte   11                              # DW_TAG_lexical_block
        .byte   1                               # DW_CHILDREN_yes
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   52                              # Abbreviation Code
        .byte   52                              # DW_TAG_variable
        .byte   0                               # DW_CHILDREN_no
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   53                              # Abbreviation Code
        .byte   8                               # DW_TAG_imported_declaration
        .byte   0                               # DW_CHILDREN_no
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   24                              # DW_AT_import
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   54                              # Abbreviation Code
        .byte   8                               # DW_TAG_imported_declaration
        .byte   0                               # DW_CHILDREN_no
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   5                               # DW_FORM_data2
        .byte   24                              # DW_AT_import
        .byte   19                              # DW_FORM_ref4
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   55                              # Abbreviation Code
        .byte   15                              # DW_TAG_pointer_type
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   56                              # Abbreviation Code
        .byte   16                              # DW_TAG_reference_type
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   57                              # Abbreviation Code
        .byte   66                              # DW_TAG_rvalue_reference_type
        .byte   0                               # DW_CHILDREN_no
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   58                              # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   1                               # DW_CHILDREN_yes
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   6                               # DW_FORM_data4
        .byte   64                              # DW_AT_frame_base
        .byte   24                              # DW_FORM_exprloc
        .ascii  "\227B"                         # DW_AT_GNU_all_call_sites
        .byte   25                              # DW_FORM_flag_present
        .byte   110                             # DW_AT_linkage_name
        .byte   14                              # DW_FORM_strp
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   63                              # DW_AT_external
        .byte   25                              # DW_FORM_flag_present
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   59                              # Abbreviation Code
        .byte   2                               # DW_TAG_class_type
        .byte   0                               # DW_CHILDREN_no
        .byte   54                              # DW_AT_calling_convention
        .byte   11                              # DW_FORM_data1
        .byte   11                              # DW_AT_byte_size
        .byte   11                              # DW_FORM_data1
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   60                              # Abbreviation Code
        .byte   5                               # DW_TAG_formal_parameter
        .byte   0                               # DW_CHILDREN_no
        .byte   2                               # DW_AT_location
        .byte   23                              # DW_FORM_sec_offset
        .byte   3                               # DW_AT_name
        .byte   14                              # DW_FORM_strp
        .byte   58                              # DW_AT_decl_file
        .byte   11                              # DW_FORM_data1
        .byte   59                              # DW_AT_decl_line
        .byte   11                              # DW_FORM_data1
        .byte   73                              # DW_AT_type
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   61                              # Abbreviation Code
        .byte   29                              # DW_TAG_inlined_subroutine
        .byte   1                               # DW_CHILDREN_yes
        .byte   49                              # DW_AT_abstract_origin
        .byte   19                              # DW_FORM_ref4
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   6                               # DW_FORM_data4
        .byte   88                              # DW_AT_call_file
        .byte   11                              # DW_FORM_data1
        .byte   89                              # DW_AT_call_line
        .byte   11                              # DW_FORM_data1
        .byte   87                              # DW_AT_call_column
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   62                              # Abbreviation Code
        .byte   11                              # DW_TAG_lexical_block
        .byte   1                               # DW_CHILDREN_yes
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   6                               # DW_FORM_data4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   63                              # Abbreviation Code
        .byte   52                              # DW_TAG_variable
        .byte   0                               # DW_CHILDREN_no
        .byte   28                              # DW_AT_const_value
        .byte   15                              # DW_FORM_udata
        .byte   49                              # DW_AT_abstract_origin
        .byte   19                              # DW_FORM_ref4
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   64                              # Abbreviation Code
        .byte   29                              # DW_TAG_inlined_subroutine
        .byte   1                               # DW_CHILDREN_yes
        .byte   49                              # DW_AT_abstract_origin
        .byte   19                              # DW_FORM_ref4
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   6                               # DW_FORM_data4
        .byte   88                              # DW_AT_call_file
        .byte   11                              # DW_FORM_data1
        .byte   89                              # DW_AT_call_line
        .byte   5                               # DW_FORM_data2
        .byte   87                              # DW_AT_call_column
        .byte   11                              # DW_FORM_data1
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   65                              # Abbreviation Code
        .byte   11                              # DW_TAG_lexical_block
        .byte   0                               # DW_CHILDREN_no
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   6                               # DW_FORM_data4
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
        .byte   1                               # Abbrev [1] 0xb:0xac7 DW_TAG_compile_unit
        .long   .Linfo_string0                  # DW_AT_producer
        .short  33                              # DW_AT_language
        .long   .Linfo_string1                  # DW_AT_name
        .long   .Lline_table_start0             # DW_AT_stmt_list
        .long   .Linfo_string2                  # DW_AT_comp_dir
        .quad   .Lfunc_begin0                   # DW_AT_low_pc
        .long   .Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
        .byte   2                               # Abbrev [2] 0x2a:0x8 DW_TAG_variable
        .long   50                              # DW_AT_type
        .byte   1                               # DW_AT_decl_file
        .short  1832                            # DW_AT_decl_line
        .byte   3                               # Abbrev [3] 0x32:0xc DW_TAG_array_type
        .long   62                              # DW_AT_type
        .byte   4                               # Abbrev [4] 0x37:0x6 DW_TAG_subrange_type
        .long   74                              # DW_AT_type
        .byte   33                              # DW_AT_count
        .byte   0                               # End Of Children Mark
        .byte   5                               # Abbrev [5] 0x3e:0x5 DW_TAG_const_type
        .long   67                              # DW_AT_type
        .byte   6                               # Abbrev [6] 0x43:0x7 DW_TAG_base_type
        .long   .Linfo_string3                  # DW_AT_name
        .byte   6                               # DW_AT_encoding
        .byte   1                               # DW_AT_byte_size
        .byte   7                               # Abbrev [7] 0x4a:0x7 DW_TAG_base_type
        .long   .Linfo_string4                  # DW_AT_name
        .byte   8                               # DW_AT_byte_size
        .byte   7                               # DW_AT_encoding
        .byte   8                               # Abbrev [8] 0x51:0x898 DW_TAG_namespace
        .long   .Linfo_string5                  # DW_AT_name
        .byte   9                               # Abbrev [9] 0x56:0x64 DW_TAG_class_type
        .long   186                             # DW_AT_containing_type
        .byte   4                               # DW_AT_calling_convention
        .long   .Linfo_string8                  # DW_AT_name
        .byte   16                              # DW_AT_byte_size
        .byte   1                               # DW_AT_decl_file
        .short  1305                            # DW_AT_decl_line
        .byte   10                              # Abbrev [10] 0x64:0x7 DW_TAG_inheritance
        .long   186                             # DW_AT_type
        .byte   0                               # DW_AT_data_member_location
        .byte   1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
        .byte   11                              # Abbrev [11] 0x6b:0xd DW_TAG_member
        .long   .Linfo_string7                  # DW_AT_name
        .long   2281                            # DW_AT_type
        .byte   1                               # DW_AT_decl_file
        .short  1317                            # DW_AT_decl_line
        .byte   8                               # DW_AT_data_member_location
        .byte   12                              # Abbrev [12] 0x78:0xf DW_TAG_subprogram
        .long   .Linfo_string8                  # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1308                            # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
        .byte   13                              # Abbrev [13] 0x81:0x5 DW_TAG_formal_parameter
        .long   2286                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   14                              # Abbrev [14] 0x87:0x1f DW_TAG_subprogram
        .long   .Linfo_string9                  # DW_AT_linkage_name
        .long   .Linfo_string10                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1310                            # DW_AT_decl_line
        .long   2281                            # DW_AT_type
        .byte   1                               # DW_AT_virtuality
        .byte   2                               # DW_AT_vtable_elem_location
        .byte   16
        .byte   2
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
        .long   86                              # DW_AT_containing_type
        .byte   13                              # Abbrev [13] 0xa0:0x5 DW_TAG_formal_parameter
        .long   2291                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   15                              # Abbrev [15] 0xa6:0x13 DW_TAG_subprogram
        .long   .Linfo_string8                  # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1314                            # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0xae:0x5 DW_TAG_formal_parameter
        .long   2286                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   16                              # Abbrev [16] 0xb3:0x5 DW_TAG_formal_parameter
        .long   2281                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   17                              # Abbrev [17] 0xba:0x5 DW_TAG_class_type
        .long   .Linfo_string6                  # DW_AT_name
                                        # DW_AT_declaration
        .byte   18                              # Abbrev [18] 0xbf:0xd DW_TAG_subprogram
        .byte   19                              # Abbrev [19] 0xc0:0xb DW_TAG_typedef
        .long   234                             # DW_AT_type
        .long   .Linfo_string17                 # DW_AT_name
        .byte   3                               # DW_AT_decl_file
        .byte   95                              # DW_AT_decl_line
        .byte   0                               # End Of Children Mark
        .byte   20                              # Abbrev [20] 0xcc:0x2b DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string15                 # DW_AT_name
        .byte   1                               # DW_AT_byte_size
        .byte   2                               # DW_AT_decl_file
        .short  2410                            # DW_AT_decl_line
        .byte   21                              # Abbrev [21] 0xd6:0x5 DW_TAG_template_type_parameter
        .long   .Linfo_string11                 # DW_AT_name
        .byte   22                              # Abbrev [22] 0xdb:0x9 DW_TAG_template_type_parameter
        .long   247                             # DW_AT_type
        .long   .Linfo_string13                 # DW_AT_name
        .byte   23                              # Abbrev [23] 0xe4:0x6 DW_TAG_inheritance
        .long   257                             # DW_AT_type
        .byte   0                               # DW_AT_data_member_location
        .byte   24                              # Abbrev [24] 0xea:0xc DW_TAG_typedef
        .long   247                             # DW_AT_type
        .long   .Linfo_string16                 # DW_AT_name
        .byte   2                               # DW_AT_decl_file
        .short  2411                            # DW_AT_decl_line
        .byte   0                               # End Of Children Mark
        .byte   25                              # Abbrev [25] 0xf7:0xa DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string12                 # DW_AT_name
        .byte   1                               # DW_AT_byte_size
        .byte   2                               # DW_AT_decl_file
        .short  2406                            # DW_AT_decl_line
        .byte   20                              # Abbrev [20] 0x101:0x18 DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string14                 # DW_AT_name
        .byte   1                               # DW_AT_byte_size
        .byte   2                               # DW_AT_decl_file
        .short  2257                            # DW_AT_decl_line
        .byte   21                              # Abbrev [21] 0x10b:0x5 DW_TAG_template_type_parameter
        .long   .Linfo_string11                 # DW_AT_name
        .byte   26                              # Abbrev [26] 0x110:0x8 DW_TAG_typedef
        .long   .Linfo_string113                # DW_AT_name
        .byte   2                               # DW_AT_decl_file
        .short  2258                            # DW_AT_decl_line
        .byte   0                               # End Of Children Mark
        .byte   18                              # Abbrev [18] 0x119:0xd DW_TAG_subprogram
        .byte   19                              # Abbrev [19] 0x11a:0xb DW_TAG_typedef
        .long   234                             # DW_AT_type
        .long   .Linfo_string17                 # DW_AT_name
        .byte   3                               # DW_AT_decl_file
        .byte   95                              # DW_AT_decl_line
        .byte   0                               # End Of Children Mark
        .byte   18                              # Abbrev [18] 0x126:0xd DW_TAG_subprogram
        .byte   19                              # Abbrev [19] 0x127:0xb DW_TAG_typedef
        .long   234                             # DW_AT_type
        .long   .Linfo_string17                 # DW_AT_name
        .byte   3                               # DW_AT_decl_file
        .byte   95                              # DW_AT_decl_line
        .byte   0                               # End Of Children Mark
        .byte   8                               # Abbrev [8] 0x133:0x48b DW_TAG_namespace
        .long   .Linfo_string18                 # DW_AT_name
        .byte   8                               # Abbrev [8] 0x138:0x485 DW_TAG_namespace
        .long   .Linfo_string19                 # DW_AT_name
        .byte   27                              # Abbrev [27] 0x13d:0x1e DW_TAG_union_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string22                 # DW_AT_name
        .byte   1                               # DW_AT_byte_size
        .byte   1                               # DW_AT_decl_file
        .short  368                             # DW_AT_decl_line
        .byte   28                              # Abbrev [28] 0x147:0x5 DW_TAG_GNU_template_parameter_pack
        .long   .Linfo_string20                 # DW_AT_name
        .byte   15                              # Abbrev [15] 0x14c:0xe DW_TAG_subprogram
        .long   .Linfo_string21                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  370                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x154:0x5 DW_TAG_formal_parameter
        .long   2301                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   27                              # Abbrev [27] 0x15b:0x3e DW_TAG_union_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string34                 # DW_AT_name
        .byte   8                               # DW_AT_byte_size
        .byte   1                               # DW_AT_decl_file
        .short  377                             # DW_AT_decl_line
        .byte   29                              # Abbrev [29] 0x165:0xb DW_TAG_GNU_template_parameter_pack
        .long   .Linfo_string20                 # DW_AT_name
        .byte   30                              # Abbrev [30] 0x16a:0x5 DW_TAG_template_type_parameter
        .long   2306                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   11                              # Abbrev [11] 0x170:0xd DW_TAG_member
        .long   .Linfo_string23                 # DW_AT_name
        .long   409                             # DW_AT_type
        .byte   1                               # DW_AT_decl_file
        .short  407                             # DW_AT_decl_line
        .byte   0                               # DW_AT_data_member_location
        .byte   11                              # Abbrev [11] 0x17d:0xd DW_TAG_member
        .long   .Linfo_string33                 # DW_AT_name
        .long   317                             # DW_AT_type
        .byte   1                               # DW_AT_decl_file
        .short  408                             # DW_AT_decl_line
        .byte   0                               # DW_AT_data_member_location
        .byte   15                              # Abbrev [15] 0x18a:0xe DW_TAG_subprogram
        .long   .Linfo_string21                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  379                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x192:0x5 DW_TAG_formal_parameter
        .long   2358                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   31                              # Abbrev [31] 0x199:0x79 DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string32                 # DW_AT_name
        .byte   8                               # DW_AT_byte_size
        .byte   1                               # DW_AT_decl_file
        .byte   219                             # DW_AT_decl_line
        .byte   22                              # Abbrev [22] 0x1a2:0x9 DW_TAG_template_type_parameter
        .long   2306                            # DW_AT_type
        .long   .Linfo_string24                 # DW_AT_name
        .byte   32                              # Abbrev [32] 0x1ab:0x6 DW_TAG_template_value_parameter
        .long   2311                            # DW_AT_type
                                        # DW_AT_default_value
        .byte   1                               # DW_AT_const_value
        .byte   33                              # Abbrev [33] 0x1b1:0xc DW_TAG_member
        .long   .Linfo_string26                 # DW_AT_name
        .long   2306                            # DW_AT_type
        .byte   1                               # DW_AT_decl_file
        .byte   239                             # DW_AT_decl_line
        .byte   0                               # DW_AT_data_member_location
        .byte   34                              # Abbrev [34] 0x1bd:0x15 DW_TAG_subprogram
        .long   .Linfo_string27                 # DW_AT_linkage_name
        .long   .Linfo_string28                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .byte   227                             # DW_AT_decl_line
        .long   2318                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
                                        # DW_AT_reference
        .byte   13                              # Abbrev [13] 0x1cc:0x5 DW_TAG_formal_parameter
        .long   2328                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   34                              # Abbrev [34] 0x1d2:0x15 DW_TAG_subprogram
        .long   .Linfo_string29                 # DW_AT_linkage_name
        .long   .Linfo_string28                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .byte   230                             # DW_AT_decl_line
        .long   2338                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
                                        # DW_AT_reference
        .byte   13                              # Abbrev [13] 0x1e1:0x5 DW_TAG_formal_parameter
        .long   2343                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   35                              # Abbrev [35] 0x1e7:0x15 DW_TAG_subprogram
        .long   .Linfo_string30                 # DW_AT_linkage_name
        .long   .Linfo_string28                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .byte   233                             # DW_AT_decl_line
        .long   2348                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
                                        # DW_AT_rvalue_reference
        .byte   13                              # Abbrev [13] 0x1f6:0x5 DW_TAG_formal_parameter
        .long   2328                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   35                              # Abbrev [35] 0x1fc:0x15 DW_TAG_subprogram
        .long   .Linfo_string31                 # DW_AT_linkage_name
        .long   .Linfo_string28                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .byte   236                             # DW_AT_decl_line
        .long   2353                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
                                        # DW_AT_rvalue_reference
        .byte   13                              # Abbrev [13] 0x20b:0x5 DW_TAG_formal_parameter
        .long   2343                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   27                              # Abbrev [27] 0x212:0x43 DW_TAG_union_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string41                 # DW_AT_name
        .byte   8                               # DW_AT_byte_size
        .byte   1                               # DW_AT_decl_file
        .short  377                             # DW_AT_decl_line
        .byte   29                              # Abbrev [29] 0x21c:0x10 DW_TAG_GNU_template_parameter_pack
        .long   .Linfo_string20                 # DW_AT_name
        .byte   30                              # Abbrev [30] 0x221:0x5 DW_TAG_template_type_parameter
        .long   2363                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x226:0x5 DW_TAG_template_type_parameter
        .long   2306                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   11                              # Abbrev [11] 0x22c:0xd DW_TAG_member
        .long   .Linfo_string23                 # DW_AT_name
        .long   597                             # DW_AT_type
        .byte   1                               # DW_AT_decl_file
        .short  407                             # DW_AT_decl_line
        .byte   0                               # DW_AT_data_member_location
        .byte   11                              # Abbrev [11] 0x239:0xd DW_TAG_member
        .long   .Linfo_string33                 # DW_AT_name
        .long   347                             # DW_AT_type
        .byte   1                               # DW_AT_decl_file
        .short  408                             # DW_AT_decl_line
        .byte   0                               # DW_AT_data_member_location
        .byte   15                              # Abbrev [15] 0x246:0xe DW_TAG_subprogram
        .long   .Linfo_string21                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  379                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x24e:0x5 DW_TAG_formal_parameter
        .long   2415                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   31                              # Abbrev [31] 0x255:0x79 DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string40                 # DW_AT_name
        .byte   8                               # DW_AT_byte_size
        .byte   1                               # DW_AT_decl_file
        .byte   219                             # DW_AT_decl_line
        .byte   22                              # Abbrev [22] 0x25e:0x9 DW_TAG_template_type_parameter
        .long   2363                            # DW_AT_type
        .long   .Linfo_string24                 # DW_AT_name
        .byte   32                              # Abbrev [32] 0x267:0x6 DW_TAG_template_value_parameter
        .long   2311                            # DW_AT_type
                                        # DW_AT_default_value
        .byte   1                               # DW_AT_const_value
        .byte   33                              # Abbrev [33] 0x26d:0xc DW_TAG_member
        .long   .Linfo_string26                 # DW_AT_name
        .long   2363                            # DW_AT_type
        .byte   1                               # DW_AT_decl_file
        .byte   239                             # DW_AT_decl_line
        .byte   0                               # DW_AT_data_member_location
        .byte   34                              # Abbrev [34] 0x279:0x15 DW_TAG_subprogram
        .long   .Linfo_string36                 # DW_AT_linkage_name
        .long   .Linfo_string28                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .byte   227                             # DW_AT_decl_line
        .long   2375                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
                                        # DW_AT_reference
        .byte   13                              # Abbrev [13] 0x288:0x5 DW_TAG_formal_parameter
        .long   2385                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   34                              # Abbrev [34] 0x28e:0x15 DW_TAG_subprogram
        .long   .Linfo_string37                 # DW_AT_linkage_name
        .long   .Linfo_string28                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .byte   230                             # DW_AT_decl_line
        .long   2395                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
                                        # DW_AT_reference
        .byte   13                              # Abbrev [13] 0x29d:0x5 DW_TAG_formal_parameter
        .long   2400                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   35                              # Abbrev [35] 0x2a3:0x15 DW_TAG_subprogram
        .long   .Linfo_string38                 # DW_AT_linkage_name
        .long   .Linfo_string28                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .byte   233                             # DW_AT_decl_line
        .long   2405                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
                                        # DW_AT_rvalue_reference
        .byte   13                              # Abbrev [13] 0x2b2:0x5 DW_TAG_formal_parameter
        .long   2385                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   35                              # Abbrev [35] 0x2b8:0x15 DW_TAG_subprogram
        .long   .Linfo_string39                 # DW_AT_linkage_name
        .long   .Linfo_string28                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .byte   236                             # DW_AT_decl_line
        .long   2410                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
                                        # DW_AT_rvalue_reference
        .byte   13                              # Abbrev [13] 0x2c7:0x5 DW_TAG_formal_parameter
        .long   2400                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   27                              # Abbrev [27] 0x2ce:0x48 DW_TAG_union_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string48                 # DW_AT_name
        .byte   8                               # DW_AT_byte_size
        .byte   1                               # DW_AT_decl_file
        .short  377                             # DW_AT_decl_line
        .byte   29                              # Abbrev [29] 0x2d8:0x15 DW_TAG_GNU_template_parameter_pack
        .long   .Linfo_string20                 # DW_AT_name
        .byte   30                              # Abbrev [30] 0x2dd:0x5 DW_TAG_template_type_parameter
        .long   2420                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x2e2:0x5 DW_TAG_template_type_parameter
        .long   2363                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x2e7:0x5 DW_TAG_template_type_parameter
        .long   2306                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   11                              # Abbrev [11] 0x2ed:0xd DW_TAG_member
        .long   .Linfo_string23                 # DW_AT_name
        .long   790                             # DW_AT_type
        .byte   1                               # DW_AT_decl_file
        .short  407                             # DW_AT_decl_line
        .byte   0                               # DW_AT_data_member_location
        .byte   11                              # Abbrev [11] 0x2fa:0xd DW_TAG_member
        .long   .Linfo_string33                 # DW_AT_name
        .long   530                             # DW_AT_type
        .byte   1                               # DW_AT_decl_file
        .short  408                             # DW_AT_decl_line
        .byte   0                               # DW_AT_data_member_location
        .byte   15                              # Abbrev [15] 0x307:0xe DW_TAG_subprogram
        .long   .Linfo_string21                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  379                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x30f:0x5 DW_TAG_formal_parameter
        .long   2472                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   31                              # Abbrev [31] 0x316:0x79 DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string47                 # DW_AT_name
        .byte   8                               # DW_AT_byte_size
        .byte   1                               # DW_AT_decl_file
        .byte   219                             # DW_AT_decl_line
        .byte   22                              # Abbrev [22] 0x31f:0x9 DW_TAG_template_type_parameter
        .long   2420                            # DW_AT_type
        .long   .Linfo_string24                 # DW_AT_name
        .byte   32                              # Abbrev [32] 0x328:0x6 DW_TAG_template_value_parameter
        .long   2311                            # DW_AT_type
                                        # DW_AT_default_value
        .byte   1                               # DW_AT_const_value
        .byte   33                              # Abbrev [33] 0x32e:0xc DW_TAG_member
        .long   .Linfo_string26                 # DW_AT_name
        .long   2420                            # DW_AT_type
        .byte   1                               # DW_AT_decl_file
        .byte   239                             # DW_AT_decl_line
        .byte   0                               # DW_AT_data_member_location
        .byte   34                              # Abbrev [34] 0x33a:0x15 DW_TAG_subprogram
        .long   .Linfo_string43                 # DW_AT_linkage_name
        .long   .Linfo_string28                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .byte   227                             # DW_AT_decl_line
        .long   2432                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
                                        # DW_AT_reference
        .byte   13                              # Abbrev [13] 0x349:0x5 DW_TAG_formal_parameter
        .long   2442                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   34                              # Abbrev [34] 0x34f:0x15 DW_TAG_subprogram
        .long   .Linfo_string44                 # DW_AT_linkage_name
        .long   .Linfo_string28                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .byte   230                             # DW_AT_decl_line
        .long   2452                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
                                        # DW_AT_reference
        .byte   13                              # Abbrev [13] 0x35e:0x5 DW_TAG_formal_parameter
        .long   2457                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   35                              # Abbrev [35] 0x364:0x15 DW_TAG_subprogram
        .long   .Linfo_string45                 # DW_AT_linkage_name
        .long   .Linfo_string28                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .byte   233                             # DW_AT_decl_line
        .long   2462                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
                                        # DW_AT_rvalue_reference
        .byte   13                              # Abbrev [13] 0x373:0x5 DW_TAG_formal_parameter
        .long   2442                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   35                              # Abbrev [35] 0x379:0x15 DW_TAG_subprogram
        .long   .Linfo_string46                 # DW_AT_linkage_name
        .long   .Linfo_string28                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .byte   236                             # DW_AT_decl_line
        .long   2467                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
                                        # DW_AT_rvalue_reference
        .byte   13                              # Abbrev [13] 0x388:0x5 DW_TAG_formal_parameter
        .long   2457                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   20                              # Abbrev [20] 0x38f:0x86 DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string68                 # DW_AT_name
        .byte   16                              # DW_AT_byte_size
        .byte   1                               # DW_AT_decl_file
        .short  495                             # DW_AT_decl_line
        .byte   36                              # Abbrev [36] 0x399:0xa DW_TAG_template_value_parameter
        .long   2311                            # DW_AT_type
        .long   .Linfo_string49                 # DW_AT_name
        .byte   1                               # DW_AT_const_value
        .byte   29                              # Abbrev [29] 0x3a3:0x15 DW_TAG_GNU_template_parameter_pack
        .long   .Linfo_string20                 # DW_AT_name
        .byte   30                              # Abbrev [30] 0x3a8:0x5 DW_TAG_template_type_parameter
        .long   2420                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x3ad:0x5 DW_TAG_template_type_parameter
        .long   2363                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x3b2:0x5 DW_TAG_template_type_parameter
        .long   2306                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   11                              # Abbrev [11] 0x3b8:0xd DW_TAG_member
        .long   .Linfo_string50                 # DW_AT_name
        .long   718                             # DW_AT_type
        .byte   1                               # DW_AT_decl_file
        .short  527                             # DW_AT_decl_line
        .byte   0                               # DW_AT_data_member_location
        .byte   11                              # Abbrev [11] 0x3c5:0xd DW_TAG_member
        .long   .Linfo_string51                 # DW_AT_name
        .long   978                             # DW_AT_type
        .byte   1                               # DW_AT_decl_file
        .short  529                             # DW_AT_decl_line
        .byte   8                               # DW_AT_data_member_location
        .byte   24                              # Abbrev [24] 0x3d2:0xc DW_TAG_typedef
        .long   1045                            # DW_AT_type
        .long   .Linfo_string62                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  528                             # DW_AT_decl_line
        .byte   15                              # Abbrev [15] 0x3de:0xe DW_TAG_subprogram
        .long   .Linfo_string63                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  498                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x3e6:0x5 DW_TAG_formal_parameter
        .long   2499                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   37                              # Abbrev [37] 0x3ec:0x12 DW_TAG_subprogram
        .long   .Linfo_string64                 # DW_AT_linkage_name
        .long   .Linfo_string65                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  510                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x3f8:0x5 DW_TAG_formal_parameter
        .long   2499                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   38                              # Abbrev [38] 0x3fe:0x16 DW_TAG_subprogram
        .long   .Linfo_string66                 # DW_AT_linkage_name
        .long   .Linfo_string67                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  514                             # DW_AT_decl_line
        .long   2311                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x40e:0x5 DW_TAG_formal_parameter
        .long   2504                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   24                              # Abbrev [24] 0x415:0xc DW_TAG_typedef
        .long   1530                            # DW_AT_type
        .long   .Linfo_string61                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  443                             # DW_AT_decl_line
        .byte   20                              # Abbrev [20] 0x421:0x90 DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string82                 # DW_AT_name
        .byte   16                              # DW_AT_byte_size
        .byte   1                               # DW_AT_decl_file
        .short  734                             # DW_AT_decl_line
        .byte   29                              # Abbrev [29] 0x42b:0x15 DW_TAG_GNU_template_parameter_pack
        .long   .Linfo_string20                 # DW_AT_name
        .byte   30                              # Abbrev [30] 0x430:0x5 DW_TAG_template_type_parameter
        .long   2420                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x435:0x5 DW_TAG_template_type_parameter
        .long   2363                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x43a:0x5 DW_TAG_template_type_parameter
        .long   2306                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   23                              # Abbrev [23] 0x440:0x6 DW_TAG_inheritance
        .long   1201                            # DW_AT_type
        .byte   0                               # DW_AT_data_member_location
        .byte   15                              # Abbrev [15] 0x446:0xe DW_TAG_subprogram
        .long   .Linfo_string78                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  739                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x44e:0x5 DW_TAG_formal_parameter
        .long   2514                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   15                              # Abbrev [15] 0x454:0x13 DW_TAG_subprogram
        .long   .Linfo_string78                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  748                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x45c:0x5 DW_TAG_formal_parameter
        .long   2514                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   16                              # Abbrev [16] 0x461:0x5 DW_TAG_formal_parameter
        .long   2519                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   15                              # Abbrev [15] 0x467:0x13 DW_TAG_subprogram
        .long   .Linfo_string78                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  749                             # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x46f:0x5 DW_TAG_formal_parameter
        .long   2514                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   16                              # Abbrev [16] 0x474:0x5 DW_TAG_formal_parameter
        .long   2529                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   38                              # Abbrev [38] 0x47a:0x1b DW_TAG_subprogram
        .long   .Linfo_string79                 # DW_AT_linkage_name
        .long   .Linfo_string80                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  750                             # DW_AT_decl_line
        .long   2534                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x48a:0x5 DW_TAG_formal_parameter
        .long   2514                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   16                              # Abbrev [16] 0x48f:0x5 DW_TAG_formal_parameter
        .long   2519                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   38                              # Abbrev [38] 0x495:0x1b DW_TAG_subprogram
        .long   .Linfo_string81                 # DW_AT_linkage_name
        .long   .Linfo_string80                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  751                             # DW_AT_decl_line
        .long   2534                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x4a5:0x5 DW_TAG_formal_parameter
        .long   2514                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   16                              # Abbrev [16] 0x4aa:0x5 DW_TAG_formal_parameter
        .long   2529                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   24                              # Abbrev [24] 0x4b1:0xc DW_TAG_typedef
        .long   1213                            # DW_AT_type
        .long   .Linfo_string77                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  730                             # DW_AT_decl_line
        .byte   20                              # Abbrev [20] 0x4bd:0x2c DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string76                 # DW_AT_name
        .byte   16                              # DW_AT_byte_size
        .byte   1                               # DW_AT_decl_file
        .short  723                             # DW_AT_decl_line
        .byte   39                              # Abbrev [39] 0x4c7:0x6 DW_TAG_template_value_parameter
        .long   2311                            # DW_AT_type
        .byte   1                               # DW_AT_const_value
        .byte   29                              # Abbrev [29] 0x4cd:0x15 DW_TAG_GNU_template_parameter_pack
        .long   .Linfo_string20                 # DW_AT_name
        .byte   30                              # Abbrev [30] 0x4d2:0x5 DW_TAG_template_type_parameter
        .long   2420                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x4d7:0x5 DW_TAG_template_type_parameter
        .long   2363                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x4dc:0x5 DW_TAG_template_type_parameter
        .long   2306                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   23                              # Abbrev [23] 0x4e2:0x6 DW_TAG_inheritance
        .long   1257                            # DW_AT_type
        .byte   0                               # DW_AT_data_member_location
        .byte   0                               # End Of Children Mark
        .byte   24                              # Abbrev [24] 0x4e9:0xc DW_TAG_typedef
        .long   1269                            # DW_AT_type
        .long   .Linfo_string75                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  676                             # DW_AT_decl_line
        .byte   20                              # Abbrev [20] 0x4f5:0x2c DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string74                 # DW_AT_name
        .byte   16                              # DW_AT_byte_size
        .byte   1                               # DW_AT_decl_file
        .short  669                             # DW_AT_decl_line
        .byte   39                              # Abbrev [39] 0x4ff:0x6 DW_TAG_template_value_parameter
        .long   2311                            # DW_AT_type
        .byte   1                               # DW_AT_const_value
        .byte   29                              # Abbrev [29] 0x505:0x15 DW_TAG_GNU_template_parameter_pack
        .long   .Linfo_string20                 # DW_AT_name
        .byte   30                              # Abbrev [30] 0x50a:0x5 DW_TAG_template_type_parameter
        .long   2420                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x50f:0x5 DW_TAG_template_type_parameter
        .long   2363                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x514:0x5 DW_TAG_template_type_parameter
        .long   2306                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   23                              # Abbrev [23] 0x51a:0x6 DW_TAG_inheritance
        .long   1313                            # DW_AT_type
        .byte   0                               # DW_AT_data_member_location
        .byte   0                               # End Of Children Mark
        .byte   24                              # Abbrev [24] 0x521:0xc DW_TAG_typedef
        .long   1325                            # DW_AT_type
        .long   .Linfo_string73                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  624                             # DW_AT_decl_line
        .byte   20                              # Abbrev [20] 0x52d:0x2c DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string72                 # DW_AT_name
        .byte   16                              # DW_AT_byte_size
        .byte   1                               # DW_AT_decl_file
        .short  617                             # DW_AT_decl_line
        .byte   39                              # Abbrev [39] 0x537:0x6 DW_TAG_template_value_parameter
        .long   2311                            # DW_AT_type
        .byte   1                               # DW_AT_const_value
        .byte   29                              # Abbrev [29] 0x53d:0x15 DW_TAG_GNU_template_parameter_pack
        .long   .Linfo_string20                 # DW_AT_name
        .byte   30                              # Abbrev [30] 0x542:0x5 DW_TAG_template_type_parameter
        .long   2420                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x547:0x5 DW_TAG_template_type_parameter
        .long   2363                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x54c:0x5 DW_TAG_template_type_parameter
        .long   2306                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   23                              # Abbrev [23] 0x552:0x6 DW_TAG_inheritance
        .long   1369                            # DW_AT_type
        .byte   0                               # DW_AT_data_member_location
        .byte   0                               # End Of Children Mark
        .byte   24                              # Abbrev [24] 0x559:0xc DW_TAG_typedef
        .long   1381                            # DW_AT_type
        .long   .Linfo_string71                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  586                             # DW_AT_decl_line
        .byte   20                              # Abbrev [20] 0x565:0x2c DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string70                 # DW_AT_name
        .byte   16                              # DW_AT_byte_size
        .byte   1                               # DW_AT_decl_file
        .short  579                             # DW_AT_decl_line
        .byte   39                              # Abbrev [39] 0x56f:0x6 DW_TAG_template_value_parameter
        .long   2311                            # DW_AT_type
        .byte   1                               # DW_AT_const_value
        .byte   29                              # Abbrev [29] 0x575:0x15 DW_TAG_GNU_template_parameter_pack
        .long   .Linfo_string20                 # DW_AT_name
        .byte   30                              # Abbrev [30] 0x57a:0x5 DW_TAG_template_type_parameter
        .long   2420                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x57f:0x5 DW_TAG_template_type_parameter
        .long   2363                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x584:0x5 DW_TAG_template_type_parameter
        .long   2306                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   23                              # Abbrev [23] 0x58a:0x6 DW_TAG_inheritance
        .long   1425                            # DW_AT_type
        .byte   0                               # DW_AT_data_member_location
        .byte   0                               # End Of Children Mark
        .byte   24                              # Abbrev [24] 0x591:0xc DW_TAG_typedef
        .long   911                             # DW_AT_type
        .long   .Linfo_string69                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  546                             # DW_AT_decl_line
        .byte   40                              # Abbrev [40] 0x59d:0x5 DW_TAG_structure_type
        .long   .Linfo_string107                # DW_AT_name
                                        # DW_AT_declaration
        .byte   24                              # Abbrev [24] 0x5a2:0xc DW_TAG_typedef
        .long   2268                            # DW_AT_type
        .long   .Linfo_string122                # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1094                            # DW_AT_decl_line
        .byte   40                              # Abbrev [40] 0x5ae:0x5 DW_TAG_structure_type
        .long   .Linfo_string124                # DW_AT_name
                                        # DW_AT_declaration
        .byte   41                              # Abbrev [41] 0x5b3:0x9 DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string125                # DW_AT_name
        .byte   1                               # DW_AT_byte_size
        .byte   1                               # DW_AT_decl_file
        .byte   163                             # DW_AT_decl_line
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   31                              # Abbrev [31] 0x5be:0x5d DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string60                 # DW_AT_name
        .byte   1                               # DW_AT_byte_size
        .byte   2                               # DW_AT_decl_file
        .byte   62                              # DW_AT_decl_line
        .byte   22                              # Abbrev [22] 0x5c7:0x9 DW_TAG_template_type_parameter
        .long   2477                            # DW_AT_type
        .long   .Linfo_string11                 # DW_AT_name
        .byte   36                              # Abbrev [36] 0x5d0:0xa DW_TAG_template_value_parameter
        .long   2477                            # DW_AT_type
        .long   .Linfo_string53                 # DW_AT_name
        .byte   3                               # DW_AT_const_value
        .byte   42                              # Abbrev [42] 0x5da:0xb DW_TAG_member
        .long   .Linfo_string54                 # DW_AT_name
        .long   2484                            # DW_AT_type
        .byte   2                               # DW_AT_decl_file
        .byte   64                              # DW_AT_decl_line
                                        # DW_AT_external
                                        # DW_AT_declaration
        .byte   43                              # Abbrev [43] 0x5e5:0x15 DW_TAG_subprogram
        .long   .Linfo_string55                 # DW_AT_linkage_name
        .long   .Linfo_string56                 # DW_AT_name
        .byte   2                               # DW_AT_decl_file
        .byte   67                              # DW_AT_decl_line
        .long   1530                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x5f4:0x5 DW_TAG_formal_parameter
        .long   2489                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   19                              # Abbrev [19] 0x5fa:0xb DW_TAG_typedef
        .long   2477                            # DW_AT_type
        .long   .Linfo_string57                 # DW_AT_name
        .byte   2                               # DW_AT_decl_file
        .byte   65                              # DW_AT_decl_line
        .byte   43                              # Abbrev [43] 0x605:0x15 DW_TAG_subprogram
        .long   .Linfo_string58                 # DW_AT_linkage_name
        .long   .Linfo_string59                 # DW_AT_name
        .byte   2                               # DW_AT_decl_file
        .byte   72                              # DW_AT_decl_line
        .long   1530                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x614:0x5 DW_TAG_formal_parameter
        .long   2489                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   31                              # Abbrev [31] 0x61b:0x94 DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string106                # DW_AT_name
        .byte   1                               # DW_AT_byte_size
        .byte   4                               # DW_AT_decl_file
        .byte   53                              # DW_AT_decl_line
        .byte   36                              # Abbrev [36] 0x624:0xa DW_TAG_template_value_parameter
        .long   2311                            # DW_AT_type
        .long   .Linfo_string83                 # DW_AT_name
        .byte   1                               # DW_AT_const_value
        .byte   22                              # Abbrev [22] 0x62e:0x9 DW_TAG_template_type_parameter
        .long   1711                            # DW_AT_type
        .long   .Linfo_string13                 # DW_AT_name
        .byte   44                              # Abbrev [44] 0x637:0xd DW_TAG_subprogram
        .long   .Linfo_string102                # DW_AT_name
        .byte   4                               # DW_AT_decl_file
        .byte   55                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x63e:0x5 DW_TAG_formal_parameter
        .long   2576                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   44                              # Abbrev [44] 0x644:0x12 DW_TAG_subprogram
        .long   .Linfo_string102                # DW_AT_name
        .byte   4                               # DW_AT_decl_file
        .byte   56                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x64b:0x5 DW_TAG_formal_parameter
        .long   2576                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   16                              # Abbrev [16] 0x650:0x5 DW_TAG_formal_parameter
        .long   2581                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   44                              # Abbrev [44] 0x656:0x12 DW_TAG_subprogram
        .long   .Linfo_string102                # DW_AT_name
        .byte   4                               # DW_AT_decl_file
        .byte   58                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x65d:0x5 DW_TAG_formal_parameter
        .long   2576                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   16                              # Abbrev [16] 0x662:0x5 DW_TAG_formal_parameter
        .long   2591                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   43                              # Abbrev [43] 0x668:0x1a DW_TAG_subprogram
        .long   .Linfo_string103                # DW_AT_linkage_name
        .long   .Linfo_string80                 # DW_AT_name
        .byte   4                               # DW_AT_decl_file
        .byte   61                              # DW_AT_decl_line
        .long   2596                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x677:0x5 DW_TAG_formal_parameter
        .long   2576                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   16                              # Abbrev [16] 0x67c:0x5 DW_TAG_formal_parameter
        .long   2581                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   43                              # Abbrev [43] 0x682:0x1a DW_TAG_subprogram
        .long   .Linfo_string104                # DW_AT_linkage_name
        .long   .Linfo_string80                 # DW_AT_name
        .byte   4                               # DW_AT_decl_file
        .byte   63                              # DW_AT_decl_line
        .long   2596                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   13                              # Abbrev [13] 0x691:0x5 DW_TAG_formal_parameter
        .long   2576                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   16                              # Abbrev [16] 0x696:0x5 DW_TAG_formal_parameter
        .long   2591                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   45                              # Abbrev [45] 0x69c:0x12 DW_TAG_subprogram
        .long   .Linfo_string102                # DW_AT_name
        .byte   4                               # DW_AT_decl_file
        .byte   67                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
                                        # DW_AT_explicit
        .byte   13                              # Abbrev [13] 0x6a3:0x5 DW_TAG_formal_parameter
        .long   2576                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   16                              # Abbrev [16] 0x6a8:0x5 DW_TAG_formal_parameter
        .long   2028                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   46                              # Abbrev [46] 0x6af:0xf6 DW_TAG_class_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string101                # DW_AT_name
        .byte   16                              # DW_AT_byte_size
        .byte   1                               # DW_AT_decl_file
        .short  1337                            # DW_AT_decl_line
        .byte   29                              # Abbrev [29] 0x6b9:0x15 DW_TAG_GNU_template_parameter_pack
        .long   .Linfo_string20                 # DW_AT_name
        .byte   30                              # Abbrev [30] 0x6be:0x5 DW_TAG_template_type_parameter
        .long   2420                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x6c3:0x5 DW_TAG_template_type_parameter
        .long   2363                            # DW_AT_type
        .byte   30                              # Abbrev [30] 0x6c8:0x5 DW_TAG_template_type_parameter
        .long   2306                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   23                              # Abbrev [23] 0x6ce:0x6 DW_TAG_inheritance
        .long   1057                            # DW_AT_type
        .byte   0                               # DW_AT_data_member_location
        .byte   23                              # Abbrev [23] 0x6d4:0x6 DW_TAG_inheritance
        .long   1563                            # DW_AT_type
        .byte   0                               # DW_AT_data_member_location
        .byte   23                              # Abbrev [23] 0x6da:0x6 DW_TAG_inheritance
        .long   1957                            # DW_AT_type
        .byte   0                               # DW_AT_data_member_location
        .byte   12                              # Abbrev [12] 0x6e0:0xf DW_TAG_subprogram
        .long   .Linfo_string89                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1403                            # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
        .byte   13                              # Abbrev [13] 0x6e9:0x5 DW_TAG_formal_parameter
        .long   2539                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   12                              # Abbrev [12] 0x6ef:0x14 DW_TAG_subprogram
        .long   .Linfo_string89                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1404                            # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
        .byte   13                              # Abbrev [13] 0x6f8:0x5 DW_TAG_formal_parameter
        .long   2539                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   16                              # Abbrev [16] 0x6fd:0x5 DW_TAG_formal_parameter
        .long   2544                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   12                              # Abbrev [12] 0x703:0x14 DW_TAG_subprogram
        .long   .Linfo_string89                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1405                            # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
        .byte   13                              # Abbrev [13] 0x70c:0x5 DW_TAG_formal_parameter
        .long   2539                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   16                              # Abbrev [16] 0x711:0x5 DW_TAG_formal_parameter
        .long   2554                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   47                              # Abbrev [47] 0x717:0x1c DW_TAG_subprogram
        .long   .Linfo_string90                 # DW_AT_linkage_name
        .long   .Linfo_string80                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1406                            # DW_AT_decl_line
        .long   2559                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
        .byte   13                              # Abbrev [13] 0x728:0x5 DW_TAG_formal_parameter
        .long   2539                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   16                              # Abbrev [16] 0x72d:0x5 DW_TAG_formal_parameter
        .long   2544                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   47                              # Abbrev [47] 0x733:0x1c DW_TAG_subprogram
        .long   .Linfo_string91                 # DW_AT_linkage_name
        .long   .Linfo_string80                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1407                            # DW_AT_decl_line
        .long   2559                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
        .byte   13                              # Abbrev [13] 0x744:0x5 DW_TAG_formal_parameter
        .long   2539                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   16                              # Abbrev [16] 0x749:0x5 DW_TAG_formal_parameter
        .long   2554                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   12                              # Abbrev [12] 0x74f:0xf DW_TAG_subprogram
        .long   .Linfo_string92                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1408                            # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
        .byte   13                              # Abbrev [13] 0x758:0x5 DW_TAG_formal_parameter
        .long   2539                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   47                              # Abbrev [47] 0x75e:0x17 DW_TAG_subprogram
        .long   .Linfo_string93                 # DW_AT_linkage_name
        .long   .Linfo_string94                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1594                            # DW_AT_decl_line
        .long   2311                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
        .byte   13                              # Abbrev [13] 0x76f:0x5 DW_TAG_formal_parameter
        .long   2564                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   47                              # Abbrev [47] 0x775:0x17 DW_TAG_subprogram
        .long   .Linfo_string95                 # DW_AT_linkage_name
        .long   .Linfo_string96                 # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1597                            # DW_AT_decl_line
        .long   2016                            # DW_AT_type
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
        .byte   13                              # Abbrev [13] 0x786:0x5 DW_TAG_formal_parameter
        .long   2564                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   48                              # Abbrev [48] 0x78c:0x18 DW_TAG_subprogram
        .long   .Linfo_string99                 # DW_AT_linkage_name
        .long   .Linfo_string100                # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1610                            # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
        .byte   1                               # DW_AT_accessibility
                                        # DW_ACCESS_public
        .byte   13                              # Abbrev [13] 0x799:0x5 DW_TAG_formal_parameter
        .long   2539                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   16                              # Abbrev [16] 0x79e:0x5 DW_TAG_formal_parameter
        .long   2559                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   31                              # Abbrev [31] 0x7a5:0x3b DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string88                 # DW_AT_name
        .byte   1                               # DW_AT_byte_size
        .byte   4                               # DW_AT_decl_file
        .byte   87                              # DW_AT_decl_line
        .byte   36                              # Abbrev [36] 0x7ae:0xa DW_TAG_template_value_parameter
        .long   2311                            # DW_AT_type
        .long   .Linfo_string84                 # DW_AT_name
        .byte   1                               # DW_AT_const_value
        .byte   36                              # Abbrev [36] 0x7b8:0xa DW_TAG_template_value_parameter
        .long   2311                            # DW_AT_type
        .long   .Linfo_string85                 # DW_AT_name
        .byte   1                               # DW_AT_const_value
        .byte   36                              # Abbrev [36] 0x7c2:0xa DW_TAG_template_value_parameter
        .long   2311                            # DW_AT_type
        .long   .Linfo_string86                 # DW_AT_name
        .byte   1                               # DW_AT_const_value
        .byte   36                              # Abbrev [36] 0x7cc:0xa DW_TAG_template_value_parameter
        .long   2311                            # DW_AT_type
        .long   .Linfo_string87                 # DW_AT_name
        .byte   1                               # DW_AT_const_value
        .byte   22                              # Abbrev [22] 0x7d6:0x9 DW_TAG_template_type_parameter
        .long   1711                            # DW_AT_type
        .long   .Linfo_string13                 # DW_AT_name
        .byte   0                               # End Of Children Mark
        .byte   24                              # Abbrev [24] 0x7e0:0xc DW_TAG_typedef
        .long   2569                            # DW_AT_type
        .long   .Linfo_string98                 # DW_AT_name
        .byte   5                               # DW_AT_decl_file
        .short  310                             # DW_AT_decl_line
        .byte   31                              # Abbrev [31] 0x7ec:0x17 DW_TAG_structure_type
        .byte   5                               # DW_AT_calling_convention
        .long   .Linfo_string105                # DW_AT_name
        .byte   1                               # DW_AT_byte_size
        .byte   4                               # DW_AT_decl_file
        .byte   42                              # DW_AT_decl_line
        .byte   45                              # Abbrev [45] 0x7f5:0xd DW_TAG_subprogram
        .long   .Linfo_string105                # DW_AT_name
        .byte   4                               # DW_AT_decl_file
        .byte   44                              # DW_AT_decl_line
                                        # DW_AT_declaration
                                        # DW_AT_external
                                        # DW_AT_explicit
        .byte   13                              # Abbrev [13] 0x7fc:0x5 DW_TAG_formal_parameter
        .long   2601                            # DW_AT_type
                                        # DW_AT_artificial
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   49                              # Abbrev [49] 0x803:0x81 DW_TAG_subprogram
        .long   .Linfo_string111                # DW_AT_linkage_name
        .long   .Linfo_string112                # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1732                            # DW_AT_decl_line
        .long   272                             # DW_AT_type
        .byte   1                               # DW_AT_inline
        .byte   22                              # Abbrev [22] 0x814:0x9 DW_TAG_template_type_parameter
        .long   1437                            # DW_AT_type
        .long   .Linfo_string108                # DW_AT_name
        .byte   22                              # Abbrev [22] 0x81d:0x9 DW_TAG_template_type_parameter
        .long   2631                            # DW_AT_type
        .long   .Linfo_string109                # DW_AT_name
        .byte   29                              # Abbrev [29] 0x826:0xb DW_TAG_GNU_template_parameter_pack
        .long   .Linfo_string110                # DW_AT_name
        .byte   30                              # Abbrev [30] 0x82b:0x5 DW_TAG_template_type_parameter
        .long   2559                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   50                              # Abbrev [50] 0x831:0xc DW_TAG_formal_parameter
        .long   .Linfo_string114                # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1732                            # DW_AT_decl_line
        .long   2754                            # DW_AT_type
        .byte   50                              # Abbrev [50] 0x83d:0xc DW_TAG_formal_parameter
        .long   .Linfo_string115                # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1732                            # DW_AT_decl_line
        .long   2559                            # DW_AT_type
        .byte   51                              # Abbrev [51] 0x849:0x3a DW_TAG_lexical_block
        .byte   52                              # Abbrev [52] 0x84a:0xc DW_TAG_variable
        .long   .Linfo_string116                # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1744                            # DW_AT_decl_line
        .long   2759                            # DW_AT_type
        .byte   52                              # Abbrev [52] 0x856:0xc DW_TAG_variable
        .long   .Linfo_string117                # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1749                            # DW_AT_decl_line
        .long   2759                            # DW_AT_type
        .byte   51                              # Abbrev [51] 0x862:0x20 DW_TAG_lexical_block
        .byte   52                              # Abbrev [52] 0x863:0xc DW_TAG_variable
        .long   .Linfo_string118                # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1764                            # DW_AT_decl_line
        .long   2559                            # DW_AT_type
        .byte   51                              # Abbrev [51] 0x86f:0x12 DW_TAG_lexical_block
        .byte   53                              # Abbrev [53] 0x870:0x8 DW_TAG_imported_declaration
        .byte   1                               # DW_AT_decl_file
        .short  1803                            # DW_AT_decl_line
        .long   1454                            # DW_AT_import
        .byte   53                              # Abbrev [53] 0x878:0x8 DW_TAG_imported_declaration
        .byte   1                               # DW_AT_decl_file
        .short  1804                            # DW_AT_decl_line
        .long   1459                            # DW_AT_import
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   49                              # Abbrev [49] 0x884:0x58 DW_TAG_subprogram
        .long   .Linfo_string119                # DW_AT_linkage_name
        .long   .Linfo_string120                # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1827                            # DW_AT_decl_line
        .long   1442                            # DW_AT_type
        .byte   1                               # DW_AT_inline
        .byte   22                              # Abbrev [22] 0x895:0x9 DW_TAG_template_type_parameter
        .long   2631                            # DW_AT_type
        .long   .Linfo_string109                # DW_AT_name
        .byte   29                              # Abbrev [29] 0x89e:0xb DW_TAG_GNU_template_parameter_pack
        .long   .Linfo_string110                # DW_AT_name
        .byte   30                              # Abbrev [30] 0x8a3:0x5 DW_TAG_template_type_parameter
        .long   2559                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   50                              # Abbrev [50] 0x8a9:0xc DW_TAG_formal_parameter
        .long   .Linfo_string114                # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1827                            # DW_AT_decl_line
        .long   2754                            # DW_AT_type
        .byte   50                              # Abbrev [50] 0x8b5:0xc DW_TAG_formal_parameter
        .long   .Linfo_string115                # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1827                            # DW_AT_decl_line
        .long   2559                            # DW_AT_type
        .byte   51                              # Abbrev [51] 0x8c1:0xe DW_TAG_lexical_block
        .byte   52                              # Abbrev [52] 0x8c2:0xc DW_TAG_variable
        .long   .Linfo_string123                # DW_AT_name
        .byte   1                               # DW_AT_decl_file
        .short  1843                            # DW_AT_decl_line
        .long   2764                            # DW_AT_type
        .byte   0                               # End Of Children Mark
        .byte   54                              # Abbrev [54] 0x8cf:0xc DW_TAG_imported_declaration
        .byte   1                               # DW_AT_decl_file
        .short  1829                            # DW_AT_decl_line
        .long   312                             # DW_AT_import
        .long   .Linfo_string19                 # DW_AT_name
        .byte   0                               # End Of Children Mark
        .byte   24                              # Abbrev [24] 0x8dc:0xc DW_TAG_typedef
        .long   272                             # DW_AT_type
        .long   .Linfo_string121                # DW_AT_name
        .byte   2                               # DW_AT_decl_file
        .short  3073                            # DW_AT_decl_line
        .byte   0                               # End Of Children Mark
        .byte   55                              # Abbrev [55] 0x8e9:0x5 DW_TAG_pointer_type
        .long   62                              # DW_AT_type
        .byte   55                              # Abbrev [55] 0x8ee:0x5 DW_TAG_pointer_type
        .long   86                              # DW_AT_type
        .byte   55                              # Abbrev [55] 0x8f3:0x5 DW_TAG_pointer_type
        .long   2296                            # DW_AT_type
        .byte   5                               # Abbrev [5] 0x8f8:0x5 DW_TAG_const_type
        .long   86                              # DW_AT_type
        .byte   55                              # Abbrev [55] 0x8fd:0x5 DW_TAG_pointer_type
        .long   317                             # DW_AT_type
        .byte   55                              # Abbrev [55] 0x902:0x5 DW_TAG_pointer_type
        .long   67                              # DW_AT_type
        .byte   6                               # Abbrev [6] 0x907:0x7 DW_TAG_base_type
        .long   .Linfo_string25                 # DW_AT_name
        .byte   2                               # DW_AT_encoding
        .byte   1                               # DW_AT_byte_size
        .byte   56                              # Abbrev [56] 0x90e:0x5 DW_TAG_reference_type
        .long   2323                            # DW_AT_type
        .byte   5                               # Abbrev [5] 0x913:0x5 DW_TAG_const_type
        .long   2306                            # DW_AT_type
        .byte   55                              # Abbrev [55] 0x918:0x5 DW_TAG_pointer_type
        .long   2333                            # DW_AT_type
        .byte   5                               # Abbrev [5] 0x91d:0x5 DW_TAG_const_type
        .long   409                             # DW_AT_type
        .byte   56                              # Abbrev [56] 0x922:0x5 DW_TAG_reference_type
        .long   2306                            # DW_AT_type
        .byte   55                              # Abbrev [55] 0x927:0x5 DW_TAG_pointer_type
        .long   409                             # DW_AT_type
        .byte   57                              # Abbrev [57] 0x92c:0x5 DW_TAG_rvalue_reference_type
        .long   2323                            # DW_AT_type
        .byte   57                              # Abbrev [57] 0x931:0x5 DW_TAG_rvalue_reference_type
        .long   2306                            # DW_AT_type
        .byte   55                              # Abbrev [55] 0x936:0x5 DW_TAG_pointer_type
        .long   347                             # DW_AT_type
        .byte   55                              # Abbrev [55] 0x93b:0x5 DW_TAG_pointer_type
        .long   2368                            # DW_AT_type
        .byte   6                               # Abbrev [6] 0x940:0x7 DW_TAG_base_type
        .long   .Linfo_string35                 # DW_AT_name
        .byte   5                               # DW_AT_encoding
        .byte   8                               # DW_AT_byte_size
        .byte   56                              # Abbrev [56] 0x947:0x5 DW_TAG_reference_type
        .long   2380                            # DW_AT_type
        .byte   5                               # Abbrev [5] 0x94c:0x5 DW_TAG_const_type
        .long   2363                            # DW_AT_type
        .byte   55                              # Abbrev [55] 0x951:0x5 DW_TAG_pointer_type
        .long   2390                            # DW_AT_type
        .byte   5                               # Abbrev [5] 0x956:0x5 DW_TAG_const_type
        .long   597                             # DW_AT_type
        .byte   56                              # Abbrev [56] 0x95b:0x5 DW_TAG_reference_type
        .long   2363                            # DW_AT_type
        .byte   55                              # Abbrev [55] 0x960:0x5 DW_TAG_pointer_type
        .long   597                             # DW_AT_type
        .byte   57                              # Abbrev [57] 0x965:0x5 DW_TAG_rvalue_reference_type
        .long   2380                            # DW_AT_type
        .byte   57                              # Abbrev [57] 0x96a:0x5 DW_TAG_rvalue_reference_type
        .long   2363                            # DW_AT_type
        .byte   55                              # Abbrev [55] 0x96f:0x5 DW_TAG_pointer_type
        .long   530                             # DW_AT_type
        .byte   55                              # Abbrev [55] 0x974:0x5 DW_TAG_pointer_type
        .long   2425                            # DW_AT_type
        .byte   6                               # Abbrev [6] 0x979:0x7 DW_TAG_base_type
        .long   .Linfo_string42                 # DW_AT_name
        .byte   5                               # DW_AT_encoding
        .byte   4                               # DW_AT_byte_size
        .byte   56                              # Abbrev [56] 0x980:0x5 DW_TAG_reference_type
        .long   2437                            # DW_AT_type
        .byte   5                               # Abbrev [5] 0x985:0x5 DW_TAG_const_type
        .long   2420                            # DW_AT_type
        .byte   55                              # Abbrev [55] 0x98a:0x5 DW_TAG_pointer_type
        .long   2447                            # DW_AT_type
        .byte   5                               # Abbrev [5] 0x98f:0x5 DW_TAG_const_type
        .long   790                             # DW_AT_type
        .byte   56                              # Abbrev [56] 0x994:0x5 DW_TAG_reference_type
        .long   2420                            # DW_AT_type
        .byte   55                              # Abbrev [55] 0x999:0x5 DW_TAG_pointer_type
        .long   790                             # DW_AT_type
        .byte   57                              # Abbrev [57] 0x99e:0x5 DW_TAG_rvalue_reference_type
        .long   2437                            # DW_AT_type
        .byte   57                              # Abbrev [57] 0x9a3:0x5 DW_TAG_rvalue_reference_type
        .long   2420                            # DW_AT_type
        .byte   55                              # Abbrev [55] 0x9a8:0x5 DW_TAG_pointer_type
        .long   718                             # DW_AT_type
        .byte   6                               # Abbrev [6] 0x9ad:0x7 DW_TAG_base_type
        .long   .Linfo_string52                 # DW_AT_name
        .byte   8                               # DW_AT_encoding
        .byte   1                               # DW_AT_byte_size
        .byte   5                               # Abbrev [5] 0x9b4:0x5 DW_TAG_const_type
        .long   2477                            # DW_AT_type
        .byte   55                              # Abbrev [55] 0x9b9:0x5 DW_TAG_pointer_type
        .long   2494                            # DW_AT_type
        .byte   5                               # Abbrev [5] 0x9be:0x5 DW_TAG_const_type
        .long   1470                            # DW_AT_type
        .byte   55                              # Abbrev [55] 0x9c3:0x5 DW_TAG_pointer_type
        .long   911                             # DW_AT_type
        .byte   55                              # Abbrev [55] 0x9c8:0x5 DW_TAG_pointer_type
        .long   2509                            # DW_AT_type
        .byte   5                               # Abbrev [5] 0x9cd:0x5 DW_TAG_const_type
        .long   911                             # DW_AT_type
        .byte   55                              # Abbrev [55] 0x9d2:0x5 DW_TAG_pointer_type
        .long   1057                            # DW_AT_type
        .byte   56                              # Abbrev [56] 0x9d7:0x5 DW_TAG_reference_type
        .long   2524                            # DW_AT_type
        .byte   5                               # Abbrev [5] 0x9dc:0x5 DW_TAG_const_type
        .long   1057                            # DW_AT_type
        .byte   57                              # Abbrev [57] 0x9e1:0x5 DW_TAG_rvalue_reference_type
        .long   1057                            # DW_AT_type
        .byte   56                              # Abbrev [56] 0x9e6:0x5 DW_TAG_reference_type
        .long   1057                            # DW_AT_type
        .byte   55                              # Abbrev [55] 0x9eb:0x5 DW_TAG_pointer_type
        .long   1711                            # DW_AT_type
        .byte   56                              # Abbrev [56] 0x9f0:0x5 DW_TAG_reference_type
        .long   2549                            # DW_AT_type
        .byte   5                               # Abbrev [5] 0x9f5:0x5 DW_TAG_const_type
        .long   1711                            # DW_AT_type
        .byte   57                              # Abbrev [57] 0x9fa:0x5 DW_TAG_rvalue_reference_type
        .long   1711                            # DW_AT_type
        .byte   56                              # Abbrev [56] 0x9ff:0x5 DW_TAG_reference_type
        .long   1711                            # DW_AT_type
        .byte   55                              # Abbrev [55] 0xa04:0x5 DW_TAG_pointer_type
        .long   2549                            # DW_AT_type
        .byte   6                               # Abbrev [6] 0xa09:0x7 DW_TAG_base_type
        .long   .Linfo_string97                 # DW_AT_name
        .byte   7                               # DW_AT_encoding
        .byte   8                               # DW_AT_byte_size
        .byte   55                              # Abbrev [55] 0xa10:0x5 DW_TAG_pointer_type
        .long   1563                            # DW_AT_type
        .byte   56                              # Abbrev [56] 0xa15:0x5 DW_TAG_reference_type
        .long   2586                            # DW_AT_type
        .byte   5                               # Abbrev [5] 0xa1a:0x5 DW_TAG_const_type
        .long   1563                            # DW_AT_type
        .byte   57                              # Abbrev [57] 0xa1f:0x5 DW_TAG_rvalue_reference_type
        .long   1563                            # DW_AT_type
        .byte   56                              # Abbrev [56] 0xa24:0x5 DW_TAG_reference_type
        .long   1563                            # DW_AT_type
        .byte   55                              # Abbrev [55] 0xa29:0x5 DW_TAG_pointer_type
        .long   2028                            # DW_AT_type
        .byte   58                              # Abbrev [58] 0xa2e:0x94 DW_TAG_subprogram
        .quad   .Lfunc_begin0                   # DW_AT_low_pc
        .long   .Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
        .byte   1                               # DW_AT_frame_base
        .byte   87
                                        # DW_AT_GNU_all_call_sites
        .long   .Linfo_string126                # DW_AT_linkage_name
        .long   .Linfo_string127                # DW_AT_name
        .byte   6                               # DW_AT_decl_file
        .byte   2                               # DW_AT_decl_line
                                        # DW_AT_external
        .byte   59                              # Abbrev [59] 0xa47:0x5 DW_TAG_class_type
        .byte   5                               # DW_AT_calling_convention
        .byte   1                               # DW_AT_byte_size
        .byte   6                               # DW_AT_decl_file
        .byte   3                               # DW_AT_decl_line
        .byte   60                              # Abbrev [60] 0xa4c:0xf DW_TAG_formal_parameter
        .long   .Ldebug_loc0                    # DW_AT_location
        .long   .Linfo_string128                # DW_AT_name
        .byte   6                               # DW_AT_decl_file
        .byte   2                               # DW_AT_decl_line
        .long   1711                            # DW_AT_type
        .byte   61                              # Abbrev [61] 0xa5b:0x66 DW_TAG_inlined_subroutine
        .long   2180                            # DW_AT_abstract_origin
        .quad   .Lfunc_begin0                   # DW_AT_low_pc
        .long   .Ltmp3-.Lfunc_begin0            # DW_AT_high_pc
        .byte   6                               # DW_AT_call_file
        .byte   3                               # DW_AT_call_line
        .byte   5                               # DW_AT_call_column
        .byte   62                              # Abbrev [62] 0xa6f:0x51 DW_TAG_lexical_block
        .quad   .Lfunc_begin0                   # DW_AT_low_pc
        .long   .Ltmp3-.Lfunc_begin0            # DW_AT_high_pc
        .byte   63                              # Abbrev [63] 0xa7c:0x6 DW_TAG_variable
        .byte   1                               # DW_AT_const_value
        .long   2242                            # DW_AT_abstract_origin
        .byte   64                              # Abbrev [64] 0xa82:0x3d DW_TAG_inlined_subroutine
        .long   2051                            # DW_AT_abstract_origin
        .quad   .Lfunc_begin0                   # DW_AT_low_pc
        .long   .Ltmp3-.Lfunc_begin0            # DW_AT_high_pc
        .byte   1                               # DW_AT_call_file
        .short  1854                            # DW_AT_call_line
        .byte   13                              # DW_AT_call_column
        .byte   62                              # Abbrev [62] 0xa97:0x27 DW_TAG_lexical_block
        .quad   .Lfunc_begin0                   # DW_AT_low_pc
        .long   .Ltmp3-.Lfunc_begin0            # DW_AT_high_pc
        .byte   63                              # Abbrev [63] 0xaa4:0x6 DW_TAG_variable
        .byte   11                              # DW_AT_const_value
        .long   2122                            # DW_AT_abstract_origin
        .byte   63                              # Abbrev [63] 0xaaa:0x6 DW_TAG_variable
        .byte   3                               # DW_AT_const_value
        .long   2134                            # DW_AT_abstract_origin
        .byte   65                              # Abbrev [65] 0xab0:0xd DW_TAG_lexical_block
        .quad   .Ltmp1                          # DW_AT_low_pc
        .long   .Ltmp3-.Ltmp1                   # DW_AT_high_pc
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   0                               # End Of Children Mark
        .byte   57                              # Abbrev [57] 0xac2:0x5 DW_TAG_rvalue_reference_type
        .long   2631                            # DW_AT_type
        .byte   5                               # Abbrev [5] 0xac7:0x5 DW_TAG_const_type
        .long   2016                            # DW_AT_type
        .byte   5                               # Abbrev [5] 0xacc:0x5 DW_TAG_const_type
        .long   2311                            # DW_AT_type
        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:
        .section        .debug_str,"MS",@progbits,1
.Linfo_string0:
        .asciz  "clang version 17.0.0 (https://github.com/llvm/llvm-project.git 28d13a6297280a1c616ccd63d5a27543d92e5b7d)" # string offset=0
.Linfo_string1:
        .asciz  "/app/example.cpp"              # string offset=105
.Linfo_string2:
        .asciz  "/app"                          # string offset=122
.Linfo_string3:
        .asciz  "char"                          # string offset=127
.Linfo_string4:
        .asciz  "__ARRAY_SIZE_TYPE__"           # string offset=132
.Linfo_string5:
        .asciz  "std"                           # string offset=152
.Linfo_string6:
        .asciz  "exception"                     # string offset=156
.Linfo_string7:
        .asciz  "_M_reason"                     # string offset=166
.Linfo_string8:
        .asciz  "bad_variant_access"            # string offset=176
.Linfo_string9:
        .asciz  "_ZNKSt18bad_variant_access4whatEv" # string offset=195
.Linfo_string10:
        .asciz  "what"                          # string offset=229
.Linfo_string11:
        .asciz  "_Tp"                           # string offset=234
.Linfo_string12:
        .asciz  "__invoke_other"                # string offset=238
.Linfo_string13:
        .asciz  "_Tag"                          # string offset=253
.Linfo_string14:
        .asciz  "__success_type<void>"          # string offset=258
.Linfo_string15:
        .asciz  "__result_of_success<void, std::__invoke_other>" # string offset=279
.Linfo_string16:
        .asciz  "__invoke_type"                 # string offset=326
.Linfo_string17:
        .asciz  "__tag"                         # string offset=340
.Linfo_string18:
        .asciz  "__detail"                      # string offset=346
.Linfo_string19:
        .asciz  "__variant"                     # string offset=355
.Linfo_string20:
        .asciz  "_Types"                        # string offset=365
.Linfo_string21:
        .asciz  "_Variadic_union"               # string offset=372
.Linfo_string22:
        .asciz  "_Variadic_union<>"             # string offset=388
.Linfo_string23:
        .asciz  "_M_first"                      # string offset=406
.Linfo_string24:
        .asciz  "_Type"                         # string offset=415
.Linfo_string25:
        .asciz  "bool"                          # string offset=421
.Linfo_string26:
        .asciz  "_M_storage"                    # string offset=426
.Linfo_string27:
        .asciz  "_ZNKRSt8__detail9__variant14_UninitializedIPcLb1EE6_M_getEv" # string offset=437
.Linfo_string28:
        .asciz  "_M_get"                        # string offset=497
.Linfo_string29:
        .asciz  "_ZNRSt8__detail9__variant14_UninitializedIPcLb1EE6_M_getEv" # string offset=504
.Linfo_string30:
        .asciz  "_ZNKOSt8__detail9__variant14_UninitializedIPcLb1EE6_M_getEv" # string offset=563
.Linfo_string31:
        .asciz  "_ZNOSt8__detail9__variant14_UninitializedIPcLb1EE6_M_getEv" # string offset=623
.Linfo_string32:
        .asciz  "_Uninitialized<char *, true>"  # string offset=682
.Linfo_string33:
        .asciz  "_M_rest"                       # string offset=711
.Linfo_string34:
        .asciz  "_Variadic_union<char *>"       # string offset=719
.Linfo_string35:
        .asciz  "long"                          # string offset=743
.Linfo_string36:
        .asciz  "_ZNKRSt8__detail9__variant14_UninitializedIPlLb1EE6_M_getEv" # string offset=748
.Linfo_string37:
        .asciz  "_ZNRSt8__detail9__variant14_UninitializedIPlLb1EE6_M_getEv" # string offset=808
.Linfo_string38:
        .asciz  "_ZNKOSt8__detail9__variant14_UninitializedIPlLb1EE6_M_getEv" # string offset=867
.Linfo_string39:
        .asciz  "_ZNOSt8__detail9__variant14_UninitializedIPlLb1EE6_M_getEv" # string offset=927
.Linfo_string40:
        .asciz  "_Uninitialized<long *, true>"  # string offset=986
.Linfo_string41:
        .asciz  "_Variadic_union<long *, char *>" # string offset=1015
.Linfo_string42:
        .asciz  "int"                           # string offset=1047
.Linfo_string43:
        .asciz  "_ZNKRSt8__detail9__variant14_UninitializedIPiLb1EE6_M_getEv" # string offset=1051
.Linfo_string44:
        .asciz  "_ZNRSt8__detail9__variant14_UninitializedIPiLb1EE6_M_getEv" # string offset=1111
.Linfo_string45:
        .asciz  "_ZNKOSt8__detail9__variant14_UninitializedIPiLb1EE6_M_getEv" # string offset=1170
.Linfo_string46:
        .asciz  "_ZNOSt8__detail9__variant14_UninitializedIPiLb1EE6_M_getEv" # string offset=1230
.Linfo_string47:
        .asciz  "_Uninitialized<int *, true>"   # string offset=1289
.Linfo_string48:
        .asciz  "_Variadic_union<int *, long *, char *>" # string offset=1317
.Linfo_string49:
        .asciz  "__trivially_destructible"      # string offset=1356
.Linfo_string50:
        .asciz  "_M_u"                          # string offset=1381
.Linfo_string51:
        .asciz  "_M_index"                      # string offset=1386
.Linfo_string52:
        .asciz  "unsigned char"                 # string offset=1395
.Linfo_string53:
        .asciz  "__v"                           # string offset=1409
.Linfo_string54:
        .asciz  "value"                         # string offset=1413
.Linfo_string55:
        .asciz  "_ZNKSt17integral_constantIhLh3EEcvhEv" # string offset=1419
.Linfo_string56:
        .asciz  "operator unsigned char"        # string offset=1457
.Linfo_string57:
        .asciz  "value_type"                    # string offset=1480
.Linfo_string58:
        .asciz  "_ZNKSt17integral_constantIhLh3EEclEv" # string offset=1491
.Linfo_string59:
        .asciz  "operator()"                    # string offset=1528
.Linfo_string60:
        .asciz  "integral_constant<unsigned char, (unsigned char)'\\x03'>" # string offset=1539
.Linfo_string61:
        .asciz  "__select_index<int *, long *, char *>" # string offset=1595
.Linfo_string62:
        .asciz  "__index_type"                  # string offset=1633
.Linfo_string63:
        .asciz  "_Variant_storage"              # string offset=1646
.Linfo_string64:
        .asciz  "_ZNSt8__detail9__variant16_Variant_storageILb1EJPiPlPcEE8_M_resetEv" # string offset=1663
.Linfo_string65:
        .asciz  "_M_reset"                      # string offset=1731
.Linfo_string66:
        .asciz  "_ZNKSt8__detail9__variant16_Variant_storageILb1EJPiPlPcEE8_M_validEv" # string offset=1740
.Linfo_string67:
        .asciz  "_M_valid"                      # string offset=1809
.Linfo_string68:
        .asciz  "_Variant_storage<true, int *, long *, char *>" # string offset=1818
.Linfo_string69:
        .asciz  "_Variant_storage_alias<int *, long *, char *>" # string offset=1864
.Linfo_string70:
        .asciz  "_Copy_ctor_base<true, int *, long *, char *>" # string offset=1910
.Linfo_string71:
        .asciz  "_Copy_ctor_alias<int *, long *, char *>" # string offset=1955
.Linfo_string72:
        .asciz  "_Move_ctor_base<true, int *, long *, char *>" # string offset=1995
.Linfo_string73:
        .asciz  "_Move_ctor_alias<int *, long *, char *>" # string offset=2040
.Linfo_string74:
        .asciz  "_Copy_assign_base<true, int *, long *, char *>" # string offset=2080
.Linfo_string75:
        .asciz  "_Copy_assign_alias<int *, long *, char *>" # string offset=2127
.Linfo_string76:
        .asciz  "_Move_assign_base<true, int *, long *, char *>" # string offset=2169
.Linfo_string77:
        .asciz  "_Move_assign_alias<int *, long *, char *>" # string offset=2216
.Linfo_string78:
        .asciz  "_Variant_base"                 # string offset=2258
.Linfo_string79:
        .asciz  "_ZNSt8__detail9__variant13_Variant_baseIJPiPlPcEEaSERKS5_" # string offset=2272
.Linfo_string80:
        .asciz  "operator="                     # string offset=2330
.Linfo_string81:
        .asciz  "_ZNSt8__detail9__variant13_Variant_baseIJPiPlPcEEaSEOS5_" # string offset=2340
.Linfo_string82:
        .asciz  "_Variant_base<int *, long *, char *>" # string offset=2397
.Linfo_string83:
        .asciz  "_Switch"                       # string offset=2434
.Linfo_string84:
        .asciz  "_Copy"                         # string offset=2442
.Linfo_string85:
        .asciz  "_CopyAssignment"               # string offset=2448
.Linfo_string86:
        .asciz  "_Move"                         # string offset=2464
.Linfo_string87:
        .asciz  "_MoveAssignment"               # string offset=2470
.Linfo_string88:
        .asciz  "_Enable_copy_move<true, true, true, true, std::variant<int *, long *, char *> >" # string offset=2486
.Linfo_string89:
        .asciz  "variant"                       # string offset=2566
.Linfo_string90:
        .asciz  "_ZNSt7variantIJPiPlPcEEaSERKS3_" # string offset=2574
.Linfo_string91:
        .asciz  "_ZNSt7variantIJPiPlPcEEaSEOS3_" # string offset=2606
.Linfo_string92:
        .asciz  "~variant"                      # string offset=2637
.Linfo_string93:
        .asciz  "_ZNKSt7variantIJPiPlPcEE22valueless_by_exceptionEv" # string offset=2646
.Linfo_string94:
        .asciz  "valueless_by_exception"        # string offset=2697
.Linfo_string95:
        .asciz  "_ZNKSt7variantIJPiPlPcEE5indexEv" # string offset=2720
.Linfo_string96:
        .asciz  "index"                         # string offset=2753
.Linfo_string97:
        .asciz  "unsigned long"                 # string offset=2759
.Linfo_string98:
        .asciz  "size_t"                        # string offset=2773
.Linfo_string99:
        .asciz  "_ZNSt7variantIJPiPlPcEE4swapERS3_" # string offset=2780
.Linfo_string100:
        .asciz  "swap"                          # string offset=2814
.Linfo_string101:
        .asciz  "variant<int *, long *, char *>" # string offset=2819
.Linfo_string102:
        .asciz  "_Enable_default_constructor"   # string offset=2850
.Linfo_string103:
        .asciz  "_ZNSt27_Enable_default_constructorILb1ESt7variantIJPiPlPcEEEaSERKS5_" # string offset=2878
.Linfo_string104:
        .asciz  "_ZNSt27_Enable_default_constructorILb1ESt7variantIJPiPlPcEEEaSEOS5_" # string offset=2947
.Linfo_string105:
        .asciz  "_Enable_default_constructor_tag" # string offset=3015
.Linfo_string106:
        .asciz  "_Enable_default_constructor<true, std::variant<int *, long *, char *> >" # string offset=3047
.Linfo_string107:
        .asciz  "__deduce_visit_result<void>"   # string offset=3119
.Linfo_string108:
        .asciz  "_Result_type"                  # string offset=3147
.Linfo_string109:
        .asciz  "_Visitor"                      # string offset=3160
.Linfo_string110:
        .asciz  "_Variants"                     # string offset=3169
.Linfo_string111:
        .asciz  "_ZSt10__do_visitINSt8__detail9__variant21__deduce_visit_resultIvEEZ1fSt7variantIJPiPlPcEEE3$_0JRS8_EEDcOT0_DpOT1_" # string offset=3179
.Linfo_string112:
        .asciz  "__do_visit<std::__detail::__variant::__deduce_visit_result<void>, (lambda at /app/example.cpp:3:16), std::variant<int *, long *, char *> &>" # string offset=3293
.Linfo_string113:
        .asciz  "type"                          # string offset=3433
.Linfo_string114:
        .asciz  "__visitor"                     # string offset=3438
.Linfo_string115:
        .asciz  "__variants"                    # string offset=3448
.Linfo_string116:
        .asciz  "__max"                         # string offset=3459
.Linfo_string117:
        .asciz  "__n"                           # string offset=3465
.Linfo_string118:
        .asciz  "__v0"                          # string offset=3469
.Linfo_string119:
        .asciz  "_ZSt5visitIZ1fSt7variantIJPiPlPcEEE3$_0JRS4_EENSt13invoke_resultIT_JDpNSt13__conditionalIX21is_lvalue_reference_vIT0_EEE4typeIRNSt19variant_alternativeILm0ENSt16remove_referenceIDTclsr9__variantE4__asclsr3stdE7declvalISA_EEEEE4typeEE4typeEOSJ_EEEE4typeEOS8_DpOSA_" # string offset=3474
.Linfo_string120:
        .asciz  "visit<(lambda at /app/example.cpp:3:16), std::variant<int *, long *, char *> &>" # string offset=3738
.Linfo_string121:
        .asciz  "invoke_result_t<(lambda at /app/example.cpp:3:16), __get_t<0, variant<int *, long *, char *> &> >" # string offset=3818
.Linfo_string122:
        .asciz  "__visit_result_t<(lambda at /app/example.cpp:3:16), std::variant<int *, long *, char *> &>" # string offset=3916
.Linfo_string123:
        .asciz  "__visit_rettypes_match"        # string offset=4007
.Linfo_string124:
        .asciz  "__variant_idx_cookie"          # string offset=4030
.Linfo_string125:
        .asciz  "__variant_cookie"              # string offset=4051
.Linfo_string126:
        .asciz  "_Z1fSt7variantIJPiPlPcEE"      # string offset=4068
.Linfo_string127:
        .asciz  "f"                             # string offset=4093
.Linfo_string128:
        .asciz  "v"                             # string offset=4095
        .ident  "clang version 17.0.0 (https://github.com/llvm/llvm-project.git 28d13a6297280a1c616ccd63d5a27543d92e5b7d)"
        .section        ".note.GNU-stack","",@progbits
        .addrsig
        .section        .debug_line,"",@progbits
.Lline_table_start0: