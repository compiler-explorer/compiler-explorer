        .section .text
.LNDBG_TX:
# mark_description "Intel(R) C Intel(R) 64 Compiler for applications running on Intel(R) 64, Version 18.0.0.128 Build 20170811";
# mark_description "-g -o /tmp/compiler-explorer-compiler1171120-54-1t2ppc6.m4k6/output.s -masm=intel -S -gxx-name=/opt/compiler";
# mark_description "-explorer/gcc-6.3.0/bin/g++";
        .intel_syntax noprefix
        .file "example.cpp"
        .text
..TXTST0:
.L_2__routine_start__Z6squarei_0:
# -- Begin  _Z6squarei
        .text
# mark_begin;

        .globl _Z6squarei
# --- square(int)
_Z6squarei:
# parameter 1(num): edi
..B1.1:                         # Preds ..B1.0
                                # Execution count [0.00e+00]
        .cfi_startproc
        .cfi_personality 0x3,__gxx_personality_v0
..___tag_value__Z6squarei.2:
..L3:
                                                          #2.21
..LN0:
        .file   1 "/tmp/compiler-explorer-compiler1171120-54-1t2ppc6.m4k6/example.cpp"
        .loc    1  2  is_stmt 1
        push      rbp                                           #2.21
        .cfi_def_cfa_offset 16
..LN1:
        mov       rbp, rsp                                      #2.21
        .cfi_def_cfa 6, 16
        .cfi_offset 6, -16
..LN2:
        sub       rsp, 16                                       #2.21
..LN3:
        mov       DWORD PTR [-16+rbp], edi                      #2.21
..LN4:
                                # LOE rbx rbp rsp r12 r13 r14 r15 rip
..B1.5:                         # Preds ..B1.1
                                # Execution count [0.00e+00]
# Begin ASM
..LN5:
        .loc    1  3  prologue_end  is_stmt 1
# Begin ASM
        label:
# End ASM                                                       #3.0
..LN6:
# End ASM
                                # LOE rbx rbp rsp r12 r13 r14 r15 rip
..B1.4:                         # Preds ..B1.5
                                # Execution count [0.00e+00]
..LN7:
        .loc    1  4  is_stmt 1
        mov       eax, DWORD PTR [-16+rbp]                      #4.18
..LN8:
        imul      eax, DWORD PTR [-16+rbp]                      #4.18
..LN9:
        .loc    1  4  epilogue_begin  is_stmt 1
        leave                                                   #4.18
        .cfi_restore 6
..LN10:
        ret                                                     #4.18
..LN11:
                                # LOE
..LN12:
        .cfi_endproc
# mark_end;
        .type   _Z6squarei,@function
        .size   _Z6squarei,.-_Z6squarei
..LN_Z6squarei.13:
.LN_Z6squarei:
        .data
# -- End  _Z6squarei
        .data
        .section .note.GNU-stack, ""
// -- Begin DWARF2 SEGMENT .debug_info
        .section .debug_info
.debug_info_seg:
        .align 1
        .4byte 0x00000074
        .2byte 0x0004
        .4byte .debug_abbrev_seg
        .byte 0x08
//      DW_TAG_compile_unit:
        .byte 0x01
//      DW_AT_comp_dir:
        .4byte .debug_str
//      DW_AT_name:
        .4byte .debug_str+0x13
//      DW_AT_producer:
        .4byte .debug_str+0x56
        .4byte .debug_str+0xc2
//      DW_AT_language:
        .byte 0x04
//      DW_AT_use_UTF8:
        .byte 0x01
//      DW_AT_low_pc:
        .8byte ..LN0
//      DW_AT_high_pc:
        .8byte ..LN_Z6squarei.13-..LN0
//      DW_AT_stmt_list:
        .4byte .debug_line_seg
//      DW_TAG_namespace:
        .byte 0x02
//      DW_AT_name:
        .4byte 0x00647473
//      DW_TAG_namespace:
        .byte 0x03
//      DW_AT_name:
        .4byte .debug_str+0x14a
//      DW_TAG_namespace:
        .byte 0x03
//      DW_AT_name:
        .4byte .debug_str+0x154
//      DW_TAG_base_type:
        .byte 0x04
//      DW_AT_byte_size:
        .byte 0x04
//      DW_AT_encoding:
        .byte 0x05
//      DW_AT_name:
        .4byte 0x00746e69
//      DW_TAG_subprogram:
        .byte 0x05
//      DW_AT_decl_line:
        .byte 0x02
//      DW_AT_decl_file:
        .byte 0x01
//      DW_AT_type:
        .4byte 0x00000041
//      DW_AT_name:
        .4byte .debug_str+0x15f
        .4byte .debug_str+0x166
//      DW_AT_low_pc:
        .8byte ..L3
//      DW_AT_high_pc:
        .8byte ..LN_Z6squarei.13-..L3
//      DW_AT_external:
        .byte 0x01
//      DW_TAG_formal_parameter:
        .byte 0x06
//      DW_AT_decl_line:
        .byte 0x02
//      DW_AT_decl_file:
        .byte 0x01
//      DW_AT_type:
        .4byte 0x00000041
//      DW_AT_name:
        .4byte 0x006d756e
//      DW_AT_location:
        .2byte 0x7602
        .byte 0x70
        .byte 0x00
        .byte 0x00
// -- Begin DWARF2 SEGMENT .debug_line
        .section .debug_line
.debug_line_seg:
        .align 1
// -- Begin DWARF2 SEGMENT .debug_abbrev
        .section .debug_abbrev
.debug_abbrev_seg:
        .align 1
        .byte 0x01
        .byte 0x11
        .byte 0x01
        .byte 0x1b
        .byte 0x0e
        .byte 0x03
        .byte 0x0e
        .byte 0x25
        .byte 0x0e
        .2byte 0x7681
        .byte 0x0e
        .byte 0x13
        .byte 0x0b
        .byte 0x53
        .byte 0x0c
        .byte 0x11
        .byte 0x01
        .byte 0x12
        .byte 0x07
        .byte 0x10
        .byte 0x17
        .2byte 0x0000
        .byte 0x02
        .byte 0x39
        .byte 0x00
        .byte 0x03
        .byte 0x08
        .2byte 0x0000
        .byte 0x03
        .byte 0x39
        .byte 0x00
        .byte 0x03
        .byte 0x0e
        .2byte 0x0000
        .byte 0x04
        .byte 0x24
        .byte 0x00
        .byte 0x0b
        .byte 0x0b
        .byte 0x3e
        .byte 0x0b
        .byte 0x03
        .byte 0x08
        .2byte 0x0000
        .byte 0x05
        .byte 0x2e
        .byte 0x01
        .byte 0x3b
        .byte 0x0b
        .byte 0x3a
        .byte 0x0b
        .byte 0x49
        .byte 0x13
        .byte 0x03
        .byte 0x0e
        .2byte 0x4087
        .byte 0x0e
        .byte 0x11
        .byte 0x01
        .byte 0x12
        .byte 0x07
        .byte 0x3f
        .byte 0x0c
        .2byte 0x0000
        .byte 0x06
        .byte 0x05
        .byte 0x00
        .byte 0x3b
        .byte 0x0b
        .byte 0x3a
        .byte 0x0b
        .byte 0x49
        .byte 0x13
        .byte 0x03
        .byte 0x08
        .byte 0x02
        .byte 0x18
        .2byte 0x0000
        .byte 0x00
// -- Begin DWARF2 SEGMENT .debug_frame
        .section .debug_frame
.debug_frame_seg:
        .align 1
// -- Begin DWARF2 SEGMENT .debug_str
        .section .debug_str,"MS",@progbits,1
.debug_str_seg:
        .align 1
        .8byte 0x656c69706d6f632f
        .8byte 0x726f6c7078652d72
        .2byte 0x7265
        .byte 0x00
        .8byte 0x6d6f632f706d742f
        .8byte 0x78652d72656c6970
        .8byte 0x632d7265726f6c70
        .8byte 0x3172656c69706d6f
        .8byte 0x352d303231313731
        .8byte 0x6370703274312d34
        .8byte 0x652f366b346d2e36
        .8byte 0x632e656c706d6178
        .2byte 0x7070
        .byte 0x00
        .8byte 0x2952286c65746e49
        .8byte 0x6c65746e49204320
        .8byte 0x4320343620295228
        .8byte 0x2072656c69706d6f
        .8byte 0x6c70706120726f66
        .8byte 0x736e6f6974616369
        .8byte 0x676e696e6e757220
        .8byte 0x65746e49206e6f20
        .8byte 0x2c3436202952286c
        .8byte 0x6e6f697372655620
        .8byte 0x2e302e302e383120
        .8byte 0x6c69754220383231
        .8byte 0x3830373130322064
        .4byte 0x000a3131
        .8byte 0x742f206f2d20672d
        .8byte 0x69706d6f632f706d
        .8byte 0x6c7078652d72656c
        .8byte 0x6d6f632d7265726f
        .8byte 0x37313172656c6970
        .8byte 0x2d34352d30323131
        .8byte 0x2e36637070327431
        .8byte 0x74756f2f366b346d
        .8byte 0x6d2d20732e747570
        .8byte 0x65746e693d6d7361
        .8byte 0x78672d20532d206c
        .8byte 0x2f3d656d616e2d78
        .8byte 0x706d6f632f74706f
        .8byte 0x7078652d72656c69
        .8byte 0x63672f7265726f6c
        .8byte 0x2f302e332e362d63
        .8byte 0x002b2b672f6e6962
        .8byte 0x78635f756e675f5f
        .2byte 0x0078
        .8byte 0x6962617878635f5f
        .2byte 0x3176
        .byte 0x00
        .4byte 0x61757173
        .2byte 0x6572
        .byte 0x00
        .8byte 0x7261757173365a5f
        .2byte 0x6965
        .byte 0x00
// -- Begin DWARF2 SEGMENT .eh_frame
        .section .eh_frame,"a",@progbits
.eh_frame_seg:
        .align 8
        .section .text
.LNDBG_TXe:
# End