        .section .text
.LNDBG_TX:
# mark_description "Intel(R) C Intel(R) 64 Compiler XE for applications running on Intel(R) 64, Version 12.1 Build 20120410";
        .file "iccKTGaIssTdIn_"
        .text
..TXTST0:
# -- Begin  main
# mark_begin;
       .align    16,0x90
        .globl main
main:
..B1.1:                         # Preds ..B1.0
..___tag_value_main.2:                                          #
..LN0:
  .file   1 "-"
   .loc    1  2  is_stmt 1
        pushq     %rbp                                          #2.12
..___tag_value_main.4:                                          #
..LN1:
        movq      %rsp, %rbp                                    #2.12
..___tag_value_main.5:                                          #
..LN2:
        andq      $-128, %rsp                                   #2.12
..LN3:
        subq      $128, %rsp                                    #2.12
..LN4:
        movl      $3, %edi                                      #2.12
..___tag_value_main.8:                                          #2.12
..LN5:
        call      __intel_new_proc_init                         #2.12
..___tag_value_main.9:                                          #
..LN6:
                                # LOE rbx r12 r13 r14 r15
..B1.6:                         # Preds ..B1.1
..LN7:
        stmxcsr   (%rsp)                                        #2.12
..LN8:
   .loc    1  3  is_stmt 1
        movl      $.L_2__STRING.0, %edi                         #3.1
..LN9:
        xorl      %eax, %eax                                    #3.1
..LN10:
   .loc    1  2  is_stmt 1
        orl       $32832, (%rsp)                                #2.12
..LN11:
        ldmxcsr   (%rsp)                                        #2.12
..___tag_value_main.10:                                         #3.1
..LN12:
   .loc    1  3  is_stmt 1
        call      printf                                        #3.1
..___tag_value_main.11:                                         #
..LN13:
                                # LOE rbx r12 r13 r14 r15
..B1.2:                         # Preds ..B1.6
..LN14:
   .loc    1  4  is_stmt 1
        movl      $.L_2__STRING.1, %edi                         #4.3
..LN15:
        xorl      %eax, %eax                                    #4.3
..___tag_value_main.12:                                         #4.3
..LN16:
        call      printf                                        #4.3
..___tag_value_main.13:                                         #
..LN17:
                                # LOE rbx r12 r13 r14 r15
..B1.3:                         # Preds ..B1.2
..LN18:
   .loc    1  5  is_stmt 1
        xorl      %eax, %eax                                    #5.1
..LN19:
        movq      %rbp, %rsp                                    #5.1
..LN20:
        popq      %rbp                                          #5.1
..___tag_value_main.15:                                         #
..LN21:
        ret                                                     #5.1
        .align    16,0x90
..___tag_value_main.19:                                         #
..LN22:
                                # LOE
..LN23:
# mark_end;
        .type   main,@function
        .size   main,.-main
..LNmain.24:
.LNmain:
        .data
# -- End  main
        .section .rodata.str1.4, "aMS",@progbits,1
        .align 4
        .align 4
.L_2__STRING.0:
        .byte   72
        .byte   101
        .byte   108
        .byte   108
        .byte   111
        .byte   32
        .byte   119
        .byte   111
        .byte   114
        .byte   108
        .byte   100
        .byte   0
        .type   .L_2__STRING.0,@object
        .size   .L_2__STRING.0,12
        .align 4
.L_2__STRING.1:
        .byte   109
        .byte   111
        .byte   111
        .byte   10
        .byte   0
        .type   .L_2__STRING.1,@object
        .size   .L_2__STRING.1,5
        .data
        .section .note.GNU-stack, ""
// -- Begin DWARF2 SEGMENT .debug_info
        .section .debug_info
.debug_info_seg:
        .align 1
        .4byte 0x000000fe
        .2byte 0x0002
        .4byte .debug_abbrev_seg
        .byte 0x08
//      DW_TAG_compile_unit:
        .byte 0x01
//      DW_AT_comp_dir:
        .8byte 0x676d2f656d6f682f
        .8byte 0x642f746c6f62646f
        .8byte 0x652d6363672f7665
        .8byte 0x007265726f6c7078
//      DW_AT_language:
        .byte 0x04
//      DW_AT_producer:
        .8byte 0x2952286c65746e49
        .8byte 0x6c65746e49204320
        .8byte 0x4320343620295228
        .8byte 0x2072656c69706d6f
        .8byte 0x6120726f66204558
        .8byte 0x69746163696c7070
        .8byte 0x6e6e757220736e6f
        .8byte 0x49206e6f20676e69
        .8byte 0x202952286c65746e
        .8byte 0x73726556202c3436
        .8byte 0x312e3231206e6f69
        .8byte 0x3220646c69754220
        .8byte 0x0a30313430323130
        .8byte 0x5320736578694620
        .8byte 0x616b6e694c656d61
        .8byte 0x4d20656d614e6567
        .8byte 0x696f507265626d65
        .4byte 0x7265746e
        .2byte 0x0a73
        .byte 0x00
//      DW_AT_stmt_list:
        .4byte .debug_line_seg
//      DW_TAG_namespace:
        .byte 0x02
//      DW_AT_name:
        .4byte 0x00647473
//      DW_TAG_namespace:
        .byte 0x02
//      DW_AT_name:
        .8byte 0x6962617878635f5f
        .2byte 0x3176
        .byte 0x00
//      DW_TAG_base_type:
        .byte 0x03
//      DW_AT_byte_size:
        .byte 0x04
//      DW_AT_encoding:
        .byte 0x05
//      DW_AT_name:
        .4byte 0x00746e69
//      DW_TAG_subprogram:
        .byte 0x04
//      DW_AT_decl_line:
        .byte 0x02
//      DW_AT_decl_column:
        .byte 0x05
//      DW_AT_decl_file:
        .byte 0x01
//      DW_AT_inline:
        .byte 0x00
//      DW_AT_accessibility:
        .byte 0x01
//      DW_AT_type:
        .4byte 0x000000d1
//      DW_AT_prototyped:
        .byte 0x01
//      DW_AT_name:
        .4byte 0x6e69616d
        .byte 0x00
        .4byte 0x6e69616d
        .byte 0x00
//      DW_AT_low_pc:
        .8byte main
//      DW_AT_high_pc:
        .8byte ..LNmain.24
//      DW_AT_external:
        .byte 0x01
        .byte 0x00
        .byte 0x00
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
        .byte 0x08
        .byte 0x13
        .byte 0x0b
        .byte 0x25
        .byte 0x08
        .byte 0x10
        .byte 0x06
        .2byte 0x0000
        .byte 0x02
        .byte 0x39
        .byte 0x00
        .byte 0x03
        .byte 0x08
        .2byte 0x0000
        .byte 0x03
        .byte 0x24
        .byte 0x00
        .byte 0x0b
        .byte 0x0b
        .byte 0x3e
        .byte 0x0b
        .byte 0x03
        .byte 0x08
        .2byte 0x0000
        .byte 0x04
        .byte 0x2e
        .byte 0x00
        .byte 0x3b
        .byte 0x0b
        .byte 0x39
        .byte 0x0b
        .byte 0x3a
        .byte 0x0b
        .byte 0x20
        .byte 0x0b
        .byte 0x32
        .byte 0x0b
        .byte 0x49
        .byte 0x13
        .byte 0x27
        .byte 0x0c
        .byte 0x03
        .byte 0x08
        .2byte 0x4087
        .byte 0x08
        .byte 0x11
        .byte 0x01
        .byte 0x12
        .byte 0x01
        .byte 0x3f
        .byte 0x0c
        .2byte 0x0000
        .byte 0x00
// -- Begin DWARF2 SEGMENT .debug_frame
        .section .debug_frame
.debug_frame_seg:
        .align 1
        .4byte 0x00000014
        .8byte 0x78010001ffffffff
        .8byte 0x0000019008070c10
        .4byte 0x00000000
        .4byte 0x00000034
        .4byte .debug_frame_seg
        .8byte ..___tag_value_main.2
        .8byte ..___tag_value_main.19-..___tag_value_main.2
        .byte 0x04
        .4byte ..___tag_value_main.4-..___tag_value_main.2
        .2byte 0x100e
        .byte 0x04
        .4byte ..___tag_value_main.5-..___tag_value_main.4
        .4byte 0x8610060c
        .2byte 0x0402
        .4byte ..___tag_value_main.15-..___tag_value_main.5
        .8byte 0x00000000c608070c
        .2byte 0x0000
// -- Begin DWARF2 SEGMENT .eh_frame
        .section .eh_frame,"a",@progbits
.eh_frame_seg:
        .align 8
        .4byte 0x0000001c
        .8byte 0x00507a0100000000
        .4byte 0x09107801
        .byte 0x00
        .8byte __gxx_personality_v0
        .4byte 0x9008070c
        .2byte 0x0001
        .byte 0x00
        .4byte 0x00000034
        .4byte 0x00000024
        .8byte ..___tag_value_main.2
        .8byte ..___tag_value_main.19-..___tag_value_main.2
        .2byte 0x0400
        .4byte ..___tag_value_main.4-..___tag_value_main.2
        .2byte 0x100e
        .byte 0x04
        .4byte ..___tag_value_main.5-..___tag_value_main.4
        .4byte 0x8610060c
        .2byte 0x0402
        .4byte ..___tag_value_main.15-..___tag_value_main.5
        .8byte 0x00000000c608070c
        .byte 0x00
        .section .text
.LNDBG_TXe:
# End
