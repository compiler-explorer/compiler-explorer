        .file   ""
        .text
.Ltext0:
        .section        .rodata.str1.1,"aMS",@progbits,1
.LC0:
        .string "Hello world"
.LC1:
        .string "moo\n"
        .section        .text.startup,"ax",@progbits
        .p2align 4,,15
        .globl  main
        .type   main, @function
main:
.LFB31:
        .file 1 "<stdin>"
        .loc 1 2 0
        .cfi_startproc
.LVL0:
        subq    $8, %rsp
.LCFI0:
        .cfi_def_cfa_offset 16
.LBB6:
.LBB7:
        .file 2 "/usr/include/x86_64-linux-gnu/bits/stdio2.h"
        .loc 2 105 0
        movl    $.LC0, %esi
        movl    $1, %edi
        xorl    %eax, %eax
        call    __printf_chk
.LVL1:
.LBE7:
.LBE6:
.LBB8:
.LBB9:
        movl    $.LC1, %esi
        movl    $1, %edi
        xorl    %eax, %eax
        call    __printf_chk
.LBE9:
.LBE8:
        .loc 1 5 0
        xorl    %eax, %eax
        addq    $8, %rsp
.LCFI1:
        .cfi_def_cfa_offset 8
        ret
        .cfi_endproc
.LFE31:
        .size   main, .-main
        .text
.Letext0:
        .file 3 "/usr/lib/gcc/x86_64-linux-gnu/4.6/include/stddef.h"
        .file 4 "/usr/include/x86_64-linux-gnu/bits/types.h"
        .file 5 "/usr/include/libio.h"
        .file 6 "/usr/include/stdio.h"
        .section        .debug_info,"",@progbits
.Ldebug_info0:
        .long   0x398
        .value  0x2
        .long   .Ldebug_abbrev0
        .byte   0x8
        .uleb128 0x1
        .long   .LASF47
        .byte   0x4
        .long   .LASF48
        .quad   0
        .quad   0
        .long   .Ldebug_ranges0+0
        .long   .Ldebug_line0
        .uleb128 0x2
        .long   .LASF7
        .byte   0x3
        .byte   0xd4
        .long   0x38
        .uleb128 0x3
        .byte   0x8
        .byte   0x7
        .long   .LASF0
        .uleb128 0x3
        .byte   0x1
        .byte   0x8
        .long   .LASF1
        .uleb128 0x3
        .byte   0x2
        .byte   0x7
        .long   .LASF2
        .uleb128 0x3
        .byte   0x4
        .byte   0x7
        .long   .LASF3
        .uleb128 0x3
        .byte   0x1
        .byte   0x6
        .long   .LASF4
        .uleb128 0x3
        .byte   0x2
        .byte   0x5
        .long   .LASF5
        .uleb128 0x4
        .byte   0x4
        .byte   0x5
        .string "int"
        .uleb128 0x3
        .byte   0x8
        .byte   0x5
        .long   .LASF6
        .uleb128 0x2
        .long   .LASF8
        .byte   0x4
        .byte   0x8d
        .long   0x69
        .uleb128 0x2
        .long   .LASF9
        .byte   0x4
        .byte   0x8e
        .long   0x69
        .uleb128 0x5
        .byte   0x8
        .uleb128 0x6
        .byte   0x8
        .long   0x8e
        .uleb128 0x3
        .byte   0x1
        .byte   0x6
        .long   .LASF10
        .uleb128 0x7
        .long   .LASF40
        .byte   0xd8
        .byte   0x5
        .value  0x111
        .long   0x262
        .uleb128 0x8
        .long   .LASF11
        .byte   0x5
        .value  0x112
        .long   0x62
        .byte   0x2
        .byte   0x23
        .uleb128 0
        .uleb128 0x8
        .long   .LASF12
        .byte   0x5
        .value  0x117
        .long   0x88
        .byte   0x2
        .byte   0x23
        .uleb128 0x8
        .uleb128 0x8
        .long   .LASF13
        .byte   0x5
        .value  0x118
        .long   0x88
        .byte   0x2
        .byte   0x23
        .uleb128 0x10
        .uleb128 0x8
        .long   .LASF14
        .byte   0x5
        .value  0x119
        .long   0x88
        .byte   0x2
        .byte   0x23
        .uleb128 0x18
        .uleb128 0x8
        .long   .LASF15
        .byte   0x5
        .value  0x11a
        .long   0x88
        .byte   0x2
        .byte   0x23
        .uleb128 0x20
        .uleb128 0x8
        .long   .LASF16
        .byte   0x5
        .value  0x11b
        .long   0x88
        .byte   0x2
        .byte   0x23
        .uleb128 0x28
        .uleb128 0x8
        .long   .LASF17
        .byte   0x5
        .value  0x11c
        .long   0x88
        .byte   0x2
        .byte   0x23
        .uleb128 0x30
        .uleb128 0x8
        .long   .LASF18
        .byte   0x5
        .value  0x11d
        .long   0x88
        .byte   0x2
        .byte   0x23
        .uleb128 0x38
        .uleb128 0x8
        .long   .LASF19
        .byte   0x5
        .value  0x11e
        .long   0x88
        .byte   0x2
        .byte   0x23
        .uleb128 0x40
        .uleb128 0x8
        .long   .LASF20
        .byte   0x5
        .value  0x120
        .long   0x88
        .byte   0x2
        .byte   0x23
        .uleb128 0x48
        .uleb128 0x8
        .long   .LASF21
        .byte   0x5
        .value  0x121
        .long   0x88
        .byte   0x2
        .byte   0x23
        .uleb128 0x50
        .uleb128 0x8
        .long   .LASF22
        .byte   0x5
        .value  0x122
        .long   0x88
        .byte   0x2
        .byte   0x23
        .uleb128 0x58
        .uleb128 0x8
        .long   .LASF23
        .byte   0x5
        .value  0x124
        .long   0x2ac
        .byte   0x2
        .byte   0x23
        .uleb128 0x60
        .uleb128 0x8
        .long   .LASF24
        .byte   0x5
        .value  0x126
        .long   0x2b2
        .byte   0x2
        .byte   0x23
        .uleb128 0x68
        .uleb128 0x8
        .long   .LASF25
        .byte   0x5
        .value  0x128
        .long   0x62
        .byte   0x2
        .byte   0x23
        .uleb128 0x70
        .uleb128 0x8
        .long   .LASF26
        .byte   0x5
        .value  0x12c
        .long   0x62
        .byte   0x2
        .byte   0x23
        .uleb128 0x74
        .uleb128 0x8
        .long   .LASF27
        .byte   0x5
        .value  0x12e
        .long   0x70
        .byte   0x2
        .byte   0x23
        .uleb128 0x78
        .uleb128 0x8
        .long   .LASF28
        .byte   0x5
        .value  0x132
        .long   0x46
        .byte   0x3
        .byte   0x23
        .uleb128 0x80
        .uleb128 0x8
        .long   .LASF29
        .byte   0x5
        .value  0x133
        .long   0x54
        .byte   0x3
        .byte   0x23
        .uleb128 0x82
        .uleb128 0x8
        .long   .LASF30
        .byte   0x5
        .value  0x134
        .long   0x2b8
        .byte   0x3
        .byte   0x23
        .uleb128 0x83
        .uleb128 0x8
        .long   .LASF31
        .byte   0x5
        .value  0x138
        .long   0x2c8
        .byte   0x3
        .byte   0x23
        .uleb128 0x88
        .uleb128 0x8
        .long   .LASF32
        .byte   0x5
        .value  0x141
        .long   0x7b
        .byte   0x3
        .byte   0x23
        .uleb128 0x90
        .uleb128 0x8
        .long   .LASF33
        .byte   0x5
        .value  0x14a
        .long   0x86
        .byte   0x3
        .byte   0x23
        .uleb128 0x98
        .uleb128 0x8
        .long   .LASF34
        .byte   0x5
        .value  0x14b
        .long   0x86
        .byte   0x3
        .byte   0x23
        .uleb128 0xa0
        .uleb128 0x8
        .long   .LASF35
        .byte   0x5
        .value  0x14c
        .long   0x86
        .byte   0x3
        .byte   0x23
        .uleb128 0xa8
        .uleb128 0x8
        .long   .LASF36
        .byte   0x5
        .value  0x14d
        .long   0x86
        .byte   0x3
        .byte   0x23
        .uleb128 0xb0
        .uleb128 0x8
        .long   .LASF37
        .byte   0x5
        .value  0x14e
        .long   0x2d
        .byte   0x3
        .byte   0x23
        .uleb128 0xb8
        .uleb128 0x8
        .long   .LASF38
        .byte   0x5
        .value  0x150
        .long   0x62
        .byte   0x3
        .byte   0x23
        .uleb128 0xc0
        .uleb128 0x8
        .long   .LASF39
        .byte   0x5
        .value  0x152
        .long   0x2ce
        .byte   0x3
        .byte   0x23
        .uleb128 0xc4
        .byte   0
        .uleb128 0x9
        .long   .LASF49
        .byte   0x5
        .byte   0xb6
        .uleb128 0xa
        .long   .LASF41
        .byte   0x18
        .byte   0x5
        .byte   0xbc
        .long   0x2a0
        .uleb128 0xb
        .long   .LASF42
        .byte   0x5
        .byte   0xbd
        .long   0x2a0
        .byte   0x2
        .byte   0x23
        .uleb128 0
        .uleb128 0xb
        .long   .LASF43
        .byte   0x5
        .byte   0xbe
        .long   0x2a6
        .byte   0x2
        .byte   0x23
        .uleb128 0x8
        .uleb128 0xb
        .long   .LASF44
        .byte   0x5
        .byte   0xc2
        .long   0x62
        .byte   0x2
        .byte   0x23
        .uleb128 0x10
        .byte   0
        .uleb128 0x6
        .byte   0x8
        .long   0x269
        .uleb128 0x6
        .byte   0x8
        .long   0x95
        .uleb128 0x6
        .byte   0x8
        .long   0x269
        .uleb128 0x6
        .byte   0x8
        .long   0x95
        .uleb128 0xc
        .long   0x8e
        .long   0x2c8
        .uleb128 0xd
        .long   0x38
        .byte   0
        .byte   0
        .uleb128 0x6
        .byte   0x8
        .long   0x262
        .uleb128 0xc
        .long   0x8e
        .long   0x2de
        .uleb128 0xd
        .long   0x38
        .byte   0x13
        .byte   0
        .uleb128 0x6
        .byte   0x8
        .long   0x2e4
        .uleb128 0xe
        .long   0x8e
        .uleb128 0xf
        .byte   0x1
        .long   .LASF50
        .byte   0x2
        .byte   0x67
        .long   0x62
        .byte   0x3
        .byte   0x1
        .long   0x308
        .uleb128 0x10
        .long   .LASF51
        .byte   0x2
        .byte   0x67
        .long   0x2de
        .uleb128 0x11
        .byte   0
        .uleb128 0x12
        .byte   0x1
        .long   .LASF52
        .byte   0x1
        .byte   0x2
        .long   0x62
        .quad   .LFB31
        .quad   .LFE31
        .long   .LLST0
        .long   0x381
        .uleb128 0x13
        .long   0x2e9
        .quad   .LBB6
        .quad   .LBE6
        .byte   0x1
        .byte   0x3
        .long   0x358
        .uleb128 0x14
        .long   0x2fb
        .byte   0xa
        .byte   0x3
        .quad   .LC0
        .byte   0x9f
        .byte   0
        .uleb128 0x15
        .long   0x2e9
        .quad   .LBB8
        .quad   .LBE8
        .byte   0x1
        .byte   0x4
        .uleb128 0x14
        .long   0x2fb
        .byte   0xa
        .byte   0x3
        .quad   .LC1
        .byte   0x9f
        .byte   0
        .byte   0
        .uleb128 0x16
        .long   .LASF45
        .byte   0x6
        .byte   0xa9
        .long   0x2a6
        .byte   0x1
        .byte   0x1
        .uleb128 0x16
        .long   .LASF46
        .byte   0x6
        .byte   0xaa
        .long   0x2a6
        .byte   0x1
        .byte   0x1
        .byte   0
        .section        .debug_abbrev,"",@progbits
.Ldebug_abbrev0:
        .uleb128 0x1
        .uleb128 0x11
        .byte   0x1
        .uleb128 0x25
        .uleb128 0xe
        .uleb128 0x13
        .uleb128 0xb
        .uleb128 0x1b
        .uleb128 0xe
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x52
        .uleb128 0x1
        .uleb128 0x55
        .uleb128 0x6
        .uleb128 0x10
        .uleb128 0x6
        .byte   0
        .byte   0
        .uleb128 0x2
        .uleb128 0x16
        .byte   0
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x3
        .uleb128 0x24
        .byte   0
        .uleb128 0xb
        .uleb128 0xb
        .uleb128 0x3e
        .uleb128 0xb
        .uleb128 0x3
        .uleb128 0xe
        .byte   0
        .byte   0
        .uleb128 0x4
        .uleb128 0x24
        .byte   0
        .uleb128 0xb
        .uleb128 0xb
        .uleb128 0x3e
        .uleb128 0xb
        .uleb128 0x3
        .uleb128 0x8
        .byte   0
        .byte   0
        .uleb128 0x5
        .uleb128 0xf
        .byte   0
        .uleb128 0xb
        .uleb128 0xb
        .byte   0
        .byte   0
        .uleb128 0x6
        .uleb128 0xf
        .byte   0
        .uleb128 0xb
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x7
        .uleb128 0x13
        .byte   0x1
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0xb
        .uleb128 0xb
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0x5
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x8
        .uleb128 0xd
        .byte   0
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0x5
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x38
        .uleb128 0xa
        .byte   0
        .byte   0
        .uleb128 0x9
        .uleb128 0x16
        .byte   0
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .byte   0
        .byte   0
        .uleb128 0xa
        .uleb128 0x13
        .byte   0x1
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0xb
        .uleb128 0xb
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0xb
        .uleb128 0xd
        .byte   0
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x38
        .uleb128 0xa
        .byte   0
        .byte   0
        .uleb128 0xc
        .uleb128 0x1
        .byte   0x1
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0xd
        .uleb128 0x21
        .byte   0
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x2f
        .uleb128 0xb
        .byte   0
        .byte   0
        .uleb128 0xe
        .uleb128 0x26
        .byte   0
        .uleb128 0x49
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0xf
        .uleb128 0x2e
        .byte   0x1
        .uleb128 0x3f
        .uleb128 0xc
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x20
        .uleb128 0xb
        .uleb128 0x34
        .uleb128 0xc
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x10
        .uleb128 0x5
        .byte   0
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x11
        .uleb128 0x18
        .byte   0
        .byte   0
        .byte   0
        .uleb128 0x12
        .uleb128 0x2e
        .byte   0x1
        .uleb128 0x3f
        .uleb128 0xc
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x1
        .uleb128 0x40
        .uleb128 0x6
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x13
        .uleb128 0x1d
        .byte   0x1
        .uleb128 0x31
        .uleb128 0x13
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x1
        .uleb128 0x58
        .uleb128 0xb
        .uleb128 0x59
        .uleb128 0xb
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x14
        .uleb128 0x5
        .byte   0
        .uleb128 0x31
        .uleb128 0x13
        .uleb128 0x2
        .uleb128 0xa
        .byte   0
        .byte   0
        .uleb128 0x15
        .uleb128 0x1d
        .byte   0x1
        .uleb128 0x31
        .uleb128 0x13
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x1
        .uleb128 0x58
        .uleb128 0xb
        .uleb128 0x59
        .uleb128 0xb
        .byte   0
        .byte   0
        .uleb128 0x16
        .uleb128 0x34
        .byte   0
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x3f
        .uleb128 0xc
        .uleb128 0x3c
        .uleb128 0xc
        .byte   0
        .byte   0
        .byte   0
        .section        .debug_loc,"",@progbits
.Ldebug_loc0:
.LLST0:
        .quad   .LFB31
        .quad   .LCFI0
        .value  0x2
        .byte   0x77
        .sleb128 8
        .quad   .LCFI0
        .quad   .LCFI1
        .value  0x2
        .byte   0x77
        .sleb128 16
        .quad   .LCFI1
        .quad   .LFE31
        .value  0x2
        .byte   0x77
        .sleb128 8
        .quad   0
        .quad   0
        .section        .debug_aranges,"",@progbits
        .long   0x2c
        .value  0x2
        .long   .Ldebug_info0
        .byte   0x8
        .byte   0
        .value  0
        .value  0
        .quad   .LFB31
        .quad   .LFE31-.LFB31
        .quad   0
        .quad   0
        .section        .debug_ranges,"",@progbits
.Ldebug_ranges0:
        .quad   .LFB31
        .quad   .LFE31
        .quad   0
        .quad   0
        .section        .debug_line,"",@progbits
.Ldebug_line0:
        .section        .debug_str,"MS",@progbits,1
.LASF19:
        .string "_IO_buf_end"
.LASF27:
        .string "_old_offset"
.LASF22:
        .string "_IO_save_end"
.LASF5:
        .string "short int"
.LASF7:
        .string "size_t"
.LASF47:
        .string "GNU C++ 4.6.3"
.LASF32:
        .string "_offset"
.LASF16:
        .string "_IO_write_ptr"
.LASF11:
        .string "_flags"
.LASF18:
        .string "_IO_buf_base"
.LASF23:
        .string "_markers"
.LASF13:
        .string "_IO_read_end"
.LASF31:
        .string "_lock"
.LASF6:
        .string "long int"
.LASF50:
        .string "printf"
.LASF28:
        .string "_cur_column"
.LASF44:
        .string "_pos"
.LASF43:
        .string "_sbuf"
.LASF40:
        .string "_IO_FILE"
.LASF1:
        .string "unsigned char"
.LASF4:
        .string "signed char"
.LASF3:
        .string "unsigned int"
.LASF41:
        .string "_IO_marker"
.LASF30:
        .string "_shortbuf"
.LASF15:
        .string "_IO_write_base"
.LASF39:
        .string "_unused2"
.LASF12:
        .string "_IO_read_ptr"
.LASF2:
        .string "short unsigned int"
.LASF10:
        .string "char"
.LASF52:
        .string "main"
.LASF42:
        .string "_next"
.LASF33:
        .string "__pad1"
.LASF34:
        .string "__pad2"
.LASF35:
        .string "__pad3"
.LASF36:
        .string "__pad4"
.LASF37:
        .string "__pad5"
.LASF51:
        .string "__fmt"
.LASF0:
        .string "long unsigned int"
.LASF17:
        .string "_IO_write_end"
.LASF9:
        .string "__off64_t"
.LASF8:
        .string "__off_t"
.LASF24:
        .string "_chain"
.LASF21:
        .string "_IO_backup_base"
.LASF45:
        .string "stdin"
.LASF26:
        .string "_flags2"
.LASF38:
        .string "_mode"
.LASF14:
        .string "_IO_read_base"
.LASF29:
        .string "_vtable_offset"
.LASF48:
        .string "/home/mgodbolt/dev/compiler-explorer"
.LASF20:
        .string "_IO_save_base"
.LASF25:
        .string "_fileno"
.LASF46:
        .string "stdout"
.LASF49:
        .string "_IO_lock_t"
        .ident  "GCC: (Ubuntu/Linaro 4.6.3-1ubuntu5) 4.6.3"
        .section        .note.GNU-stack,"",@progbits
