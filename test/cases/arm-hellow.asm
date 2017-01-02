        .syntax unified
        .arch armv7-a
        .eabi_attribute 27, 3
        .fpu vfpv3-d16
        .eabi_attribute 20, 1
        .eabi_attribute 21, 1
        .eabi_attribute 23, 3
        .eabi_attribute 24, 1
        .eabi_attribute 25, 1
        .eabi_attribute 26, 2
        .eabi_attribute 30, 2
        .eabi_attribute 34, 1
        .eabi_attribute 18, 4
        .thumb
        .file   ""
        .text
.Ltext0:
        .cfi_sections   .debug_frame
        .section        .text.startup,"ax",%progbits
        .align  2
        .global main
        .thumb
        .thumb_func
        .type   main, %function
main:
        .fnstart
.LFB31:
        .file 1 "<stdin>"
        .loc 1 2 0
        .cfi_startproc
        @ args = 0, pretend = 0, frame = 0
        @ frame_needed = 0, uses_anonymous_args = 0
.LVL0:
        push    {r3, lr}
        .save {r3, lr}
.LCFI0:
        .cfi_def_cfa_offset 8
        .cfi_offset 14, -4
        .cfi_offset 3, -8
.LBB6:
.LBB7:
        .file 2 "/usr/arm-linux-gnueabi/include/bits/stdio2.h"
        .loc 2 105 0
        movs    r0, #1
        movw    r1, #:lower16:.LC0
        movt    r1, #:upper16:.LC0
        bl      __printf_chk
.LVL1:
.LBE7:
.LBE6:
.LBB8:
.LBB9:
        movs    r0, #1
        movw    r1, #:lower16:.LC1
        movt    r1, #:upper16:.LC1
        bl      __printf_chk
.LBE9:
.LBE8:
        .loc 1 5 0
        movs    r0, #0
        pop     {r3, pc}
        .cfi_endproc
.LFE31:
        .fnend
        .size   main, .-main
        .section        .rodata.str1.4,"aMS",%progbits,1
        .align  2
.LC0:
        .ascii  "Hello world\000"
.LC1:
        .ascii  "moo\012\000"
        .text
.Letext0:
        .file 3 "/usr/lib/gcc/arm-linux-gnueabi/4.6/include/stddef.h"
        .file 4 "/usr/arm-linux-gnueabi/include/bits/types.h"
        .file 5 "/usr/arm-linux-gnueabi/include/libio.h"
        .file 6 "/usr/arm-linux-gnueabi/include/stdio.h"
        .section        .debug_info,"",%progbits
.Ldebug_info0:
        .4byte  0x37d
        .2byte  0x2
        .4byte  .Ldebug_abbrev0
        .byte   0x4
        .uleb128 0x1
        .4byte  .LASF50
        .byte   0x4
        .4byte  .LASF51
        .4byte  0
        .4byte  0
        .4byte  .Ldebug_ranges0+0
        .4byte  .Ldebug_line0
        .uleb128 0x2
        .4byte  .LASF8
        .byte   0x3
        .byte   0xd4
        .4byte  0x30
        .uleb128 0x3
        .byte   0x4
        .byte   0x7
        .4byte  .LASF0
        .uleb128 0x3
        .byte   0x1
        .byte   0x8
        .4byte  .LASF1
        .uleb128 0x3
        .byte   0x2
        .byte   0x7
        .4byte  .LASF2
        .uleb128 0x3
        .byte   0x4
        .byte   0x7
        .4byte  .LASF3
        .uleb128 0x3
        .byte   0x1
        .byte   0x6
        .4byte  .LASF4
        .uleb128 0x3
        .byte   0x2
        .byte   0x5
        .4byte  .LASF5
        .uleb128 0x4
        .byte   0x4
        .byte   0x5
        .ascii  "int\000"
        .uleb128 0x3
        .byte   0x8
        .byte   0x5
        .4byte  .LASF6
        .uleb128 0x3
        .byte   0x8
        .byte   0x7
        .4byte  .LASF7
        .uleb128 0x2
        .4byte  .LASF9
        .byte   0x4
        .byte   0x38
        .4byte  0x61
        .uleb128 0x2
        .4byte  .LASF10
        .byte   0x4
        .byte   0x8d
        .4byte  0x85
        .uleb128 0x3
        .byte   0x4
        .byte   0x5
        .4byte  .LASF11
        .uleb128 0x2
        .4byte  .LASF12
        .byte   0x4
        .byte   0x8e
        .4byte  0x6f
        .uleb128 0x5
        .byte   0x4
        .uleb128 0x6
        .byte   0x4
        .4byte  0x9f
        .uleb128 0x3
        .byte   0x1
        .byte   0x8
        .4byte  .LASF13
        .uleb128 0x7
        .4byte  .LASF43
        .byte   0x98
        .byte   0x5
        .2byte  0x111
        .4byte  0x267
        .uleb128 0x8
        .4byte  .LASF14
        .byte   0x5
        .2byte  0x112
        .4byte  0x5a
        .byte   0x2
        .byte   0x23
        .uleb128 0
        .uleb128 0x8
        .4byte  .LASF15
        .byte   0x5
        .2byte  0x117
        .4byte  0x99
        .byte   0x2
        .byte   0x23
        .uleb128 0x4
        .uleb128 0x8
        .4byte  .LASF16
        .byte   0x5
        .2byte  0x118
        .4byte  0x99
        .byte   0x2
        .byte   0x23
        .uleb128 0x8
        .uleb128 0x8
        .4byte  .LASF17
        .byte   0x5
        .2byte  0x119
        .4byte  0x99
        .byte   0x2
        .byte   0x23
        .uleb128 0xc
        .uleb128 0x8
        .4byte  .LASF18
        .byte   0x5
        .2byte  0x11a
        .4byte  0x99
        .byte   0x2
        .byte   0x23
        .uleb128 0x10
        .uleb128 0x8
        .4byte  .LASF19
        .byte   0x5
        .2byte  0x11b
        .4byte  0x99
        .byte   0x2
        .byte   0x23
        .uleb128 0x14
        .uleb128 0x8
        .4byte  .LASF20
        .byte   0x5
        .2byte  0x11c
        .4byte  0x99
        .byte   0x2
        .byte   0x23
        .uleb128 0x18
        .uleb128 0x8
        .4byte  .LASF21
        .byte   0x5
        .2byte  0x11d
        .4byte  0x99
        .byte   0x2
        .byte   0x23
        .uleb128 0x1c
        .uleb128 0x8
        .4byte  .LASF22
        .byte   0x5
        .2byte  0x11e
        .4byte  0x99
        .byte   0x2
        .byte   0x23
        .uleb128 0x20
        .uleb128 0x8
        .4byte  .LASF23
        .byte   0x5
        .2byte  0x120
        .4byte  0x99
        .byte   0x2
        .byte   0x23
        .uleb128 0x24
        .uleb128 0x8
        .4byte  .LASF24
        .byte   0x5
        .2byte  0x121
        .4byte  0x99
        .byte   0x2
        .byte   0x23
        .uleb128 0x28
        .uleb128 0x8
        .4byte  .LASF25
        .byte   0x5
        .2byte  0x122
        .4byte  0x99
        .byte   0x2
        .byte   0x23
        .uleb128 0x2c
        .uleb128 0x8
        .4byte  .LASF26
        .byte   0x5
        .2byte  0x124
        .4byte  0x2b1
        .byte   0x2
        .byte   0x23
        .uleb128 0x30
        .uleb128 0x8
        .4byte  .LASF27
        .byte   0x5
        .2byte  0x126
        .4byte  0x2b7
        .byte   0x2
        .byte   0x23
        .uleb128 0x34
        .uleb128 0x8
        .4byte  .LASF28
        .byte   0x5
        .2byte  0x128
        .4byte  0x5a
        .byte   0x2
        .byte   0x23
        .uleb128 0x38
        .uleb128 0x8
        .4byte  .LASF29
        .byte   0x5
        .2byte  0x12c
        .4byte  0x5a
        .byte   0x2
        .byte   0x23
        .uleb128 0x3c
        .uleb128 0x8
        .4byte  .LASF30
        .byte   0x5
        .2byte  0x12e
        .4byte  0x7a
        .byte   0x2
        .byte   0x23
        .uleb128 0x40
        .uleb128 0x8
        .4byte  .LASF31
        .byte   0x5
        .2byte  0x132
        .4byte  0x3e
        .byte   0x2
        .byte   0x23
        .uleb128 0x44
        .uleb128 0x8
        .4byte  .LASF32
        .byte   0x5
        .2byte  0x133
        .4byte  0x4c
        .byte   0x2
        .byte   0x23
        .uleb128 0x46
        .uleb128 0x8
        .4byte  .LASF33
        .byte   0x5
        .2byte  0x134
        .4byte  0x2bd
        .byte   0x2
        .byte   0x23
        .uleb128 0x47
        .uleb128 0x8
        .4byte  .LASF34
        .byte   0x5
        .2byte  0x138
        .4byte  0x2cd
        .byte   0x2
        .byte   0x23
        .uleb128 0x48
        .uleb128 0x8
        .4byte  .LASF35
        .byte   0x5
        .2byte  0x141
        .4byte  0x8c
        .byte   0x2
        .byte   0x23
        .uleb128 0x50
        .uleb128 0x8
        .4byte  .LASF36
        .byte   0x5
        .2byte  0x14a
        .4byte  0x97
        .byte   0x2
        .byte   0x23
        .uleb128 0x58
        .uleb128 0x8
        .4byte  .LASF37
        .byte   0x5
        .2byte  0x14b
        .4byte  0x97
        .byte   0x2
        .byte   0x23
        .uleb128 0x5c
        .uleb128 0x8
        .4byte  .LASF38
        .byte   0x5
        .2byte  0x14c
        .4byte  0x97
        .byte   0x2
        .byte   0x23
        .uleb128 0x60
        .uleb128 0x8
        .4byte  .LASF39
        .byte   0x5
        .2byte  0x14d
        .4byte  0x97
        .byte   0x2
        .byte   0x23
        .uleb128 0x64
        .uleb128 0x8
        .4byte  .LASF40
        .byte   0x5
        .2byte  0x14e
        .4byte  0x25
        .byte   0x2
        .byte   0x23
        .uleb128 0x68
        .uleb128 0x8
        .4byte  .LASF41
        .byte   0x5
        .2byte  0x150
        .4byte  0x5a
        .byte   0x2
        .byte   0x23
        .uleb128 0x6c
        .uleb128 0x8
        .4byte  .LASF42
        .byte   0x5
        .2byte  0x152
        .4byte  0x2d3
        .byte   0x2
        .byte   0x23
        .uleb128 0x70
        .byte   0
        .uleb128 0x9
        .4byte  .LASF52
        .byte   0x5
        .byte   0xb6
        .uleb128 0xa
        .4byte  .LASF44
        .byte   0xc
        .byte   0x5
        .byte   0xbc
        .4byte  0x2a5
        .uleb128 0xb
        .4byte  .LASF45
        .byte   0x5
        .byte   0xbd
        .4byte  0x2a5
        .byte   0x2
        .byte   0x23
        .uleb128 0
        .uleb128 0xb
        .4byte  .LASF46
        .byte   0x5
        .byte   0xbe
        .4byte  0x2ab
        .byte   0x2
        .byte   0x23
        .uleb128 0x4
        .uleb128 0xb
        .4byte  .LASF47
        .byte   0x5
        .byte   0xc2
        .4byte  0x5a
        .byte   0x2
        .byte   0x23
        .uleb128 0x8
        .byte   0
        .uleb128 0x6
        .byte   0x4
        .4byte  0x26e
        .uleb128 0x6
        .byte   0x4
        .4byte  0xa6
        .uleb128 0x6
        .byte   0x4
        .4byte  0x26e
        .uleb128 0x6
        .byte   0x4
        .4byte  0xa6
        .uleb128 0xc
        .4byte  0x9f
        .4byte  0x2cd
        .uleb128 0xd
        .4byte  0x30
        .byte   0
        .byte   0
        .uleb128 0x6
        .byte   0x4
        .4byte  0x267
        .uleb128 0xc
        .4byte  0x9f
        .4byte  0x2e3
        .uleb128 0xd
        .4byte  0x30
        .byte   0x27
        .byte   0
        .uleb128 0x6
        .byte   0x4
        .4byte  0x2e9
        .uleb128 0xe
        .4byte  0x9f
        .uleb128 0xf
        .byte   0x1
        .4byte  .LASF53
        .byte   0x2
        .byte   0x67
        .4byte  0x5a
        .byte   0x3
        .byte   0x1
        .4byte  0x30d
        .uleb128 0x10
        .4byte  .LASF54
        .byte   0x2
        .byte   0x67
        .4byte  0x2e3
        .uleb128 0x11
        .byte   0
        .uleb128 0x12
        .byte   0x1
        .4byte  .LASF55
        .byte   0x1
        .byte   0x2
        .4byte  0x5a
        .4byte  .LFB31
        .4byte  .LFE31
        .4byte  .LLST0
        .4byte  0x366
        .uleb128 0x13
        .4byte  0x2ee
        .4byte  .LBB6
        .4byte  .LBE6
        .byte   0x1
        .byte   0x3
        .4byte  0x349
        .uleb128 0x14
        .4byte  0x300
        .byte   0x6
        .byte   0x3
        .4byte  .LC0
        .byte   0x9f
        .byte   0
        .uleb128 0x15
        .4byte  0x2ee
        .4byte  .LBB8
        .4byte  .LBE8
        .byte   0x1
        .byte   0x4
        .uleb128 0x14
        .4byte  0x300
        .byte   0x6
        .byte   0x3
        .4byte  .LC1
        .byte   0x9f
        .byte   0
        .byte   0
        .uleb128 0x16
        .4byte  .LASF48
        .byte   0x6
        .byte   0xa9
        .4byte  0x2ab
        .byte   0x1
        .byte   0x1
        .uleb128 0x16
        .4byte  .LASF49
        .byte   0x6
        .byte   0xaa
        .4byte  0x2ab
        .byte   0x1
        .byte   0x1
        .byte   0
        .section        .debug_abbrev,"",%progbits
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
        .section        .debug_loc,"",%progbits
.Ldebug_loc0:
.LLST0:
        .4byte  .LFB31
        .4byte  .LCFI0
        .2byte  0x2
        .byte   0x7d
        .sleb128 0
        .4byte  .LCFI0
        .4byte  .LFE31
        .2byte  0x2
        .byte   0x7d
        .sleb128 8
        .4byte  0
        .4byte  0
        .section        .debug_aranges,"",%progbits
        .4byte  0x1c
        .2byte  0x2
        .4byte  .Ldebug_info0
        .byte   0x4
        .byte   0
        .2byte  0
        .2byte  0
        .4byte  .LFB31
        .4byte  .LFE31-.LFB31
        .4byte  0
        .4byte  0
        .section        .debug_ranges,"",%progbits
.Ldebug_ranges0:
        .4byte  .LFB31
        .4byte  .LFE31
        .4byte  0
        .4byte  0
        .section        .debug_line,"",%progbits
.Ldebug_line0:
        .section        .debug_str,"MS",%progbits,1
.LASF22:
        .ascii  "_IO_buf_end\000"
.LASF9:
        .ascii  "__quad_t\000"
.LASF30:
        .ascii  "_old_offset\000"
.LASF25:
        .ascii  "_IO_save_end\000"
.LASF5:
        .ascii  "short int\000"
.LASF8:
        .ascii  "size_t\000"
.LASF50:
        .ascii  "GNU C++ 4.6.3\000"
.LASF35:
        .ascii  "_offset\000"
.LASF19:
        .ascii  "_IO_write_ptr\000"
.LASF14:
        .ascii  "_flags\000"
.LASF21:
        .ascii  "_IO_buf_base\000"
.LASF26:
        .ascii  "_markers\000"
.LASF16:
        .ascii  "_IO_read_end\000"
.LASF6:
        .ascii  "long long int\000"
.LASF34:
        .ascii  "_lock\000"
.LASF11:
        .ascii  "long int\000"
.LASF53:
        .ascii  "printf\000"
.LASF31:
        .ascii  "_cur_column\000"
.LASF47:
        .ascii  "_pos\000"
.LASF46:
        .ascii  "_sbuf\000"
.LASF43:
        .ascii  "_IO_FILE\000"
.LASF1:
        .ascii  "unsigned char\000"
.LASF4:
        .ascii  "signed char\000"
.LASF7:
        .ascii  "long long unsigned int\000"
.LASF0:
        .ascii  "unsigned int\000"
.LASF44:
        .ascii  "_IO_marker\000"
.LASF33:
        .ascii  "_shortbuf\000"
.LASF18:
        .ascii  "_IO_write_base\000"
.LASF42:
        .ascii  "_unused2\000"
.LASF15:
        .ascii  "_IO_read_ptr\000"
.LASF2:
        .ascii  "short unsigned int\000"
.LASF13:
        .ascii  "char\000"
.LASF55:
        .ascii  "main\000"
.LASF45:
        .ascii  "_next\000"
.LASF36:
        .ascii  "__pad1\000"
.LASF37:
        .ascii  "__pad2\000"
.LASF38:
        .ascii  "__pad3\000"
.LASF39:
        .ascii  "__pad4\000"
.LASF40:
        .ascii  "__pad5\000"
.LASF54:
        .ascii  "__fmt\000"
.LASF3:
        .ascii  "long unsigned int\000"
.LASF20:
        .ascii  "_IO_write_end\000"
.LASF12:
        .ascii  "__off64_t\000"
.LASF10:
        .ascii  "__off_t\000"
.LASF27:
        .ascii  "_chain\000"
.LASF24:
        .ascii  "_IO_backup_base\000"
.LASF48:
        .ascii  "stdin\000"
.LASF29:
        .ascii  "_flags2\000"
.LASF41:
        .ascii  "_mode\000"
.LASF17:
        .ascii  "_IO_read_base\000"
.LASF32:
        .ascii  "_vtable_offset\000"
.LASF51:
        .ascii  "/home/mgodbolt/dev/compiler-explorer\000"
.LASF23:
        .ascii  "_IO_save_base\000"
.LASF28:
        .ascii  "_fileno\000"
.LASF49:
        .ascii  "stdout\000"
.LASF52:
        .ascii  "_IO_lock_t\000"
        .ident  "GCC: (Ubuntu/Linaro 4.6.3-1ubuntu5) 4.6.3"
        .section        .note.GNU-stack,"",%progbits
