  .section .mdebug.abi32
  .previous
  .nan legacy
  .module fp=32
  .module nooddspreg
  .abicalls
  .option pic0
  .text
$Ltext0:
  .section .rodata.str1.4,"aMS",@progbits,1
  .align 2
$LC0:
  .ascii "hello world\000"
  .text
  .align 2
  .globl main
$LFB12 = .
  .file 1 "/tmp/compiler-explorer-compiler118023-63-r26q30.aivx/example.cpp"
  .loc 1 2 0
  .cfi_startproc
  .set nomips16
  .set nomicromips
  .ent main
  .type main, @function
main:
  .frame $sp,32,$31 # vars= 0, regs= 1/0, args= 16, gp= 8
  .mask 0x80000000,-4
  .fmask 0x00000000,0
  .set noreorder
  .set nomacro
  addiu $sp,$sp,-32
  .cfi_def_cfa_offset 32
  sw $31,28($sp)
  .cfi_offset 31, -4
  .loc 1 3 0
  lui $4,%hi($LC0)
  addiu $4,$4,%lo($LC0)
  jal puts
  nop

$LVL0 = .
  .loc 1 4 0
  move $2,$0
  lw $31,28($sp)
  nop
  j $31
  addiu $sp,$sp,32

  .cfi_def_cfa_offset 0
  .cfi_restore 31
  .set macro
  .set reorder
  .end main
  .cfi_endproc
$LFE12:
  .size main, .-main
$Letext0:
  .file 2 "/opt/compiler-explorer/mips/gcc-5.4.0/mips-unknown-linux-gnu/lib/gcc/mips-unknown-linux-gnu/5.4.0/include/stddef.h"
  .file 3 "/opt/compiler-explorer/mips/gcc-5.4.0/mips-unknown-linux-gnu/mips-unknown-linux-gnu/sysroot/usr/include/bits/types.h"
  .file 4 "/opt/compiler-explorer/mips/gcc-5.4.0/mips-unknown-linux-gnu/mips-unknown-linux-gnu/sysroot/usr/include/libio.h"
  .file 5 "/opt/compiler-explorer/mips/gcc-5.4.0/mips-unknown-linux-gnu/mips-unknown-linux-gnu/sysroot/usr/include/stdio.h"
  .section .debug_info,"",@progbits
$Ldebug_info0:
  .4byte 0x2dd
  .2byte 0x4
  .4byte $Ldebug_abbrev0
  .byte 0x4
  .uleb128 0x1
  .4byte $LASF51
  .byte 0x4
  .4byte $LASF52
  .4byte $Ltext0
  .4byte $Letext0-$Ltext0
  .4byte $Ldebug_line0
  .uleb128 0x2
  .4byte $LASF8
  .byte 0x2
  .byte 0xd8
  .4byte 0x2c
  .uleb128 0x3
  .byte 0x4
  .byte 0x7
  .4byte $LASF0
  .uleb128 0x3
  .byte 0x1
  .byte 0x8
  .4byte $LASF1
  .uleb128 0x3
  .byte 0x2
  .byte 0x7
  .4byte $LASF2
  .uleb128 0x3
  .byte 0x4
  .byte 0x7
  .4byte $LASF3
  .uleb128 0x3
  .byte 0x1
  .byte 0x6
  .4byte $LASF4
  .uleb128 0x3
  .byte 0x2
  .byte 0x5
  .4byte $LASF5
  .uleb128 0x4
  .byte 0x4
  .byte 0x5
  .ascii "int\000"
  .uleb128 0x3
  .byte 0x8
  .byte 0x5
  .4byte $LASF6
  .uleb128 0x3
  .byte 0x8
  .byte 0x7
  .4byte $LASF7
  .uleb128 0x2
  .4byte $LASF9
  .byte 0x3
  .byte 0x37
  .4byte 0x5d
  .uleb128 0x2
  .4byte $LASF10
  .byte 0x3
  .byte 0x8c
  .4byte 0x81
  .uleb128 0x3
  .byte 0x4
  .byte 0x5
  .4byte $LASF11
  .uleb128 0x2
  .4byte $LASF12
  .byte 0x3
  .byte 0x8d
  .4byte 0x6b
  .uleb128 0x3
  .byte 0x4
  .byte 0x7
  .4byte $LASF13
  .uleb128 0x5
  .byte 0x4
  .uleb128 0x6
  .byte 0x4
  .4byte 0xa2
  .uleb128 0x3
  .byte 0x1
  .byte 0x6
  .4byte $LASF14
  .uleb128 0x7
  .4byte $LASF44
  .byte 0x98
  .byte 0x4
  .byte 0xf1
  .4byte 0x226
  .uleb128 0x8
  .4byte $LASF15
  .byte 0x4
  .byte 0xf2
  .4byte 0x56
  .byte 0
  .uleb128 0x8
  .4byte $LASF16
  .byte 0x4
  .byte 0xf7
  .4byte 0x9c
  .byte 0x4
  .uleb128 0x8
  .4byte $LASF17
  .byte 0x4
  .byte 0xf8
  .4byte 0x9c
  .byte 0x8
  .uleb128 0x8
  .4byte $LASF18
  .byte 0x4
  .byte 0xf9
  .4byte 0x9c
  .byte 0xc
  .uleb128 0x8
  .4byte $LASF19
  .byte 0x4
  .byte 0xfa
  .4byte 0x9c
  .byte 0x10
  .uleb128 0x8
  .4byte $LASF20
  .byte 0x4
  .byte 0xfb
  .4byte 0x9c
  .byte 0x14
  .uleb128 0x8
  .4byte $LASF21
  .byte 0x4
  .byte 0xfc
  .4byte 0x9c
  .byte 0x18
  .uleb128 0x8
  .4byte $LASF22
  .byte 0x4
  .byte 0xfd
  .4byte 0x9c
  .byte 0x1c
  .uleb128 0x8
  .4byte $LASF23
  .byte 0x4
  .byte 0xfe
  .4byte 0x9c
  .byte 0x20
  .uleb128 0x9
  .4byte $LASF24
  .byte 0x4
  .2byte 0x100
  .4byte 0x9c
  .byte 0x24
  .uleb128 0x9
  .4byte $LASF25
  .byte 0x4
  .2byte 0x101
  .4byte 0x9c
  .byte 0x28
  .uleb128 0x9
  .4byte $LASF26
  .byte 0x4
  .2byte 0x102
  .4byte 0x9c
  .byte 0x2c
  .uleb128 0x9
  .4byte $LASF27
  .byte 0x4
  .2byte 0x104
  .4byte 0x25e
  .byte 0x30
  .uleb128 0x9
  .4byte $LASF28
  .byte 0x4
  .2byte 0x106
  .4byte 0x264
  .byte 0x34
  .uleb128 0x9
  .4byte $LASF29
  .byte 0x4
  .2byte 0x108
  .4byte 0x56
  .byte 0x38
  .uleb128 0x9
  .4byte $LASF30
  .byte 0x4
  .2byte 0x10c
  .4byte 0x56
  .byte 0x3c
  .uleb128 0x9
  .4byte $LASF31
  .byte 0x4
  .2byte 0x10e
  .4byte 0x76
  .byte 0x40
  .uleb128 0x9
  .4byte $LASF32
  .byte 0x4
  .2byte 0x112
  .4byte 0x3a
  .byte 0x44
  .uleb128 0x9
  .4byte $LASF33
  .byte 0x4
  .2byte 0x113
  .4byte 0x48
  .byte 0x46
  .uleb128 0x9
  .4byte $LASF34
  .byte 0x4
  .2byte 0x114
  .4byte 0x26a
  .byte 0x47
  .uleb128 0x9
  .4byte $LASF35
  .byte 0x4
  .2byte 0x118
  .4byte 0x27a
  .byte 0x48
  .uleb128 0x9
  .4byte $LASF36
  .byte 0x4
  .2byte 0x121
  .4byte 0x88
  .byte 0x50
  .uleb128 0x9
  .4byte $LASF37
  .byte 0x4
  .2byte 0x129
  .4byte 0x9a
  .byte 0x58
  .uleb128 0x9
  .4byte $LASF38
  .byte 0x4
  .2byte 0x12a
  .4byte 0x9a
  .byte 0x5c
  .uleb128 0x9
  .4byte $LASF39
  .byte 0x4
  .2byte 0x12b
  .4byte 0x9a
  .byte 0x60
  .uleb128 0x9
  .4byte $LASF40
  .byte 0x4
  .2byte 0x12c
  .4byte 0x9a
  .byte 0x64
  .uleb128 0x9
  .4byte $LASF41
  .byte 0x4
  .2byte 0x12e
  .4byte 0x21
  .byte 0x68
  .uleb128 0x9
  .4byte $LASF42
  .byte 0x4
  .2byte 0x12f
  .4byte 0x56
  .byte 0x6c
  .uleb128 0x9
  .4byte $LASF43
  .byte 0x4
  .2byte 0x131
  .4byte 0x280
  .byte 0x70
  .byte 0
  .uleb128 0xa
  .4byte $LASF53
  .byte 0x4
  .byte 0x96
  .uleb128 0x7
  .4byte $LASF45
  .byte 0xc
  .byte 0x4
  .byte 0x9c
  .4byte 0x25e
  .uleb128 0x8
  .4byte $LASF46
  .byte 0x4
  .byte 0x9d
  .4byte 0x25e
  .byte 0
  .uleb128 0x8
  .4byte $LASF47
  .byte 0x4
  .byte 0x9e
  .4byte 0x264
  .byte 0x4
  .uleb128 0x8
  .4byte $LASF48
  .byte 0x4
  .byte 0xa2
  .4byte 0x56
  .byte 0x8
  .byte 0
  .uleb128 0x6
  .byte 0x4
  .4byte 0x22d
  .uleb128 0x6
  .byte 0x4
  .4byte 0xa9
  .uleb128 0xb
  .4byte 0xa2
  .4byte 0x27a
  .uleb128 0xc
  .4byte 0x93
  .byte 0
  .byte 0
  .uleb128 0x6
  .byte 0x4
  .4byte 0x226
  .uleb128 0xb
  .4byte 0xa2
  .4byte 0x290
  .uleb128 0xc
  .4byte 0x93
  .byte 0x27
  .byte 0
  .uleb128 0xd
  .4byte $LASF54
  .byte 0x1
  .byte 0x2
  .4byte 0x56
  .4byte $LFB12
  .4byte $LFE12-$LFB12
  .uleb128 0x1
  .byte 0x9c
  .4byte 0x2bd
  .uleb128 0xe
  .4byte $LVL0
  .4byte 0x2d3
  .uleb128 0xf
  .uleb128 0x1
  .byte 0x54
  .uleb128 0x5
  .byte 0x3
  .4byte $LC0
  .byte 0
  .byte 0
  .uleb128 0x10
  .4byte $LASF49
  .byte 0x5
  .byte 0xab
  .4byte 0x264
  .uleb128 0x10
  .4byte $LASF50
  .byte 0x5
  .byte 0xac
  .4byte 0x264
  .uleb128 0x11
  .4byte $LASF55
  .4byte $LASF56
  .4byte $LASF55
  .byte 0
  .section .debug_abbrev,"",@progbits
$Ldebug_abbrev0:
  .uleb128 0x1
  .uleb128 0x11
  .byte 0x1
  .uleb128 0x25
  .uleb128 0xe
  .uleb128 0x13
  .uleb128 0xb
  .uleb128 0x3
  .uleb128 0xe
  .uleb128 0x11
  .uleb128 0x1
  .uleb128 0x12
  .uleb128 0x6
  .uleb128 0x10
  .uleb128 0x17
  .byte 0
  .byte 0
  .uleb128 0x2
  .uleb128 0x16
  .byte 0
  .uleb128 0x3
  .uleb128 0xe
  .uleb128 0x3a
  .uleb128 0xb
  .uleb128 0x3b
  .uleb128 0xb
  .uleb128 0x49
  .uleb128 0x13
  .byte 0
  .byte 0
  .uleb128 0x3
  .uleb128 0x24
  .byte 0
  .uleb128 0xb
  .uleb128 0xb
  .uleb128 0x3e
  .uleb128 0xb
  .uleb128 0x3
  .uleb128 0xe
  .byte 0
  .byte 0
  .uleb128 0x4
  .uleb128 0x24
  .byte 0
  .uleb128 0xb
  .uleb128 0xb
  .uleb128 0x3e
  .uleb128 0xb
  .uleb128 0x3
  .uleb128 0x8
  .byte 0
  .byte 0
  .uleb128 0x5
  .uleb128 0xf
  .byte 0
  .uleb128 0xb
  .uleb128 0xb
  .byte 0
  .byte 0
  .uleb128 0x6
  .uleb128 0xf
  .byte 0
  .uleb128 0xb
  .uleb128 0xb
  .uleb128 0x49
  .uleb128 0x13
  .byte 0
  .byte 0
  .uleb128 0x7
  .uleb128 0x13
  .byte 0x1
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
  .byte 0
  .byte 0
  .uleb128 0x8
  .uleb128 0xd
  .byte 0
  .uleb128 0x3
  .uleb128 0xe
  .uleb128 0x3a
  .uleb128 0xb
  .uleb128 0x3b
  .uleb128 0xb
  .uleb128 0x49
  .uleb128 0x13
  .uleb128 0x38
  .uleb128 0xb
  .byte 0
  .byte 0
  .uleb128 0x9
  .uleb128 0xd
  .byte 0
  .uleb128 0x3
  .uleb128 0xe
  .uleb128 0x3a
  .uleb128 0xb
  .uleb128 0x3b
  .uleb128 0x5
  .uleb128 0x49
  .uleb128 0x13
  .uleb128 0x38
  .uleb128 0xb
  .byte 0
  .byte 0
  .uleb128 0xa
  .uleb128 0x16
  .byte 0
  .uleb128 0x3
  .uleb128 0xe
  .uleb128 0x3a
  .uleb128 0xb
  .uleb128 0x3b
  .uleb128 0xb
  .byte 0
  .byte 0
  .uleb128 0xb
  .uleb128 0x1
  .byte 0x1
  .uleb128 0x49
  .uleb128 0x13
  .uleb128 0x1
  .uleb128 0x13
  .byte 0
  .byte 0
  .uleb128 0xc
  .uleb128 0x21
  .byte 0
  .uleb128 0x49
  .uleb128 0x13
  .uleb128 0x2f
  .uleb128 0xb
  .byte 0
  .byte 0
  .uleb128 0xd
  .uleb128 0x2e
  .byte 0x1
  .uleb128 0x3f
  .uleb128 0x19
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
  .uleb128 0x6
  .uleb128 0x40
  .uleb128 0x18
  .uleb128 0x2117
  .uleb128 0x19
  .uleb128 0x1
  .uleb128 0x13
  .byte 0
  .byte 0
  .uleb128 0xe
  .uleb128 0x4109
  .byte 0x1
  .uleb128 0x11
  .uleb128 0x1
  .uleb128 0x31
  .uleb128 0x13
  .byte 0
  .byte 0
  .uleb128 0xf
  .uleb128 0x410a
  .byte 0
  .uleb128 0x2
  .uleb128 0x18
  .uleb128 0x2111
  .uleb128 0x18
  .byte 0
  .byte 0
  .uleb128 0x10
  .uleb128 0x34
  .byte 0
  .uleb128 0x3
  .uleb128 0xe
  .uleb128 0x3a
  .uleb128 0xb
  .uleb128 0x3b
  .uleb128 0xb
  .uleb128 0x49
  .uleb128 0x13
  .uleb128 0x3f
  .uleb128 0x19
  .uleb128 0x3c
  .uleb128 0x19
  .byte 0
  .byte 0
  .uleb128 0x11
  .uleb128 0x2e
  .byte 0
  .uleb128 0x3f
  .uleb128 0x19
  .uleb128 0x3c
  .uleb128 0x19
  .uleb128 0x6e
  .uleb128 0xe
  .uleb128 0x3
  .uleb128 0xe
  .uleb128 0x6e
  .uleb128 0xe
  .byte 0
  .byte 0
  .byte 0
  .section .debug_aranges,"",@progbits
  .4byte 0x1c
  .2byte 0x2
  .4byte $Ldebug_info0
  .byte 0x4
  .byte 0
  .2byte 0
  .2byte 0
  .4byte $Ltext0
  .4byte $Letext0-$Ltext0
  .4byte 0
  .4byte 0
  .section .debug_line,"",@progbits
$Ldebug_line0:
  .section .debug_str,"MS",@progbits,1
$LASF23:
  .ascii "_IO_buf_end\000"
$LASF9:
  .ascii "__quad_t\000"
$LASF31:
  .ascii "_old_offset\000"
$LASF56:
  .ascii "__builtin_puts\000"
$LASF26:
  .ascii "_IO_save_end\000"
$LASF5:
  .ascii "short int\000"
$LASF8:
  .ascii "size_t\000"
$LASF13:
  .ascii "sizetype\000"
$LASF36:
  .ascii "_offset\000"
$LASF20:
  .ascii "_IO_write_ptr\000"
$LASF15:
  .ascii "_flags\000"
$LASF22:
  .ascii "_IO_buf_base\000"
$LASF27:
  .ascii "_markers\000"
$LASF17:
  .ascii "_IO_read_end\000"
$LASF51:
  .ascii "GNU C++ 5.4.0 -meb -march=mips1 -mabi=32 -mhard-float -m"
  .ascii "llsc -mplt -mips1 -mno-shared -g -O\000"
$LASF6:
  .ascii "long long int\000"
$LASF35:
  .ascii "_lock\000"
$LASF11:
  .ascii "long int\000"
$LASF32:
  .ascii "_cur_column\000"
$LASF52:
  .ascii "/tmp/compiler-explorer-compiler118023-63-r26q30.aivx/exa"
  .ascii "mple.cpp\000"
$LASF48:
  .ascii "_pos\000"
$LASF47:
  .ascii "_sbuf\000"
$LASF44:
  .ascii "_IO_FILE\000"
$LASF1:
  .ascii "unsigned char\000"
$LASF4:
  .ascii "signed char\000"
$LASF7:
  .ascii "long long unsigned int\000"
$LASF0:
  .ascii "unsigned int\000"
$LASF45:
  .ascii "_IO_marker\000"
$LASF34:
  .ascii "_shortbuf\000"
$LASF55:
  .ascii "puts\000"
$LASF19:
  .ascii "_IO_write_base\000"
$LASF43:
  .ascii "_unused2\000"
$LASF16:
  .ascii "_IO_read_ptr\000"
$LASF2:
  .ascii "short unsigned int\000"
$LASF14:
  .ascii "char\000"
$LASF54:
  .ascii "main\000"
$LASF46:
  .ascii "_next\000"
$LASF37:
  .ascii "__pad1\000"
$LASF38:
  .ascii "__pad2\000"
$LASF39:
  .ascii "__pad3\000"
$LASF40:
  .ascii "__pad4\000"
$LASF41:
  .ascii "__pad5\000"
$LASF3:
  .ascii "long unsigned int\000"
$LASF21:
  .ascii "_IO_write_end\000"
$LASF12:
  .ascii "__off64_t\000"
$LASF10:
  .ascii "__off_t\000"
$LASF28:
  .ascii "_chain\000"
$LASF25:
  .ascii "_IO_backup_base\000"
$LASF49:
  .ascii "stdin\000"
$LASF30:
  .ascii "_flags2\000"
$LASF42:
  .ascii "_mode\000"
$LASF18:
  .ascii "_IO_read_base\000"
$LASF33:
  .ascii "_vtable_offset\000"
$LASF24:
  .ascii "_IO_save_base\000"
$LASF29:
  .ascii "_fileno\000"
$LASF50:
  .ascii "stdout\000"
$LASF53:
  .ascii "_IO_lock_t\000"
  .ident "GCC: (crosstool-NG crosstool-ng-1.23.0-rc2) 5.4.0"
