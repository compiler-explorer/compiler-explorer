	.file	"example.c"
	.intel_syntax noprefix
	.text
.Ltext0:
	.file 0 "/home/partouf/Documents/intsizetest" "example.c"
	.globl	size
	.bss
	.align 4
	.type	size, @object
	.size	size, 4
size:
	.zero	4
	.text
.Letext0:
	.file 1 "example.c"
	.section	.debug_info,"",@progbits
.Ldebug_info0:
	.long	0x38
	.value	0x5
	.byte	0x1
	.byte	0x8
	.long	.Ldebug_abbrev0
	.uleb128 0x1
	.long	.LASF2
	.byte	0x1d
	.long	.LASF0
	.long	.LASF1
	.long	.Ldebug_line0
	.uleb128 0x2
	.long	.LASF3
	.byte	0x1
	.byte	0x1
	.byte	0x5
	.long	0x34
	.uleb128 0x9
	.byte	0x3
	.quad	size
	.uleb128 0x3
	.byte	0x4
	.byte	0x5
	.string	"int"
	.byte	0
	.section	.debug_abbrev,"",@progbits
.Ldebug_abbrev0:
	.uleb128 0x1
	.uleb128 0x11
	.byte	0x1
	.uleb128 0x25
	.uleb128 0xe
	.uleb128 0x13
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x1f
	.uleb128 0x1b
	.uleb128 0x1f
	.uleb128 0x10
	.uleb128 0x17
	.byte	0
	.byte	0
	.uleb128 0x2
	.uleb128 0x34
	.byte	0
	.uleb128 0x3
	.uleb128 0xe
	.uleb128 0x3a
	.uleb128 0xb
	.uleb128 0x3b
	.uleb128 0xb
	.uleb128 0x39
	.uleb128 0xb
	.uleb128 0x49
	.uleb128 0x13
	.uleb128 0x3f
	.uleb128 0x19
	.uleb128 0x2
	.uleb128 0x18
	.byte	0
	.byte	0
	.uleb128 0x3
	.uleb128 0x24
	.byte	0
	.uleb128 0xb
	.uleb128 0xb
	.uleb128 0x3e
	.uleb128 0xb
	.uleb128 0x3
	.uleb128 0x8
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_aranges,"",@progbits
	.long	0x1c
	.value	0x2
	.long	.Ldebug_info0
	.byte	0x8
	.byte	0
	.value	0
	.value	0
	.quad	0
	.quad	0
	.section	.debug_line,"",@progbits
.Ldebug_line0:
	.section	.debug_str,"MS",@progbits,1
.LASF2:
	.string	"GNU C17 12.2.0 -masm=intel -mtune=generic -march=x86-64 -g -O2"
.LASF3:
	.string	"size"
	.section	.debug_line_str,"MS",@progbits,1
.LASF0:
	.string	"example.c"
.LASF1:
	.string	"/home/partouf/Documents/intsizetest"
	.ident	"GCC: (Compiler-Explorer-Build-gcc--binutils-2.38) 12.2.0"
	.section	.note.GNU-stack,"",@progbits