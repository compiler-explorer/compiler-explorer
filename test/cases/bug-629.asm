        .file   "example.cpp"
        .intel_syntax noprefix
        .text
.Ltext0:
        .globl  y
        .data
        .align 4
        .type   y, @object
        .size   y, 4
y:
        .long   100
        .globl  d
        .align 4
        .type   d, @object
        .size   d, 4
d:
        .long   98
        .globl  f
        .align 8
        .type   f, @object
        .size   f, 8
f:
        .long   3435973837
        .long   1092337616
        .globl  v
        .bss
        .type   v, @object
        .size   v, 1
v:
        .zero   1
        .globl  dd
        .align 4
        .type   dd, @object
        .size   dd, 4
dd:
        .zero   4
        .globl  yuu
        .align 4
        .type   yuu, @object
        .size   yuu, 4
yuu:
        .zero   4
        .globl  jj
        .align 8
        .type   jj, @object
        .size   jj, 8
jj:
        .zero   8
        .globl  q
        .align 4
        .type   q, @object
        .size   q, 4
q:
        .zero   4
        .text
        .globl  x()
        .type   x(), @function
x():
.LFB0:
        .file 1 "/tmp/compiler-explorer-compiler1171024-7041-8xnwpl.6xrlw/example.cpp"
        .loc 1 11 0
        .cfi_startproc
        push    rbp
        .cfi_def_cfa_offset 16
        .cfi_offset 6, -16
        mov     rbp, rsp
        .cfi_def_cfa_register 6
        .loc 1 12 0
        mov     eax, DWORD PTR d[rip]
        .loc 1 13 0
        pop     rbp
        .cfi_def_cfa 7, 8
        ret
        .cfi_endproc
.LFE0:
        .size   x(), .-x()
.Letext0:
        .section        .debug_info,"",@progbits
.Ldebug_info0:
        .long   0x10f
        .value  0x4
        .long   .Ldebug_abbrev0
        .byte   0x8
        .uleb128 0x1
        .long   .LASF5
        .byte   0x4
        .long   .LASF6
        .long   .LASF7
        .quad   .Ltext0
        .quad   .Letext0-.Ltext0
        .long   .Ldebug_line0
        .uleb128 0x2
        .string "y"
        .byte   0x1
        .byte   0x2
        .long   0x40
        .uleb128 0x9
        .byte   0x3
        .quad   y
        .uleb128 0x3
        .byte   0x4
        .byte   0x5
        .string "int"
        .uleb128 0x2
        .string "d"
        .byte   0x1
        .byte   0x3
        .long   0x40
        .uleb128 0x9
        .byte   0x3
        .quad   d
        .uleb128 0x2
        .string "f"
        .byte   0x1
        .byte   0x4
        .long   0x6d
        .uleb128 0x9
        .byte   0x3
        .quad   f
        .uleb128 0x4
        .byte   0x8
        .byte   0x4
        .long   .LASF0
        .uleb128 0x2
        .string "v"
        .byte   0x1
        .byte   0x5
        .long   0x87
        .uleb128 0x9
        .byte   0x3
        .quad   v
        .uleb128 0x4
        .byte   0x1
        .byte   0x6
        .long   .LASF1
        .uleb128 0x2
        .string "dd"
        .byte   0x1
        .byte   0x6
        .long   0xa2
        .uleb128 0x9
        .byte   0x3
        .quad   dd
        .uleb128 0x4
        .byte   0x4
        .byte   0x4
        .long   .LASF2
        .uleb128 0x2
        .string "yuu"
        .byte   0x1
        .byte   0x7
        .long   0x40
        .uleb128 0x9
        .byte   0x3
        .quad   yuu
        .uleb128 0x2
        .string "jj"
        .byte   0x1
        .byte   0x8
        .long   0xd2
        .uleb128 0x9
        .byte   0x3
        .quad   jj
        .uleb128 0x4
        .byte   0x8
        .byte   0x5
        .long   .LASF3
        .uleb128 0x2
        .string "q"
        .byte   0x1
        .byte   0x9
        .long   0xec
        .uleb128 0x9
        .byte   0x3
        .quad   q
        .uleb128 0x4
        .byte   0x4
        .byte   0x7
        .long   .LASF4
        .uleb128 0x5
        .string "x"
        .byte   0x1
        .byte   0xb
        .long   .LASF8
        .long   0x40
        .quad   .LFB0
        .quad   .LFE0-.LFB0
        .uleb128 0x1
        .byte   0x9c
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
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x1b
        .uleb128 0xe
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x7
        .uleb128 0x10
        .uleb128 0x17
        .byte   0
        .byte   0
        .uleb128 0x2
        .uleb128 0x34
        .byte   0
        .uleb128 0x3
        .uleb128 0x8
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x3f
        .uleb128 0x19
        .uleb128 0x2
        .uleb128 0x18
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
        .uleb128 0x8
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
        .uleb128 0xe
        .byte   0
        .byte   0
        .uleb128 0x5
        .uleb128 0x2e
        .byte   0
        .uleb128 0x3f
        .uleb128 0x19
        .uleb128 0x3
        .uleb128 0x8
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x6e
        .uleb128 0xe
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x7
        .uleb128 0x40
        .uleb128 0x18
        .uleb128 0x2117
        .uleb128 0x19
        .byte   0
        .byte   0
        .byte   0
        .section        .debug_aranges,"",@progbits
        .long   0x2c
        .value  0x2
        .long   .Ldebug_info0
        .byte   0x8
        .byte   0
        .value  0
        .value  0
        .quad   .Ltext0
        .quad   .Letext0-.Ltext0
        .quad   0
        .quad   0
        .section        .debug_line,"",@progbits
.Ldebug_line0:
        .section        .debug_str,"MS",@progbits,1
.LASF4:
        .string "unsigned int"
.LASF5:
        .string "GNU C++14 7.2.0 -masm=intel -mtune=generic -march=x86-64 -g -fstack-protector"
.LASF8:
        .string "x()"
.LASF2:
        .string "float"
.LASF3:
        .string "long int"
.LASF0:
        .string "double"
.LASF7:
        .string "/home/chedy/Documents/DevEnv/Projects/compiler-explorer"
.LASF6:
        .string "/tmp/compiler-explorer-compiler1171024-7041-8xnwpl.6xrlw/example.cpp"
.LASF1:
        .string "char"
        .ident  "GCC: (Ubuntu 7.2.0-1ubuntu1~14.04) 7.2.0"
        .section        .note.GNU-stack,"",@progbits
