        .file   "example.cpp"
        .intel_syntax noprefix
        .text
.Ltext0:
        .globl  _Z3foov
        .type   _Z3foov, @function
_Z3foov:
.LFB17:
        .file 1 "/tmp/compiler-explorer-compiler119330-63-1ccesdf.nzyy/example.cpp"
        .loc 1 3 11
        .cfi_startproc
        .loc 1 3 13
        .loc 1 3 24 is_stmt 0
        mov     eax, 42
        ret
        .cfi_endproc
.LFE17:
        .size   _Z3foov, .-_Z3foov
        .globl  _Z3barv
        .type   _Z3barv, @function
_Z3barv:
.LFB18:
        .loc 1 4 11 is_stmt 1
        .cfi_startproc
        .loc 1 4 13
.LVL0:
.LBB6:
.LBB7:
        .file 2 "/opt/compiler-explorer/gcc-8.3.0/include/c++/8.3.0/typeinfo"
        .loc 2 100 7
        .loc 2 100 14 is_stmt 0
        mov     rax, QWORD PTR _ZTIi[rip+8]
        .loc 2 100 31
        cmp     BYTE PTR [rax], 42
        sete    dl
        movzx   edx, dl
.LVL1:
.LBE7:
.LBE6:
        .loc 1 4 40
        movsx   eax, BYTE PTR [rax+rdx]
        .loc 1 4 43
        ret
        .cfi_endproc
.LFE18:
        .size   _Z3barv, .-_Z3barv
        .globl  _Z3bazv
        .type   _Z3bazv, @function
_Z3bazv:
.LFB19:
        .loc 1 6 11 is_stmt 1
        .cfi_startproc
        .loc 1 6 13
.LVL2:
        .loc 2 100 7
        .loc 1 6 45 is_stmt 0
        mov     eax, 53
        ret
        .cfi_endproc
.LFE19:
        .size   _Z3bazv, .-_Z3bazv
.Letext0:
        .file 3 "/opt/compiler-explorer/gcc-8.3.0/include/c++/8.3.0/x86_64-linux-gnu/bits/c++config.h"
        .file 4 "<built-in>"
        .section        .debug_info,"",@progbits
.Ldebug_info0:
        .long   0x153
        .value  0x4
        .long   .Ldebug_abbrev0
        .byte   0x8
        .uleb128 0x1
        .long   .LASF4
        .byte   0x4
        .long   .LASF5
        .long   .LASF6
        .quad   .Ltext0
        .quad   .Letext0-.Ltext0
        .long   .Ldebug_line0
        .uleb128 0x2
        .string "std"
        .byte   0x4
        .byte   0
        .long   0x75
        .uleb128 0x3
        .long   .LASF0
        .byte   0x3
        .value  0x104
        .byte   0x41
        .uleb128 0x4
        .byte   0x3
        .value  0x104
        .byte   0x41
        .long   0x38
        .uleb128 0x5
        .long   .LASF7
        .long   0x6f
        .uleb128 0x6
        .long   .LASF8
        .byte   0x2
        .byte   0x63
        .byte   0x11
        .long   .LASF9
        .long   0x125
        .byte   0x1
        .long   0x68
        .uleb128 0x7
        .long   0x137
        .byte   0
        .byte   0
        .uleb128 0x8
        .long   0x4a
        .byte   0
        .uleb128 0x9
        .long   .LASF10
        .byte   0x3
        .value  0x106
        .byte   0xb
        .long   0x95
        .uleb128 0x3
        .long   .LASF0
        .byte   0x3
        .value  0x108
        .byte   0x41
        .uleb128 0x4
        .byte   0x3
        .value  0x108
        .byte   0x41
        .long   0x82
        .byte   0
        .uleb128 0xa
        .string "baz"
        .byte   0x1
        .byte   0x6
        .byte   0x5
        .long   .LASF1
        .long   0xb7
        .quad   .LFB19
        .quad   .LFE19-.LFB19
        .uleb128 0x1
        .byte   0x9c
        .uleb128 0xb
        .byte   0x4
        .byte   0x5
        .string "int"
        .uleb128 0xc
        .string "bar"
        .byte   0x1
        .byte   0x4
        .byte   0x5
        .long   .LASF11
        .long   0xb7
        .quad   .LFB18
        .quad   .LFE18-.LFB18
        .uleb128 0x1
        .byte   0x9c
        .long   0x103
        .uleb128 0xd
        .long   0x142
        .quad   .LBB6
        .quad   .LBE6-.LBB6
        .byte   0x1
        .byte   0x4
        .byte   0x24
        .uleb128 0xe
        .long   0x14c
        .byte   0
        .byte   0
        .uleb128 0xa
        .string "foo"
        .byte   0x1
        .byte   0x3
        .byte   0x5
        .long   .LASF2
        .long   0xb7
        .quad   .LFB17
        .quad   .LFE17-.LFB17
        .uleb128 0x1
        .byte   0x9c
        .uleb128 0xf
        .byte   0x8
        .long   0x132
        .uleb128 0x10
        .byte   0x1
        .byte   0x6
        .long   .LASF3
        .uleb128 0x8
        .long   0x12b
        .uleb128 0xf
        .byte   0x8
        .long   0x6f
        .uleb128 0x8
        .long   0x137
        .uleb128 0x11
        .long   0x53
        .long   0x14c
        .byte   0x3
        .uleb128 0x12
        .long   .LASF12
        .long   0x13d
        .byte   0
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
        .uleb128 0x39
        .byte   0x1
        .uleb128 0x3
        .uleb128 0x8
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x3
        .uleb128 0x39
        .byte   0
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0x5
        .uleb128 0x39
        .uleb128 0xb
        .uleb128 0x89
        .uleb128 0x19
        .byte   0
        .byte   0
        .uleb128 0x4
        .uleb128 0x3a
        .byte   0
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0x5
        .uleb128 0x39
        .uleb128 0xb
        .uleb128 0x18
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x5
        .uleb128 0x2
        .byte   0x1
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3c
        .uleb128 0x19
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x6
        .uleb128 0x2e
        .byte   0x1
        .uleb128 0x3f
        .uleb128 0x19
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x39
        .uleb128 0xb
        .uleb128 0x6e
        .uleb128 0xe
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x32
        .uleb128 0xb
        .uleb128 0x3c
        .uleb128 0x19
        .uleb128 0x64
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x7
        .uleb128 0x5
        .byte   0
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x34
        .uleb128 0x19
        .byte   0
        .byte   0
        .uleb128 0x8
        .uleb128 0x26
        .byte   0
        .uleb128 0x49
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x9
        .uleb128 0x39
        .byte   0x1
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0x5
        .uleb128 0x39
        .uleb128 0xb
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0xa
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
        .uleb128 0x39
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
        .uleb128 0xb
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
        .uleb128 0xc
        .uleb128 0x2e
        .byte   0x1
        .uleb128 0x3f
        .uleb128 0x19
        .uleb128 0x3
        .uleb128 0x8
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x39
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
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0xd
        .uleb128 0x1d
        .byte   0x1
        .uleb128 0x31
        .uleb128 0x13
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x7
        .uleb128 0x58
        .uleb128 0xb
        .uleb128 0x59
        .uleb128 0xb
        .uleb128 0x57
        .uleb128 0xb
        .byte   0
        .byte   0
        .uleb128 0xe
        .uleb128 0x5
        .byte   0
        .uleb128 0x31
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0xf
        .uleb128 0xf
        .byte   0
        .uleb128 0xb
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x10
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
        .uleb128 0x11
        .uleb128 0x2e
        .byte   0x1
        .uleb128 0x47
        .uleb128 0x13
        .uleb128 0x64
        .uleb128 0x13
        .uleb128 0x20
        .uleb128 0xb
        .byte   0
        .byte   0
        .uleb128 0x12
        .uleb128 0x5
        .byte   0
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x34
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
.LASF1:
        .string "_Z3bazv"
.LASF2:
        .string "_Z3foov"
.LASF7:
        .string "type_info"
.LASF10:
        .string "__gnu_cxx"
.LASF3:
        .string "char"
.LASF5:
        .string "/tmp/compiler-explorer-compiler119330-63-1ccesdf.nzyy/example.cpp"
.LASF11:
        .string "_Z3barv"
.LASF9:
        .string "_ZNKSt9type_info4nameEv"
.LASF12:
        .string "this"
.LASF0:
        .string "__cxx11"
.LASF4:
        .string "GNU C++14 8.3.0 -masm=intel -mtune=generic -march=x86-64 -g -O1"
.LASF8:
        .string "name"
.LASF6:
        .string "/tmp/compiler-explorer-compiler119330-63-1ccesdf.nzyy"
        .ident  "GCC: (Compiler-Explorer-Build) 8.3.0"
        .section        .note.GNU-stack,"",@progbits