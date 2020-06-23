        .file   "example.cpp"
        .intel_syntax noprefix
        .text
.Ltext0:
        .globl  square(int)
        .type   square(int), @function
square(int):
.LFB0:
        .file 1 "./example.cpp"
        .loc 1 2 21
        .cfi_startproc
        endbr64
        push    rbp
        .cfi_def_cfa_offset 16
        .cfi_offset 6, -16
        mov     rbp, rsp
        .cfi_def_cfa_register 6
        mov     DWORD PTR [rbp-4], edi
        .loc 1 3 18
        mov     eax, DWORD PTR [rbp-4]
        imul    eax, eax
        .loc 1 4 1
        pop     rbp
        .cfi_def_cfa 7, 8
        ret
        .cfi_endproc
.LFE0:
        .size   square(int), .-square(int)
.Letext0:
        .section        .debug_info,"",@progbits
.Ldebug_info0:
        .long   0x67
        .value  0x4
        .long   .Ldebug_abbrev0
        .byte   0x8
        .uleb128 0x1
        .long   .LASF0
        .byte   0x4
        .long   .LASF1
        .long   .LASF2
        .quad   .Ltext0
        .quad   .Letext0-.Ltext0
        .long   .Ldebug_line0
        .uleb128 0x2
        .long   .LASF3
        .byte   0x1
        .byte   0x2
        .byte   0x5
        .long   .LASF4
        .long   0x63
        .quad   .LFB0
        .quad   .LFE0-.LFB0
        .uleb128 0x1
        .byte   0x9c
        .long   0x63
        .uleb128 0x3
        .string "num"
        .byte   0x1
        .byte   0x2
        .byte   0x10
        .long   0x63
        .uleb128 0x2
        .byte   0x91
        .sleb128 -20
        .byte   0
        .uleb128 0x4
        .byte   0x4
        .byte   0x5
        .string "int"
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
        .uleb128 0x3
        .uleb128 0x5
        .byte   0
        .uleb128 0x3
        .uleb128 0x8
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x39
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x2
        .uleb128 0x18
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
.LASF0:
        .string "GNU C++14 10.1.0 -masm=intel -mtune=generic -march=x86-64 -g -fcf-protection=full"
.LASF4:
        .string "square(int)"
.LASF3:
        .string "square"
.LASF2:
        .string "/home/ce"
.LASF1:
        .string "./example.cpp"
        .ident  "GCC: (Compiler-Explorer-Build) 10.1.0"
        .section        .note.GNU-stack,"",@progbits
        .section        .note.gnu.property,"a"
        .align 8
        .long    1f - 0f
        .long    4f - 1f
        .long    5
0:
        .string  "GNU"
1:
        .align 8
        .long    0xc0000002
        .long    3f - 2f
2:
        .long    0x3
3:
        .align 8
4:
