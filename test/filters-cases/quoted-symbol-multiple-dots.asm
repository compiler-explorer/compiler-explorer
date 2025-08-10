; https://godbolt.org/z/87G9vf3qq
        .file   "example.cpp"
        .intel_syntax noprefix
# GNU C++23 (Compiler-Explorer-Build-gcc-53c64dc5a8fb3d85d67b2f7555a11d9afc4dbee1-binutils-2.44) version 16.0.0 20250810 (experimental) (x86_64-linux-gnu)
#       compiled by GNU C version 11.4.0, GMP version 6.3.0, MPFR version 4.2.2, MPC version 1.3.1, isl version isl-0.24-GMP

# GGC heuristics: --param ggc-min-expand=30 --param ggc-min-heapsize=4096
# options passed: -masm=intel -mtune=generic -march=x86-64 -g -O3 -std=c++23
        .text
.Ltext0:
        .file 0 "/app" "/app/example.cpp"
        .p2align 4
        .type   "_Z8functionib.constprop.0", @function
"_Z8functionib.constprop.0":
.LVL0:
.LFB2:
        .file 1 "/app/example.cpp"
        .loc 1 1 24 view -0
        .cfi_startproc
        .loc 1 2 5 view .LVU1
        .loc 1 3 9 view .LVU2
# /app/example.cpp:3:         return num * num;
        .loc 1 3 22 is_stmt 0 view .LVU3
        imul    edi, edi        # _2, num
.LVL1:
# /app/example.cpp:8: }
        .loc 1 8 1 view .LVU4
        mov     eax, edi  #, _2
        ret
        .cfi_endproc
.LFE2:
        .size   "_Z8functionib.constprop.0", .-"_Z8functionib.constprop.0"
        .p2align 4
        .type   "_Z8functionib.constprop.1", @function
"_Z8functionib.constprop.1":
.LVL2:
.LFB3:
        .loc 1 1 24 is_stmt 1 view -0
        .cfi_startproc
        .loc 1 2 5 view .LVU6
        .loc 1 6 9 view .LVU7
# /app/example.cpp:6:         return num + num;
        .loc 1 6 22 is_stmt 0 view .LVU8
        lea     eax, [rdi+rdi]    # _3,
# /app/example.cpp:8: }
        .loc 1 8 1 view .LVU9
        ret
        .cfi_endproc
.LFE3:
        .size   "_Z8functionib.constprop.1", .-"_Z8functionib.constprop.1"
        .p2align 4
        .globl  "_Z8functionib"
        .type   "_Z8functionib", @function
"_Z8functionib":
.LVL3:
.LFB0:
        .loc 1 1 69 is_stmt 1 view -0
        .cfi_startproc
        .loc 1 2 5 view .LVU11
# /app/example.cpp:3:         return num * num;
        .loc 1 3 22 is_stmt 0 view .LVU12
        mov     eax, edi  # tmp103, num
        imul    eax, edi        # tmp103, num
        add     edi, edi  # tmp102
.LVL4:
        .loc 1 3 22 view .LVU13
        test    sil, sil        # b
        cmove   eax, edi      # tmp102,, <retval>
# /app/example.cpp:8: }
        .loc 1 8 1 view .LVU14
        ret
        .cfi_endproc
.LFE0:
        .size   "_Z8functionib", .-"_Z8functionib"
        .section        .text.startup,"ax",@progbits
        .p2align 4
        .globl  "main"
        .type   "main", @function
"main":
.LVL5:
.LFB1:
        .loc 1 10 36 is_stmt 1 view -0
        .cfi_startproc
        .loc 1 11 5 view .LVU16
# /app/example.cpp:11:     return function(argc, false) + function(argc, true);
        .loc 1 11 20 is_stmt 0 view .LVU17
        call    "_Z8functionib.constprop.1"     #
.LVL6:
        mov     edx, eax  # _1,
# /app/example.cpp:11:     return function(argc, false) + function(argc, true);
        .loc 1 11 44 discriminator 1 view .LVU18
        call    "_Z8functionib.constprop.0"     #
.LVL7:
# /app/example.cpp:11:     return function(argc, false) + function(argc, true);
        .loc 1 11 55 discriminator 2 view .LVU19
        add     eax, edx  # _5, _1
# /app/example.cpp:12: }
        .loc 1 12 1 view .LVU20
        ret
        .cfi_endproc
.LFE1:
        .size   "main", .-"main"
        .text
.Letext0:
        .section        .debug_info,"",@progbits
.Ldebug_info0:
        .long   0x17c
        .value  0x5
        .byte   0x1
        .byte   0x8
        .long   .Ldebug_abbrev0
        .uleb128 0xb
        .long   .LASF4
        .byte   0x21
        .byte   0x4
        .long   0x3163e
        .long   .LASF0
        .long   .LASF1
        .long   .LLRL3
        .quad   0
        .long   .Ldebug_line0
        .uleb128 0xc
        .long   .LASF5
        .byte   0x1
        .byte   0xa
        .byte   0x6
        .long   0xa8
        .quad   .LFB1
        .quad   .LFE1-.LFB1
        .uleb128 0x1
        .byte   0x9c
        .long   0xa8
        .uleb128 0xd
        .long   .LASF6
        .byte   0x1
        .byte   0xa
        .byte   0xf
        .long   0xa8
        .long   .LLST2
        .long   .LVUS2
        .uleb128 0xe
        .long   0xb4
        .uleb128 0x1
        .byte   0x54
        .uleb128 0xf
        .quad   .LVL6
        .long   0x126
        .long   0x8b
        .uleb128 0x1
        .uleb128 0x1
        .byte   0x55
        .uleb128 0x2
        .byte   0x75
        .sleb128 0
        .uleb128 0x2
        .long   0xe3
        .uleb128 0x1
        .byte   0x30
        .byte   0
        .uleb128 0x10
        .quad   .LVL7
        .long   0xf7
        .uleb128 0x1
        .uleb128 0x1
        .byte   0x55
        .uleb128 0x3
        .byte   0xa3
        .uleb128 0x1
        .byte   0x55
        .uleb128 0x2
        .long   0xe3
        .uleb128 0x1
        .byte   0x31
        .byte   0
        .byte   0
        .uleb128 0x11
        .byte   0x4
        .byte   0x5
        .string "int"
        .uleb128 0x3
        .long   0xa8
        .uleb128 0x4
        .long   0xb9
        .uleb128 0x4
        .long   0xbe
        .uleb128 0x5
        .byte   0x6
        .long   .LASF2
        .uleb128 0x12
        .long   .LASF7
        .byte   0x1
        .byte   0x1
        .byte   0x18
        .long   .LASF8
        .long   0xa8
        .byte   0x1
        .long   0xec
        .uleb128 0x6
        .string "num"
        .byte   0x2b
        .long   0xaf
        .uleb128 0x6
        .string "b"
        .byte   0x3b
        .long   0xf2
        .byte   0
        .uleb128 0x5
        .byte   0x2
        .long   .LASF3
        .uleb128 0x3
        .long   0xec
        .uleb128 0x7
        .long   0xc4
        .quad   .LFB2
        .quad   .LFE2-.LFB2
        .uleb128 0x1
        .byte   0x9c
        .long   0x126
        .uleb128 0x8
        .long   0xd9
        .long   .LLST0
        .long   .LVUS0
        .uleb128 0x9
        .long   0xe3
        .byte   0x1
        .byte   0
        .uleb128 0x7
        .long   0xc4
        .quad   .LFB3
        .quad   .LFE3-.LFB3
        .uleb128 0x1
        .byte   0x9c
        .long   0x14f
        .uleb128 0xa
        .long   0xd9
        .uleb128 0x1
        .byte   0x55
        .uleb128 0x9
        .long   0xe3
        .byte   0
        .byte   0
        .uleb128 0x13
        .long   0xc4
        .long   .LASF8
        .quad   .LFB0
        .quad   .LFE0-.LFB0
        .uleb128 0x1
        .byte   0x9c
        .uleb128 0x8
        .long   0xd9
        .long   .LLST1
        .long   .LVUS1
        .uleb128 0xa
        .long   0xe3
        .uleb128 0x1
        .byte   0x54
        .byte   0
        .byte   0
        .section        .debug_abbrev,"",@progbits
.Ldebug_abbrev0:
        .uleb128 0x1
        .uleb128 0x49
        .byte   0
        .uleb128 0x2
        .uleb128 0x18
        .uleb128 0x7e
        .uleb128 0x18
        .byte   0
        .byte   0
        .uleb128 0x2
        .uleb128 0x49
        .byte   0
        .uleb128 0x80
        .uleb128 0x13
        .uleb128 0x7e
        .uleb128 0x18
        .byte   0
        .byte   0
        .uleb128 0x3
        .uleb128 0x26
        .byte   0
        .uleb128 0x49
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x4
        .uleb128 0xf
        .byte   0
        .uleb128 0xb
        .uleb128 0x21
        .sleb128 8
        .uleb128 0x49
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x5
        .uleb128 0x24
        .byte   0
        .uleb128 0xb
        .uleb128 0x21
        .sleb128 1
        .uleb128 0x3e
        .uleb128 0xb
        .uleb128 0x3
        .uleb128 0xe
        .byte   0
        .byte   0
        .uleb128 0x6
        .uleb128 0x5
        .byte   0
        .uleb128 0x3
        .uleb128 0x8
        .uleb128 0x3a
        .uleb128 0x21
        .sleb128 1
        .uleb128 0x3b
        .uleb128 0x21
        .sleb128 1
        .uleb128 0x39
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x7
        .uleb128 0x2e
        .byte   0x1
        .uleb128 0x31
        .uleb128 0x13
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x7
        .uleb128 0x40
        .uleb128 0x18
        .uleb128 0x7a
        .uleb128 0x19
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x8
        .uleb128 0x5
        .byte   0
        .uleb128 0x31
        .uleb128 0x13
        .uleb128 0x2
        .uleb128 0x17
        .uleb128 0x2137
        .uleb128 0x17
        .byte   0
        .byte   0
        .uleb128 0x9
        .uleb128 0x5
        .byte   0
        .uleb128 0x31
        .uleb128 0x13
        .uleb128 0x1c
        .uleb128 0xb
        .byte   0
        .byte   0
        .uleb128 0xa
        .uleb128 0x5
        .byte   0
        .uleb128 0x31
        .uleb128 0x13
        .uleb128 0x2
        .uleb128 0x18
        .byte   0
        .byte   0
        .uleb128 0xb
        .uleb128 0x11
        .byte   0x1
        .uleb128 0x25
        .uleb128 0xe
        .uleb128 0x13
        .uleb128 0xb
        .uleb128 0x90
        .uleb128 0xb
        .uleb128 0x91
        .uleb128 0x6
        .uleb128 0x3
        .uleb128 0x1f
        .uleb128 0x1b
        .uleb128 0x1f
        .uleb128 0x55
        .uleb128 0x17
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x10
        .uleb128 0x17
        .byte   0
        .byte   0
        .uleb128 0xc
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
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x7
        .uleb128 0x40
        .uleb128 0x18
        .uleb128 0x7a
        .uleb128 0x19
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0xd
        .uleb128 0x5
        .byte   0
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
        .uleb128 0x2
        .uleb128 0x17
        .uleb128 0x2137
        .uleb128 0x17
        .byte   0
        .byte   0
        .uleb128 0xe
        .uleb128 0x5
        .byte   0
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x2
        .uleb128 0x18
        .byte   0
        .byte   0
        .uleb128 0xf
        .uleb128 0x48
        .byte   0x1
        .uleb128 0x7d
        .uleb128 0x1
        .uleb128 0x7f
        .uleb128 0x13
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x10
        .uleb128 0x48
        .byte   0x1
        .uleb128 0x7d
        .uleb128 0x1
        .uleb128 0x7f
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x11
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
        .uleb128 0x12
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
        .uleb128 0x20
        .uleb128 0xb
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x13
        .uleb128 0x2e
        .byte   0x1
        .uleb128 0x31
        .uleb128 0x13
        .uleb128 0x6e
        .uleb128 0xe
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x7
        .uleb128 0x40
        .uleb128 0x18
        .uleb128 0x7a
        .uleb128 0x19
        .byte   0
        .byte   0
        .byte   0
        .section        .debug_loclists,"",@progbits
        .long   .Ldebug_loc3-.Ldebug_loc2
.Ldebug_loc2:
        .value  0x5
        .byte   0x8
        .byte   0
        .long   0
.Ldebug_loc0:
.LVUS2:
        .uleb128 0
        .uleb128 .LVU19
        .uleb128 .LVU19
        .uleb128 0
.LLST2:
        .byte   0x6
        .quad   .LVL5
        .byte   0x4
        .uleb128 .LVL5-.LVL5
        .uleb128 .LVL7-1-.LVL5
        .uleb128 0x1
        .byte   0x55
        .byte   0x4
        .uleb128 .LVL7-1-.LVL5
        .uleb128 .LFE1-.LVL5
        .uleb128 0x4
        .byte   0xa3
        .uleb128 0x1
        .byte   0x55
        .byte   0x9f
        .byte   0
.LVUS0:
        .uleb128 0
        .uleb128 .LVU4
        .uleb128 .LVU4
        .uleb128 0
.LLST0:
        .byte   0x6
        .quad   .LVL0
        .byte   0x4
        .uleb128 .LVL0-.LVL0
        .uleb128 .LVL1-.LVL0
        .uleb128 0x1
        .byte   0x55
        .byte   0x4
        .uleb128 .LVL1-.LVL0
        .uleb128 .LFE2-.LVL0
        .uleb128 0x4
        .byte   0xa3
        .uleb128 0x1
        .byte   0x55
        .byte   0x9f
        .byte   0
.LVUS1:
        .uleb128 0
        .uleb128 .LVU13
        .uleb128 .LVU13
        .uleb128 0
.LLST1:
        .byte   0x6
        .quad   .LVL3
        .byte   0x4
        .uleb128 .LVL3-.LVL3
        .uleb128 .LVL4-.LVL3
        .uleb128 0x1
        .byte   0x55
        .byte   0x4
        .uleb128 .LVL4-.LVL3
        .uleb128 .LFE0-.LVL3
        .uleb128 0x4
        .byte   0xa3
        .uleb128 0x1
        .byte   0x55
        .byte   0x9f
        .byte   0
.Ldebug_loc3:
        .section        .debug_aranges,"",@progbits
        .long   0x3c
        .value  0x2
        .long   .Ldebug_info0
        .byte   0x8
        .byte   0
        .value  0
        .value  0
        .quad   .Ltext0
        .quad   .Letext0-.Ltext0
        .quad   .LFB1
        .quad   .LFE1-.LFB1
        .quad   0
        .quad   0
        .section        .debug_rnglists,"",@progbits
.Ldebug_ranges0:
        .long   .Ldebug_ranges3-.Ldebug_ranges2
.Ldebug_ranges2:
        .value  0x5
        .byte   0x8
        .byte   0
        .long   0
.LLRL3:
        .byte   0x7
        .quad   .Ltext0
        .uleb128 .Letext0-.Ltext0
        .byte   0x7
        .quad   .LFB1
        .uleb128 .LFE1-.LFB1
        .byte   0
.Ldebug_ranges3:
        .section        .debug_line,"",@progbits
.Ldebug_line0:
        .section        .debug_str,"MS",@progbits,1
.LASF8:
        .string "_Z8functionib"
.LASF7:
        .string "function"
.LASF4:
        .string "GNU C++23 16.0.0 20250810 (experimental) -masm=intel -mtune=generic -march=x86-64 -g -O3 -std=c++23"
.LASF6:
        .string "argc"
.LASF3:
        .string "bool"
.LASF5:
        .string "main"
.LASF2:
        .string "char"
        .section        .debug_line_str,"MS",@progbits,1
.LASF0:
        .string "/app/example.cpp"
.LASF1:
        .string "/app"
        .ident  "GCC: (Compiler-Explorer-Build-gcc-53c64dc5a8fb3d85d67b2f7555a11d9afc4dbee1-binutils-2.44) 16.0.0 20250810 (experimental)"
        .section        .note.GNU-stack,"",@progbits
