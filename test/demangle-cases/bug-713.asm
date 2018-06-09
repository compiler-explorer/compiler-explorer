        .file   "example.cpp"
        .text
.Ltext0:
        .section        .text._ZN6NormalD2Ev,"axG",@progbits,_ZN6NormalD5Ev,comdat
        .align 2
        .p2align 4,,15
        .weak   _ZN6NormalD2Ev
        .type   _ZN6NormalD2Ev, @function
_ZN6NormalD2Ev:
.LFB6:
        .file 1 "/tmp/compiler-explorer-compiler118012-54-4arhbo.s5ady/example.cpp"
        .loc 1 8 0
        .cfi_startproc
.LVL0:
.LBB11:
        .loc 1 8 0
        movq    $_ZTV6Normal+16, (%rdi)
        addq    $8, %rdi
.LVL1:
        jmp     _ZN3FooD1Ev
.LVL2:
.LBE11:
        .cfi_endproc
.LFE6:
        .size   _ZN6NormalD2Ev, .-_ZN6NormalD2Ev
        .weak   _ZN6NormalD1Ev
        .set    _ZN6NormalD1Ev,_ZN6NormalD2Ev
        .section        .text._ZN6NormalD0Ev,"axG",@progbits,_ZN6NormalD5Ev,comdat
        .align 2
        .p2align 4,,15
        .weak   _ZN6NormalD0Ev
        .type   _ZN6NormalD0Ev, @function
_ZN6NormalD0Ev:
.LFB8:
        .loc 1 8 0
        .cfi_startproc
.LVL3:
        pushq   %rbx
        .cfi_def_cfa_offset 16
        .cfi_offset 3, -16
        .loc 1 8 0
        movq    %rdi, %rbx
.LBB14:
.LBB15:
        movq    $_ZTV6Normal+16, (%rdi)
        leaq    8(%rdi), %rdi
.LVL4:
        call    _ZN3FooD1Ev
.LVL5:
.LBE15:
.LBE14:
        movq    %rbx, %rdi
        movl    $16, %esi
        popq    %rbx
        .cfi_def_cfa_offset 8
.LVL6:
        jmp     _ZdlPvm
.LVL7:
        .cfi_endproc
.LFE8:
        .size   _ZN6NormalD0Ev, .-_ZN6NormalD0Ev
        .text
        .p2align 4,,15
        .globl  _Z7caller1v
        .type   _Z7caller1v, @function
_Z7caller1v:
.LFB0:
        .loc 1 14 0
        .cfi_startproc
        subq    $24, %rsp
        .cfi_def_cfa_offset 32
.LVL8:
.LBB16:
.LBB17:
        .loc 1 6 0
        leaq    8(%rsp), %rdi
        movq    $_ZTV6Normal+16, (%rsp)
        call    _ZN3FooC1Ev
.LVL9:
.LBE17:
.LBE16:
.LBB18:
.LBB19:
        .loc 1 8 0
        leaq    8(%rsp), %rdi
        movq    $_ZTV6Normal+16, (%rsp)
        call    _ZN3FooD1Ev
.LVL10:
.LBE19:
.LBE18:
        .loc 1 16 0
        addq    $24, %rsp
        .cfi_def_cfa_offset 8
        ret
        .cfi_endproc
.LFE0:
        .size   _Z7caller1v, .-_Z7caller1v
        .p2align 4,,15
        .globl  _Z7caller2P6Normal
        .type   _Z7caller2P6Normal, @function
_Z7caller2P6Normal:
.LFB4:
        .loc 1 18 0
        .cfi_startproc
.LVL11:
        .loc 1 19 0
        testq   %rdi, %rdi
        je      .L7
        .loc 1 19 0 is_stmt 0 discriminator 1
        movq    (%rdi), %rax
        movq    8(%rax), %rax
        cmpq    $_ZN6NormalD0Ev, %rax
        jne     .L9
        .loc 1 18 0 is_stmt 1
        pushq   %rbx
        .cfi_def_cfa_offset 16
        .cfi_offset 3, -16
        movq    %rdi, %rbx
.LVL12:
.LBB24:
.LBB25:
.LBB26:
.LBB27:
        .loc 1 8 0
        movq    $_ZTV6Normal+16, (%rdi)
        leaq    8(%rdi), %rdi
.LVL13:
        call    _ZN3FooD1Ev
.LVL14:
.LBE27:
.LBE26:
        movq    %rbx, %rdi
        movl    $16, %esi
.LBE25:
.LBE24:
        .loc 1 20 0
        popq    %rbx
        .cfi_restore 3
        .cfi_def_cfa_offset 8
.LVL15:
.LBB29:
.LBB28:
        .loc 1 8 0
        jmp     _ZdlPvm
.LVL16:
        .p2align 4,,10
        .p2align 3
.L7:
        rep ret
        .p2align 4,,10
        .p2align 3
.L9:
.LBE28:
.LBE29:
        .loc 1 19 0 discriminator 1
        jmp     *%rax
.LVL17:
        .cfi_endproc
.LFE4:
        .size   _Z7caller2P6Normal, .-_Z7caller2P6Normal
        .weak   _ZTS6Normal
        .section        .rodata._ZTS6Normal,"aG",@progbits,_ZTS6Normal,comdat
        .align 8
        .type   _ZTS6Normal, @object
        .size   _ZTS6Normal, 8
_ZTS6Normal:
        .string "6Normal"
        .weak   _ZTI6Normal
        .section        .rodata._ZTI6Normal,"aG",@progbits,_ZTI6Normal,comdat
        .align 8
        .type   _ZTI6Normal, @object
        .size   _ZTI6Normal, 16
_ZTI6Normal:
        .quad   _ZTVN10__cxxabiv117__class_type_infoE+16
        .quad   _ZTS6Normal
        .weak   _ZTV6Normal
        .section        .rodata._ZTV6Normal,"aG",@progbits,_ZTV6Normal,comdat
        .align 8
        .type   _ZTV6Normal, @object
        .size   _ZTV6Normal, 32
_ZTV6Normal:
        .quad   0
        .quad   _ZTI6Normal
        .quad   _ZN6NormalD1Ev
        .quad   _ZN6NormalD0Ev
        .text
.Letext0:
        .section        .debug_info,"",@progbits
.Ldebug_info0:
        .long   0x379
        .value  0x4
        .long   .Ldebug_abbrev0
        .byte   0x8
        .uleb128 0x1
        .long   .LASF9
        .byte   0x4
        .long   .LASF10
        .long   .LASF11
        .long   .Ldebug_ranges0+0x30
        .quad   0
        .long   .Ldebug_line0
        .uleb128 0x2
        .string "Foo"
        .byte   0x1
        .byte   0x1
        .byte   0x1
        .long   0x69
        .uleb128 0x3
        .string "Foo"
        .byte   0x1
        .byte   0x2
        .long   .LASF12
        .long   0x48
        .long   0x4e
        .uleb128 0x4
        .long   0x69
        .byte   0
        .uleb128 0x5
        .long   .LASF13
        .byte   0x1
        .byte   0x3
        .long   .LASF14
        .long   0x5d
        .uleb128 0x4
        .long   0x69
        .uleb128 0x4
        .long   0x6f
        .byte   0
        .byte   0
        .uleb128 0x6
        .byte   0x8
        .long   0x29
        .uleb128 0x7
        .byte   0x4
        .byte   0x5
        .string "int"
        .uleb128 0x8
        .long   0x6f
        .uleb128 0x9
        .long   .LASF2
        .byte   0x10
        .byte   0x1
        .byte   0x6
        .long   0x7b
        .long   0xf9
        .uleb128 0xa
        .long   .LASF0
        .long   0x109
        .byte   0
        .byte   0x1
        .uleb128 0xb
        .long   .LASF1
        .byte   0x1
        .byte   0xb
        .long   0x29
        .byte   0x8
        .uleb128 0xc
        .long   .LASF2
        .long   .LASF3
        .byte   0x1
        .long   0xb4
        .long   0xbf
        .uleb128 0x4
        .long   0x119
        .uleb128 0xd
        .long   0x124
        .byte   0
        .uleb128 0xc
        .long   .LASF2
        .long   .LASF4
        .byte   0x1
        .long   0xd1
        .long   0xd7
        .uleb128 0x4
        .long   0x119
        .byte   0
        .uleb128 0xe
        .long   .LASF15
        .byte   0x1
        .byte   0x8
        .long   .LASF16
        .byte   0x1
        .long   0x7b
        .byte   0x1
        .byte   0x1
        .long   0xed
        .uleb128 0x4
        .long   0x119
        .uleb128 0x4
        .long   0x6f
        .byte   0
        .byte   0
        .uleb128 0x8
        .long   0x7b
        .uleb128 0xf
        .long   0x6f
        .long   0x109
        .uleb128 0x10
        .byte   0
        .uleb128 0x6
        .byte   0x8
        .long   0x10f
        .uleb128 0x11
        .byte   0x8
        .long   .LASF17
        .long   0xfe
        .uleb128 0x6
        .byte   0x8
        .long   0x7b
        .uleb128 0x8
        .long   0x119
        .uleb128 0x12
        .byte   0x8
        .long   0xf9
        .uleb128 0x13
        .long   0xd7
        .byte   0x2
        .long   0x138
        .long   0x14b
        .uleb128 0x14
        .long   .LASF5
        .long   0x11f
        .uleb128 0x14
        .long   .LASF6
        .long   0x76
        .byte   0
        .uleb128 0x15
        .long   0x12a
        .long   .LASF7
        .long   0x172
        .quad   .LFB8
        .quad   .LFE8-.LFB8
        .uleb128 0x1
        .byte   0x9c
        .long   0x172
        .long   0x1cb
        .uleb128 0x16
        .long   0x138
        .long   .LLST1
        .uleb128 0x17
        .long   0x12a
        .quad   .LBB14
        .quad   .LBE14-.LBB14
        .byte   0x1
        .byte   0x8
        .long   0x1b0
        .uleb128 0x16
        .long   0x138
        .long   .LLST2
        .uleb128 0x18
        .quad   .LVL5
        .uleb128 0x19
        .uleb128 0x1
        .byte   0x55
        .uleb128 0x2
        .byte   0x73
        .sleb128 8
        .byte   0
        .byte   0
        .uleb128 0x1a
        .quad   .LVL7
        .long   0x36f
        .uleb128 0x19
        .uleb128 0x1
        .byte   0x55
        .uleb128 0x3
        .byte   0xf3
        .uleb128 0x1
        .byte   0x55
        .uleb128 0x19
        .uleb128 0x1
        .byte   0x54
        .uleb128 0x1
        .byte   0x40
        .byte   0
        .byte   0
        .uleb128 0x15
        .long   0x12a
        .long   .LASF8
        .long   0x1f2
        .quad   .LFB6
        .quad   .LFE6-.LFB6
        .uleb128 0x1
        .byte   0x9c
        .long   0x1f2
        .long   0x20f
        .uleb128 0x16
        .long   0x138
        .long   .LLST0
        .uleb128 0x1b
        .quad   .LVL2
        .uleb128 0x19
        .uleb128 0x1
        .byte   0x55
        .uleb128 0x5
        .byte   0xf3
        .uleb128 0x1
        .byte   0x55
        .byte   0x23
        .uleb128 0x8
        .byte   0
        .byte   0
        .uleb128 0x1c
        .long   .LASF18
        .byte   0x1
        .byte   0x12
        .long   .LASF19
        .quad   .LFB4
        .quad   .LFE4-.LFB4
        .uleb128 0x1
        .byte   0x9c
        .long   0x2aa
        .uleb128 0x1d
        .string "n"
        .byte   0x1
        .byte   0x12
        .long   0x119
        .long   .LLST5
        .uleb128 0x1e
        .long   0x12a
        .quad   .LBB24
        .long   .Ldebug_ranges0+0
        .byte   0x1
        .byte   0x13
        .uleb128 0x16
        .long   0x138
        .long   .LLST6
        .uleb128 0x17
        .long   0x12a
        .quad   .LBB26
        .quad   .LBE26-.LBB26
        .byte   0x1
        .byte   0x8
        .long   0x28e
        .uleb128 0x16
        .long   0x138
        .long   .LLST7
        .uleb128 0x18
        .quad   .LVL14
        .uleb128 0x19
        .uleb128 0x1
        .byte   0x55
        .uleb128 0x2
        .byte   0x73
        .sleb128 8
        .byte   0
        .byte   0
        .uleb128 0x1a
        .quad   .LVL16
        .long   0x36f
        .uleb128 0x19
        .uleb128 0x1
        .byte   0x55
        .uleb128 0x3
        .byte   0xf3
        .uleb128 0x1
        .byte   0x55
        .uleb128 0x19
        .uleb128 0x1
        .byte   0x54
        .uleb128 0x1
        .byte   0x40
        .byte   0
        .byte   0
        .byte   0
        .uleb128 0x1f
        .long   .LASF20
        .byte   0x1
        .byte   0xe
        .long   .LASF21
        .quad   .LFB0
        .quad   .LFE0-.LFB0
        .uleb128 0x1
        .byte   0x9c
        .long   0x33e
        .uleb128 0x20
        .string "n"
        .byte   0x1
        .byte   0xf
        .long   0x7b
        .uleb128 0x2
        .byte   0x91
        .sleb128 -32
        .uleb128 0x17
        .long   0x33e
        .quad   .LBB16
        .quad   .LBE16-.LBB16
        .byte   0x1
        .byte   0xf
        .long   0x30c
        .uleb128 0x16
        .long   0x34e
        .long   .LLST3
        .uleb128 0x18
        .quad   .LVL9
        .uleb128 0x19
        .uleb128 0x1
        .byte   0x55
        .uleb128 0x2
        .byte   0x91
        .sleb128 -24
        .byte   0
        .byte   0
        .uleb128 0x21
        .long   0x12a
        .quad   .LBB18
        .quad   .LBE18-.LBB18
        .byte   0x1
        .byte   0xf
        .uleb128 0x16
        .long   0x138
        .long   .LLST4
        .uleb128 0x18
        .quad   .LVL10
        .uleb128 0x19
        .uleb128 0x1
        .byte   0x55
        .uleb128 0x2
        .byte   0x91
        .sleb128 -24
        .byte   0
        .byte   0
        .byte   0
        .uleb128 0x22
        .long   0xbf
        .byte   0x1
        .byte   0x6
        .byte   0x2
        .long   0x34e
        .long   0x358
        .uleb128 0x14
        .long   .LASF5
        .long   0x11f
        .byte   0
        .uleb128 0x23
        .long   0x33e
        .long   .LASF22
        .long   0x369
        .long   0x36f
        .uleb128 0x24
        .long   0x34e
        .byte   0
        .uleb128 0x25
        .long   .LASF23
        .long   .LASF24
        .long   .LASF23
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
        .uleb128 0x55
        .uleb128 0x17
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x10
        .uleb128 0x17
        .byte   0
        .byte   0
        .uleb128 0x2
        .uleb128 0x13
        .byte   0x1
        .uleb128 0x3
        .uleb128 0x8
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
        .uleb128 0x3
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
        .uleb128 0x6e
        .uleb128 0xe
        .uleb128 0x3c
        .uleb128 0x19
        .uleb128 0x64
        .uleb128 0x13
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x4
        .uleb128 0x5
        .byte   0
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x34
        .uleb128 0x19
        .byte   0
        .byte   0
        .uleb128 0x5
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
        .uleb128 0x6e
        .uleb128 0xe
        .uleb128 0x3c
        .uleb128 0x19
        .uleb128 0x64
        .uleb128 0x13
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
        .uleb128 0x8
        .uleb128 0x26
        .byte   0
        .uleb128 0x49
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x9
        .uleb128 0x2
        .byte   0x1
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0xb
        .uleb128 0xb
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x1d
        .uleb128 0x13
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0xa
        .uleb128 0xd
        .byte   0
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x38
        .uleb128 0xb
        .uleb128 0x34
        .uleb128 0x19
        .uleb128 0x32
        .uleb128 0xb
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
        .uleb128 0xb
        .byte   0
        .byte   0
        .uleb128 0xc
        .uleb128 0x2e
        .byte   0x1
        .uleb128 0x3f
        .uleb128 0x19
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x6e
        .uleb128 0xe
        .uleb128 0x34
        .uleb128 0x19
        .uleb128 0x32
        .uleb128 0xb
        .uleb128 0x3c
        .uleb128 0x19
        .uleb128 0x64
        .uleb128 0x13
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0xd
        .uleb128 0x5
        .byte   0
        .uleb128 0x49
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0xe
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
        .uleb128 0x6e
        .uleb128 0xe
        .uleb128 0x4c
        .uleb128 0xb
        .uleb128 0x1d
        .uleb128 0x13
        .uleb128 0x32
        .uleb128 0xb
        .uleb128 0x3c
        .uleb128 0x19
        .uleb128 0x8b
        .uleb128 0xb
        .uleb128 0x64
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0xf
        .uleb128 0x15
        .byte   0x1
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x10
        .uleb128 0x18
        .byte   0
        .byte   0
        .byte   0
        .uleb128 0x11
        .uleb128 0xf
        .byte   0
        .uleb128 0xb
        .uleb128 0xb
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x49
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x12
        .uleb128 0x10
        .byte   0
        .uleb128 0xb
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x13
        .uleb128 0x2e
        .byte   0x1
        .uleb128 0x47
        .uleb128 0x13
        .uleb128 0x20
        .uleb128 0xb
        .uleb128 0x64
        .uleb128 0x13
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x14
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
        .uleb128 0x15
        .uleb128 0x2e
        .byte   0x1
        .uleb128 0x31
        .uleb128 0x13
        .uleb128 0x6e
        .uleb128 0xe
        .uleb128 0x64
        .uleb128 0x13
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x7
        .uleb128 0x40
        .uleb128 0x18
        .uleb128 0x64
        .uleb128 0x13
        .uleb128 0x2117
        .uleb128 0x19
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x16
        .uleb128 0x5
        .byte   0
        .uleb128 0x31
        .uleb128 0x13
        .uleb128 0x2
        .uleb128 0x17
        .byte   0
        .byte   0
        .uleb128 0x17
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
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x18
        .uleb128 0x4109
        .byte   0x1
        .uleb128 0x11
        .uleb128 0x1
        .byte   0
        .byte   0
        .uleb128 0x19
        .uleb128 0x410a
        .byte   0
        .uleb128 0x2
        .uleb128 0x18
        .uleb128 0x2111
        .uleb128 0x18
        .byte   0
        .byte   0
        .uleb128 0x1a
        .uleb128 0x4109
        .byte   0x1
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x2115
        .uleb128 0x19
        .uleb128 0x31
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x1b
        .uleb128 0x4109
        .byte   0x1
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x2115
        .uleb128 0x19
        .byte   0
        .byte   0
        .uleb128 0x1c
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
        .uleb128 0x6e
        .uleb128 0xe
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x7
        .uleb128 0x40
        .uleb128 0x18
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x1d
        .uleb128 0x5
        .byte   0
        .uleb128 0x3
        .uleb128 0x8
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x2
        .uleb128 0x17
        .byte   0
        .byte   0
        .uleb128 0x1e
        .uleb128 0x1d
        .byte   0x1
        .uleb128 0x31
        .uleb128 0x13
        .uleb128 0x52
        .uleb128 0x1
        .uleb128 0x55
        .uleb128 0x17
        .uleb128 0x58
        .uleb128 0xb
        .uleb128 0x59
        .uleb128 0xb
        .byte   0
        .byte   0
        .uleb128 0x1f
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
        .uleb128 0x6e
        .uleb128 0xe
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
        .uleb128 0x20
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
        .uleb128 0x2
        .uleb128 0x18
        .byte   0
        .byte   0
        .uleb128 0x21
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
        .byte   0
        .byte   0
        .uleb128 0x22
        .uleb128 0x2e
        .byte   0x1
        .uleb128 0x47
        .uleb128 0x13
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x20
        .uleb128 0xb
        .uleb128 0x64
        .uleb128 0x13
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x23
        .uleb128 0x2e
        .byte   0x1
        .uleb128 0x31
        .uleb128 0x13
        .uleb128 0x6e
        .uleb128 0xe
        .uleb128 0x64
        .uleb128 0x13
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x24
        .uleb128 0x5
        .byte   0
        .uleb128 0x31
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x25
        .uleb128 0x2e
        .byte   0
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
        .byte   0
        .byte   0
        .byte   0
        .section        .debug_loc,"",@progbits
.Ldebug_loc0:
.LLST1:
        .quad   .LVL3
        .quad   .LVL4
        .value  0x1
        .byte   0x55
        .quad   .LVL4
        .quad   .LVL6
        .value  0x1
        .byte   0x53
        .quad   .LVL6
        .quad   .LVL7-1
        .value  0x1
        .byte   0x55
        .quad   .LVL7-1
        .quad   .LFE8
        .value  0x4
        .byte   0xf3
        .uleb128 0x1
        .byte   0x55
        .byte   0x9f
        .quad   0
        .quad   0
.LLST2:
        .quad   .LVL3
        .quad   .LVL4
        .value  0x1
        .byte   0x55
        .quad   .LVL4
        .quad   .LVL5
        .value  0x1
        .byte   0x53
        .quad   0
        .quad   0
.LLST0:
        .quad   .LVL0
        .quad   .LVL1
        .value  0x1
        .byte   0x55
        .quad   .LVL1
        .quad   .LVL2-1
        .value  0x3
        .byte   0x75
        .sleb128 -8
        .byte   0x9f
        .quad   .LVL2-1
        .quad   .LFE6
        .value  0x4
        .byte   0xf3
        .uleb128 0x1
        .byte   0x55
        .byte   0x9f
        .quad   0
        .quad   0
.LLST5:
        .quad   .LVL11
        .quad   .LVL13
        .value  0x1
        .byte   0x55
        .quad   .LVL13
        .quad   .LVL15
        .value  0x1
        .byte   0x53
        .quad   .LVL15
        .quad   .LVL16-1
        .value  0x1
        .byte   0x55
        .quad   .LVL16-1
        .quad   .LVL16
        .value  0x4
        .byte   0xf3
        .uleb128 0x1
        .byte   0x55
        .byte   0x9f
        .quad   .LVL16
        .quad   .LVL17-1
        .value  0x1
        .byte   0x55
        .quad   .LVL17-1
        .quad   .LFE4
        .value  0x4
        .byte   0xf3
        .uleb128 0x1
        .byte   0x55
        .byte   0x9f
        .quad   0
        .quad   0
.LLST6:
        .quad   .LVL12
        .quad   .LVL13
        .value  0x1
        .byte   0x55
        .quad   .LVL13
        .quad   .LVL15
        .value  0x1
        .byte   0x53
        .quad   .LVL15
        .quad   .LVL16-1
        .value  0x1
        .byte   0x55
        .quad   .LVL16-1
        .quad   .LVL16
        .value  0x4
        .byte   0xf3
        .uleb128 0x1
        .byte   0x55
        .byte   0x9f
        .quad   0
        .quad   0
.LLST7:
        .quad   .LVL12
        .quad   .LVL13
        .value  0x1
        .byte   0x55
        .quad   .LVL13
        .quad   .LVL14
        .value  0x1
        .byte   0x53
        .quad   0
        .quad   0
.LLST3:
        .quad   .LVL8
        .quad   .LVL9
        .value  0x1
        .byte   0x57
        .quad   0
        .quad   0
.LLST4:
        .quad   .LVL9
        .quad   .LVL10
        .value  0x1
        .byte   0x57
        .quad   0
        .quad   0
        .section        .debug_aranges,"",@progbits
        .long   0x4c
        .value  0x2
        .long   .Ldebug_info0
        .byte   0x8
        .byte   0
        .value  0
        .value  0
        .quad   .Ltext0
        .quad   .Letext0-.Ltext0
        .quad   .LFB6
        .quad   .LFE6-.LFB6
        .quad   .LFB8
        .quad   .LFE8-.LFB8
        .quad   0
        .quad   0
        .section        .debug_ranges,"",@progbits
.Ldebug_ranges0:
        .quad   .LBB24
        .quad   .LBE24
        .quad   .LBB29
        .quad   .LBE29
        .quad   0
        .quad   0
        .quad   .Ltext0
        .quad   .Letext0
        .quad   .LFB6
        .quad   .LFE6
        .quad   .LFB8
        .quad   .LFE8
        .quad   0
        .quad   0
        .section        .debug_line,"",@progbits
.Ldebug_line0:
        .section        .debug_str,"MS",@progbits,1
.LASF21:
        .string "_Z7caller1v"
.LASF16:
        .string "_ZN6NormalD4Ev"
.LASF3:
        .string "_ZN6NormalC4ERKS_"
.LASF9:
        .string "GNU C++14 7.2.0 -mtune=generic -march=x86-64 -g -O2 -std=c++1z"
.LASF22:
        .string "_ZN6NormalC2Ev"
.LASF17:
        .string "__vtbl_ptr_type"
.LASF1:
        .string "foo_"
.LASF0:
        .string "_vptr.Normal"
.LASF23:
        .string "_ZdlPvm"
.LASF13:
        .string "~Foo"
.LASF14:
        .string "_ZN3FooD4Ev"
.LASF11:
        .string "/compiler-explorer"
.LASF4:
        .string "_ZN6NormalC4Ev"
.LASF5:
        .string "this"
.LASF2:
        .string "Normal"
.LASF15:
        .string "~Normal"
.LASF10:
        .string "/tmp/compiler-explorer-compiler118012-54-4arhbo.s5ady/example.cpp"
.LASF6:
        .string "__in_chrg"
.LASF7:
        .string "_ZN6NormalD0Ev"
.LASF19:
        .string "_Z7caller2P6Normal"
.LASF20:
        .string "caller1"
.LASF18:
        .string "caller2"
.LASF8:
        .string "_ZN6NormalD2Ev"
.LASF12:
        .string "_ZN3FooC4Ev"
.LASF24:
        .string "operator delete"
        .ident  "GCC: (GCC-Explorer-Build) 7.2.0"
        .section        .note.GNU-stack,"",@progbits