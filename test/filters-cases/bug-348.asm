        .file   "example.cpp"
        .arch msp430f169
        .cpu 430
        .mpy none

        .section        .debug_abbrev,"",@progbits
.Ldebug_abbrev0:
        .section        .debug_info,"",@progbits
.Ldebug_info0:
        .section        .debug_line,"",@progbits
.Ldebug_line0:
        .text
.Ltext0:
        .p2align 1,0
.global square(int)
        .type   square(int),@function
/***********************
 * Function `square(int)'
 ***********************/
square(int):
.LFB0:
.LM1:
        push    r10
.LCFI0:
        push    r4
.LCFI1:
        mov     r1, r4
.LCFI2:
        add     #4, r4
.LCFI3:
        sub     #2, r1
.LCFI4:
        mov     r15, -6(r4)
.LM2:
        mov     -6(r4), r10
        mov     -6(r4), r12
        call    #__mulhi3
        mov     r14, r15
.LM3:
        add     #2, r1
        pop     r4
        pop     r10
        ret
.LFE0:
.Lfe1:
        .size   square(int),.Lfe1-square(int)
;; End of function

        .section        .debug_frame,"",@progbits
.Lframe0:
        .4byte  .LECIE0-.LSCIE0
.LSCIE0:
        .4byte  0xffffffff
        .byte   0x1
        .string ""
        .uleb128 0x1
        .sleb128 -2
        .byte   0x0
        .byte   0xc
        .uleb128 0x1
        .uleb128 0x2
        .byte   0x80
        .uleb128 0x1
        .p2align 1,0
.LECIE0:
.LSFDE0:
        .4byte  .LEFDE0-.LASFDE0
.LASFDE0:
        .4byte  .Lframe0
        .2byte  .LFB0
        .2byte  .LFE0-.LFB0
        .byte   0x4
        .4byte  .LCFI0-.LFB0
        .byte   0xe
        .uleb128 0x4
        .byte   0x4
        .4byte  .LCFI1-.LCFI0
        .byte   0xe
        .uleb128 0x6
        .byte   0x84
        .uleb128 0x3
        .byte   0x8a
        .uleb128 0x2
        .byte   0x4
        .4byte  .LCFI2-.LCFI1
        .byte   0xd
        .uleb128 0x4
        .byte   0x4
        .4byte  .LCFI3-.LCFI2
        .byte   0xe
        .uleb128 0x2
        .p2align 1,0
.LEFDE0:
        .text
.Letext0:
        .section        .debug_loc,"",@progbits
.Ldebug_loc0:
.LLST0:
        .2byte  .LFB0-.Ltext0
        .2byte  .LCFI0-.Ltext0
        .2byte  0x2
        .byte   0x71
        .sleb128 2
        .2byte  .LCFI0-.Ltext0
        .2byte  .LCFI1-.Ltext0
        .2byte  0x2
        .byte   0x71
        .sleb128 4
        .2byte  .LCFI1-.Ltext0
        .2byte  .LCFI2-.Ltext0
        .2byte  0x2
        .byte   0x71
        .sleb128 6
        .2byte  .LCFI2-.Ltext0
        .2byte  .LCFI3-.Ltext0
        .2byte  0x2
        .byte   0x74
        .sleb128 6
        .2byte  .LCFI3-.Ltext0
        .2byte  .LFE0-.Ltext0
        .2byte  0x2
        .byte   0x74
        .sleb128 2
        .2byte  0x0
        .2byte  0x0
        .section        .debug_info
        .4byte  0x4c
        .2byte  0x2
        .4byte  .Ldebug_abbrev0
        .byte   0x2
        .uleb128 0x1
        .4byte  .LASF0
        .byte   0x4
        .4byte  .LASF1
        .2byte  .Ltext0
        .2byte  .Letext0
        .4byte  .Ldebug_line0
        .uleb128 0x2
        .byte   0x1
        .4byte  .LASF2
        .byte   0x1
        .byte   0x2
        .4byte  .LASF3
        .4byte  0x48
        .2byte  .LFB0
        .2byte  .LFE0
        .4byte  .LLST0
        .4byte  0x48
        .uleb128 0x3
        .string "num"
        .byte   0x1
        .byte   0x2
        .4byte  0x48
        .byte   0x2
        .byte   0x91
        .sleb128 0
        .byte   0x0
        .uleb128 0x4
        .byte   0x2
        .byte   0x5
        .string "int"
        .byte   0x0
        .section        .debug_abbrev
        .uleb128 0x1
        .uleb128 0x11
        .byte   0x1
        .uleb128 0x25
        .uleb128 0xe
        .uleb128 0x13
        .uleb128 0xb
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x1
        .uleb128 0x10
        .uleb128 0x6
        .byte   0x0
        .byte   0x0
        .uleb128 0x2
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
        .uleb128 0x2007
        .uleb128 0xe
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
        .byte   0x0
        .byte   0x0
        .uleb128 0x3
        .uleb128 0x5
        .byte   0x0
        .uleb128 0x3
        .uleb128 0x8
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x2
        .uleb128 0xa
        .byte   0x0
        .byte   0x0
        .uleb128 0x4
        .uleb128 0x24
        .byte   0x0
        .uleb128 0xb
        .uleb128 0xb
        .uleb128 0x3e
        .uleb128 0xb
        .uleb128 0x3
        .uleb128 0x8
        .byte   0x0
        .byte   0x0
        .byte   0x0
        .section        .debug_pubnames,"",@progbits
        .4byte  0x19
        .2byte  0x2
        .4byte  .Ldebug_info0
        .4byte  0x50
        .4byte  0x1d
        .string "square"
        .4byte  0x0
        .section        .debug_aranges,"",@progbits
        .4byte  0x10
        .2byte  0x2
        .4byte  .Ldebug_info0
        .byte   0x2
        .byte   0x0
        .2byte  .Ltext0
        .2byte  .Letext0-.Ltext0
        .2byte  0x0
        .2byte  0x0
        .section        .debug_line
        .4byte  .LELT0-.LSLT0
.LSLT0:
        .2byte  0x2
        .4byte  .LELTP0-.LASLTP0
.LASLTP0:
        .byte   0x1
        .byte   0x1
        .byte   0xf6
        .byte   0xf5
        .byte   0xa
        .byte   0x0
        .byte   0x1
        .byte   0x1
        .byte   0x1
        .byte   0x1
        .byte   0x0
        .byte   0x0
        .byte   0x0
        .byte   0x1
        .ascii  "/tmp/compiler-explorer-compiler118016-56-1e03ruw.ddj4"
        .byte   0
        .byte   0x0
        .string "example.cpp"
        .uleb128 0x1
        .uleb128 0x0
        .uleb128 0x0
        .byte   0x0
.LELTP0:
        .byte   0x0
        .uleb128 0x3
        .byte   0x2
        .2byte  .LM1
        .byte   0x15
        .byte   0x0
        .uleb128 0x3
        .byte   0x2
        .2byte  .LM2
        .byte   0x15
        .byte   0x0
        .uleb128 0x3
        .byte   0x2
        .2byte  .LM3
        .byte   0x15
        .byte   0x0
        .uleb128 0x3
        .byte   0x2
        .2byte  .Letext0
        .byte   0x0
        .uleb128 0x1
        .byte   0x1
.LELT0:
        .section        .debug_str,"MS",@progbits,1
.LASF0:
        .string "GNU C++ 4.5.3"
.LASF1:
        .ascii  "/"
        .string "tmp/compiler-explorer-compiler118016-56-1e03ruw.ddj4/example.cpp"
.LASF3:
        .string "square(int)"
.LASF2:
        .string "square"