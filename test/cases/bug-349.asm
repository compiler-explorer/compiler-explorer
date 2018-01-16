        .file   "example.cpp"
.text
.Ltext0:
        .balign 2
        .global foo()
        .type   foo(), @function
foo():
.LFB0:
        .file 1 "/tmp/compiler-explorer-compiler118016-63-en7728.30nfm/example.cpp"
        .loc 1 1 0
; start of function
; framesize_regs:     2
; framesize_locals:   0
; framesize_outgoing: 0
; framesize:          2
; elim ap -> fp       4
; elim fp -> sp       0
; saved regs: R4
        ; start of prologue
        PUSHM.W #1, R4
.LCFI0:
        MOV.W   R1, R4
.LCFI1:
        ; end of prologue
        .loc 1 2 0
        MOV.B   #3, R12
        .loc 1 3 0
        ; start of epilogue
        POPM.W  #1, r4
        RET
.LFE0:
        .size   foo(), .-foo()
        .section        .debug_frame,"",@progbits
.Lframe0:
        .4byte  .LECIE0-.LSCIE0
.LSCIE0:
        .4byte  0xffffffff
        .byte   0x3
        .string ""
        .uleb128 0x1
        .sleb128 -2
        .uleb128 0
        .byte   0xc
        .uleb128 0x1
        .uleb128 0x2
        .byte   0x80
        .uleb128 0x1
        .balign 4
.LECIE0:
.LSFDE0:
        .4byte  .LEFDE0-.LASFDE0
.LASFDE0:
        .4byte  .Lframe0
        .4byte  .LFB0
        .4byte  .LFE0-.LFB0
        .byte   0x4
        .4byte  .LCFI0-.LFB0
        .byte   0xe
        .uleb128 0x4
        .byte   0x84
        .uleb128 0x2
        .byte   0x4
        .4byte  .LCFI1-.LCFI0
        .byte   0xd
        .uleb128 0x4
        .balign 4
.LEFDE0:
.text
.Letext0:
        .section        .debug_info,"",@progbits
.Ldebug_info0:
        .4byte  0x3e
        .2byte  0x4
        .4byte  .Ldebug_abbrev0
        .byte   0x4
        .uleb128 0x1
        .4byte  .LASF0
        .byte   0x4
        .4byte  .LASF1
        .4byte  .Ltext0
        .4byte  .Letext0-.Ltext0
        .4byte  .Ldebug_line0
        .uleb128 0x2
        .string "foo"
        .byte   0x1
        .byte   0x1
        .4byte  .LASF2
        .4byte  0x3a
        .4byte  .LFB0
        .4byte  .LFE0-.LFB0
        .uleb128 0x1
        .byte   0x9c
        .uleb128 0x3
        .byte   0x2
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
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x6
        .uleb128 0x10
        .uleb128 0x17
        .byte   0
        .byte   0
        .uleb128 0x2
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
        .uleb128 0x6
        .uleb128 0x40
        .uleb128 0x18
        .uleb128 0x2117
        .uleb128 0x19
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
        .byte   0
        .section        .debug_aranges,"",@progbits
        .4byte  0x1c
        .2byte  0x2
        .4byte  .Ldebug_info0
        .byte   0x4
        .byte   0
        .2byte  0
        .2byte  0
        .4byte  .Ltext0
        .4byte  .Letext0-.Ltext0
        .4byte  0
        .4byte  0
        .section        .debug_line,"",@progbits
.Ldebug_line0:
        .section        .debug_str,"MS",@progbits,1
.LASF2:
        .string "foo()"
.LASF0:
        .string "GNU C++14 6.2.1 20161212 -g -std=c++1z"
.LASF1:
        .string "/tmp/compiler-explorer-compiler118016-63-en7728.30nfm/example.cpp"
        .ident  "GCC: (SOMNIUM Technologies Limited - msp430-gcc 6.2.1.16) 6.2.1 20161212"