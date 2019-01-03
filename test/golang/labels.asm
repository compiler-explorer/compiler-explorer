# command-line-arguments
"".Closures STEXT size=82 args=0x0 locals=0x18
	0x0000 00000 (labels.go:5)	TEXT	"".Closures(SB), $24-0
	0x0000 00000 (labels.go:5)	MOVQ	(TLS), CX
	0x0009 00009 (labels.go:5)	CMPQ	SP, 16(CX)
	0x000d 00013 (labels.go:5)	JLS	75
	0x000f 00015 (labels.go:5)	SUBQ	$24, SP
	0x0013 00019 (labels.go:5)	MOVQ	BP, 16(SP)
	0x0018 00024 (labels.go:5)	LEAQ	16(SP), BP
	0x001d 00029 (labels.go:5)	FUNCDATA	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	0x001d 00029 (labels.go:5)	FUNCDATA	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	0x001d 00029 (labels.go:5)	FUNCDATA	$3, gclocals·9fb7f0986f647f17cb53dda1484e0f7a(SB)
	0x001d 00029 (labels.go:6)	PCDATA	$2, $0
	0x001d 00029 (labels.go:6)	PCDATA	$0, $0
	0x001d 00029 (labels.go:6)	CMPQ	"".N(SB), $1
	0x0025 00037 (labels.go:6)	JNE	49
	0x0027 00039 (<unknown line number>)	PCDATA	$2, $-2
	0x0027 00039 (<unknown line number>)	PCDATA	$0, $-2
	0x0027 00039 (<unknown line number>)	MOVQ	16(SP), BP
	0x002c 00044 (<unknown line number>)	ADDQ	$24, SP
	0x0030 00048 (<unknown line number>)	RET
	0x0031 00049 (labels.go:7)	PCDATA	$2, $0
	0x0031 00049 (labels.go:7)	PCDATA	$0, $0
	0x0031 00049 (labels.go:7)	MOVL	$0, (SP)
	0x0038 00056 (labels.go:7)	PCDATA	$2, $1
	0x0038 00056 (labels.go:7)	LEAQ	"".Closures.func1·f(SB), AX
	0x003f 00063 (labels.go:7)	PCDATA	$2, $0
	0x003f 00063 (labels.go:7)	MOVQ	AX, 8(SP)
	0x0044 00068 (labels.go:7)	CALL	runtime.newproc(SB)
	0x0049 00073 (labels.go:7)	JMP	39
	0x004b 00075 (labels.go:7)	NOP
	0x004b 00075 (labels.go:5)	PCDATA	$0, $-1
	0x004b 00075 (labels.go:5)	PCDATA	$2, $-1
	0x004b 00075 (labels.go:5)	CALL	runtime.morestack_noctxt(SB)
	0x0050 00080 (labels.go:5)	JMP	0
	0x0000 64 48 8b 0c 25 00 00 00 00 48 3b 61 10 76 3c 48  dH..%....H;a.v<H
	0x0010 83 ec 18 48 89 6c 24 10 48 8d 6c 24 10 48 83 3d  ...H.l$.H.l$.H.=
	0x0020 00 00 00 00 01 75 0a 48 8b 6c 24 10 48 83 c4 18  .....u.H.l$.H...
	0x0030 c3 c7 04 24 00 00 00 00 48 8d 05 00 00 00 00 48  ...$....H......H
	0x0040 89 44 24 08 e8 00 00 00 00 eb dc e8 00 00 00 00  .D$.............
	0x0050 eb ae                                            ..
	rel 5+4 t=16 TLS+0
	rel 32+4 t=15 "".N+-1
	rel 59+4 t=15 "".Closures.func1·f+0
	rel 69+4 t=8 runtime.newproc+0
	rel 76+4 t=8 runtime.morestack_noctxt+0
"".Closures_func1_1 STEXT size=81 args=0x0 locals=0x10
	0x0000 00000 (labels.go:21)	TEXT	"".Closures_func1_1(SB), $16-0
	0x0000 00000 (labels.go:21)	MOVQ	(TLS), CX
	0x0009 00009 (labels.go:21)	CMPQ	SP, 16(CX)
	0x000d 00013 (labels.go:21)	JLS	74
	0x000f 00015 (labels.go:21)	SUBQ	$16, SP
	0x0013 00019 (labels.go:21)	MOVQ	BP, 8(SP)
	0x0018 00024 (labels.go:21)	LEAQ	8(SP), BP
	0x001d 00029 (labels.go:21)	FUNCDATA	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	0x001d 00029 (labels.go:21)	FUNCDATA	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	0x001d 00029 (labels.go:21)	FUNCDATA	$3, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	0x001d 00029 (labels.go:22)	PCDATA	$2, $0
	0x001d 00029 (labels.go:22)	PCDATA	$0, $0
	0x001d 00029 (labels.go:22)	CMPQ	"".N(SB), $1
	0x0025 00037 (labels.go:22)	JNE	49
	0x0027 00039 (<unknown line number>)	PCDATA	$2, $-2
	0x0027 00039 (<unknown line number>)	PCDATA	$0, $-2
	0x0027 00039 (<unknown line number>)	MOVQ	8(SP), BP
	0x002c 00044 (<unknown line number>)	ADDQ	$16, SP
	0x0030 00048 (<unknown line number>)	RET
	0x0031 00049 (labels.go:23)	PCDATA	$2, $0
	0x0031 00049 (labels.go:23)	PCDATA	$0, $0
	0x0031 00049 (labels.go:23)	CALL	runtime.printlock(SB)
	0x0036 00054 (labels.go:23)	MOVQ	$1, (SP)
	0x003e 00062 (labels.go:23)	CALL	runtime.printint(SB)
	0x0043 00067 (labels.go:23)	CALL	runtime.printunlock(SB)
	0x0048 00072 (labels.go:23)	JMP	39
	0x004a 00074 (labels.go:23)	NOP
	0x004a 00074 (labels.go:21)	PCDATA	$0, $-1
	0x004a 00074 (labels.go:21)	PCDATA	$2, $-1
	0x004a 00074 (labels.go:21)	CALL	runtime.morestack_noctxt(SB)
	0x004f 00079 (labels.go:21)	JMP	0
	0x0000 64 48 8b 0c 25 00 00 00 00 48 3b 61 10 76 3b 48  dH..%....H;a.v;H
	0x0010 83 ec 10 48 89 6c 24 08 48 8d 6c 24 08 48 83 3d  ...H.l$.H.l$.H.=
	0x0020 00 00 00 00 01 75 0a 48 8b 6c 24 08 48 83 c4 10  .....u.H.l$.H...
	0x0030 c3 e8 00 00 00 00 48 c7 04 24 01 00 00 00 e8 00  ......H..$......
	0x0040 00 00 00 e8 00 00 00 00 eb dd e8 00 00 00 00 eb  ................
	0x0050 af                                               .
	rel 5+4 t=16 TLS+0
	rel 32+4 t=15 "".N+-1
	rel 50+4 t=8 runtime.printlock+0
	rel 63+4 t=8 runtime.printint+0
	rel 68+4 t=8 runtime.printunlock+0
	rel 75+4 t=8 runtime.morestack_noctxt+0
"".αβ STEXT size=81 args=0x0 locals=0x10
	0x0000 00000 (labels.go:27)	TEXT	"".αβ(SB), $16-0
	0x0000 00000 (labels.go:27)	MOVQ	(TLS), CX
	0x0009 00009 (labels.go:27)	CMPQ	SP, 16(CX)
	0x000d 00013 (labels.go:27)	JLS	74
	0x000f 00015 (labels.go:27)	SUBQ	$16, SP
	0x0013 00019 (labels.go:27)	MOVQ	BP, 8(SP)
	0x0018 00024 (labels.go:27)	LEAQ	8(SP), BP
	0x001d 00029 (labels.go:27)	FUNCDATA	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	0x001d 00029 (labels.go:27)	FUNCDATA	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	0x001d 00029 (labels.go:27)	FUNCDATA	$3, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	0x001d 00029 (labels.go:28)	PCDATA	$2, $0
	0x001d 00029 (labels.go:28)	PCDATA	$0, $0
	0x001d 00029 (labels.go:28)	CMPQ	"".N(SB), $1
	0x0025 00037 (labels.go:28)	JNE	49
	0x0027 00039 (<unknown line number>)	PCDATA	$2, $-2
	0x0027 00039 (<unknown line number>)	PCDATA	$0, $-2
	0x0027 00039 (<unknown line number>)	MOVQ	8(SP), BP
	0x002c 00044 (<unknown line number>)	ADDQ	$16, SP
	0x0030 00048 (<unknown line number>)	RET
	0x0031 00049 (labels.go:29)	PCDATA	$2, $0
	0x0031 00049 (labels.go:29)	PCDATA	$0, $0
	0x0031 00049 (labels.go:29)	CALL	runtime.printlock(SB)
	0x0036 00054 (labels.go:29)	MOVQ	$1, (SP)
	0x003e 00062 (labels.go:29)	CALL	runtime.printint(SB)
	0x0043 00067 (labels.go:29)	CALL	runtime.printunlock(SB)
	0x0048 00072 (labels.go:29)	JMP	39
	0x004a 00074 (labels.go:29)	NOP
	0x004a 00074 (labels.go:27)	PCDATA	$0, $-1
	0x004a 00074 (labels.go:27)	PCDATA	$2, $-1
	0x004a 00074 (labels.go:27)	CALL	runtime.morestack_noctxt(SB)
	0x004f 00079 (labels.go:27)	JMP	0
	0x0000 64 48 8b 0c 25 00 00 00 00 48 3b 61 10 76 3b 48  dH..%....H;a.v;H
	0x0010 83 ec 10 48 89 6c 24 08 48 8d 6c 24 08 48 83 3d  ...H.l$.H.l$.H.=
	0x0020 00 00 00 00 01 75 0a 48 8b 6c 24 08 48 83 c4 10  .....u.H.l$.H...
	0x0030 c3 e8 00 00 00 00 48 c7 04 24 01 00 00 00 e8 00  ......H..$......
	0x0040 00 00 00 e8 00 00 00 00 eb dd e8 00 00 00 00 eb  ................
	0x0050 af                                               .
	rel 5+4 t=16 TLS+0
	rel 32+4 t=15 "".N+-1
	rel 50+4 t=8 runtime.printlock+0
	rel 63+4 t=8 runtime.printint+0
	rel 68+4 t=8 runtime.printunlock+0
	rel 75+4 t=8 runtime.morestack_noctxt+0
"".Closures.func1.1 STEXT size=81 args=0x0 locals=0x10
	0x0000 00000 (labels.go:8)	TEXT	"".Closures.func1.1(SB), $16-0
	0x0000 00000 (labels.go:8)	MOVQ	(TLS), CX
	0x0009 00009 (labels.go:8)	CMPQ	SP, 16(CX)
	0x000d 00013 (labels.go:8)	JLS	74
	0x000f 00015 (labels.go:8)	SUBQ	$16, SP
	0x0013 00019 (labels.go:8)	MOVQ	BP, 8(SP)
	0x0018 00024 (labels.go:8)	LEAQ	8(SP), BP
	0x001d 00029 (labels.go:8)	FUNCDATA	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	0x001d 00029 (labels.go:8)	FUNCDATA	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	0x001d 00029 (labels.go:8)	FUNCDATA	$3, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	0x001d 00029 (labels.go:9)	PCDATA	$2, $0
	0x001d 00029 (labels.go:9)	PCDATA	$0, $0
	0x001d 00029 (labels.go:9)	CMPQ	"".N(SB), $4
	0x0025 00037 (labels.go:9)	JNE	49
	0x0027 00039 (<unknown line number>)	PCDATA	$2, $-2
	0x0027 00039 (<unknown line number>)	PCDATA	$0, $-2
	0x0027 00039 (<unknown line number>)	MOVQ	8(SP), BP
	0x002c 00044 (<unknown line number>)	ADDQ	$16, SP
	0x0030 00048 (<unknown line number>)	RET
	0x0031 00049 (labels.go:10)	PCDATA	$2, $0
	0x0031 00049 (labels.go:10)	PCDATA	$0, $0
	0x0031 00049 (labels.go:10)	CALL	runtime.printlock(SB)
	0x0036 00054 (labels.go:10)	MOVQ	$1, (SP)
	0x003e 00062 (labels.go:10)	CALL	runtime.printint(SB)
	0x0043 00067 (labels.go:10)	CALL	runtime.printunlock(SB)
	0x0048 00072 (labels.go:10)	JMP	39
	0x004a 00074 (labels.go:10)	NOP
	0x004a 00074 (labels.go:8)	PCDATA	$0, $-1
	0x004a 00074 (labels.go:8)	PCDATA	$2, $-1
	0x004a 00074 (labels.go:8)	CALL	runtime.morestack_noctxt(SB)
	0x004f 00079 (labels.go:8)	JMP	0
	0x0000 64 48 8b 0c 25 00 00 00 00 48 3b 61 10 76 3b 48  dH..%....H;a.v;H
	0x0010 83 ec 10 48 89 6c 24 08 48 8d 6c 24 08 48 83 3d  ...H.l$.H.l$.H.=
	0x0020 00 00 00 00 04 75 0a 48 8b 6c 24 08 48 83 c4 10  .....u.H.l$.H...
	0x0030 c3 e8 00 00 00 00 48 c7 04 24 01 00 00 00 e8 00  ......H..$......
	0x0040 00 00 00 e8 00 00 00 00 eb dd e8 00 00 00 00 eb  ................
	0x0050 af                                               .
	rel 5+4 t=16 TLS+0
	rel 32+4 t=15 "".N+-1
	rel 50+4 t=8 runtime.printlock+0
	rel 63+4 t=8 runtime.printint+0
	rel 68+4 t=8 runtime.printunlock+0
	rel 75+4 t=8 runtime.morestack_noctxt+0
"".Closures.func1 STEXT size=90 args=0x0 locals=0x10
	0x0000 00000 (labels.go:7)	TEXT	"".Closures.func1(SB), $16-0
	0x0000 00000 (labels.go:7)	MOVQ	(TLS), CX
	0x0009 00009 (labels.go:7)	CMPQ	SP, 16(CX)
	0x000d 00013 (labels.go:7)	JLS	83
	0x000f 00015 (labels.go:7)	SUBQ	$16, SP
	0x0013 00019 (labels.go:7)	MOVQ	BP, 8(SP)
	0x0018 00024 (labels.go:7)	LEAQ	8(SP), BP
	0x001d 00029 (labels.go:7)	FUNCDATA	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	0x001d 00029 (labels.go:7)	FUNCDATA	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	0x001d 00029 (labels.go:7)	FUNCDATA	$3, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	0x001d 00029 (labels.go:14)	PCDATA	$2, $0
	0x001d 00029 (labels.go:14)	PCDATA	$0, $0
	0x001d 00029 (labels.go:14)	MOVQ	"".N(SB), AX
	0x0024 00036 (labels.go:14)	CMPQ	AX, $3
	0x0028 00040 (labels.go:14)	JEQ	48
	0x002a 00042 (labels.go:15)	CMPQ	AX, $4
	0x002e 00046 (labels.go:15)	JNE	58
	0x0030 00048 (<unknown line number>)	PCDATA	$2, $-2
	0x0030 00048 (<unknown line number>)	PCDATA	$0, $-2
	0x0030 00048 (<unknown line number>)	MOVQ	8(SP), BP
	0x0035 00053 (<unknown line number>)	ADDQ	$16, SP
	0x0039 00057 (<unknown line number>)	RET
	0x003a 00058 (labels.go:15)	PCDATA	$2, $0
	0x003a 00058 (labels.go:15)	PCDATA	$0, $0
	0x003a 00058 (labels.go:15)	CALL	runtime.printlock(SB)
	0x003f 00063 (labels.go:15)	MOVQ	$1, (SP)
	0x0047 00071 (labels.go:15)	CALL	runtime.printint(SB)
	0x004c 00076 (labels.go:15)	CALL	runtime.printunlock(SB)
	0x0051 00081 (labels.go:15)	JMP	48
	0x0053 00083 (labels.go:15)	NOP
	0x0053 00083 (labels.go:7)	PCDATA	$0, $-1
	0x0053 00083 (labels.go:7)	PCDATA	$2, $-1
	0x0053 00083 (labels.go:7)	CALL	runtime.morestack_noctxt(SB)
	0x0058 00088 (labels.go:7)	JMP	0
	0x0000 64 48 8b 0c 25 00 00 00 00 48 3b 61 10 76 44 48  dH..%....H;a.vDH
	0x0010 83 ec 10 48 89 6c 24 08 48 8d 6c 24 08 48 8b 05  ...H.l$.H.l$.H..
	0x0020 00 00 00 00 48 83 f8 03 74 06 48 83 f8 04 75 0a  ....H...t.H...u.
	0x0030 48 8b 6c 24 08 48 83 c4 10 c3 e8 00 00 00 00 48  H.l$.H.........H
	0x0040 c7 04 24 01 00 00 00 e8 00 00 00 00 e8 00 00 00  ..$.............
	0x0050 00 eb dd e8 00 00 00 00 eb a6                    ..........
	rel 5+4 t=16 TLS+0
	rel 32+4 t=15 "".N+0
	rel 59+4 t=8 runtime.printlock+0
	rel 72+4 t=8 runtime.printint+0
	rel 77+4 t=8 runtime.printunlock+0
	rel 84+4 t=8 runtime.morestack_noctxt+0
go.info."".Closures.func1.1$abstract SDWARFINFO dupok size=44
	0x0000 03 63 6f 6d 6d 61 6e 64 2d 6c 69 6e 65 2d 61 72  .command-line-ar
	0x0010 67 75 6d 65 6e 74 73 2e 43 6c 6f 73 75 72 65 73  guments.Closures
	0x0020 2e 66 75 6e 63 31 2e 31 00 01 01 00              .func1.1....
go.loc."".Closures SDWARFLOC size=0
go.info."".Closures SDWARFINFO size=37
	0x0000 02 22 22 2e 43 6c 6f 73 75 72 65 73 00 00 00 00  ."".Closures....
	0x0010 00 00 00 00 00 00 00 00 00 00 00 00 00 01 9c 00  ................
	0x0020 00 00 00 01 00                                   .....
	rel 13+8 t=1 "".Closures+0
	rel 21+8 t=1 "".Closures+82
	rel 31+4 t=29 gofile..labels.go+0
go.range."".Closures SDWARFRANGE size=0
go.isstmt."".Closures SDWARFMISC size=0
	0x0000 04 0f 04 0e 03 08 01 0c 02 07 01 13 02 07 00     ...............
go.loc."".Closures_func1_1 SDWARFLOC size=0
go.info."".Closures_func1_1 SDWARFINFO size=45
	0x0000 02 22 22 2e 43 6c 6f 73 75 72 65 73 5f 66 75 6e  ."".Closures_fun
	0x0010 63 31 5f 31 00 00 00 00 00 00 00 00 00 00 00 00  c1_1............
	0x0020 00 00 00 00 00 01 9c 00 00 00 00 01 00           .............
	rel 21+8 t=1 "".Closures_func1_1+0
	rel 29+8 t=1 "".Closures_func1_1+81
	rel 39+4 t=29 gofile..labels.go+0
go.range."".Closures_func1_1 SDWARFRANGE size=0
go.isstmt."".Closures_func1_1 SDWARFMISC size=0
	0x0000 04 0f 04 0e 03 08 01 0c 02 05 01 14 02 07 00     ...............
go.loc."".αβ SDWARFLOC size=0
go.info."".αβ SDWARFINFO size=33
	0x0000 02 22 22 2e ce b1 ce b2 00 00 00 00 00 00 00 00  ."".............
	0x0010 00 00 00 00 00 00 00 00 00 01 9c 00 00 00 00 01  ................
	0x0020 00                                               .
	rel 9+8 t=1 "".αβ+0
	rel 17+8 t=1 "".αβ+81
	rel 27+4 t=29 gofile..labels.go+0
go.range."".αβ SDWARFRANGE size=0
go.isstmt."".αβ SDWARFMISC size=0
	0x0000 04 0f 04 0e 03 08 01 0c 02 05 01 14 02 07 00     ...............
go.loc."".Closures.func1.1 SDWARFLOC size=0
go.info."".Closures.func1.1 SDWARFINFO size=24
	0x0000 04 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	0x0010 00 00 00 00 00 01 9c 00                          ........
	rel 1+4 t=28 go.info."".Closures.func1.1$abstract+0
	rel 5+8 t=1 "".Closures.func1.1+0
	rel 13+8 t=1 "".Closures.func1.1+81
go.range."".Closures.func1.1 SDWARFRANGE size=0
go.isstmt."".Closures.func1.1 SDWARFMISC size=0
	0x0000 04 0f 04 0e 03 08 01 0c 02 05 01 14 02 07 00     ...............
go.loc."".Closures.func1 SDWARFLOC size=0
go.info."".Closures.func1 SDWARFINFO size=58
	0x0000 02 22 22 2e 43 6c 6f 73 75 72 65 73 2e 66 75 6e  ."".Closures.fun
	0x0010 63 31 00 00 00 00 00 00 00 00 00 00 00 00 00 00  c1..............
	0x0020 00 00 00 01 9c 00 00 00 00 01 06 00 00 00 00 00  ................
	0x0030 00 00 00 00 00 00 00 0f 00 00                    ..........
	rel 19+8 t=1 "".Closures.func1+0
	rel 27+8 t=1 "".Closures.func1+90
	rel 37+4 t=29 gofile..labels.go+0
	rel 43+4 t=28 go.info."".Closures.func1.1$abstract+0
	rel 47+4 t=28 go.range."".Closures.func1+0
	rel 51+4 t=29 gofile..labels.go+0
go.range."".Closures.func1 SDWARFRANGE size=64
	0x0000 ff ff ff ff ff ff ff ff 00 00 00 00 00 00 00 00  ................
	0x0010 2a 00 00 00 00 00 00 00 30 00 00 00 00 00 00 00  *.......0.......
	0x0020 3a 00 00 00 00 00 00 00 53 00 00 00 00 00 00 00  :.......S.......
	0x0030 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
	rel 8+8 t=1 "".Closures.func1+0
go.isstmt."".Closures.func1 SDWARFMISC size=0
	0x0000 04 0f 04 0e 03 07 01 06 02 04 01 0c 02 05 01 14  ................
	0x0010 02 07 00                                         ...
"".N SNOPTRBSS size=8
"".Closures.func1·f SRODATA dupok size=8
	0x0000 00 00 00 00 00 00 00 00                          ........
	rel 0+8 t=1 "".Closures.func1+0
"".Closures.func1.1·f SRODATA dupok size=8
	0x0000 00 00 00 00 00 00 00 00                          ........
	rel 0+8 t=1 "".Closures.func1.1+0
gclocals·33cdeccccebe80329f1fdbee7f5874cb SRODATA dupok size=8
	0x0000 01 00 00 00 00 00 00 00                          ........
gclocals·9fb7f0986f647f17cb53dda1484e0f7a SRODATA dupok size=10
	0x0000 02 00 00 00 01 00 00 00 00 01                    ..........
