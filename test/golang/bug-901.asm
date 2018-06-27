"".Fun STEXT nosplit size=14 args=0x0 locals=0x0
	0x0000 00000 (test.go:3)	TEXT	"".Fun(SB), NOSPLIT, $0-0
	0x0000 00000 (test.go:3)	FUNCDATA	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	0x0000 00000 (test.go:3)	FUNCDATA	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	0x0000 00000 (test.go:3)	XORL	AX, AX
	0x0002 00002 (test.go:4)	JMP	7
	0x0004 00004 (test.go:4)	INCQ	AX
	0x0007 00007 (test.go:4)	CMPQ	AX, $10
	0x000b 00011 (test.go:4)	JLT	4
	0x000d 00013 (<unknown line number>)	RET
	0x0000 31 c0 eb 03 48 ff c0 48 83 f8 0a 7c f7 c3        1...H..H...|..
go.info."".Fun SDWARFINFO size=32
	0x0000 02 22 22 2e 46 75 6e 00 00 00 00 00 00 00 00 00  ."".Fun.........
	0x0010 00 00 00 00 00 00 00 00 01 9c 00 00 00 00 01 00  ................
	rel 8+8 t=1 "".Fun+0
	rel 16+8 t=1 "".Fun+14
	rel 26+4 t=29 gofile../workspace/go/src/gopractice/io/test.go+0
go.range."".Fun SDWARFRANGE size=0
gclocals·33cdeccccebe80329f1fdbee7f5874cb SRODATA dupok size=8
	0x0000 01 00 00 00 00 00 00 00                          ........
