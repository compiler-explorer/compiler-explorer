Closures_pc0:
	.file 1 "labels.go"
	.loc 1 5 0
	TEXT	"".Closures(SB), $24-0
	MOVQ	(TLS), CX
	CMPQ	SP, 16(CX)
	JLS	Closures_pc75
	SUBQ	$24, SP
	MOVQ	BP, 16(SP)
	LEAQ	16(SP), BP
	FUNCDATA	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	FUNCDATA	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	FUNCDATA	$3, gclocals·9fb7f0986f647f17cb53dda1484e0f7a(SB)
	.loc 1 6 0
	PCDATA	$2, $0
	PCDATA	$0, $0
	CMPQ	"".N(SB), $1
	JNE	Closures_pc49
Closures_pc39:
	PCDATA	$2, $-2
	PCDATA	$0, $-2
	MOVQ	16(SP), BP
	ADDQ	$24, SP
	RET
Closures_pc49:
	.loc 1 7 0
	PCDATA	$2, $0
	PCDATA	$0, $0
	MOVL	$0, (SP)
	PCDATA	$2, $1
	LEAQ	"".Closures.func1·f(SB), AX
	PCDATA	$2, $0
	MOVQ	AX, 8(SP)
	CALL	runtime.newproc(SB)
	JMP	Closures_pc39
Closures_pc75:
	NOP
	.loc 1 5 0
	PCDATA	$0, $-1
	PCDATA	$2, $-1
	CALL	runtime.morestack_noctxt(SB)
	JMP	Closures_pc0
Closures_func1_1_pc0:
	.loc 1 21 0
	TEXT	"".Closures_func1_1(SB), $16-0
	MOVQ	(TLS), CX
	CMPQ	SP, 16(CX)
	JLS	Closures_func1_1_pc74
	SUBQ	$16, SP
	MOVQ	BP, 8(SP)
	LEAQ	8(SP), BP
	FUNCDATA	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	FUNCDATA	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	FUNCDATA	$3, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	.loc 1 22 0
	PCDATA	$2, $0
	PCDATA	$0, $0
	CMPQ	"".N(SB), $1
	JNE	Closures_func1_1_pc49
Closures_func1_1_pc39:
	PCDATA	$2, $-2
	PCDATA	$0, $-2
	MOVQ	8(SP), BP
	ADDQ	$16, SP
	RET
Closures_func1_1_pc49:
	.loc 1 23 0
	PCDATA	$2, $0
	PCDATA	$0, $0
	CALL	runtime.printlock(SB)
	MOVQ	$1, (SP)
	CALL	runtime.printint(SB)
	CALL	runtime.printunlock(SB)
	JMP	Closures_func1_1_pc39
Closures_func1_1_pc74:
	NOP
	.loc 1 21 0
	PCDATA	$0, $-1
	PCDATA	$2, $-1
	CALL	runtime.morestack_noctxt(SB)
	JMP	Closures_func1_1_pc0
αβ_pc0:
	.loc 1 27 0
	TEXT	"".αβ(SB), $16-0
	MOVQ	(TLS), CX
	CMPQ	SP, 16(CX)
	JLS	αβ_pc74
	SUBQ	$16, SP
	MOVQ	BP, 8(SP)
	LEAQ	8(SP), BP
	FUNCDATA	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	FUNCDATA	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	FUNCDATA	$3, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	.loc 1 28 0
	PCDATA	$2, $0
	PCDATA	$0, $0
	CMPQ	"".N(SB), $1
	JNE	αβ_pc49
αβ_pc39:
	PCDATA	$2, $-2
	PCDATA	$0, $-2
	MOVQ	8(SP), BP
	ADDQ	$16, SP
	RET
αβ_pc49:
	.loc 1 29 0
	PCDATA	$2, $0
	PCDATA	$0, $0
	CALL	runtime.printlock(SB)
	MOVQ	$1, (SP)
	CALL	runtime.printint(SB)
	CALL	runtime.printunlock(SB)
	JMP	αβ_pc39
αβ_pc74:
	NOP
	.loc 1 27 0
	PCDATA	$0, $-1
	PCDATA	$2, $-1
	CALL	runtime.morestack_noctxt(SB)
	JMP	αβ_pc0
Closures_func1_1_pc0_1:
	.loc 1 8 0
	TEXT	"".Closures.func1.1(SB), $16-0
	MOVQ	(TLS), CX
	CMPQ	SP, 16(CX)
	JLS	Closures_func1_1_pc74_1
	SUBQ	$16, SP
	MOVQ	BP, 8(SP)
	LEAQ	8(SP), BP
	FUNCDATA	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	FUNCDATA	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	FUNCDATA	$3, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	.loc 1 9 0
	PCDATA	$2, $0
	PCDATA	$0, $0
	CMPQ	"".N(SB), $4
	JNE	Closures_func1_1_pc49_1
Closures_func1_1_pc39_1:
	PCDATA	$2, $-2
	PCDATA	$0, $-2
	MOVQ	8(SP), BP
	ADDQ	$16, SP
	RET
Closures_func1_1_pc49_1:
	.loc 1 10 0
	PCDATA	$2, $0
	PCDATA	$0, $0
	CALL	runtime.printlock(SB)
	MOVQ	$1, (SP)
	CALL	runtime.printint(SB)
	CALL	runtime.printunlock(SB)
	JMP	Closures_func1_1_pc39_1
Closures_func1_1_pc74_1:
	NOP
	.loc 1 8 0
	PCDATA	$0, $-1
	PCDATA	$2, $-1
	CALL	runtime.morestack_noctxt(SB)
	JMP	Closures_func1_1_pc0_1
Closures_func1_pc0:
	.loc 1 7 0
	TEXT	"".Closures.func1(SB), $16-0
	MOVQ	(TLS), CX
	CMPQ	SP, 16(CX)
	JLS	Closures_func1_pc83
	SUBQ	$16, SP
	MOVQ	BP, 8(SP)
	LEAQ	8(SP), BP
	FUNCDATA	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	FUNCDATA	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	FUNCDATA	$3, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	.loc 1 14 0
	PCDATA	$2, $0
	PCDATA	$0, $0
	MOVQ	"".N(SB), AX
	CMPQ	AX, $3
	JEQ	Closures_func1_pc48
	.loc 1 15 0
	CMPQ	AX, $4
	JNE	Closures_func1_pc58
Closures_func1_pc48:
	PCDATA	$2, $-2
	PCDATA	$0, $-2
	MOVQ	8(SP), BP
	ADDQ	$16, SP
	RET
Closures_func1_pc58:
	PCDATA	$2, $0
	PCDATA	$0, $0
	CALL	runtime.printlock(SB)
	MOVQ	$1, (SP)
	CALL	runtime.printint(SB)
	CALL	runtime.printunlock(SB)
	JMP	Closures_func1_pc48
Closures_func1_pc83:
	NOP
	.loc 1 7 0
	PCDATA	$0, $-1
	PCDATA	$2, $-1
	CALL	runtime.morestack_noctxt(SB)
	JMP	Closures_func1_pc0
