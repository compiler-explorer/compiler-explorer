Closures_pc0:
	.file 1 "labels.go"
	.loc 1 5 0
	text	"".Closures(SB), $24-0
	movq	(TLS), CX
	cmpq	SP, 16(CX)
	jls	Closures_pc75
	subq	$24, SP
	movq	BP, 16(SP)
	leaq	16(SP), BP
	funcdata	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	funcdata	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	funcdata	$3, gclocals·9fb7f0986f647f17cb53dda1484e0f7a(SB)
	.loc 1 6 0
	pcdata	$2, $0
	pcdata	$0, $0
	cmpq	"".N(SB), $1
	jne	Closures_pc49
Closures_pc39:
	pcdata	$2, $-2
	pcdata	$0, $-2
	movq	16(SP), BP
	addq	$24, SP
	ret
Closures_pc49:
	.loc 1 7 0
	pcdata	$2, $0
	pcdata	$0, $0
	movl	$0, (SP)
	pcdata	$2, $1
	leaq	"".Closures.func1·f(SB), AX
	pcdata	$2, $0
	movq	AX, 8(SP)
	call	runtime.newproc(SB)
	jmp	Closures_pc39
Closures_pc75:
	nop
	.loc 1 5 0
	pcdata	$0, $-1
	pcdata	$2, $-1
	call	runtime.morestack_noctxt(SB)
	jmp	Closures_pc0
Closures_func1_1_pc0:
	.loc 1 21 0
	text	"".Closures_func1_1(SB), $16-0
	movq	(TLS), CX
	cmpq	SP, 16(CX)
	jls	Closures_func1_1_pc74
	subq	$16, SP
	movq	BP, 8(SP)
	leaq	8(SP), BP
	funcdata	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	funcdata	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	funcdata	$3, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	.loc 1 22 0
	pcdata	$2, $0
	pcdata	$0, $0
	cmpq	"".N(SB), $1
	jne	Closures_func1_1_pc49
Closures_func1_1_pc39:
	pcdata	$2, $-2
	pcdata	$0, $-2
	movq	8(SP), BP
	addq	$16, SP
	ret
Closures_func1_1_pc49:
	.loc 1 23 0
	pcdata	$2, $0
	pcdata	$0, $0
	call	runtime.printlock(SB)
	movq	$1, (SP)
	call	runtime.printint(SB)
	call	runtime.printunlock(SB)
	jmp	Closures_func1_1_pc39
Closures_func1_1_pc74:
	nop
	.loc 1 21 0
	pcdata	$0, $-1
	pcdata	$2, $-1
	call	runtime.morestack_noctxt(SB)
	jmp	Closures_func1_1_pc0
αβ_pc0:
	.loc 1 27 0
	text	"".αβ(SB), $16-0
	movq	(TLS), CX
	cmpq	SP, 16(CX)
	jls	αβ_pc74
	subq	$16, SP
	movq	BP, 8(SP)
	leaq	8(SP), BP
	funcdata	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	funcdata	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	funcdata	$3, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	.loc 1 28 0
	pcdata	$2, $0
	pcdata	$0, $0
	cmpq	"".N(SB), $1
	jne	αβ_pc49
αβ_pc39:
	pcdata	$2, $-2
	pcdata	$0, $-2
	movq	8(SP), BP
	addq	$16, SP
	ret
αβ_pc49:
	.loc 1 29 0
	pcdata	$2, $0
	pcdata	$0, $0
	call	runtime.printlock(SB)
	movq	$1, (SP)
	call	runtime.printint(SB)
	call	runtime.printunlock(SB)
	jmp	αβ_pc39
αβ_pc74:
	nop
	.loc 1 27 0
	pcdata	$0, $-1
	pcdata	$2, $-1
	call	runtime.morestack_noctxt(SB)
	jmp	αβ_pc0
Closures_func1_1_pc0_1:
	.loc 1 8 0
	text	"".Closures.func1.1(SB), $16-0
	movq	(TLS), CX
	cmpq	SP, 16(CX)
	jls	Closures_func1_1_pc74_1
	subq	$16, SP
	movq	BP, 8(SP)
	leaq	8(SP), BP
	funcdata	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	funcdata	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	funcdata	$3, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	.loc 1 9 0
	pcdata	$2, $0
	pcdata	$0, $0
	cmpq	"".N(SB), $4
	jne	Closures_func1_1_pc49_1
Closures_func1_1_pc39_1:
	pcdata	$2, $-2
	pcdata	$0, $-2
	movq	8(SP), BP
	addq	$16, SP
	ret
Closures_func1_1_pc49_1:
	.loc 1 10 0
	pcdata	$2, $0
	pcdata	$0, $0
	call	runtime.printlock(SB)
	movq	$1, (SP)
	call	runtime.printint(SB)
	call	runtime.printunlock(SB)
	jmp	Closures_func1_1_pc39_1
Closures_func1_1_pc74_1:
	nop
	.loc 1 8 0
	pcdata	$0, $-1
	pcdata	$2, $-1
	call	runtime.morestack_noctxt(SB)
	jmp	Closures_func1_1_pc0_1
Closures_func1_pc0:
	.loc 1 7 0
	text	"".Closures.func1(SB), $16-0
	movq	(TLS), CX
	cmpq	SP, 16(CX)
	jls	Closures_func1_pc83
	subq	$16, SP
	movq	BP, 8(SP)
	leaq	8(SP), BP
	funcdata	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	funcdata	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	funcdata	$3, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	.loc 1 14 0
	pcdata	$2, $0
	pcdata	$0, $0
	movq	"".N(SB), AX
	cmpq	AX, $3
	jeq	Closures_func1_pc48
	.loc 1 15 0
	cmpq	AX, $4
	jne	Closures_func1_pc58
Closures_func1_pc48:
	pcdata	$2, $-2
	pcdata	$0, $-2
	movq	8(SP), BP
	addq	$16, SP
	ret
Closures_func1_pc58:
	pcdata	$2, $0
	pcdata	$0, $0
	call	runtime.printlock(SB)
	movq	$1, (SP)
	call	runtime.printint(SB)
	call	runtime.printunlock(SB)
	jmp	Closures_func1_pc48
Closures_func1_pc83:
	nop
	.loc 1 7 0
	pcdata	$0, $-1
	pcdata	$2, $-1
	call	runtime.morestack_noctxt(SB)
	jmp	Closures_func1_pc0
