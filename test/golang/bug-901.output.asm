	.file 1 "test.go"
	.loc 1 3 0
	text	"".Fun(SB), NOSPLIT, $0-0
	funcdata	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	funcdata	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	xorl	AX, AX
	.loc 1 4 0
	jmp	7
	incq	AX
	cmpq	AX, $10
	jlt	4
	ret