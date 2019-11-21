	.file 1 "test.go"
	.loc 1 3 0
	text	"".Fun(SB), NOSPLIT, $0-0
	funcdata	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	funcdata	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	xorl	AX, AX
	.loc 1 4 0
	jmp	Fun_pc7
Fun_pc4:
	incq	AX
Fun_pc7:
	cmpq	AX, $10
	jlt	Fun_pc4
	ret