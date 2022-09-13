	.file 1 "test.go"
	.loc 1 3 0
	TEXT	"".Fun(SB), NOSPLIT, $0-0
	FUNCDATA	$0, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	FUNCDATA	$1, gclocals·33cdeccccebe80329f1fdbee7f5874cb(SB)
	XORL	AX, AX
	.loc 1 4 0
	JMP	Fun_pc7
Fun_pc4:
	INCQ	AX
Fun_pc7:
	CMPQ	AX, $10
	JLT	Fun_pc4
	RET