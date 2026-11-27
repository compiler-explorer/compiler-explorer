	.intel_syntax noprefix
	.file 0 "src/example.mach"
	.globl example.example.main
	.hidden example.example.main
	.type example.example.main, @function
	.size example.example.main, .Ltmp7 - example.example.main
	.globl std.print.println
	.type std.print.println, @function
	.globl example.util.fmt.twice
	.type example.util.fmt.twice, @function
	.local litem0
	.text
	.balign 16
	.balign 16, 204
example.example.main:
.Ltmp1:
	.loc 0 0 0 is_stmt 0
.Ltmp0:
	.cfi_startproc
	lea rsp, [rsp - 72]
	.loc 0 5 5 is_stmt 1
	.cfi_def_cfa rsp, 80
	lea rdi, [rsp]
	lea rsi, [rip + .Lc.00000000]
	call std.print.println
	.loc 0 6 9
	mov edi, 3
	call example.util.fmt.twice
.Ltmp5:
	.loc 0 4 9
	.cfi_remember_state
	lea rsp, [rsp + 72]
	.cfi_def_cfa rsp, 8
	ret
	.cfi_restore_state
	.cfi_endproc
.Ltmp7:
	.section .rodata, "a", @progbits
	.balign 1
.Lc.00000000:
litem0:
	.byte 0x68, 0x69, 0x00
