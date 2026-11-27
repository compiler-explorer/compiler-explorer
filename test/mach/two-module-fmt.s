	.intel_syntax noprefix
	.file 0 "src/util/fmt.mach"
	.globl example.util.fmt.twice
	.hidden example.util.fmt.twice
	.type example.util.fmt.twice, @function
	.size example.util.fmt.twice, .Ltmp7 - example.util.fmt.twice
	.globl example.util.fmt.unused
	.hidden example.util.fmt.unused
	.type example.util.fmt.unused, @function
	.size example.util.fmt.unused, .Ltmp16 - example.util.fmt.unused
	.text
	.balign 16
	.balign 16, 204
example.util.fmt.twice:
.Ltmp1:
.Ltmp2:
	.loc 0 1 9
.Ltmp0:
	.cfi_startproc
	mov qword ptr [rsp - 24], rdi
	.loc 0 2 9
	imul rax, rdi, 2
.Ltmp5:
	.loc 0 1 9
	.cfi_remember_state
	ret
	.cfi_restore_state
	.cfi_endproc
.Ltmp8:
.Ltmp7:
	.section .text, "ax", @progbits, unique, 1
	.balign 16
	.balign 16, 204
example.util.fmt.unused:
.Ltmp10:
.Ltmp11:
	.loc 0 5 9
.Ltmp9:
	.cfi_startproc
	mov qword ptr [rsp - 24], rdi
	.loc 0 6 9
	add rdi, 7
.Ltmp14:
	.loc 0 5 9
	mov rax, rdi
	.cfi_remember_state
	ret
	.cfi_restore_state
	.cfi_endproc
.Ltmp17:
.Ltmp16:
