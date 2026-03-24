	.text
	.file	"example.cpp"
	.globl	_Z3fooi
	.type	_Z3fooi,@function
_Z3fooi:
	.cfi_startproc
	cmpb	$3, %dil
	ja	.LBB0_6
	movl	%edi, %eax
	jmpq	*.LJTI0_0(,%rax,8)
.LBB0_2:
	movl	$10, %eax
	retq
.LBB0_3:
	movl	$20, %eax
	retq
.LBB0_4:
	movl	$30, %eax
	retq
.LBB0_5:
	movl	$40, %eax
	retq
.LBB0_6:
	xorl	%eax, %eax
	retq
.Lfunc_end0:
	.size	_Z3fooi, .Lfunc_end0-_Z3fooi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	3, 0x90
.LJTI0_0:
	.quad	.LBB0_2
	.quad	.LBB0_3
	.quad	.LBB0_4
	.quad	.LBB0_5
	.section	.debug_loclists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0
.Ldebug_list_header_start0:
	.short	5
	.byte	8
	.byte	0
	.long	0
.Ldebug_list_header_end0:
	.section	.debug_info,"",@progbits
	.long	42
	.ident	"clang version 21.0.0"
	.section	".note.GNU-stack","",@progbits
