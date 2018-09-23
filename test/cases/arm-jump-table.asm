	.text
	.file	"arm-jump-table.cpp"
	.globl	_Z13switchexampleh      // -- Begin function _Z13switchexampleh
	.p2align	2
	.type	_Z13switchexampleh,@function
_Z13switchexampleh:                     // @_Z13switchexampleh
// %bb.0:
                                        // kill: def $w0 killed $w0 def $x0
	and	w8, w0, #0xff
	cmp	w8, #139                // =139
	b.hi	.LBB0_3
// %bb.1:
	adrp	x9, .LJTI0_0
	and	x8, x0, #0xff
	add	x9, x9, :lo12:.LJTI0_0
	ldr	x8, [x9, x8, lsl #3]
	mov	w0, #123
	br	x8
.LBB0_2:
	orr	w0, wzr, #0x3
	ret
.LBB0_3:
	cmp	w8, #255                // =255
	b.ne	.LBB0_5
// %bb.4:
	mov	w0, #149
	ret
.LBB0_5:
	orr	w0, wzr, #0x1
.LBB0_6:
	ret
.LBB0_7:
	orr	w0, wzr, #0x7c
	ret
.LBB0_8:
	mov	w0, #125
	ret
.LBB0_9:
	orr	w0, wzr, #0x7e
	ret
.LBB0_10:
	orr	w0, wzr, #0x7f
	ret
.LBB0_11:
	orr	w0, wzr, #0x80
	ret
.LBB0_12:
	mov	w0, #129
	ret
.LBB0_13:
	mov	w0, #130
	ret
.LBB0_14:
	mov	w0, #131
	ret
.LBB0_15:
	mov	w0, #132
	ret
.LBB0_16:
	mov	w0, #133
	ret
.LBB0_17:
	mov	w0, #134
	ret
.LBB0_18:
	sub	sp, sp, #16             // =16
	orr	w8, wzr, #0x40
	strb	w8, [sp, #12]
	mov	w0, #145
	add	sp, sp, #16             // =16
	ret
.LBB0_19:
	orr	w0, wzr, #0x2
	ret
.LBB0_20:
	orr	w0, wzr, #0x4
	ret
.LBB0_21:
	mov	w0, #146
	ret
.LBB0_22:
	mov	w0, #147
	ret
.LBB0_23:
	mov	w0, #148
	ret
.Lfunc_end0:
	.size	_Z13switchexampleh, .Lfunc_end0-_Z13switchexampleh
	.section	.rodata,"a",@progbits
	.p2align	3
.LJTI0_0:
	.xword	.LBB0_6
	.xword	.LBB0_5
	.xword	.LBB0_7
	.xword	.LBB0_5
	.xword	.LBB0_8
	.xword	.LBB0_5
	.xword	.LBB0_9
	.xword	.LBB0_5
	.xword	.LBB0_10
	.xword	.LBB0_5
	.xword	.LBB0_11
	.xword	.LBB0_5
	.xword	.LBB0_12
	.xword	.LBB0_5
	.xword	.LBB0_13
	.xword	.LBB0_5
	.xword	.LBB0_14
	.xword	.LBB0_5
	.xword	.LBB0_15
	.xword	.LBB0_5
	.xword	.LBB0_16
	.xword	.LBB0_5
	.xword	.LBB0_17
	.xword	.LBB0_5
	.xword	.LBB0_18
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_19
	.xword	.LBB0_2
	.xword	.LBB0_2
	.xword	.LBB0_20
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_5
	.xword	.LBB0_21
	.xword	.LBB0_22
	.xword	.LBB0_23
                                        // -- End function

	.ident	"clang version 7.0.0 (trunk 331741) (llvm/trunk 331740)"
	.section	".note.GNU-stack","",@progbits
