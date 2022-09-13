	.cpu arm7tdmi
	.eabi_attribute 20, 1
	.eabi_attribute 21, 1
	.eabi_attribute 23, 3
	.eabi_attribute 24, 1
	.eabi_attribute 25, 1
	.eabi_attribute 26, 1
	.eabi_attribute 30, 2
	.eabi_attribute 34, 0
	.eabi_attribute 18, 4
	.file	"example.cpp"
	.text
.Ltext0:
	.cfi_sections	.debug_frame
	.align	2
	.global	_Z4revcc
	.arch armv4t
	.syntax unified
	.arm
	.fpu softvfp
	.type	_Z4revcc, %function
_Z4revcc:
	.fnstart
.LVL0:
.LFB0:
	.file 1 "/tmp/example.cpp"
	.loc 1 1 19 view -0
	.cfi_startproc
	@ Function supports interworking.
	@ args = 0, pretend = 0, frame = 0
	@ frame_needed = 0, uses_anonymous_args = 0
	@ link register save eliminated.
	.loc 1 2 3 view .LVU1
	sub	r3, r0, #65
	cmp	r3, #19
	ldrls	pc, [pc, r3, asl #2]
	b	.L7
.L4:
	.word	.L8
	.word	.L7
	.word	.L6
	.word	.L7
	.word	.L7
	.word	.L7
	.word	.L5
	.word	.L7
	.word	.L7
	.word	.L7
	.word	.L7
	.word	.L7
	.word	.L7
	.word	.L7
	.word	.L7
	.word	.L7
	.word	.L7
	.word	.L7
	.word	.L7
	.word	.L3
.L8:
	.loc 1 12 14 is_stmt 0 view .LVU2
	mov	r0, #84
.LVL1:
.L7:
	.loc 1 14 1 view .LVU3
	bx	lr
.LVL2:
.L3:
	.loc 1 9 5 is_stmt 1 view .LVU4
	.loc 1 10 7 view .LVU5
	.loc 1 10 14 is_stmt 0 view .LVU6
	mov	r0, #65
.LVL3:
	.loc 1 10 14 view .LVU7
	bx	lr
.LVL4:
.L5:
	.loc 1 7 5 is_stmt 1 view .LVU8
	.loc 1 8 7 view .LVU9
	.loc 1 8 14 is_stmt 0 view .LVU10
	mov	r0, #67
.LVL5:
	.loc 1 8 14 view .LVU11
	bx	lr
.LVL6:
.L6:
	.loc 1 6 14 view .LVU12
	mov	r0, #71
.LVL7:
	.loc 1 6 14 view .LVU13
	bx	lr
	.cfi_endproc
.LFE0:
	.cantunwind
	.fnend
	.size	_Z4revcc, .-_Z4revcc
.Letext0:
	.section	.debug_info,"",%progbits
.Ldebug_info0:
	.4byte	0x5a
	.2byte	0x4
