	.file	"example.cpp"
__SREG__ = 0x3f
__SP_H__ = 0x3e
__SP_L__ = 0x3d
__CCP__ = 0x34
__tmp_reg__ = 0
__zero_reg__ = 1
	.stabs	"/tmp/",100,0,4,.Ltext0
	.stabs	"/tmp/example.cpp",100,0,4,.Ltext0
	.text
.Ltext0:
	.stabs	"gcc2_compiled.",60,0,0,0
	.stabs	"__builtin_va_list:t(0,1)=*(0,2)=(0,2)",128,0,0,0
	.stabs	"complex long double:t(0,3)=R3;8;0;",128,0,0,0
	.stabs	"complex double:t(0,4)=R3;8;0;",128,0,0,0
	.stabs	"complex float:t(0,5)=R3;8;0;",128,0,0,0
	.stabs	"complex int:t(0,6)=s4real:(0,7)=r(0,7);-32768;32767;,0,16;imag:(0,7),16,16;;",128,0,0,0
	.stabs	"long long unsigned int:t(0,8)=@s64;r(0,8);0;01777777777777777777777;",128,0,0,0
	.stabs	"long unsigned int:t(0,9)=@s32;r(0,9);0;037777777777;",128,0,0,0
	.stabs	"unsigned int:t(0,10)=r(0,10);0;0177777;",128,0,0,0
	.stabs	"unsigned char:t(0,11)=@s8;r(0,11);0;255;",128,0,0,0
	.stabs	"long long int:t(0,12)=@s64;r(0,12);01000000000000000000000;0777777777777777777777;",128,0,0,0
	.stabs	"long int:t(0,13)=@s32;r(0,13);020000000000;017777777777;",128,0,0,0
	.stabs	"int:t(0,7)",128,0,0,0
	.stabs	"signed char:t(0,14)=@s8;r(0,14);-128;127;",128,0,0,0
	.stabs	"char:t(0,15)=r(0,15);0;127;",128,0,0,0
	.stabs	"signed:t(0,7)",128,0,0,0
	.stabs	"unsigned long:t(0,9)",128,0,0,0
	.stabs	"long long unsigned:t(0,8)",128,0,0,0
	.stabs	"short int:t(0,16)=r(0,16);-32768;32767;",128,0,0,0
	.stabs	"short unsigned int:t(0,17)=r(0,17);0;0177777;",128,0,0,0
	.stabs	"unsigned short:t(0,17)",128,0,0,0
	.stabs	"float:t(0,18)=r(0,7);4;0;",128,0,0,0
	.stabs	"double:t(0,19)=r(0,7);4;0;",128,0,0,0
	.stabs	"long double:t(0,20)=r(0,7);4;0;",128,0,0,0
	.stabs	"void:t(0,2)",128,0,0,0
	.stabs	"wchar_t:t(0,21)=r(0,21);-32768;32767;",128,0,0,0
	.stabs	"bool:t(0,22)=@s8;-16;",128,0,0,0
	.stabs	"__vtbl_ptr_type:t(0,23)=*(0,24)=f(0,7)",128,0,0,0
	.stabs	"_Z12testFunctionPii:F(0,7)",36,0,1,_Z12testFunctionPii
	.stabs	"input:p(0,25)=*(0,7)",160,0,1,5
	.stabs	"length:p(0,7)",160,0,1,7
.global	_Z12testFunctionPii
	.type	_Z12testFunctionPii, @function
_Z12testFunctionPii:
	.stabd	46,0,0
	.stabn	68,0,1,.LM0-.LFBB1
.LM0:
.LFBB1:
	push r29
	push r28
	in r28,__SP_L__
	in r29,__SP_H__
	sbiw r28,8
	in __tmp_reg__,__SREG__
	cli
	out __SP_H__,r29
	out __SREG__,__tmp_reg__
	out __SP_L__,r28
/* prologue: function */
/* frame size = 8 */
/* stack size = 10 */
.L__stack_usage = 10
	std Y+6,r25
	std Y+5,r24
	std Y+8,r23
	std Y+7,r22
.LBB2:
	.stabn	68,0,2,.LM1-.LFBB1
.LM1:
	std Y+2,__zero_reg__
	std Y+1,__zero_reg__
.LBB3:
	.stabn	68,0,3,.LM2-.LFBB1
.LM2:
	std Y+4,__zero_reg__
	std Y+3,__zero_reg__
	rjmp .L2
.L4:
	.stabn	68,0,4,.LM3-.LFBB1
.LM3:
	ldd r24,Y+3
	ldd r25,Y+4
	lsl r24
	rol r25
	ldd r18,Y+5
	ldd r19,Y+6
	add r24,r18
	adc r25,r19
	mov r30,r24
	mov r31,r25
	ld r24,Z
	ldd r25,Z+1
	ldd r18,Y+1
	ldd r19,Y+2
	add r24,r18
	adc r25,r19
	std Y+2,r25
	std Y+1,r24
	.stabn	68,0,3,.LM4-.LFBB1
.LM4:
	ldd r24,Y+3
	ldd r25,Y+4
	adiw r24,1
	std Y+4,r25
	std Y+3,r24
.L2:
	.stabn	68,0,3,.LM5-.LFBB1
.LM5:
	ldi r20,lo8(1)
	ldd r18,Y+3
	ldd r19,Y+4
	ldd r24,Y+7
	ldd r25,Y+8
	cp r18,r24
	cpc r19,r25
	brlt .L3
	ldi r20,lo8(0)
.L3:
	tst r20
	brne .L4
.LBE3:
	.stabn	68,0,6,.LM6-.LFBB1
.LM6:
	ldd r24,Y+1
	ldd r25,Y+2
/* epilogue start */
.LBE2:
	.stabn	68,0,7,.LM7-.LFBB1
.LM7:
	adiw r28,8
	in __tmp_reg__,__SREG__
	cli
	out __SP_H__,r29
	out __SREG__,__tmp_reg__
	out __SP_L__,r28
	pop r28
	pop r29
	ret
	.size	_Z12testFunctionPii, .-_Z12testFunctionPii
	.stabs	"sum:(0,7)",128,0,2,1
	.stabn	192,0,0,.LBB2-.LFBB1
	.stabs	"i:(0,7)",128,0,3,3
	.stabn	192,0,0,.LBB3-.LFBB1
	.stabn	224,0,0,.LBE3-.LFBB1
	.stabn	224,0,0,.LBE2-.LFBB1
.Lscope1:
	.stabs	"",36,0,0,.Lscope1-.LFBB1
	.stabd	78,0,0
	.stabs	"",100,0,0,.Letext0
.Letext0:
