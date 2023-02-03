	; Module start
	.compiler_version	"TASKING SmartCode v10.1r1 - TriCore C compiler Build 21121068 SN-09003536"
	.compiler_invocation	"ctc --core=tc1.8 -g -o cpp_demo.asm .\\cpp_demo.ic"
	.compiler_name		"ctc"
	;source	'.\\cpp_demo.cpp'

	
$TC18
	
	.sdecl	'.text.cpp_demo._Z11printfhellov',code,cluster('_Z11printfhellov')
	.sect	'.text.cpp_demo._Z11printfhellov'
	.align	2
	
	.global	_Z11printfhellov
; Function _Z11printfhellov
.L5:
_Z11printfhellov:	.type	func
	movh.a	a4,#@his(.1.str)
	lea	a4,[a4]@los(.1.str)
.L36:
	j	printf
.L22:
	
___Z11printfhellov_function_end:
	.size	_Z11printfhellov,___Z11printfhellov_function_end-_Z11printfhellov
.L16:
	; End of function
	
	.sdecl	'.text.cpp_demo.main',code,cluster('main')
	.sect	'.text.cpp_demo.main'
	.align	2
	
	.global	main
; Function main
.L7:
main:	.type	func
	call	_main
.L41:
	mov	d15,#0
	mov.a	a15,#9
.L2:
	add	d15,d15,#100
	loop	a15,.L2
.L26:
	call	_Z11printfhellov
.L42:
	sha	d2,d15,#2
	ret
.L24:
	
__main_function_end:
	.size	main,__main_function_end-main
.L21:
	; End of function
	
	.sdecl	'.rodata.cpp_demo..1.str',data,rom
	.sect	'.rodata.cpp_demo..1.str'
.1.str:	.type	object
	.size	.1.str,13
	.byte	104,101,108,108
	.byte	111,119,111,114
	.byte	108,100,33,33
	.space	1
	.calls	'_Z11printfhellov','printf'
	.calls	'main','_main'
	.calls	'main','_Z11printfhellov'
	.calls	'_Z11printfhellov','',0
	.extern	printf
	.extern	_main
	.extern	__printf_int
	.calls	'main','',0
	.sdecl	'.debug_info',debug
	.sect	'.debug_info'
.L9:
	.word	345
	.half	3
	.word	.L10
	.byte	4
.L8:
	.byte	1
	.byte	'.\\cpp_demo.cpp',0
	.byte	'TASKING TriCore C compiler',0
	.byte	'C:\\Users\\QXZ3F7O\\Documents\\work\\compiler-explorer\\test\\tasking\\',0,12,1
	.word	.L11
	.byte	2
	.byte	'unsigned int',0,4,7,3
	.word	124
	.byte	4
	.word	124
	.byte	5
	.byte	'__cmpswapw',0
	.word	140
	.byte	1,1,1,1,6
	.byte	'p',0
	.word	145
	.byte	6
	.byte	'value',0
	.word	124
	.byte	6
	.byte	'compare',0
	.word	124
	.byte	0
.L23:
	.byte	2
	.byte	'int',0,4,5,7
	.word	202
	.byte	8
	.byte	'__c11_atomic_thread_fence',0,1,1,1,1,9
	.word	209
	.byte	0,2
	.byte	'char',0,1,6,7
	.word	251
	.byte	4
	.word	259
	.byte	10
	.word	264
	.byte	11
	.byte	'printf',0,1,141,1,12
	.word	202
	.byte	1,1,1,1,12,1,141,1,30
	.word	269
	.byte	13,1,141,1,41,0,14
	.byte	'_main',0,2,8,36,1,1,1,1,15,1,4
	.word	323
	.byte	16
	.byte	'__codeptr',0,2,1,1
	.word	325
	.byte	0
	.sdecl	'.debug_abbrev',debug
	.sect	'.debug_abbrev'
.L10:
	.byte	1,17,1,3,8,37,8,27,8,19,15,128,70,12,16,6,0,0,2,36,0,3,8,11,15,62,15,0,0,3,53,0,73,19,0,0,4,15,0,73,19
	.byte	0,0,5,46,1,3,8,73,19,54,15,39,12,63,12,60,12,0,0,6,5,0,3,8,73,19,0,0,7,38,0,73,19,0,0,8,46,1,3,8,54,15
	.byte	39,12,63,12,60,12,0,0,9,5,0,73,19,0,0,10,55,0,73,19,0,0,11,46,1,3,8,58,15,59,15,57,15,73,19,54,15,39,12
	.byte	63,12,60,12,0,0,12,5,0,58,15,59,15,57,15,73,19,0,0,13,24,0,58,15,59,15,57,15,0,0,14,46,0,3,8,58,15,59
	.byte	15,57,15,54,15,39,12,63,12,60,12,0,0,15,21,0,54,15,0,0,16,22,0,3,8,58,15,59,15,57,15,73,19,0,0,0
	.sdecl	'.debug_line',debug
	.sect	'.debug_line'
.L11:
	.word	.L29-.L28
.L28:
	.half	3
	.word	.L31-.L30
.L30:
	.byte	2,1,-4,9,10,0,1,1,1,1,0,0,0,1
	.byte	'C:\\Users\\QXZ3F7O\\Desktop\\Smartcode_v10.1r1\\ctc\\include\\',0,0
	.byte	'stdio.h',0,1,0,0
	.byte	'.\\cpp_demo.cpp',0,0,0,0,0
.L31:
.L29:
	.sdecl	'.debug_info',debug,cluster('_Z11printfhellov')
	.sect	'.debug_info'
.L12:
	.word	176
	.half	3
	.word	.L13
	.byte	4,1
	.byte	'.\\cpp_demo.cpp',0
	.byte	'TASKING TriCore C compiler',0
	.byte	'C:\\Users\\QXZ3F7O\\Documents\\work\\compiler-explorer\\test\\tasking\\',0,12,1
	.word	.L15,.L14
	.byte	2
	.word	.L8
	.byte	3
	.byte	'_Z11printfhellov',0,1,3,6,1,1,1
	.word	.L5,.L22,.L4
	.byte	4
	.word	.L5,.L22
	.byte	0,0
	.sdecl	'.debug_abbrev',debug,cluster('_Z11printfhellov')
	.sect	'.debug_abbrev'
.L13:
	.byte	1,17,1,3,8,37,8,27,8,19,15,128,70,12,85,6,16,6,0,0,2,61,0,24,16,0,0,3,46,1,3,8,58,15,59,15,57,15,54,15
	.byte	39,12,63,12,17,1,18,1,64,6,0,0,4,11,0,17,1,18,1,0,0,0
	.sdecl	'.debug_line',debug,cluster('_Z11printfhellov')
	.sect	'.debug_line'
.L14:
	.word	.L33-.L32
.L32:
	.half	3
	.word	.L35-.L34
.L34:
	.byte	2,1,-4,9,10,0,1,1,1,1,0,0,0,1,0
	.byte	'.\\cpp_demo.cpp',0,0,0,0,0
.L35:
	.byte	5,23,7,0,5,2
	.word	.L5
	.byte	3,4,1,5,9,9
	.half	.L36-.L5
	.byte	1,5,1,7,9
	.half	.L16-.L36
	.byte	3,1,0,1,1
.L33:
	.sdecl	'.debug_ranges',debug,cluster('_Z11printfhellov')
	.sect	'.debug_ranges'
.L15:
	.word	-1,.L5,0,.L16-.L5,0,0
	.sdecl	'.debug_info',debug,cluster('main')
	.sect	'.debug_info'
.L17:
	.word	228
	.half	3
	.word	.L18
	.byte	4,1
	.byte	'.\\cpp_demo.cpp',0
	.byte	'TASKING TriCore C compiler',0
	.byte	'C:\\Users\\QXZ3F7O\\Documents\\work\\compiler-explorer\\test\\tasking\\',0,12,1
	.word	.L20,.L19
	.byte	2
	.word	.L8
	.byte	3
	.byte	'main',0,1,8,5
	.word	.L23
	.byte	1,1,1
	.word	.L7,.L24,.L6
	.byte	4
	.word	.L7,.L24
	.byte	5
	.byte	'__61813_9_j',0,1,10,10
	.word	.L23,.L25
	.byte	4
	.word	.L2,.L26
	.byte	5
	.byte	'__61814_13_i',0,1,11,10
	.word	.L23,.L27
	.byte	0,0,0,0
	.sdecl	'.debug_abbrev',debug,cluster('main')
	.sect	'.debug_abbrev'
.L18:
	.byte	1,17,1,3,8,37,8,27,8,19,15,128,70,12,85,6,16,6,0,0,2,61,0,24,16,0,0,3,46,1,3,8,58,15,59,15,57,15,73,16
	.byte	54,15,39,12,63,12,17,1,18,1,64,6,0,0,4,11,1,17,1,18,1,0,0,5,52,0,3,8,58,15,59,15,57,15,73,16,2,6,0,0,0
	.sdecl	'.debug_line',debug,cluster('main')
	.sect	'.debug_line'
.L19:
	.word	.L38-.L37
.L37:
	.half	3
	.word	.L40-.L39
.L39:
	.byte	2,1,-4,9,10,0,1,1,1,1,0,0,0,1,0
	.byte	'.\\cpp_demo.cpp',0,0,0,0,0
.L40:
	.byte	5,6,7,0,5,2
	.word	.L7
	.byte	3,8,1,5,13,9
	.half	.L41-.L7
	.byte	3,1,1,5,68,3,1,1,5,13,9
	.half	.L2-.L41
	.byte	3,1,1,5,68,3,127,1,5,17,7,9
	.half	.L26-.L2
	.byte	3,4,1,5,8,9
	.half	.L42-.L26
	.byte	3,2,1,5,1,3,1,1,7,9
	.half	.L21-.L42
	.byte	0,1,1
.L38:
	.sdecl	'.debug_ranges',debug,cluster('main')
	.sect	'.debug_ranges'
.L20:
	.word	-1,.L7,0,.L21-.L7,0,0
	.sdecl	'.debug_loc',debug,cluster('_Z11printfhellov')
	.sect	'.debug_loc'
.L4:
	.word	-1,.L5,0,.L22-.L5
	.half	2
	.byte	138,0
	.word	0,0
	.sdecl	'.debug_loc',debug,cluster('main')
	.sect	'.debug_loc'
.L25:
	.word	-1,.L7,.L2-.L7,.L24-.L7
	.half	1
	.byte	95
	.word	0,0
.L27:
	.word	0,0
.L6:
	.word	-1,.L7,0,.L24-.L7
	.half	2
	.byte	138,0
	.word	0,0
	.sdecl	'.debug_frame',debug
	.sect	'.debug_frame'
.L43:
	.word	48
	.word	-1
	.byte	3,0,2,1,27,12,26,0,8,27,8,26,8,28,8,29,8,30,8,31,8,34,8,35,8,32,8,33,8,16,8,17,8,24,8,25,8,36,8,37,8,38
	.byte	8,39
	.sdecl	'.debug_frame',debug,cluster('_Z11printfhellov')
	.sect	'.debug_frame'
	.word	12
	.word	.L43,.L5,.L22-.L5
	.sdecl	'.debug_frame',debug,cluster('main')
	.sect	'.debug_frame'
	.word	12
	.word	.L43,.L7,.L24-.L7
	; Module end
