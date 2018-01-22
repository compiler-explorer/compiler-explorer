	.file "output.pas"
# Begin asmlist al_begin

.section .debug_line
	.type	.Ldebug_linesection0,@object
.Ldebug_linesection0:
	.type	.Ldebug_line0,@object
.Ldebug_line0:

.section .debug_abbrev
	.type	.Ldebug_abbrevsection0,@object
.Ldebug_abbrevsection0:
	.type	.Ldebug_abbrev0,@object
.Ldebug_abbrev0:

.section .text.b_DEBUGSTART_$OUTPUT
.globl	DEBUGSTART_$OUTPUT
	.type	DEBUGSTART_$OUTPUT,@object
DEBUGSTART_$OUTPUT:
# End asmlist al_begin
# Begin asmlist al_procedures

.section .text.n_output_$$_square$smallint$$smallint
	.balign 16,0x90
.globl	OUTPUT_$$_SQUARE$SMALLINT$$SMALLINT
	.type	OUTPUT_$$_SQUARE$SMALLINT$$SMALLINT,@function
OUTPUT_$$_SQUARE$SMALLINT$$SMALLINT:
.Lc1:
.Ll1:
# [output.pas]
# [12] begin
	pushq	%rbp
.Lc3:
.Lc4:
	movq	%rsp,%rbp
.Lc5:
	leaq	-16(%rsp),%rsp
# Var num located at rbp-8, size=OS_S16
# Var $result located at rbp-12, size=OS_S16
	movw	%di,-8(%rbp)
.Ll2:
# [13] Square := num * num + 14;
	movswl	-8(%rbp),%edx
	movswl	-8(%rbp),%eax
	imull	%edx,%eax
	leal	14(%eax),%eax
	movw	%ax,-12(%rbp)
.Ll3:
# [14] end;
	movswl	-12(%rbp),%eax
	leave
	ret
.Lc2:
.Lt1:
.Le0:
	.size	OUTPUT_$$_SQUARE$SMALLINT$$SMALLINT, .Le0 - OUTPUT_$$_SQUARE$SMALLINT$$SMALLINT
.Ll4:
# End asmlist al_procedures
# Begin asmlist al_dwarf_frame

.section .debug_frame
.Lc6:
	.long	.Lc8-.Lc7
.Lc7:
	.long	-1
	.byte	1
	.byte	0
	.uleb128	1
	.sleb128	-4
	.byte	16
	.byte	12
	.uleb128	7
	.uleb128	8
	.byte	5
	.uleb128	16
	.uleb128	2
	.balign 4,0
.Lc8:
	.long	.Lc10-.Lc9
.Lc9:
	.quad	.Lc6
	.quad	.Lc1
	.quad	.Lc2-.Lc1
	.byte	4
	.long	.Lc3-.Lc1
	.byte	14
	.uleb128	16
	.byte	4
	.long	.Lc4-.Lc3
	.byte	5
	.uleb128	6
	.uleb128	4
	.byte	4
	.long	.Lc5-.Lc4
	.byte	13
	.uleb128	6
	.balign 4,0
.Lc10:
# End asmlist al_dwarf_frame
# Begin asmlist al_dwarf_info

.section .debug_info
	.type	.Ldebug_info0,@object
.Ldebug_info0:
	.long	.Ledebug_info0-.Lf1
.Lf1:
	.short	2
	.long	.Ldebug_abbrev0
	.byte	8
	.uleb128	1
# [11] function Square(const num: Integer): Integer;
	.ascii	"output.pas\000"
	.ascii	"Free Pascal 3.0.2+dfsg-5ubuntu1 2017/09/14\000"
	.ascii	"/tmp/compiler-explorer-compiler118020-15958-ivh3sj."
	.ascii	"7n4td/\000"
	.byte	9
	.byte	3
	.long	.Ldebug_line0
	.quad	DEBUGSTART_$OUTPUT
	.quad	DEBUGEND_$OUTPUT
# Syms - Begin unit OUTPUT has index 3
# Symbol OUTPUT
# Symbol SYSTEM
# Symbol SQUARE
# Syms - End unit OUTPUT has index 3
# Syms - Begin Staticsymtable
# Symbol OUTPUT_$$_init
# Syms - End Staticsymtable
# Procdef Square(const SmallInt):SmallInt;
	.uleb128	2
	.ascii	"SQUARE\000"
	.byte	1
	.byte	65
	.byte	1
	.quad	_$OUTPUT$_Ld1
	.quad	OUTPUT_$$_SQUARE$SMALLINT$$SMALLINT
	.quad	.Lt1
# Symbol NUM
	.uleb128	3
	.ascii	"NUM\000"
	.byte	2
	.byte	118
	.sleb128	-8
	.quad	_$OUTPUT$_Ld1
# Symbol result
	.uleb128	4
	.ascii	"result\000"
	.byte	2
	.byte	118
	.sleb128	-12
	.quad	_$OUTPUT$_Ld1
# Symbol SQUARE
	.uleb128	4
	.ascii	"SQUARE\000"
	.byte	2
	.byte	118
	.sleb128	-12
	.quad	_$OUTPUT$_Ld1
	.byte	0
# Defs - Begin unit SYSTEM has index 1
# Definition SmallInt
.globl	_$OUTPUT$_Ld1
	.type	_$OUTPUT$_Ld1,@object
_$OUTPUT$_Ld1:
	.uleb128	5
	.ascii	"SMALLINT\000"
	.quad	.La1
	.type	.La1,@object
.La1:
	.uleb128	6
	.ascii	"SMALLINT\000"
	.byte	5
	.byte	2
.globl	_$OUTPUT$_Ld2
	.type	_$OUTPUT$_Ld2,@object
_$OUTPUT$_Ld2:
	.uleb128	7
	.quad	_$OUTPUT$_Ld1
# Defs - End unit SYSTEM has index 1
# Defs - Begin unit OUTPUT has index 3
# Defs - End unit OUTPUT has index 3
# Defs - Begin Staticsymtable
# Defs - End Staticsymtable
	.byte	0
	.type	.Ledebug_info0,@object
.Ledebug_info0:
# End asmlist al_dwarf_info
# Begin asmlist al_dwarf_abbrev

.section .debug_abbrev
# Abbrev 1
	.uleb128	1
	.uleb128	17
	.byte	1
	.uleb128	3
	.uleb128	8
	.uleb128	37
	.uleb128	8
	.uleb128	27
	.uleb128	8
	.uleb128	19
	.uleb128	11
	.uleb128	66
	.uleb128	11
	.uleb128	16
	.uleb128	6
	.uleb128	17
	.uleb128	1
	.uleb128	18
	.uleb128	1
	.byte	0
	.byte	0
# Abbrev 2
	.uleb128	2
	.uleb128	46
	.byte	1
	.uleb128	3
	.uleb128	8
	.uleb128	39
	.uleb128	12
	.uleb128	54
	.uleb128	11
	.uleb128	63
	.uleb128	12
	.uleb128	73
	.uleb128	16
	.uleb128	17
	.uleb128	1
	.uleb128	18
	.uleb128	1
	.byte	0
	.byte	0
# Abbrev 3
	.uleb128	3
	.uleb128	5
	.byte	0
	.uleb128	3
	.uleb128	8
	.uleb128	2
	.uleb128	10
	.uleb128	73
	.uleb128	16
	.byte	0
	.byte	0
# Abbrev 4
	.uleb128	4
	.uleb128	52
	.byte	0
	.uleb128	3
	.uleb128	8
	.uleb128	2
	.uleb128	10
	.uleb128	73
	.uleb128	16
	.byte	0
	.byte	0
# Abbrev 5
	.uleb128	5
	.uleb128	22
	.byte	0
	.uleb128	3
	.uleb128	8
	.uleb128	73
	.uleb128	16
	.byte	0
	.byte	0
# Abbrev 6
	.uleb128	6
	.uleb128	36
	.byte	0
	.uleb128	3
	.uleb128	8
	.uleb128	62
	.uleb128	11
	.uleb128	11
	.uleb128	11
	.byte	0
	.byte	0
# Abbrev 7
	.uleb128	7
	.uleb128	16
	.byte	0
	.uleb128	73
	.uleb128	16
	.byte	0
	.byte	0
	.byte	0
# End asmlist al_dwarf_abbrev
# Begin asmlist al_dwarf_line

.section .debug_line
# === header start ===
	.long	.Ledebug_line0-.Lf2
.Lf2:
	.short	2
	.long	.Lehdebug_line0-.Lf3
.Lf3:
	.byte	1
	.byte	1
	.byte	1
	.byte	255
	.byte	13
	.byte	0
	.byte	1
	.byte	1
	.byte	1
	.byte	1
	.byte	0
	.byte	0
	.byte	0
	.byte	1
	.byte	0
	.byte	0
	.byte	1
# include_directories
	.byte	0
# file_names
# [17] 
	.ascii	"output.pas\000"
	.uleb128	0
	.uleb128	0
	.uleb128	0
	.byte	0
	.type	.Lehdebug_line0,@object
.Lehdebug_line0:
# === header end ===
# function: OUTPUT_$$_SQUARE$SMALLINT$$SMALLINT
# [12:1]
	.byte	0
	.uleb128	9
	.byte	2
	.quad	.Ll1
	.byte	5
	.uleb128	1
	.byte	23
# [13:13]
	.byte	2
	.uleb128	.Ll2-.Ll1
	.byte	5
	.uleb128	13
	.byte	13
# [14:1]
	.byte	2
	.uleb128	.Ll3-.Ll2
	.byte	5
	.uleb128	1
	.byte	13
	.byte	0
	.uleb128	9
	.byte	2
	.quad	.Ll4
	.byte	0
	.byte	1
	.byte	1
# ###################
	.type	.Ledebug_line0,@object
.Ledebug_line0:
# End asmlist al_dwarf_line
# Begin asmlist al_end

.section .text.z_DEBUGEND_$OUTPUT
.globl	DEBUGEND_$OUTPUT
	.type	DEBUGEND_$OUTPUT,@object
DEBUGEND_$OUTPUT:
# End asmlist al_end
.section .note.GNU-stack,"",%progbits

