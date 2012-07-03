LongLong:
	.word	123456
	.word	0
	.type	Long, %object
	.size	Long, 4
Long:
	.word	2345
	.type	Int, %object
	.size	Int, 4
Int:
	.word	123
	.type	Short, %object
	.size	Short, 2
Short:
	.short	4660
	.type	Char, %object
	.size	Char, 1
Char:
	.byte	-128
