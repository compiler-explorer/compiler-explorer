        .globl  LongLong
LongLong:
        .long   0x0001e240,0x00000000
        .type   LongLong,@object
        .size   LongLong,8
        .align 4
        .globl Int
Int:
        .long   123
        .type   Int,@object
        .size   Int,4
        .align 2
        .globl Short
Short:
        .word   4660
        .type   Short,@object
        .size   Short,2
        .align 1
        .globl Char
Char:
        .byte   -128
        .type   Char,@object
        .size   Char,1
        .data
