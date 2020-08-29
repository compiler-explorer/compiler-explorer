        .globl  LongLong
LongLong:
        .quad   123456
        .globl  Long
        .align 8
        .type   Long, @object
        .size   Long, 8
Long:
        .quad   2345
        .globl  Int
        .align 4
        .type   Int, @object
        .size   Int, 4
Int:
        .long   123
        .globl  Short
        .align 2
        .type   Short, @object
        .size   Short, 2
Short:
        .value  4660
        .globl  Char
        .type   Char, @object
        .size   Char, 1
Char:
        .byte   -128
