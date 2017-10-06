        .global LongLong
LongLong:
        .word   123456
        .word   0
        .type   Long, %object
        .size   Long, 4
        .global Long
Long:
        .word   2345
        .type   Int, %object
        .size   Int, 4
        .global Int
Int:
        .word   123
        .type   Short, %object
        .size   Short, 2
        .global Short
Short:
        .short  4660
        .type   Char, %object
        .size   Char, 1
        .global Char
Char:
        .byte   -128
