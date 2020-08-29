        .globl  Char
Char:
        .byte   128                     # 0x80
        .size   Char, 1
        .type   Short,@object           # @Short
        .globl  Short
        .align  2
Short:
        .short  4660                    # 0x1234
        .size   Short, 2
        .type   Int,@object             # @Int
        .globl  Int
        .align  4
Int:
        .long   123                     # 0x7b
        .size   Int, 4
        .type   Long,@object            # @Long
        .globl  Long
        .align  8
Long:
        .quad   2345                    # 0x929
        .size   Long, 8
        .type   LongLong,@object        # @LongLong
        .globl  LongLong
        .align  8
