        .text
        .intel_syntax noprefix
        .file   "example.c"
my_own_address:
.Ltmp0:
        .quad   .Ltmp0

        .file   1 "/tmp/compiler-explorer-compiler1181017-55-1emkn80.7vnff" "/tmp/compiler-explorer-compiler1181017-55-1emkn80.7vnff/example.c"
        .type   ptr,@object             # @ptr
        .data
        .globl  ptr
        .p2align        3
ptr:
        .quad   my_own_address
        .size   ptr, 8
