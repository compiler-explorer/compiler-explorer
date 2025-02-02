        .arch armv8-a
        .file   "example.cpp"
        .text
        .global a
        .section        .rodata
        .align  3
.LC0:
        .string "a"
        .base64 "AGIA"
        .data
        .align  3
        .type   a, %object
        .size   a, 8
a:
        .xword  .LC0
        .ident  "GCC: (GNU) 15.0.1 20250115 (experimental)"
        .section        .note.GNU-stack,"",@progbits
