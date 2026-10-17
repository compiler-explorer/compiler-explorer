        .file   "example.cpp"
# GNU C++17 (crosstool-NG esp-14.2.0_20241119) version 14.2.0 (xtensa-esp-elf)
#       compiled by GNU C version 4.9.2, GMP version 6.2.1, MPFR version 4.2.1, MPC version 1.2.1, isl version isl-0.26-GMP

# GGC heuristics: --param ggc-min-expand=100 --param ggc-min-heapsize=131072
# options passed: -mdynconfig=xtensa_esp32.so -g -O2
        .text
.Ltext0:
        .file 0 "/app" "/app/example.cpp"
        .align  4
        .global _Z1fm
        .type   _Z1fm, @function
_Z1fm:
.LVL0:
.LFB3:
        .file 1 "/app/example.cpp"
        .loc 1 3 23 view -0
        .loc 1 3 23 is_stmt 0 view .LVU1
        entry   sp, 32        #
.LCFI0:
        .loc 1 4 5 is_stmt 1 view .LVU2
.LVL1:
        .loc 1 5 5 view .LVU3
.LBB2:
        .loc 1 5 25 discriminator 1 view .LVU4
.LBE2:
# /app/example.cpp:4:     uint32_t ret = 0;
        .loc 1 4 14 is_stmt 0 view .LVU5
        movi.n  a10, 0      # <retval>,
.LBB3:
# /app/example.cpp:5:     for(uint32_t b = 1; b; b<<=1){
        .loc 1 5 18 view .LVU6
        movi.n  a8, 1       # b,
# /app/example.cpp:7:             ret++;
        .loc 1 7 16 view .LVU7
        mov.n   a13, a10      # tmp52, <retval>
        movi.n  a9, 0x20    # tmp53,
        loop    a9, .L3_LEND    # tmp53,
.LVL2:
.L3:
        .loc 1 6 9 is_stmt 1 view .LVU8
# /app/example.cpp:6:         if(b & x){
        .loc 1 6 14 is_stmt 0 view .LVU9
        and     a12, a2, a8       # _18, x, b
# /app/example.cpp:7:             ret++;
        .loc 1 7 16 view .LVU10
        movi.n  a11, 1      # tmp50,
        moveqz  a11, a13, a12       # tmp50, tmp52, _18
        add.n   a10, a10, a11 # <retval>, <retval>, tmp50
.LVL3:
        .loc 1 5 5 is_stmt 1 discriminator 2 view .LVU11
# /app/example.cpp:5:     for(uint32_t b = 1; b; b<<=1){
        .loc 1 5 29 is_stmt 0 discriminator 2 view .LVU12
        add.n   a8, a8, a8    # b, b, b
.LVL4:
        .loc 1 5 25 is_stmt 1 discriminator 1 view .LVU13
        .L3_LEND:             #
.LBE3:
        .loc 1 10 5 view .LVU14
# /app/example.cpp:11: }
        .loc 1 11 1 is_stmt 0 view .LVU15
        mov.n   a2, a10       #, <retval>
.LVL5:
        .loc 1 11 1 view .LVU16
        retw.n
.LFE3:
        .size   _Z1fm, .-_Z1fm
        .ident  "GCC: (crosstool-NG esp-14.2.0_20241119) 14.2.0"
