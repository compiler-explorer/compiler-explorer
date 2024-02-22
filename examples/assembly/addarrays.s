.globl addarrays_serial
.type addarrays_serial, %function

# void addarrays_serial(int *res, int *A, int *B, int N)
# Set res[i] = A[i] + B[i] for i from [0, N)
# x0 -> res
# x1 -> A
# x2 -> B
# w3 -> N
# w4 -> loop counter

addarrays_serial:
    mov w4, wzr
.loop_begin:
    cmp w4, w3
    beq .loop_exit
    ldr w5, [x1, w4, uxtw 2]
    ldr w6, [x2, w4, uxtw 2]
    add w5, w5, w6
    str w5, [x0, w4, uxtw 2]
    add w4, w4, 1
    b .loop_begin
.loop_exit:
    ret

.global addarrays_sve
.type addarrays_sve, %function

# void addarrays_sve (int *restrict res, int *A, int *B, int N)
# Set res[i] = A[i] + B[i] for i from [0, N) using SVE vectors.
# x0 -> res
# x1 -> A
# x2 -> B
# x3 -> N
# x4 -> loop counter

addarrays_sve:
    mov x4, xzr
.loop_begin_2:
    whilelt p0.s, x4, x3
    b.none .loop_exit_2
    ld1w z0.s, p0/z, [x1, x4, lsl 2]
    ld1w z1.s, p0/z, [x2, x4, lsl 2]
    add z0.s, z0.s, z1.s
    st1w z0.s, p0, [x0, x4, lsl 2]
    incw x4
    b .loop_begin_2
.loop_exit_2:
    ret

.global addarrays_neon
.type addarrays_neon, %function

addarrays_neon:
    mov w4, 0
    lsr w5, w3, 2
.vector_loop:
    cmp w4, w5
    beq .vector_loop_exit
    ldr q30, [x1, x4, lsl 4]
    ldr q31, [x2, x4, lsl 4]
    add v30.4s, v30.4s, v31.4s
    str q30, [x0, x4, lsl 4]
    add w4, w4, 1
    b .vector_loop
.vector_loop_exit:
    lsl w4, w5, 2
.tail_loop:
    cmp w4, w3
    beq .tail_loop_exit
    ldr w6, [x1, w4, uxtw 2]
    ldr w7, [x2, w4, uxtw 2]
    add w6, w6, w7
    str w6, [x0, w4, uxtw 2]
    add w4, w4, 1
    b .tail_loop
.tail_loop_exit:
    ret

.globl main
.type main, %function
main:
    adrp x0, res
    add x0, x0, :lo12:res
    adrp x1, A
    add x1, x1, :lo12:A
    adrp x2, B
    add x2, x2, :lo12:B
    mov x3, 10
    bl addarrays_neon
    adrp x0, res_ref
    add x0, x0, :lo12:res_ref
    adrp x1, res
    add x1, x1, :lo12:res
    mov w2, 40
    bl memcmp
    bl _exit

.globl A
.type A, %object
A: .word 10, 13, 5, 8, 1, 42, 65, 17, 21, 24

.globl B
.type B, %object
B: .word 19, 12, 31, 42, 3, 9, 25, 69, 87, 93 

.global res_ref
.type res_ref, %object
res_ref: .word 29, 25, 36, 50, 4, 51, 90, 86, 108, 117

.globl res
.bss
.type  res, %object
.size res, 40
res:
    .zero 40
