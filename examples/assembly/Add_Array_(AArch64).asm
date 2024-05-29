# This file demonstrates adding 2 arrays using serial, SVE and NEON instructions.
# To assemble in Compiler Explorer select AArch64 binutils and add
# "-march=armv8.2-a+sve" to the compiler options.
#
# To compile locally use "aarch64-linux-gnu-gcc -mcpu=generic+sve" (or -march=armv8.2-a+sve),
# the compiler driver will set up the call main.

# All functions have the following prototype:
# void addarrays_{serial,neon,sve} (int *restrict res, int *A, int *B, long N)
# x0 -> res
# x1 -> A
# x2 -> B
# x3 -> N

.globl addarrays_serial
.type addarrays_serial, %function

addarrays_serial:
    # x4 is the loop counter
    mov x4, xzr
    b .cond
.loop_body:
    # Each int is 4 bytes, so we use lsl 2 (left shift by 2)
    ldr w5, [x1, x4, lsl 2]
    ldr w6, [x2, x4, lsl 2]
    add w5, w5, w6
    str w5, [x0, x4, lsl 2]
    add x4, x4, 1
.cond:
    cmp x4, x3
    blt .loop_body
    ret

.global addarrays_neon
.type addarrays_neon, %function

addarrays_neon:
    mov w4, 0
    # w5 contains the number of vector iterations.
    # For example, if w3 (ie N) is 10,
    # then w5 = 10 / 4 = 2.
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
    # Iterate over remaining elements serially, starting from w5*4.
    # For example, if N = 10, we complete 2 vector iterations,
    # and the first scalar iteration will start from 2 * 4 = 8.
    # So the scalar loop will iterate from [8, 10).
    # The scalar loop is identical to the above written addarrays_serial.
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

.global addarrays_sve
.type addarrays_sve, %function

# We start by building predicate in p0 whose elements are set to 1,
# for values of induction variable < N.
# Then we load elements from input arrays x1, x2 into SVE registers z0, z1
# for elements corresponding to active lanes of p0 (set to 1), 
# and zero out the rest (/z).
# The result is then added and computed in z0, and only elements from z0
# corresponding to active lanes of p0 are stored into res.
# Finally, the induction variable x4 is incremented by number of 32-bit words
# in the vector.
# If none of the predicate elements are active, the loop is terminated.
#
# Note that the code doesn't rely on knowing vector length, and can
# thus "scale" to different vector lengths set by hardware without
# needing recompilation, and also doesn't need a tail scalar loop.
#
# For example: 
# Let's consider adding following arrays:
# A = [ 10, 13, 5, 8, 1, 42, 65, 17, 21, 24 ]
# B = [ 19, 12, 31, 42, 3, 9, 25, 69, 87, 93 ]
# Let result be the output array for storing sum of individual elements
# from A and B.
# N = 10
#
# Case 1: Vector length = 256 bits, that is, 8 number of 32-bit elements.
# 1st iteration:
# p0.s = [x4 < x3, x4+1 < x3, ... x4+<len-1> < x3] = [ 1, 1, 1, 1, 1, 1, 1, 1]
# Since first element of p0 is active, we branch to loop body,
# Since all lanes of p0 are active, load 8 elements from A starting from &A[x4]
# z0.s = [ A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7] ]
# Similarly,
# z1.s = [ B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7] ]
# Add and store result in z0:
# z0.s = [ z0.s[0] + z0.s[1], ... +, z0.s[7] + z1.s[7] ] 
# result = [ z0.s[0], z0.s[1], z0.s[2], z0.s[3], z0.s[4],
#	     z0.s[5], z0.s[6], z0.s[7] ]
# Finally incw x4 will increment x4 by 8, thus x4 = 8.
#
# 2nd iteration:
# x4 = 8
# p0.s = [ 1, 1, 0, 0, 0, 0, 0, 0 ]
# Since only first two lanes are active in p0, we load &A[8], &A[9]
# in z0 and zero out rest of elements (because of /z): 
# z0.s = [ A[8], A[9], 0, 0, 0, 0, 0, 0 ]
# Similarly, z1.s = [ B[8], B[9], 0, 0, 0, 0, 0, 0 ]
# Compute the result:
# z0.s = [ z0.s[0] + z1.s[0], z0.s[1] + z1.s[1], 0, 0, 0, 0, 0, 0 ] 
# and since first two elements of p0 are active:
# we store z0.s[0] into result[8] and z0.s[1] in result[9] repsectively.
# Increment x4 by 8, thus x4 = 16.
#
# 3rd iteration:
# x4 = 16
# Since x4 > N, p0 will be all false predicate.
# Since all lanes of p0 are inactive, the loop is terminated.
#
# Case2: Vector length = 512 bits, that is, 16 number of 32-bit elements.
# 1st iteration:
# p0.s = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 ]
# z0.s = [ A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7],
#	   A[8], A[9], 0, 0, 0, 0, 0, 0 ]
# z1.s = [ B[0], B[1], B[2], B[3], B[4], B[5], B[6], B[7],
#	   B[8], B[9], 0, 0, 0, 0, 0, 0 ]
# Compute sum in z0:
# z0.s = [ z0.s[0] + z1.s[0], ..., z0.s[9] + z1.s[9], 0, 0, 0, 0, 0, 0 ] 
# Store first 9 elements from z0 since first 9 lanes of p0 are active.
# result = [ z0.s[0], z0.s[1], z0.s[2], z0.s[3], z0.s[4], z0.s[5], z0.s[6],
#	     z0.s[7], z0.s[8], z0.s[9] ]
# Increment x4 by 16, thus x4 = 16.
#
# 2nd iteration:
# p0.s = all false predicate since x4 > N.
# Since all lanes of p0 are inactive, terminate the loop.

addarrays_sve:
    mov x4, xzr
    b .cond_2
.loop_body_2:
    # Load elements in z0 from x1 (which is A), corresponding
    # to active lanes of p0 and zero out the rest.
    ld1w z0.s, p0/z, [x1, x4, lsl 2]
    # Similarly, load elements from x2 (which is B) into z1.
    ld1w z1.s, p0/z, [x2, x4, lsl 2]
    add z0.s, z0.s, z1.s
    # Store elements of result computed in z0, corresponding to
    # active lanes of p0 in x0 (x0 is result).
    st1w z0.s, p0, [x0, x4, lsl 2]
    # Increment x4 by number of 32-bit elements in the vector.
    incw x4
.cond_2:
    # Build predicate p0.s = x4 < x3
    whilelt p0.s, x4, x3
    # Branch to beginning of loop body if the first bit in p0 is active.
    b.first .loop_body_2
    ret

.globl main
.type main, %function
main:
    # Load address for res, A, B in x0, x1, x2 respectively
    # and call one of the addarrays function, which will
    # store the computed result in res.
    adrp x0, result
    add x0, x0, :lo12:result
    adrp x1, A
    add x1, x1, :lo12:A
    adrp x2, B
    add x2, x2, :lo12:B
    mov x3, 10
    bl addarrays_sve

    # Call memcmp (result_ref, result, 40) to verify the result.
    adrp x0, result_ref
    add x0, x0, :lo12:result_ref
    adrp x1, result
    add x1, x1, :lo12:result
    mov w2, 40
    bl memcmp

    # The return value of memcmp is computed in w0. Pass that
    # as argument to _exit, so it becomes the exit status of
    # the process. 
    bl _exit

.globl A
.type A, %object
A: .word 10, 13, 5, 8, 1, 42, 65, 17, 21, 24

.globl B
.type B, %object
B: .word 19, 12, 31, 42, 3, 9, 25, 69, 87, 93 

.global result_ref
.type result_ref, %object
result_ref: .word 29, 25, 36, 50, 4, 51, 90, 86, 108, 117

.globl result
.bss
.type  result, %object
.size result, 40
result:
    .zero 40
