; Assemble with NASM and -f elf64

%define q(w,x,y,z) ((((z) << 6) | ((y) << 4) | ((x) << 2) | (w)))

; Sums up 32-bit integers in array.
;
; 1st argument (rdi) = Pointer to start of array.
; 2nd argument (rsi) = Length of array (number of elements, not bytes).
;
; This function may read up to 28 bytes past the end of the array, similar
; to glibc's AVX2 memchr implementation.
;
; Unlike the "Sum over array (Optimized)" C++ example also available in CE,
; this function makes no assumptions about the alignment or size of the array.
;
; Compared to clang's `-O3 -mavx2` ASM generated for the "Sum over array" C++
; example, this code is generally faster when the length of the array is not a
; multiple of the unrolled size (i.e., the number of elements processed by one
; iteration of the unrolled loop), and about the same speed when it evenly fits
; into the unrolled loop size. This is because clang's ASM does not use vector
; operations to reduce the array as much as possible, i.e. it will use a scalar
; loop to process <= 31 elements after the unrolled loop, even when vector
; instructions could be used to reduce the "residual" to <= 7 elements. This
; code always uses vector instructions to add up elements; it has no scalar loop
; to clean up the remaining elements at all.
sum_array_avx2:
  ; If the length (rsi) is not zero, skip past
  ; this return 0 statement.
  test rsi, rsi
  jnz .continue
  xor eax, eax
  ret
.continue:
  ; Zero out the first accumulator register. This register
  ; is always needed no matter what branch we take.
  vpxor xmm0, xmm0

  ; Copy rsi to rdx and rcx. We store the residual number
  ; of elements in rdx, and the number of residual vector
  ; adds in rcx (needed because we unroll the add loop 4x,
  ; and we want to avoid summing the remaining elements
  ; with scalar instructions).
  mov rdx, rsi
  mov rcx, rsi

  ; Get residual number of elements. We can use a 32-bit
  ; instruction here because `x & 7` always clears the
  ; upper 32 bits of x anyway, which is what the 32-bit
  ; version of `and` does. We can use 32-bit instructions
  ; with this register from now on.
  ; edx = len - 8 * (len / 8)
  and edx, 8 - 1

  ; Mask out bits representing the number of residual
  ; elements, to get number of vector add operations.
  ; There are 8 32-bit integers in a ymm register.
  ; rcx = 8 * (len / 8)
  ;
  ; `and` sets the ZF; if there are no vector add iterations,
  ; jump to the label that handles the final residual elements
  ; after the unrolled loop and after the residual vector adds.
  ; We jump to .residual_gt0 because we already handled the case
  ; that there are 0 elements, so we can skip the check if there
  ; are 0 vector add iterations.
  and rcx, -8
  jz .residual_gt0

  ; If we got here, we need to zero out 2 more registers.
  vpxor xmm1, xmm1
  vpxor xmm2, xmm2

  ; rsi = 32 * (len / 32)
  ; This effectively sets rsi to the number of elements we can
  ; process with our 4x unrolled loop. If 0, we skip the unrolled loop.
  and rsi, -(4*8)
  jz .lt32

  ; It is always true that rcx > rsi. rcx - rsi = number of residual
  ; vector adds needed after the main unrolled loop.
  sub rcx, rsi

  ; If we got here, we need to zero out the last register,
  ; because we need it in the unrolled loop.
  vpxor xmm3, xmm3

  ; Point rdi to the next element after the elements processed by the
  ; unrolled loop.
  lea rdi, [rdi + 4*rsi]
  ; We negate rsi here (unrolled length) and add to it until it becomes
  ; 0. We use a negative offset to reuse the ZF set by `add`, as opposed
  ; to having an extra `cmp` instruction.
  neg rsi
.loop:
  ; [<end pointer> + <negative offset> + <local offset>]
  vpaddd ymm0, ymm0, [rdi + 4*rsi + 0*(4*8)]
  vpaddd ymm1, ymm1, [rdi + 4*rsi + 1*(4*8)]
  vpaddd ymm2, ymm2, [rdi + 4*rsi + 2*(4*8)]
  vpaddd ymm3, ymm3, [rdi + 4*rsi + 3*(4*8)]
  add rsi, 32
  ; If the negative offset isn't 0, we can keep iterating.
  jnz .loop
  ; This addition only needs to happen when we do the main unrolled loop.
  vpaddd ymm2, ymm3
.lt32:
  ; Skip over the necessary amount of residual vector adds
  ; based on rcx. The content of rcx here is actually
  ; always 0, 8, 16, or 24, so we only need to check ecx.
  test ecx, ecx
  jz .residual
  cmp ecx, 8
  je .r1
  cmp ecx, 16
  je .r2
  ; Add up remaining vectors. We do this in reverse so that the above
  ; instructions can jump to anywhere in between these instructions.
  vpaddd ymm2, ymm2, [rdi + 2*(4*8)]
.r2:
  vpaddd ymm1, ymm1, [rdi + 1*(4*8)]
.r1:
  vpaddd ymm0, ymm0, [rdi + 0*(4*8)]
.residual:
  ; Sum up ymm0-2 into ymm0.
  vpaddd ymm1, ymm2
  vpaddd ymm0, ymm1

  ; Skip to the end if the number of residual elements is zero.
  test edx, edx
  jz .hsum
.residual_gt0:
  ; Multiply by 32 (size of one row of LUT).
  shl edx, 5
  ; rdx is never 0 here, so we need to subtract the length of
  ; a row since we omit the first row from the table (which
  ; would be all zeros) since it is never used. This means
  ; that if rdx=1, we access the first row of the table.
  vmovdqa ymm4, [mask_lut + rdx - 32]
  ; Zero elements past the bounds of the array based on mask in ymm4.
  ; rdi points to the element after the elements processed by the unrolled
  ; loop, thus we need to add sizeof(int)*rcx to get a pointer to the first
  ; actual residual element.
  ;
  ; This reads up to 28 bytes past the end of the array.
  vpand   ymm4, ymm4, [rdi + 4*rcx]
  vpaddd  ymm0, ymm4
.hsum:
  ; Horizontal reduction of 32-bit integers in ymm0.
  vextracti128    xmm1, ymm0, 1
  vpaddd  xmm0, xmm0, xmm1
  vpshufd xmm1, xmm0, q(2,3,2,3)
  vpaddd  xmm0, xmm0, xmm1
  vpshufd xmm1, xmm0, q(1,1,1,1)
  vpaddd  xmm0, xmm1, xmm0
  vmovd   eax, xmm0
  ret

; Lookup table for masking residual elements.
align 32
mask_lut:      dd \
 -1,  0,  0,  0,  0,  0,  0,  0, \
 -1, -1,  0,  0,  0,  0,  0,  0, \
 -1, -1, -1,  0,  0,  0,  0,  0, \
 -1, -1, -1, -1,  0,  0,  0,  0, \
 -1, -1, -1, -1, -1,  0,  0,  0, \
 -1, -1, -1, -1, -1, -1,  0,  0, \
 -1, -1, -1, -1, -1, -1, -1,  0
