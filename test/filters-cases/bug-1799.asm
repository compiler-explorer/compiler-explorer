  .cpu arm7tdmi
  .eabi_attribute 20, 1
  .eabi_attribute 21, 1
  .eabi_attribute 23, 3
  .eabi_attribute 24, 1
  .eabi_attribute 25, 1
  .eabi_attribute 26, 1
  .eabi_attribute 30, 2
  .eabi_attribute 34, 0
  .eabi_attribute 18, 4
  .file "example.cpp"
  .text
.Ltext0:
  .cfi_sections .debug_frame
  .align 2
  .global DaysInMonth2()
  .arch armv4t
  .syntax unified
  .arm
  .fpu softvfp
  .type DaysInMonth2(), %function
DaysInMonth2():
  .fnstart
.LFB2:
  .file 1 "/app/example.cpp"
  .loc 1 22 20 view -0
  .cfi_startproc
  .loc 1 22 22 view .LVU1
  .loc 1 22 40 is_stmt 0 view .LVU2
  ldr r3, .L8
  ldr r3, [r3]
  ldr r2, .L8+4
.LBB8:
  .loc 1 13 18 view .LVU3
  cmp r3, #1
.LBE8:
  .loc 1 22 40 view .LVU4
  ldr r2, [r2]
.LVL0:
.LBB13:
.LBI8:
  .loc 1 10 15 is_stmt 1 view .LVU5
  .loc 1 13 3 view .LVU6
  .loc 1 13 18 is_stmt 0 view .LVU7
  beq .L7
.L2:
  .loc 1 17 3 is_stmt 1 view .LVU8
  .loc 1 17 23 is_stmt 0 view .LVU9
  ldr r2, .L8+8
.LVL1:
  .loc 1 17 23 view .LVU10
  ldr r0, [r2, r3, lsl #2]
  bx lr
.LVL2:
.L7:
.LBB10:
.LBI10:
  .loc 1 1 16 is_stmt 1 view .LVU11
  .loc 1 4 3 view .LVU12
  .loc 1 4 26 is_stmt 0 view .LVU13
  tst r2, #3
  bne .L2
  .loc 1 4 36 view .LVU14
  ldr r1, .L8+12
  smull ip, r0, r1, r2
  asr r1, r2, #31
  rsb r1, r1, r0, asr #3
  add r1, r1, r1, lsl #2
  add r1, r1, r1, lsl #2
  .loc 1 4 26 view .LVU15
  cmp r2, r1
  bne .L5
  .loc 1 4 47 view .LVU16
  tst r2, #15
  bne .L2
.L5:
  .loc 1 4 47 view .LVU17
.LBE10:
  .loc 1 14 12 view .LVU18
  mov r0, #29
.LVL3:
  .loc 1 14 12 view .LVU19
.LBE13:
  .loc 1 22 48 view .LVU20
  bx lr
  .align 2
.L8:
  .word b
  .word a
  .word .LANCHOR0
  .word 1374389535
  .cfi_endproc
.LFE2:
  .cantunwind
  .fnend
  .size DaysInMonth2(), .-DaysInMonth2()
  .section .rodata
  .align 2
  .set .LANCHOR0,. + 0
  .type sg_days, %object
  .size sg_days, 48
sg_days:
  .word 31
  .word 28
  .word 31
  .word 30
  .word 31
  .word 30
  .word 31
  .word 31
  .word 30
  .word 31
  .word 30
  .word 31
  .text
.Letext0:
  .section .debug_info,"",%progbits
.Ldebug_info0:
  .4byte 0x135
  .2byte 0x4
