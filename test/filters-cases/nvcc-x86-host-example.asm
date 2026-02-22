        .file   "example.cudafe1.cpp"
        .text
.Ltext0:
#APP
        .section .nv_fatbin, "a"
.align 8
fatbinData:
.quad 0x00100001ba55ed50,0x0000000000000fa8,0x0000005001010002,0x0000000000000d08
.quad 0x0000000000000000,0x0000003400010007,0x0000000f00000040,0x0000000000000011
.quad 0x0000000000000000,0x0000000000000000,0x6178652f7070612f,0x0075632e656c706d
.quad 0x33010102464c457f,0x0000000000000007,0x0000007800be0002,0x0000000000000000
.quad 0x0000000000000c60,0x0000000000000920,0x0038004000340534,0x0001000d00400003
.quad 0x7472747368732e00,0x747274732e006261,0x746d79732e006261,0x746d79732e006261
.quad 0x78646e68735f6261,0x7466752e766e2e00,0x2e007972746e652e,0x006f666e692e766e
.quad 0x5a5f2e747865742e,0x5065726175717336,0x692e766e2e006969,0x73365a5f2e6f666e
.quad 0x6969506572617571,0x6168732e766e2e00,0x73365a5f2e646572,0x6969506572617571
.quad 0x6e6f632e766e2e00,0x5f2e30746e617473,0x657261757173365a,0x6265642e00696950
.quad 0x00656e696c5f6775,0x6265642e6c65722e,0x00656e696c5f6775,0x756265645f766e2e
.quad 0x735f656e696c5f67,0x6c65722e00737361,0x756265645f766e2e,0x735f656e696c5f67
.quad 0x5f766e2e00737361,0x74705f6775626564,0x2e00007478745f78,0x6261747274736873
.quad 0x6261747274732e00,0x6261746d79732e00,0x6261746d79732e00,0x2e0078646e68735f
.quad 0x652e7466752e766e,0x766e2e007972746e,0x5a5f006f666e692e,0x5065726175717336
.text

#NO_APP
        .type   _ZL26__cudaUnregisterBinaryUtilv, @function
_ZL26__cudaUnregisterBinaryUtilv:
.LFB1304:
        .file 1 "/opt/compiler-explorer/cuda/12.0.1/bin/../targets/x86_64-linux/include/crt/host_runtime.h"
        .loc 1 257 1 view -0
        .cfi_startproc
        subq    $8, %rsp
        .cfi_def_cfa_offset 16
        movq    _ZL20__cudaFatCubinHandle(%rip), %rdi
        call    __cudaUnregisterFatBinary
        addq    $8, %rsp
        .cfi_def_cfa_offset 8
        ret
        .cfi_endproc
.LFE1304:
        .size   _ZL26__cudaUnregisterBinaryUtilv, .-_ZL26__cudaUnregisterBinaryUtilv
        .globl  main
        .type   main, @function
main:
.LFB1305:
        .file 2 "/app/example.cu"
        .loc 2 23 1 view -0
        .cfi_startproc
        subq    $8, %rsp
        .cfi_def_cfa_offset 16
        movl    $0, %edi
        movl    $0, %esi
        call    _Z6squarePii
        xorl    %eax, %eax
        addq    $8, %rsp
        .cfi_def_cfa_offset 8
        ret
        .cfi_endproc
.LFE1305:
        .size   main, .-main
        .section        .rodata.str1.1,"aMS",@progbits,1
.LC0:
        .string "_Z6squarePii"
        .text
        .type   _ZL24__sti____cudaRegisterAllv, @function
_ZL24__sti____cudaRegisterAllv:
.LFB1329:
        .file 3 "/app/example.fatbin.c"
        .loc 3 2 44 is_stmt 1 view -0
        .cfi_startproc
        subq    $8, %rsp
        .cfi_def_cfa_offset 16
        movl    $_ZL15__fatDeviceText, %edi
        call    __cudaRegisterFatBinary
        movq    %rax, %rdi
        movq    %rax, _ZL20__cudaFatCubinHandle(%rip)
        pushq   $0
        .cfi_def_cfa_offset 24
        movl    $0, %r9d
        movl    $-1, %r8d
        movl    $.LC0, %ecx
        movq    %rcx, %rdx
        movl    $_Z6squarePii, %esi
        call    __cudaRegisterFunction
        addq    $8, %rsp
        .cfi_def_cfa_offset 16
        movq    _ZL20__cudaFatCubinHandle(%rip), %rdi
        call    __cudaRegisterFatBinaryEnd
        movl    $_ZL26__cudaUnregisterBinaryUtilv, %edi
        call    atexit
        addq    $8, %rsp
        .cfi_def_cfa_offset 8
        ret
        .cfi_endproc
.LFE1329:
        .size   _ZL24__sti____cudaRegisterAllv, .-_ZL24__sti____cudaRegisterAllv
        .section        .init_array,"aw"
        .align 8
        .quad   _ZL24__sti____cudaRegisterAllv
        .section        .nvFatBinSegment,"a"
        .align 8
        .type   _ZL15__fatDeviceText, @object
        .size   _ZL15__fatDeviceText, 24
_ZL15__fatDeviceText:
        .long   1180844977
        .long   1
        .quad   fatbinData
        .quad   0
        .local  _ZL20__cudaFatCubinHandle
        .comm   _ZL20__cudaFatCubinHandle,8,8
