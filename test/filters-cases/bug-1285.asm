        .text
        .file   "example.cpp"
        .file   1 "/tmp/compiler-explorer-compiler119221-62-2dbhpb.3gqfx" "example.cpp"
        .file   2 "/opt/compiler-explorer/gcc-8.3.0/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0" "bitset"
        .file   3 "/opt/compiler-explorer/gcc-8.3.0/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0/x86_64-linux-gnu/bits" "c++config.h"
        .file   4 "/opt/compiler-explorer/gcc-8.3.0/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0/debug" "debug.h"
        .file   5 "/usr/include/x86_64-linux-gnu/bits/types" "__mbstate_t.h"
        .file   6 "/usr/include/x86_64-linux-gnu/bits/types" "mbstate_t.h"
        .file   7 "/opt/compiler-explorer/gcc-8.3.0/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0" "cwchar"
        .file   8 "/usr/include/x86_64-linux-gnu/bits/types" "wint_t.h"
        .file   9 "/usr/include" "wchar.h"
        .file   10 "/usr/include/x86_64-linux-gnu/bits" "libio.h"
        .file   11 "/usr/include/x86_64-linux-gnu/bits" "types.h"
        .file   12 "/opt/compiler-explorer/clang-trunk-20190319/lib/clang/9.0.0/include" "stddef.h"
        .file   13 "/usr/include/x86_64-linux-gnu/bits/types" "__FILE.h"
        .file   14 "/usr/include/x86_64-linux-gnu/bits" "stdint-intn.h"
        .file   15 "/opt/compiler-explorer/gcc-8.3.0/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0" "cstdint"
        .file   16 "/usr/include" "stdint.h"
        .file   17 "/usr/include/x86_64-linux-gnu/bits" "stdint-uintn.h"
        .file   18 "/opt/compiler-explorer/gcc-8.3.0/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0/bits" "exception_ptr.h"
        .file   19 "/opt/compiler-explorer/gcc-8.3.0/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0/ext" "new_allocator.h"
        .file   20 "/opt/compiler-explorer/gcc-8.3.0/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0" "clocale"
        .file   21 "/usr/include" "locale.h"
        .file   22 "/usr/include" "ctype.h"
        .file   23 "/opt/compiler-explorer/gcc-8.3.0/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0" "cctype"
        .file   24 "/usr/include" "stdlib.h"
        .file   25 "/opt/compiler-explorer/gcc-8.3.0/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0/bits" "std_abs.h"
        .file   26 "/opt/compiler-explorer/gcc-8.3.0/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0" "cstdlib"
        .file   27 "/usr/include/x86_64-linux-gnu/bits" "stdlib-float.h"
        .file   28 "/usr/include/x86_64-linux-gnu/bits" "stdlib-bsearch.h"
        .file   29 "/usr/include/x86_64-linux-gnu/bits/types" "FILE.h"
        .file   30 "/opt/compiler-explorer/gcc-8.3.0/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0" "cstdio"
        .file   31 "/usr/include/x86_64-linux-gnu/bits" "_G_config.h"
        .file   32 "/usr/include" "stdio.h"
        .file   33 "/usr/include/x86_64-linux-gnu/bits" "stdio.h"
        .globl  _Z16apply_known_maskRSt6bitsetILm64EE # -- Begin function _Z16apply_known_maskRSt6bitsetILm64EE
        .p2align        4, 0x90
        .type   _Z16apply_known_maskRSt6bitsetILm64EE,@function
_Z16apply_known_maskRSt6bitsetILm64EE:  # @_Z16apply_known_maskRSt6bitsetILm64EE
.Lfunc_begin0:
        .loc    1 7 0                   # example.cpp:7:0
        .cfi_startproc
# %bb.0:
        #DEBUG_VALUE: apply_known_mask:bits <- $rdi
        #DEBUG_VALUE: apply_known_mask:bits <- $rdi
        #DEBUG_VALUE: operator&=:this <- $rdi
        #DEBUG_VALUE: operator&=:this <- $rdi
        #DEBUG_VALUE: _M_do_and:this <- $rdi
        #DEBUG_VALUE: _M_do_and:this <- $rdi
        #DEBUG_VALUE: operator&=:__rhs <- undef
        #DEBUG_VALUE: _M_do_and:__x <- undef
        #DEBUG_VALUE: apply_known_mask:mask <- [DW_OP_deref] undef
        .loc    2 433 14 prologue_end   # /opt/compiler-explorer/gcc-8.3.0/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0/bitset:433:14
        andq    $775946532, (%rdi)      # imm = 0x2E400124
.Ltmp0:
        .loc    1 15 1                  # example.cpp:15:1
        retq
.Ltmp1:
.Lfunc_end0:
        .size   _Z16apply_known_maskRSt6bitsetILm64EE, .Lfunc_end0-_Z16apply_known_maskRSt6bitsetILm64EE
        .cfi_endproc
