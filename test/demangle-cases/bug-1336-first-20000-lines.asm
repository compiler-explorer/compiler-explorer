_ZSt19piecewise_construct:
  .zero 1
_ZN9__gnu_cxx5__ops16__iter_less_iterEv:
  push rbp
  mov rbp, rsp
  pop rbp
  ret
_ZN9__gnu_cxx5__ops14_Iter_less_valC2ENS0_15_Iter_less_iterE:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  nop
  pop rbp
  ret
_ZN9__gnu_cxx5__ops15__iter_less_valEv:
  push rbp
  mov rbp, rsp
  pop rbp
  ret
_ZN9__gnu_cxx5__ops15__val_comp_iterENS0_15_Iter_less_iterE:
  push rbp
  mov rbp, rsp
  pop rbp
  ret
_ZN9__gnu_cxx5__ops20__iter_equal_to_iterEv:
  push rbp
  mov rbp, rsp
  pop rbp
  ret
_ZSt4__lgl:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  bsr rax, rax
  xor rax, 63
  cdqe
  mov edx, 63
  sub rdx, rax
  mov rax, rdx
  pop rbp
  ret
_ZNKSt9type_infoeqERKS_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rax+8]
  cmp rdx, rax
  je .L13
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+8]
  movzx eax, BYTE PTR [rax]
  cmp al, 42
  je .L14
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+8]
  mov rsi, rdx
  mov rdi, rax
  call strcmp
  test eax, eax
  jne .L14
.L13:
  mov eax, 1
  jmp .L15
.L14:
  mov eax, 0
.L15:
  leave
  ret
_ZnwmPv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  pop rbp
  ret
_ZdlPvS_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  nop
  pop rbp
  ret
_ZNSt11char_traitsIcE6assignERcRKc:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  movzx edx, BYTE PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax], dl
  nop
  pop rbp
  ret
_ZNSt11char_traitsIcE7compareEPKcS2_m:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov eax, 0
  test al, al
  je .L22
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZN9__gnu_cxx11char_traitsIcE7compareEPKcS3_m
  jmp .L23
.L22:
  cmp QWORD PTR [rbp-24], 0
  jne .L24
  mov eax, 0
  jmp .L23
.L24:
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call memcmp
  nop
.L23:
  leave
  ret
_ZNSt11char_traitsIcE6lengthEPKc:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rbp-8], rax
  mov eax, 0
  test al, al
  je .L27
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZN9__gnu_cxx11char_traitsIcE6lengthEPKc
  jmp .L28
.L27:
  mov rax, QWORD PTR [rbp-24]
  mov rcx, -1
  mov rdx, rax
  mov eax, 0
  mov rdi, rdx
  repnz scasb
  mov rax, rcx
  not rax
  sub rax, 1
  nop
.L28:
  leave
  ret
_ZNSt11char_traitsIcE4moveEPcPKcm:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  cmp QWORD PTR [rbp-24], 0
  jne .L30
  mov rax, QWORD PTR [rbp-8]
  jmp .L31
.L30:
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call memmove
  nop
.L31:
  leave
  ret
_ZNSt11char_traitsIcE4copyEPcPKcm:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  cmp QWORD PTR [rbp-24], 0
  jne .L33
  mov rax, QWORD PTR [rbp-8]
  jmp .L34
.L33:
  mov rdx, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rbp-16]
  mov rcx, rdx
  mov rsi, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdx, rax
  mov rdi, rcx
  call memcpy
  nop
.L34:
  leave
  ret
_ZNSt11char_traitsIcE6assignEPcmc:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov eax, edx
  mov BYTE PTR [rbp-20], al
  cmp QWORD PTR [rbp-16], 0
  jne .L36
  mov rax, QWORD PTR [rbp-8]
  jmp .L37
.L36:
  movsx ecx, BYTE PTR [rbp-20]
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call memset
  nop
.L37:
  leave
  ret
_ZZL18__gthread_active_pvE20__gthread_active_ptr:
  .quad _ZL28__gthrw___pthread_key_createPjPFvPvE
_ZL18__gthread_active_pv:
  push rbp
  mov rbp, rsp
  mov rax, QWORD PTR _ZZL18__gthread_active_pvE20__gthread_active_ptr[rip]
  test rax, rax
  setne al
  movzx eax, al
  pop rbp
  ret
_ZN9__gnu_cxxL18__exchange_and_addEPVii:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov DWORD PTR [rbp-12], esi
  mov edx, DWORD PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  lock xadd DWORD PTR [rax], edx
  mov eax, edx
  pop rbp
  ret
_ZN9__gnu_cxxL25__exchange_and_add_singleEPii:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-24], rdi
  mov DWORD PTR [rbp-28], esi
  mov rax, QWORD PTR [rbp-24]
  mov eax, DWORD PTR [rax]
  mov DWORD PTR [rbp-4], eax
  mov rax, QWORD PTR [rbp-24]
  mov edx, DWORD PTR [rax]
  mov eax, DWORD PTR [rbp-28]
  add edx, eax
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax], edx
  mov eax, DWORD PTR [rbp-4]
  pop rbp
  ret
_ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov DWORD PTR [rbp-12], esi
  call _ZL18__gthread_active_pv
  test eax, eax
  setne al
  test al, al
  je .L45
  mov edx, DWORD PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, edx
  mov rdi, rax
  call _ZN9__gnu_cxxL18__exchange_and_addEPVii
  jmp .L46
.L45:
  mov edx, DWORD PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, edx
  mov rdi, rax
  call _ZN9__gnu_cxxL25__exchange_and_add_singleEPii
  nop
.L46:
  leave
  ret
_ZStanSt13_Ios_FmtflagsS_:
  push rbp
  mov rbp, rsp
  mov DWORD PTR [rbp-4], edi
  mov DWORD PTR [rbp-8], esi
  mov eax, DWORD PTR [rbp-4]
  and eax, DWORD PTR [rbp-8]
  pop rbp
  ret
_ZStorSt13_Ios_FmtflagsS_:
  push rbp
  mov rbp, rsp
  mov DWORD PTR [rbp-4], edi
  mov DWORD PTR [rbp-8], esi
  mov eax, DWORD PTR [rbp-4]
  or eax, DWORD PTR [rbp-8]
  pop rbp
  ret
_ZStcoSt13_Ios_Fmtflags:
  push rbp
  mov rbp, rsp
  mov DWORD PTR [rbp-4], edi
  mov eax, DWORD PTR [rbp-4]
  not eax
  pop rbp
  ret
_ZStoRRSt13_Ios_FmtflagsS_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov DWORD PTR [rbp-12], esi
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax]
  mov edx, DWORD PTR [rbp-12]
  mov esi, edx
  mov edi, eax
  call _ZStorSt13_Ios_FmtflagsS_
  mov edx, eax
  mov rax, QWORD PTR [rbp-8]
  mov DWORD PTR [rax], edx
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZStaNRSt13_Ios_FmtflagsS_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov DWORD PTR [rbp-12], esi
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax]
  mov edx, DWORD PTR [rbp-12]
  mov esi, edx
  mov edi, eax
  call _ZStanSt13_Ios_FmtflagsS_
  mov edx, eax
  mov rax, QWORD PTR [rbp-8]
  mov DWORD PTR [rax], edx
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSt8ios_base4setfESt13_Ios_FmtflagsS0_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov DWORD PTR [rbp-28], esi
  mov DWORD PTR [rbp-32], edx
  mov rax, QWORD PTR [rbp-24]
  mov eax, DWORD PTR [rax+24]
  mov DWORD PTR [rbp-4], eax
  mov eax, DWORD PTR [rbp-32]
  mov edi, eax
  call _ZStcoSt13_Ios_Fmtflags
  mov edx, eax
  mov rax, QWORD PTR [rbp-24]
  add rax, 24
  mov esi, edx
  mov rdi, rax
  call _ZStaNRSt13_Ios_FmtflagsS_
  mov edx, DWORD PTR [rbp-32]
  mov eax, DWORD PTR [rbp-28]
  mov esi, edx
  mov edi, eax
  call _ZStanSt13_Ios_FmtflagsS_
  mov edx, eax
  mov rax, QWORD PTR [rbp-24]
  add rax, 24
  mov esi, edx
  mov rdi, rax
  call _ZStoRRSt13_Ios_FmtflagsS_
  mov eax, DWORD PTR [rbp-4]
  leave
  ret
_ZSt3hexRSt8ios_base:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov edx, 74
  mov esi, 8
  mov rdi, rax
  call _ZNSt8ios_base4setfESt13_Ios_FmtflagsS0_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZSt3octRSt8ios_base:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov edx, 74
  mov esi, 64
  mov rdi, rax
  call _ZNSt8ios_base4setfESt13_Ios_FmtflagsS0_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNKSt5ctypeIcE7toupperEc:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov eax, esi
  mov BYTE PTR [rbp-12], al
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  add rax, 16
  mov rax, QWORD PTR [rax]
  movsx ecx, BYTE PTR [rbp-12]
  mov rdx, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rdx
  call rax
  leave
  ret
_ZNKSt5ctypeIcE7tolowerEc:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov eax, esi
  mov BYTE PTR [rbp-12], al
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  add rax, 32
  mov rax, QWORD PTR [rax]
  movsx ecx, BYTE PTR [rbp-12]
  mov rdx, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rdx
  call rax
  leave
  ret
_ZNKSt5ctypeIcE7tolowerEPcPKc:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  add rax, 40
  mov rax, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-24]
  mov rsi, QWORD PTR [rbp-16]
  mov rcx, QWORD PTR [rbp-8]
  mov rdi, rcx
  call rax
  leave
  ret
_ZNKSt5ctypeIcE5widenEc:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov eax, esi
  mov BYTE PTR [rbp-12], al
  mov rax, QWORD PTR [rbp-8]
  movzx eax, BYTE PTR [rax+56]
  test al, al
  je .L70
  movzx eax, BYTE PTR [rbp-12]
  movzx eax, al
  mov rdx, QWORD PTR [rbp-8]
  cdqe
  movzx eax, BYTE PTR [rdx+57+rax]
  jmp .L71
.L70:
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt5ctypeIcE13_M_widen_initEv
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  add rax, 48
  mov rax, QWORD PTR [rax]
  movsx ecx, BYTE PTR [rbp-12]
  mov rdx, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rdx
  call rax
  nop
.L71:
  leave
  ret
_ZNKSt5ctypeIcE6narrowEcc:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov BYTE PTR [rbp-28], al
  mov eax, edx
  mov BYTE PTR [rbp-32], al
  movzx eax, BYTE PTR [rbp-28]
  movzx eax, al
  mov rdx, QWORD PTR [rbp-24]
  cdqe
  movzx eax, BYTE PTR [rdx+313+rax]
  test al, al
  je .L73
  movzx eax, BYTE PTR [rbp-28]
  movzx eax, al
  mov rdx, QWORD PTR [rbp-24]
  cdqe
  movzx eax, BYTE PTR [rdx+313+rax]
  jmp .L74
.L73:
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax]
  add rax, 64
  mov rax, QWORD PTR [rax]
  movsx edx, BYTE PTR [rbp-32]
  movsx esi, BYTE PTR [rbp-28]
  mov rcx, QWORD PTR [rbp-24]
  mov rdi, rcx
  call rax
  mov BYTE PTR [rbp-1], al
  movzx eax, BYTE PTR [rbp-1]
  cmp al, BYTE PTR [rbp-32]
  je .L75
  movzx eax, BYTE PTR [rbp-28]
  movzx eax, al
  mov rdx, QWORD PTR [rbp-24]
  cdqe
  movzx ecx, BYTE PTR [rbp-1]
  mov BYTE PTR [rdx+313+rax], cl
.L75:
  movzx eax, BYTE PTR [rbp-1]
.L74:
  leave
  ret
_ZNKSt5ctypeIcE2isEtc:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov eax, esi
  mov WORD PTR [rbp-12], ax
  mov eax, edx
  mov BYTE PTR [rbp-16], al
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+48]
  movzx edx, BYTE PTR [rbp-16]
  movzx edx, dl
  add rdx, rdx
  add rax, rdx
  movzx eax, WORD PTR [rax]
  and ax, WORD PTR [rbp-12]
  test ax, ax
  setne al
  pop rbp
  ret
_ZNSt15_Rb_tree_headerC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov DWORD PTR [rax], 0
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt15_Rb_tree_header8_M_resetEv
  nop
  leave
  ret
_ZNSt15_Rb_tree_header8_M_resetEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+8], 0
  mov rdx, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+16], rdx
  mov rdx, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+24], rdx
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+32], 0
  nop
  pop rbp
  ret
_ZZNSt19_Sp_make_shared_tag5_S_tiEvE5__tag:
  .zero 16
_ZNSt19_Sp_make_shared_tag5_S_tiEv:
  push rbp
  mov rbp, rsp
  mov eax, OFFSET FLAT:_ZZNSt19_Sp_make_shared_tag5_S_tiEvE5__tag
  pop rbp
  ret
_ZSt16__deque_buf_sizem:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  cmp QWORD PTR [rbp-8], 511
  ja .L83
  mov eax, 512
  mov edx, 0
  div QWORD PTR [rbp-8]
  jmp .L85
.L83:
  mov eax, 1
.L85:
  pop rbp
  ret
_ZNSt9_Any_data9_M_accessEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNKSt9_Any_data9_M_accessEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt14_Function_baseC2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+16], 0
  nop
  pop rbp
  ret
_ZNSt14_Function_baseD2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+16]
  test rax, rax
  je .L93
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+16]
  mov rsi, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rbp-8]
  mov edx, 3
  mov rdi, rcx
  call rax
.L93:
  nop
  leave
  ret
_ZNKSt14_Function_base8_M_emptyEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+16]
  test rax, rax
  sete al
  pop rbp
  ret
_ZNSt15regex_constantsanENS_18syntax_option_typeES0_:
  push rbp
  mov rbp, rsp
  mov DWORD PTR [rbp-4], edi
  mov DWORD PTR [rbp-8], esi
  mov eax, DWORD PTR [rbp-4]
  and eax, DWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt15regex_constantsorENS_18syntax_option_typeES0_:
  push rbp
  mov rbp, rsp
  mov DWORD PTR [rbp-4], edi
  mov DWORD PTR [rbp-8], esi
  mov eax, DWORD PTR [rbp-4]
  or eax, DWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt15regex_constantsanENS_15match_flag_typeES0_:
  push rbp
  mov rbp, rsp
  mov DWORD PTR [rbp-4], edi
  mov DWORD PTR [rbp-8], esi
  mov eax, DWORD PTR [rbp-4]
  and eax, DWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt15regex_constantsorENS_15match_flag_typeES0_:
  push rbp
  mov rbp, rsp
  mov DWORD PTR [rbp-4], edi
  mov DWORD PTR [rbp-8], esi
  mov eax, DWORD PTR [rbp-4]
  or eax, DWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt15regex_constantscoENS_15match_flag_typeE:
  push rbp
  mov rbp, rsp
  mov DWORD PTR [rbp-4], edi
  mov eax, DWORD PTR [rbp-4]
  not eax
  pop rbp
  ret
_ZNSt11regex_errorC2ENSt15regex_constants10error_typeEPKc:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov DWORD PTR [rbp-12], esi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt13runtime_errorC2EPKc
  mov edx, OFFSET FLAT:_ZTVSt11regex_error+16
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-8]
  mov edx, DWORD PTR [rbp-12]
  mov DWORD PTR [rax+16], edx
  nop
  leave
  ret
_ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 16
  mov DWORD PTR [rbp-20], edi
  mov QWORD PTR [rbp-32], rsi
  mov edi, 24
  call __cxa_allocate_exception
  mov rbx, rax
  mov rdx, QWORD PTR [rbp-32]
  mov eax, DWORD PTR [rbp-20]
  mov esi, eax
  mov rdi, rbx
  call _ZNSt11regex_errorC1ENSt15regex_constants10error_typeEPKc
  mov edx, OFFSET FLAT:_ZNSt11regex_errorD1Ev
  mov esi, OFFSET FLAT:_ZTISt11regex_error
  mov rdi, rbx
  call __cxa_throw
  mov r12, rax
  mov rdi, rbx
  call __cxa_free_exception
  mov rax, r12
  mov rdi, rax
  call _Unwind_Resume
_ZNSt8__detail11_State_baseC2ENS_7_OpcodeE:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov DWORD PTR [rbp-12], esi
  mov rax, QWORD PTR [rbp-8]
  mov edx, DWORD PTR [rbp-12]
  mov DWORD PTR [rax], edx
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+8], -1
  nop
  pop rbp
  ret
_ZNSt8__detail11_State_base10_M_has_altEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax]
  cmp eax, 1
  je .L112
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax]
  cmp eax, 2
  je .L112
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax]
  cmp eax, 7
  jne .L113
.L112:
  mov eax, 1
  jmp .L114
.L113:
  mov eax, 0
.L114:
  pop rbp
  ret
_ZNSt8__detail9_NFA_baseC2ENSt15regex_constants18syntax_option_typeE:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov DWORD PTR [rbp-12], esi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt6vectorImSaImEEC1Ev
  mov rax, QWORD PTR [rbp-8]
  mov edx, DWORD PTR [rbp-12]
  mov DWORD PTR [rax+24], edx
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+32], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+40], 0
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+48], 0
  nop
  leave
  ret
_ZNKSt8__detail9_NFA_base8_M_startEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+32]
  pop rbp
  ret
_ZNKSt8__detail9_NFA_base12_M_sub_countEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+40]
  pop rbp
  ret
_ZNKSt17integral_constantIbLb1EEcvbEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov eax, 1
  pop rbp
  ret
_ZSt7forwardIcEOT_RNSt16remove_referenceIS0_E4typeE:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt4pairIccEC1IccLb1EEEOT_OT0_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZSt7forwardIcEOT_RNSt16remove_referenceIS0_E4typeE
  movzx edx, BYTE PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax], dl
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt7forwardIcEOT_RNSt16remove_referenceIS0_E4typeE
  movzx edx, BYTE PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+1], dl
  nop
  leave
  ret
.LC0:
  .string "^$\\.*+?()[]{}|"
.LC1:
  .string ".[\\*^$"
.LC2:
  .string ".[\\()*+?{|^$"
.LC3:
  .string ".[\\()*+?{|^$\n"
.LC4:
  .string ".[\\*^$\n"
_ZNSt8__detail12_ScannerBaseC2ENSt15regex_constants18syntax_option_typeE:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov DWORD PTR [rbp-12], esi
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], 0
  mov QWORD PTR [rax+8], 0
  mov QWORD PTR [rax+16], 0
  mov QWORD PTR [rax+24], 0
  mov QWORD PTR [rax+32], 0
  mov QWORD PTR [rax+40], 0
  mov QWORD PTR [rax+48], 0
  mov QWORD PTR [rax+56], 0
  mov QWORD PTR [rax+64], 0
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax], 94
  mov rax, QWORD PTR [rbp-8]
  mov DWORD PTR [rax+4], 22
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+8], 36
  mov rax, QWORD PTR [rbp-8]
  mov DWORD PTR [rax+12], 23
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+16], 46
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+24], 42
  mov rax, QWORD PTR [rbp-8]
  mov DWORD PTR [rax+28], 20
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+32], 43
  mov rax, QWORD PTR [rbp-8]
  mov DWORD PTR [rax+36], 21
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+40], 63
  mov rax, QWORD PTR [rbp-8]
  mov DWORD PTR [rax+44], 18
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+48], 124
  mov rax, QWORD PTR [rbp-8]
  mov DWORD PTR [rax+52], 19
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+56], 10
  mov rax, QWORD PTR [rbp-8]
  mov DWORD PTR [rax+60], 19
  mov rax, QWORD PTR [rbp-8]
  mov DWORD PTR [rax+68], 19
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+72], 0
  mov QWORD PTR [rax+80], 0
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+72], 48
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+74], 98
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+75], 8
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+76], 102
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+77], 12
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+78], 110
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+79], 10
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+80], 114
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+81], 13
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+82], 116
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+83], 9
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+84], 118
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+85], 11
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+88], 0
  mov QWORD PTR [rax+96], 0
  mov DWORD PTR [rax+104], 0
  mov WORD PTR [rax+108], 0
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+88], 34
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+89], 34
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+90], 47
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+91], 47
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+92], 92
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+93], 92
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+94], 97
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+95], 7
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+96], 98
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+97], 8
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+98], 102
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+99], 12
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+100], 110
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+101], 10
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+102], 114
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+103], 13
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+104], 116
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+105], 9
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+106], 118
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+107], 11
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+112], OFFSET FLAT:.LC0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+120], OFFSET FLAT:.LC1
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+128], OFFSET FLAT:.LC2
  mov rax, QWORD PTR [rbp-8]
  mov DWORD PTR [rax+136], 0
  mov rax, QWORD PTR [rbp-8]
  mov edx, DWORD PTR [rbp-12]
  mov DWORD PTR [rax+140], edx
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt8__detail12_ScannerBase10_M_is_ecmaEv
  test al, al
  je .L127
  mov rax, QWORD PTR [rbp-8]
  add rax, 72
  jmp .L128
.L127:
  mov rax, QWORD PTR [rbp-8]
  add rax, 88
.L128:
  mov rdx, QWORD PTR [rbp-8]
  mov QWORD PTR [rdx+152], rax
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt8__detail12_ScannerBase10_M_is_ecmaEv
  test al, al
  je .L129
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+112]
  jmp .L130
.L129:
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+140]
  mov esi, 32
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  je .L131
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+120]
  jmp .L130
.L131:
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+140]
  mov esi, 64
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  je .L133
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+128]
  jmp .L130
.L133:
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+140]
  mov esi, 256
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  jne .L135
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+140]
  mov esi, 512
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  jne .L136
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+140]
  mov esi, 128
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  je .L137
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+128]
  jmp .L130
.L137:
  mov eax, 0
  jmp .L130
.L136:
  mov eax, OFFSET FLAT:.LC3
  jmp .L130
.L135:
  mov eax, OFFSET FLAT:.LC4
.L130:
  mov rdx, QWORD PTR [rbp-8]
  mov QWORD PTR [rdx+160], rax
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+168], 0
  nop
  leave
  ret
_ZNSt8__detail12_ScannerBase14_M_find_escapeEc:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov BYTE PTR [rbp-28], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+152]
  mov QWORD PTR [rbp-8], rax
.L145:
  mov rax, QWORD PTR [rbp-8]
  movzx eax, BYTE PTR [rax]
  test al, al
  je .L142
  mov rax, QWORD PTR [rbp-8]
  movzx eax, BYTE PTR [rax]
  cmp BYTE PTR [rbp-28], al
  jne .L143
  mov rax, QWORD PTR [rbp-8]
  add rax, 1
  jmp .L144
.L143:
  add QWORD PTR [rbp-8], 2
  jmp .L145
.L142:
  mov eax, 0
.L144:
  pop rbp
  ret
_ZNKSt8__detail12_ScannerBase10_M_is_ecmaEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+140]
  mov esi, 16
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  setne al
  leave
  ret
_ZNKSt8__detail12_ScannerBase11_M_is_basicEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov esi, 256
  mov edi, 32
  call _ZNSt15regex_constantsorENS_18syntax_option_typeES0_
  mov edx, eax
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+140]
  mov esi, edx
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  setne al
  leave
  ret
_ZNKSt8__detail12_ScannerBase9_M_is_awkEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+140]
  mov esi, 128
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  setne al
  leave
  ret
_ZSt3minImERKT_S2_S2_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  cmp rdx, rax
  jnb .L153
  mov rax, QWORD PTR [rbp-16]
  jmp .L154
.L153:
  mov rax, QWORD PTR [rbp-8]
.L154:
  pop rbp
  ret
_ZN9__gnu_cxx14__alloc_traitsISaIcEcE27_S_propagate_on_move_assignEv:
  push rbp
  mov rbp, rsp
  mov eax, 1
  pop rbp
  ret
_ZN9__gnu_cxx14__alloc_traitsISaIcEcE15_S_always_equalEv:
  push rbp
  mov rbp, rsp
  mov eax, 1
  pop rbp
  ret
.LC5:
  .string "[a-fA-F0-9]{4}"
.LC6:
  .string "012z"
main:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 56
  lea rax, [rbp-48]
  mov edx, 16
  mov esi, OFFSET FLAT:.LC5
  mov rdi, rax
  call _ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEEC1EPKcNSt15regex_constants18syntax_option_typeE
  lea rax, [rbp-48]
  mov edx, 0
  mov rsi, rax
  mov edi, OFFSET FLAT:.LC6
  call _ZSt11regex_matchIcNSt7__cxx1112regex_traitsIcEEEbPKT_RKNS0_11basic_regexIS3_T0_EENSt15regex_constants15match_flag_typeE
  movzx eax, al
  mov esi, eax
  mov edi, OFFSET FLAT:_ZSt4cout
  call _ZNSolsEb
  mov esi, OFFSET FLAT:_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
  mov rdi, rax
  call _ZNSolsEPFRSoS_E
  mov BYTE PTR [rbp-51], 1
  mov BYTE PTR [rbp-50], 1
  mov BYTE PTR [rbp-49], 0
  lea rax, [rbp-51]
  add rax, 3
  lea rcx, [rbp-51]
  mov edx, 1
  mov rsi, rax
  mov rdi, rcx
  call _ZSt10accumulateIPbiSt7bit_andIbEET0_T_S4_S3_T1_
  mov esi, eax
  mov edi, OFFSET FLAT:_ZSt4cout
  call _ZNSolsEi
  lea rax, [rbp-48]
  mov rdi, rax
  call _ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEED1Ev
  mov eax, 0
  jmp .L163
  mov rbx, rax
  lea rax, [rbp-48]
  mov rdi, rax
  call _ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L163:
  add rsp, 56
  pop rbx
  pop rbp
  ret
_ZN9__gnu_cxx11char_traitsIcE2ltERKcS3_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  movzx edx, BYTE PTR [rax]
  mov rax, QWORD PTR [rbp-16]
  movzx eax, BYTE PTR [rax]
  cmp dl, al
  setl al
  pop rbp
  ret
_ZN9__gnu_cxx11char_traitsIcE7compareEPKcS3_m:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov QWORD PTR [rbp-8], 0
.L171:
  mov rax, QWORD PTR [rbp-8]
  cmp rax, QWORD PTR [rbp-40]
  jnb .L167
  mov rdx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-8]
  add rdx, rax
  mov rcx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-8]
  add rax, rcx
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx11char_traitsIcE2ltERKcS3_
  test al, al
  je .L168
  mov eax, -1
  jmp .L169
.L168:
  mov rdx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-8]
  add rdx, rax
  mov rcx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-8]
  add rax, rcx
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx11char_traitsIcE2ltERKcS3_
  test al, al
  je .L170
  mov eax, 1
  jmp .L169
.L170:
  add QWORD PTR [rbp-8], 1
  jmp .L171
.L167:
  mov eax, 0
.L169:
  leave
  ret
_ZN9__gnu_cxx11char_traitsIcE6lengthEPKc:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-8], 0
.L174:
  mov BYTE PTR [rbp-9], 0
  mov rdx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-8]
  add rdx, rax
  lea rax, [rbp-9]
  mov rsi, rax
  mov rdi, rdx
  call _ZN9__gnu_cxx11char_traitsIcE2eqERKcS3_
  xor eax, 1
  test al, al
  je .L173
  add QWORD PTR [rbp-8], 1
  jmp .L174
.L173:
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZN9__gnu_cxx11char_traitsIcE2eqERKcS3_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  movzx edx, BYTE PTR [rax]
  mov rax, QWORD PTR [rbp-16]
  movzx eax, BYTE PTR [rax]
  cmp dl, al
  sete al
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSaIcED2Ev
  nop
  leave
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD1Ev
  nop
  leave
  ret
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4dataEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  leave
  ret
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+8]
  pop rbp
  ret
_ZNSt6vectorImSaImEEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseImSaImEEC2Ev
  nop
  leave
  ret
_ZNSt6vectorImSaImEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseImSaImEE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt8_DestroyIPmmEvT_S1_RSaIT0_E
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseImSaImEED2Ev
  nop
  leave
  ret
_ZSt3maxImERKT_S2_S2_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rax]
  cmp rdx, rax
  jnb .L187
  mov rax, QWORD PTR [rbp-16]
  jmp .L188
.L187:
  mov rax, QWORD PTR [rbp-8]
.L188:
  pop rbp
  ret
_ZNSt14__shared_countILN9__gnu_cxx12_Lock_policyE2EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  test rax, rax
  je .L191
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  mov rdi, rax
  call _ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv
.L191:
  nop
  leave
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2Ev:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-40], rdi
  mov rbx, QWORD PTR [rbp-40]
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaIcEC1Ev
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv
  mov rcx, rax
  lea rax, [rbp-17]
  mov rdx, rax
  mov rsi, rcx
  mov rdi, rbx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC1EPcOS3_
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaIcED1Ev
  mov rax, QWORD PTR [rbp-40]
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm
  nop
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2EOS4_:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 16
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rbx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv
  mov rdi, rax
  call _ZSt4moveIRSaIcEEONSt16remove_referenceIT_E4typeEOS3_
  mov r12, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv
  mov rdx, r12
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC1EPcOS3_
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_is_localEv
  test al, al
  je .L194
  mov rax, QWORD PTR [rbp-32]
  lea rcx, [rax+16]
  mov rax, QWORD PTR [rbp-24]
  add rax, 16
  mov edx, 16
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt11char_traitsIcE4copyEPcPKcm
  jmp .L195
.L194:
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc
  mov rax, QWORD PTR [rbp-32]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm
.L195:
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-32]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc
  mov rax, QWORD PTR [rbp-32]
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm
  nop
  add rsp, 16
  pop rbx
  pop r12
  pop rbp
  ret
_ZN9__gnu_cxx14__alloc_traitsISaIcEcE27_S_propagate_on_copy_assignEv:
  push rbp
  mov rbp, rsp
  mov eax, 0
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSERKS4_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 56
  mov QWORD PTR [rbp-56], rdi
  mov QWORD PTR [rbp-64], rsi
  call _ZN9__gnu_cxx14__alloc_traitsISaIcEcE27_S_propagate_on_copy_assignEv
  test al, al
  je .L199
  call _ZN9__gnu_cxx14__alloc_traitsISaIcEcE15_S_always_equalEv
  xor eax, 1
  test al, al
  je .L200
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_is_localEv
  xor eax, 1
  test al, al
  je .L200
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv
  mov rsi, rbx
  mov rdi, rax
  call _ZStneIcEbRKSaIT_ES3_
  test al, al
  je .L200
  mov eax, 1
  jmp .L201
.L200:
  mov eax, 0
.L201:
  test al, al
  je .L202
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv
  cmp rax, 15
  setbe al
  test al, al
  je .L203
  mov rax, QWORD PTR [rbp-56]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc
  mov rax, QWORD PTR [rbp-56]
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm
  jmp .L202
.L203:
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv
  mov rdx, rax
  lea rax, [rbp-33]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSaIcEC1ERKS_
  mov rax, QWORD PTR [rbp-24]
  lea rdx, [rax+1]
  lea rax, [rbp-33]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaIcEE8allocateERS0_m
  mov QWORD PTR [rbp-32], rax
  mov rax, QWORD PTR [rbp-56]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm
  mov rdx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc
  mov rdx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm
  mov rdx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm
  lea rax, [rbp-33]
  mov rdi, rax
  call _ZNSaIcED1Ev
.L202:
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv
  mov rsi, rbx
  mov rdi, rax
  call _ZSt15__alloc_on_copyISaIcEEvRT_RKS1_
.L199:
  mov rdx, QWORD PTR [rbp-64]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignERKS4_
  jmp .L208
  mov rbx, rax
  lea rax, [rbp-33]
  mov rdi, rax
  call _ZNSaIcED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L208:
  add rsp, 56
  pop rbx
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEpLEc:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov eax, esi
  mov BYTE PTR [rbp-12], al
  movsx edx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, edx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5clearEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm
  nop
  leave
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2ERKS4_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  mov rbx, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv
  mov rdx, rax
  lea rax, [rbp-17]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx14__alloc_traitsISaIcEcE17_S_select_on_copyERKS1_
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv
  mov rcx, rax
  lea rax, [rbp-17]
  mov rdx, rax
  mov rsi, rcx
  mov rdi, rbx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC1EPcOS3_
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaIcED1Ev
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv
  add rbx, rax
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rcx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdx, rbx
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_
  jmp .L215
  mov rbx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L215:
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5emptyEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv
  test rax, rax
  sete al
  leave
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSEOS4_:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 32
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_is_localEv
  xor eax, 1
  test al, al
  je .L219
  call _ZN9__gnu_cxx14__alloc_traitsISaIcEcE27_S_propagate_on_move_assignEv
  test al, al
  je .L219
  call _ZN9__gnu_cxx14__alloc_traitsISaIcEcE15_S_always_equalEv
  xor eax, 1
  test al, al
  je .L219
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv
  mov rsi, rbx
  mov rdi, rax
  call _ZStneIcEbRKSaIT_ES3_
  test al, al
  je .L219
  mov eax, 1
  jmp .L220
.L219:
  mov eax, 0
.L220:
  test al, al
  je .L221
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc
  mov rax, QWORD PTR [rbp-40]
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm
.L221:
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv
  mov rsi, rbx
  mov rdi, rax
  call _ZSt15__alloc_on_moveISaIcEEvRT_S2_
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_is_localEv
  test al, al
  je .L222
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv
  test rax, rax
  setne al
  test al, al
  je .L223
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rdx, r12
  mov rsi, rbx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_copyEPcPKcm
.L223:
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm
  jmp .L224
.L222:
  call _ZN9__gnu_cxx14__alloc_traitsISaIcEcE27_S_propagate_on_move_assignEv
  test al, al
  jne .L225
  call _ZN9__gnu_cxx14__alloc_traitsISaIcEcE15_S_always_equalEv
  test al, al
  jne .L225
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv
  mov rsi, rbx
  mov rdi, rax
  call _ZSteqIcEbRKSaIT_ES3_
  test al, al
  je .L226
.L225:
  mov eax, 1
  jmp .L227
.L226:
  mov eax, 0
.L227:
  test al, al
  je .L228
  mov QWORD PTR [rbp-24], 0
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_is_localEv
  xor eax, 1
  test al, al
  je .L229
  call _ZN9__gnu_cxx14__alloc_traitsISaIcEcE15_S_always_equalEv
  test al, al
  je .L230
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+16]
  mov QWORD PTR [rbp-32], rax
  jmp .L229
.L230:
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm
.L229:
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm
  mov rax, QWORD PTR [rbp-48]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm
  cmp QWORD PTR [rbp-24], 0
  je .L232
  mov rdx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-48]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc
  mov rdx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-48]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm
  jmp .L224
.L232:
  mov rax, QWORD PTR [rbp-48]
  lea rdx, [rax+16]
  mov rax, QWORD PTR [rbp-48]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc
  jmp .L224
.L228:
  mov rdx, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignERKS4_
.L224:
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5clearEv
  mov rax, QWORD PTR [rbp-40]
  add rsp, 32
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEEC2EPKcNSt15regex_constants18syntax_option_typeE:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov DWORD PTR [rbp-20], edx
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZNSt11char_traitsIcE6lengthEPKc
  mov rdx, rax
  mov rax, QWORD PTR [rbp-16]
  lea rdi, [rdx+rax]
  mov edx, DWORD PTR [rbp-20]
  mov rsi, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov ecx, edx
  mov rdx, rdi
  mov rdi, rax
  call _ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEEC1IPKcEET_S7_NSt15regex_constants18syntax_option_typeE
  nop
  leave
  ret
_ZNSt12__shared_ptrIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 8
  mov rdi, rax
  call _ZNSt14__shared_countILN9__gnu_cxx12_Lock_policyE2EED1Ev
  nop
  leave
  ret
_ZNSt10shared_ptrIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12__shared_ptrIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EED2Ev
  nop
  leave
  ret
_ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 16
  mov rdi, rax
  call _ZNSt10shared_ptrIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEED1Ev
  mov rax, QWORD PTR [rbp-8]
  add rax, 8
  mov rdi, rax
  call _ZNSt6localeD1Ev
  nop
  leave
  ret
_ZSt11regex_matchIcNSt7__cxx1112regex_traitsIcEEEbPKT_RKNS0_11basic_regexIS3_T0_EENSt15regex_constants15match_flag_typeE:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov DWORD PTR [rbp-20], edx
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt7__cxx1112regex_traitsIcE6lengthEPKc
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  lea rsi, [rdx+rax]
  mov ecx, DWORD PTR [rbp-20]
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZSt11regex_matchIPKccNSt7__cxx1112regex_traitsIcEEEbT_S5_RKNS2_11basic_regexIT0_T1_EENSt15regex_constants15match_flag_typeE
  leave
  ret
_ZNKSt7bit_andIbEclERKbS2_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-16]
  movzx eax, BYTE PTR [rax]
  movzx edx, al
  mov rax, QWORD PTR [rbp-24]
  movzx eax, BYTE PTR [rax]
  movzx eax, al
  and eax, edx
  test eax, eax
  setne al
  pop rbp
  ret
_ZSt10accumulateIPbiSt7bit_andIbEET0_T_S4_S3_T1_:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov DWORD PTR [rbp-36], edx
.L245:
  mov rax, QWORD PTR [rbp-24]
  cmp rax, QWORD PTR [rbp-32]
  je .L244
  cmp DWORD PTR [rbp-36], 0
  setne al
  mov BYTE PTR [rbp-1], al
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-1]
  mov rsi, rax
  lea rdi, [rbp+16]
  call _ZNKSt7bit_andIbEclERKbS2_
  movzx eax, al
  mov DWORD PTR [rbp-36], eax
  add QWORD PTR [rbp-24], 1
  jmp .L245
.L244:
  mov eax, DWORD PTR [rbp-36]
  leave
  ret
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IPcvEET_S7_RKS3_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov QWORD PTR [rbp-48], rcx
  mov rbx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv
  mov rcx, rax
  mov rax, QWORD PTR [rbp-48]
  mov rdx, rax
  mov rsi, rcx
  mov rdi, rbx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC1EPcRKS3_
  mov rdx, QWORD PTR [rbp-40]
  mov rcx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_
  jmp .L252
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderD1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L252:
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_is_localEv
  xor eax, 1
  test al, al
  je .L255
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm
.L255:
  nop
  leave
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 16
  mov rdi, rax
  call _ZNSt14pointer_traitsIPcE10pointer_toERc
  leave
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC2EPcRKS3_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rdx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSaIcEC2ERKS_
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov QWORD PTR [rax], rdx
  nop
  leave
  ret
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+8]
  pop rbp
  ret
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8capacityEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_is_localEv
  test al, al
  je .L262
  mov eax, 15
  jmp .L264
.L262:
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+16]
.L264:
  leave
  ret
_ZNSt12_Vector_baseImSaImEE12_Vector_implD2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSaImED2Ev
  nop
  leave
  ret
_ZNSt12_Vector_baseImSaImEEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseImSaImEE12_Vector_implC1Ev
  nop
  leave
  ret
_ZNSt12_Vector_baseImSaImEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  sub rdx, rax
  mov rax, rdx
  sar rax, 3
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt12_Vector_baseImSaImEE13_M_deallocateEPmm
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseImSaImEE12_Vector_implD1Ev
  nop
  leave
  ret
_ZNSt12_Vector_baseImSaImEE19_M_get_Tp_allocatorEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt8_DestroyIPmmEvT_S1_RSaIT0_E:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt8_DestroyIPmEvT_S1_
  nop
  leave
  ret
_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_releaseEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 8
  mov esi, -1
  mov rdi, rax
  call _ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii
  cmp eax, 1
  sete al
  test al, al
  je .L273
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  add rax, 16
  mov rax, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-8]
  mov rdi, rdx
  call rax
  mov rax, QWORD PTR [rbp-8]
  add rax, 12
  mov esi, -1
  mov rdi, rax
  call _ZN9__gnu_cxxL27__exchange_and_add_dispatchEPii
  cmp eax, 1
  sete al
  test al, al
  je .L273
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  add rax, 24
  mov rax, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-8]
  mov rdi, rdx
  call rax
.L273:
  nop
  leave
  ret
_ZNSt14__shared_countILN9__gnu_cxx12_Lock_policyE2EE7_M_swapERS2_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-32]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  nop
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC2EPcOS3_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt4moveIRSaIcEEONSt16remove_referenceIT_E4typeEOS3_
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSaIcEC2ERKS_
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov QWORD PTR [rax], rdx
  nop
  leave
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rdx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm
  mov BYTE PTR [rbp-1], 0
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-32]
  add rdx, rax
  lea rax, [rbp-1]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt11char_traitsIcE6assignERcRKc
  nop
  leave
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt4moveIRSaIcEEONSt16remove_referenceIT_E4typeEOS3_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_is_localEv:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv
  cmp rbx, rax
  sete al
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov QWORD PTR [rax], rdx
  nop
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov QWORD PTR [rax+16], rdx
  nop
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov QWORD PTR [rax+8], rdx
  nop
  pop rbp
  ret
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZStneIcEbRKSaIT_ES3_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov eax, 0
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 16
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-32]
  lea rbx, [rax+1]
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv
  mov rdx, rbx
  mov rsi, r12
  mov rdi, rax
  call _ZNSt16allocator_traitsISaIcEE10deallocateERS0_Pcm
  nop
  add rsp, 16
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt16allocator_traitsISaIcEE8allocateERS0_m:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov edx, 0
  mov rsi, rcx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorIcE8allocateEmPKv
  leave
  ret
_ZSt15__alloc_on_copyISaIcEEvRT_RKS1_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rdx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt18__do_alloc_on_copyISaIcEEvRT_RKS1_St17integral_constantIbLb0EE
  nop
  leave
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignERKS4_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9push_backEc:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-40], rdi
  mov eax, esi
  mov BYTE PTR [rbp-44], al
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-24]
  lea rbx, [rax+1]
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8capacityEv
  cmp rbx, rax
  seta al
  test al, al
  je .L297
  mov rsi, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-40]
  mov r8d, 1
  mov ecx, 0
  mov edx, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
.L297:
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-24]
  add rdx, rax
  lea rax, [rbp-44]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt11char_traitsIcE6assignERcRKc
  mov rax, QWORD PTR [rbp-24]
  lea rdx, [rax+1]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm
  nop
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZN9__gnu_cxx14__alloc_traitsISaIcEcE17_S_select_on_copyERKS1_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaIcEE37select_on_container_copy_constructionERKS0_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_construct_auxIPcEEvT_S7_St12__false_type
  nop
  leave
  ret
_ZSt15__alloc_on_moveISaIcEEvRT_S2_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rdx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt18__do_alloc_on_moveISaIcEEvRT_S2_St17integral_constantIbLb1EE
  nop
  leave
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_copyEPcPKcm:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  cmp QWORD PTR [rbp-24], 1
  jne .L303
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt11char_traitsIcE6assignERcRKc
  jmp .L305
.L303:
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt11char_traitsIcE4copyEPcPKcm
.L305:
  nop
  leave
  ret
_ZSteqIcEbRKSaIT_ES3_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov eax, 1
  pop rbp
  ret
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE15_M_check_lengthEmmPKc:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov QWORD PTR [rbp-48], rcx
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8max_sizeEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-32]
  sub rax, rdx
  add rax, rbx
  cmp QWORD PTR [rbp-40], rax
  seta al
  test al, al
  je .L310
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZSt20__throw_length_errorPKc
.L310:
  nop
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov eax, edx
  mov BYTE PTR [rbp-36], al
  movsx ebx, BYTE PTR [rbp-36]
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv
  mov rsi, rax
  mov rdx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov r8d, ebx
  mov rcx, rdx
  mov rdx, rsi
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEixEm:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-16]
  add rax, rdx
  leave
  ret
.LC7:
  .string "%s: __pos (which is %zu) > this->size() (which is %zu)"
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8_M_checkEmPKc:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv
  cmp QWORD PTR [rbp-16], rax
  seta al
  test al, al
  je .L316
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv
  mov rcx, rax
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rax
  mov edi, OFFSET FLAT:.LC7
  mov eax, 0
  call _ZSt24__throw_out_of_range_fmtPKcz
.L316:
  mov rax, QWORD PTR [rbp-16]
  leave
  ret
_ZNSt14__shared_countILN9__gnu_cxx12_Lock_policyE2EEC2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], 0
  nop
  pop rbp
  ret
_ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEEC2IPKcEET_S7_NSt15regex_constants18syntax_option_typeE:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 56
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  mov QWORD PTR [rbp-56], rdx
  mov DWORD PTR [rbp-60], ecx
  lea rax, [rbp-24]
  mov rdi, rax
  call _ZNSt6localeC1Ev
  lea rax, [rbp-56]
  mov rdi, rax
  call _ZSt4moveIRPKcEONSt16remove_referenceIT_E4typeEOS4_
  mov rbx, QWORD PTR [rax]
  lea rax, [rbp-48]
  mov rdi, rax
  call _ZSt4moveIRPKcEONSt16remove_referenceIT_E4typeEOS4_
  mov rsi, QWORD PTR [rax]
  mov ecx, DWORD PTR [rbp-60]
  lea rdx, [rbp-24]
  mov rax, QWORD PTR [rbp-40]
  mov r8d, ecx
  mov rcx, rdx
  mov rdx, rbx
  mov rdi, rax
  call _ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEEC1IPKcEET_S7_St6localeNSt15regex_constants18syntax_option_typeE
  lea rax, [rbp-24]
  mov rdi, rax
  call _ZNSt6localeD1Ev
  jmp .L322
  mov rbx, rax
  lea rax, [rbp-24]
  mov rdi, rax
  call _ZNSt6localeD1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L322:
  add rsp, 56
  pop rbx
  pop rbp
  ret
_ZNSt7__cxx1112regex_traitsIcE6lengthEPKc:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt11char_traitsIcE6lengthEPKc
  leave
  ret
_ZSt11regex_matchIPKccNSt7__cxx1112regex_traitsIcEEEbT_S5_RKNS2_11basic_regexIT0_T1_EENSt15regex_constants15match_flag_typeE:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 88
  mov QWORD PTR [rbp-72], rdi
  mov QWORD PTR [rbp-80], rsi
  mov QWORD PTR [rbp-88], rdx
  mov DWORD PTR [rbp-92], ecx
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaINSt7__cxx119sub_matchIPKcEEEC1Ev
  lea rdx, [rbp-17]
  lea rax, [rbp-64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1113match_resultsIPKcSaINS_9sub_matchIS2_EEEEC1ERKS5_
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaINSt7__cxx119sub_matchIPKcEEED1Ev
  mov edi, DWORD PTR [rbp-92]
  mov rcx, QWORD PTR [rbp-88]
  lea rdx, [rbp-64]
  mov rsi, QWORD PTR [rbp-80]
  mov rax, QWORD PTR [rbp-72]
  mov r8d, edi
  mov rdi, rax
  call _ZSt11regex_matchIPKcSaINSt7__cxx119sub_matchIS1_EEEcNS2_12regex_traitsIcEEEbT_S8_RNS2_13match_resultsIS8_T0_EERKNS2_11basic_regexIT1_T2_EENSt15regex_constants15match_flag_typeE
  mov ebx, eax
  nop
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt7__cxx1113match_resultsIPKcSaINS_9sub_matchIS2_EEEED1Ev
  mov eax, ebx
  jmp .L329
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt7__cxx1113match_resultsIPKcSaINS_9sub_matchIS2_EEEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L329:
  add rsp, 88
  pop rbx
  pop rbp
  ret
_ZNSt14pointer_traitsIPcE10pointer_toERc:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZSt9addressofIcEPT_RS0_
  leave
  ret
_ZNSt12_Vector_baseImSaImEE12_Vector_implC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSaImEC2Ev
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+8], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+16], 0
  nop
  leave
  ret
_ZNSaImED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorImED2Ev
  nop
  leave
  ret
_ZNSt12_Vector_baseImSaImEE13_M_deallocateEPmm:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  cmp QWORD PTR [rbp-16], 0
  je .L336
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaImEE10deallocateERS0_Pmm
.L336:
  nop
  leave
  ret
_ZSt8_DestroyIPmEvT_S1_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Destroy_auxILb1EE9__destroyIPmEEvT_S3_
  nop
  leave
  ret
_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE10_M_destroyEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  cmp QWORD PTR [rbp-8], 0
  je .L340
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  add rax, 8
  mov rax, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-8]
  mov rdi, rdx
  call rax
.L340:
  nop
  leave
  ret
_ZNSt9_Any_data9_M_accessIPKSt9type_infoEERT_v:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt9_Any_data9_M_accessEv
  leave
  ret
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 16
  mov rdi, rax
  call _ZNSt14pointer_traitsIPKcE10pointer_toERS0_
  leave
  ret
_ZNSt16allocator_traitsISaIcEE10deallocateERS0_Pcm:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcm
  nop
  leave
  ret
_ZN9__gnu_cxx13new_allocatorIcE8allocateEmPKv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNK9__gnu_cxx13new_allocatorIcE8max_sizeEv
  cmp QWORD PTR [rbp-16], rax
  seta al
  test al, al
  je .L349
  call _ZSt17__throw_bad_allocv
.L349:
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _Znwm
  nop
  leave
  ret
_ZSt18__do_alloc_on_copyISaIcEEvRT_RKS1_St17integral_constantIbLb0EE:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  nop
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 56
  mov QWORD PTR [rbp-56], rdi
  mov QWORD PTR [rbp-64], rsi
  mov rax, QWORD PTR [rbp-56]
  cmp rax, QWORD PTR [rbp-64]
  je .L355
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8capacityEv
  mov QWORD PTR [rbp-32], rax
  mov rax, QWORD PTR [rbp-24]
  cmp rax, QWORD PTR [rbp-32]
  jbe .L353
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rbp-48], rax
  mov rdx, QWORD PTR [rbp-32]
  lea rcx, [rbp-48]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm
  mov QWORD PTR [rbp-40], rax
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
  mov rdx, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc
  mov rdx, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm
.L353:
  cmp QWORD PTR [rbp-24], 0
  je .L354
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rcx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdx, rax
  mov rsi, rbx
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_copyEPcPKcm
.L354:
  mov rdx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm
.L355:
  nop
  add rsp, 56
  pop rbx
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm:
  push rbp
  mov rbp, rsp
  sub rsp, 80
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  mov QWORD PTR [rbp-56], rdx
  mov QWORD PTR [rbp-64], rcx
  mov QWORD PTR [rbp-72], r8
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv
  sub rax, QWORD PTR [rbp-48]
  sub rax, QWORD PTR [rbp-56]
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-72]
  add rax, rdx
  sub rax, QWORD PTR [rbp-56]
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8capacityEv
  mov rdx, rax
  lea rcx, [rbp-24]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm
  mov QWORD PTR [rbp-16], rax
  cmp QWORD PTR [rbp-48], 0
  je .L357
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rcx, rax
  mov rdx, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rbp-16]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_copyEPcPKcm
.L357:
  cmp QWORD PTR [rbp-64], 0
  je .L358
  cmp QWORD PTR [rbp-72], 0
  je .L358
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-48]
  lea rcx, [rdx+rax]
  mov rdx, QWORD PTR [rbp-72]
  mov rax, QWORD PTR [rbp-64]
  mov rsi, rax
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_copyEPcPKcm
.L358:
  cmp QWORD PTR [rbp-8], 0
  je .L359
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rcx, rax
  mov rdx, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rbp-56]
  add rax, rdx
  lea rsi, [rcx+rax]
  mov rdx, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rbp-72]
  add rdx, rax
  mov rax, QWORD PTR [rbp-16]
  lea rcx, [rdx+rax]
  mov rax, QWORD PTR [rbp-8]
  mov rdx, rax
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_copyEPcPKcm
.L359:
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc
  mov rdx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm
  nop
  leave
  ret
_ZNSt16allocator_traitsISaIcEE37select_on_container_copy_constructionERKS0_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSaIcEC1ERKS_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_construct_auxIPcEEvT_S7_St12__false_type:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov rdx, QWORD PTR [rbp-40]
  mov rcx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag
  nop
  leave
  ret
_ZSt18__do_alloc_on_moveISaIcEEvRT_S2_St17integral_constantIbLb1EE:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZSt4moveIRSaIcEEONSt16remove_referenceIT_E4typeEOS3_
  nop
  leave
  ret
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8max_sizeEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv
  mov rdi, rax
  call _ZNSt16allocator_traitsISaIcEE8max_sizeERKS0_
  sub rax, 1
  shr rax
  leave
  ret
.LC8:
  .string "basic_string::_M_replace_aux"
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE14_M_replace_auxEmmmc:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 88
  mov QWORD PTR [rbp-56], rdi
  mov QWORD PTR [rbp-64], rsi
  mov QWORD PTR [rbp-72], rdx
  mov QWORD PTR [rbp-80], rcx
  mov eax, r8d
  mov BYTE PTR [rbp-84], al
  mov rdx, QWORD PTR [rbp-80]
  mov rsi, QWORD PTR [rbp-72]
  mov rax, QWORD PTR [rbp-56]
  mov ecx, OFFSET FLAT:.LC8
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE15_M_check_lengthEmmPKc
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv
  mov QWORD PTR [rbp-24], rax
  mov rdx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-80]
  add rax, rdx
  sub rax, QWORD PTR [rbp-72]
  mov QWORD PTR [rbp-32], rax
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8capacityEv
  cmp QWORD PTR [rbp-32], rax
  setbe al
  test al, al
  je .L367
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-64]
  add rax, rdx
  mov QWORD PTR [rbp-40], rax
  mov rax, QWORD PTR [rbp-24]
  sub rax, QWORD PTR [rbp-64]
  sub rax, QWORD PTR [rbp-72]
  mov QWORD PTR [rbp-48], rax
  cmp QWORD PTR [rbp-48], 0
  je .L369
  mov rax, QWORD PTR [rbp-72]
  cmp rax, QWORD PTR [rbp-80]
  je .L369
  mov rdx, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rbp-72]
  lea rsi, [rdx+rax]
  mov rdx, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rbp-80]
  lea rcx, [rdx+rax]
  mov rax, QWORD PTR [rbp-48]
  mov rdx, rax
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_moveEPcPKcm
  jmp .L369
.L367:
  mov rcx, QWORD PTR [rbp-80]
  mov rdx, QWORD PTR [rbp-72]
  mov rsi, QWORD PTR [rbp-64]
  mov rax, QWORD PTR [rbp-56]
  mov r8, rcx
  mov ecx, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_mutateEmmPKcm
.L369:
  cmp QWORD PTR [rbp-80], 0
  je .L370
  movsx ebx, BYTE PTR [rbp-84]
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-64]
  lea rcx, [rdx+rax]
  mov rax, QWORD PTR [rbp-80]
  mov edx, ebx
  mov rsi, rax
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_S_assignEPcmc
.L370:
  mov rdx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm
  mov rax, QWORD PTR [rbp-56]
  add rsp, 88
  pop rbx
  pop rbp
  ret
.LC9:
  .string "basic_string::_M_create"
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov rax, QWORD PTR [rbp-32]
  mov rbx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8max_sizeEv
  cmp rbx, rax
  seta al
  test al, al
  je .L373
  mov edi, OFFSET FLAT:.LC9
  call _ZSt20__throw_length_errorPKc
.L373:
  mov rax, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rax]
  cmp QWORD PTR [rbp-40], rax
  jnb .L374
  mov rax, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-40]
  add rdx, rdx
  cmp rax, rdx
  jnb .L374
  mov rax, QWORD PTR [rbp-40]
  lea rdx, [rax+rax]
  mov rax, QWORD PTR [rbp-32]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-32]
  mov rbx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8max_sizeEv
  cmp rbx, rax
  seta al
  test al, al
  je .L374
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8max_sizeEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-32]
  mov QWORD PTR [rax], rdx
.L374:
  mov rax, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rax]
  lea rbx, [rax+1]
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv
  mov rsi, rbx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaIcEE8allocateERS0_m
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8_M_limitEmm:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv
  sub rax, QWORD PTR [rbp-32]
  cmp QWORD PTR [rbp-40], rax
  setb al
  mov BYTE PTR [rbp-1], al
  cmp BYTE PTR [rbp-1], 0
  je .L377
  mov rax, QWORD PTR [rbp-40]
  jmp .L379
.L377:
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv
  sub rax, QWORD PTR [rbp-32]
.L379:
  leave
  ret
_ZSt4moveIRPKcEONSt16remove_referenceIT_E4typeEOS4_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEEC2IPKcEET_S7_St6localeNSt15regex_constants18syntax_option_typeE:
  push rbp
  mov rbp, rsp
  push r13
  push r12
  push rbx
  sub rsp, 56
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  mov QWORD PTR [rbp-56], rdx
  mov QWORD PTR [rbp-64], rcx
  mov DWORD PTR [rbp-68], r8d
  mov rax, QWORD PTR [rbp-40]
  mov edx, DWORD PTR [rbp-68]
  mov DWORD PTR [rax], edx
  mov rax, QWORD PTR [rbp-40]
  lea rbx, [rax+8]
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZSt4moveIRSt6localeEONSt16remove_referenceIT_E4typeEOS3_
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt6localeC1ERKS_
  mov rax, QWORD PTR [rbp-40]
  mov r13d, DWORD PTR [rax]
  mov rax, QWORD PTR [rbp-40]
  lea r12, [rax+8]
  lea rax, [rbp-56]
  mov rdi, rax
  call _ZSt4moveIRPKcEONSt16remove_referenceIT_E4typeEOS4_
  mov rbx, QWORD PTR [rax]
  lea rax, [rbp-48]
  mov rdi, rax
  call _ZSt4moveIRPKcEONSt16remove_referenceIT_E4typeEOS4_
  mov rax, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-40]
  lea rdi, [rdx+16]
  mov r8d, r13d
  mov rcx, r12
  mov rdx, rbx
  mov rsi, rax
  call _ZNSt8__detail13__compile_nfaINSt7__cxx1112regex_traitsIcEEPKcEENSt9enable_ifIXsrNS_27__is_contiguous_normal_iterIT0_EE5valueESt10shared_ptrIKNS_4_NFAIT_EEEE4typeES8_S8_RKNSC_11locale_typeENSt15regex_constants18syntax_option_typeE
  jmp .L385
  mov rbx, rax
  mov rax, QWORD PTR [rbp-40]
  add rax, 8
  mov rdi, rax
  call _ZNSt6localeD1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L385:
  add rsp, 56
  pop rbx
  pop r12
  pop r13
  pop rbp
  ret
_ZNSaINSt7__cxx119sub_matchIPKcEEEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorINSt7__cxx119sub_matchIPKcEEEC2Ev
  nop
  leave
  ret
_ZNSaINSt7__cxx119sub_matchIPKcEEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorINSt7__cxx119sub_matchIPKcEEED2Ev
  nop
  leave
  ret
_ZNSt7__cxx1113match_resultsIPKcSaINS_9sub_matchIS2_EEEEC2ERKS5_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEC2ERKS5_
  nop
  leave
  ret
_ZNSt7__cxx1113match_resultsIPKcSaINS_9sub_matchIS2_EEEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EED2Ev
  nop
  leave
  ret
_ZSt11regex_matchIPKcSaINSt7__cxx119sub_matchIS1_EEEcNS2_12regex_traitsIcEEEbT_S8_RNS2_13match_resultsIS8_T0_EERKNS2_11basic_regexIT1_T2_EENSt15regex_constants15match_flag_typeE:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov QWORD PTR [rbp-32], rcx
  mov DWORD PTR [rbp-36], r8d
  mov edi, DWORD PTR [rbp-36]
  mov rcx, QWORD PTR [rbp-32]
  mov rdx, QWORD PTR [rbp-24]
  mov rsi, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov r8d, edi
  mov rdi, rax
  call _ZNSt8__detail17__regex_algo_implIPKcSaINSt7__cxx119sub_matchIS2_EEEcNS3_12regex_traitsIcEELNS_20_RegexExecutorPolicyE0ELb1EEEbT_SA_RNS3_13match_resultsISA_T0_EERKNS3_11basic_regexIT1_T2_EENSt15regex_constants15match_flag_typeE
  leave
  ret
_ZSt9addressofIcEPT_RS0_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZSt11__addressofIcEPT_RS0_
  leave
  ret
_ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  lea rax, [rbp-8]
  mov rdi, rax
  call _ZSt19__iterator_categoryIPKcENSt15iterator_traitsIT_E17iterator_categoryERKS3_
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt10__distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_St26random_access_iterator_tag
  leave
  ret
_ZNSaImEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorImEC2Ev
  nop
  leave
  ret
_ZN9__gnu_cxx13new_allocatorImED2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  nop
  pop rbp
  ret
_ZNSt16allocator_traitsISaImEE10deallocateERS0_Pmm:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorImE10deallocateEPmm
  nop
  leave
  ret
_ZNSt12_Destroy_auxILb1EE9__destroyIPmEEvT_S3_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  nop
  pop rbp
  ret
_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EED2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov edx, OFFSET FLAT:_ZTVSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE+16
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  nop
  pop rbp
  ret
_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EED0Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EED1Ev
  mov rax, QWORD PTR [rbp-8]
  mov esi, 16
  mov rdi, rax
  call _ZdlPvm
  leave
  ret
_ZNSt14pointer_traitsIPKcE10pointer_toERS0_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZSt9addressofIKcEPT_RS1_
  leave
  ret
_ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcm:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZdlPv
  leave
  ret
_ZNK9__gnu_cxx13new_allocatorIcE8max_sizeEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, -1
  pop rbp
  ret
_ZSt8distanceIPcENSt15iterator_traitsIT_E15difference_typeES2_S2_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  lea rax, [rbp-8]
  mov rdi, rax
  call _ZSt19__iterator_categoryIPcENSt15iterator_traitsIT_E17iterator_categoryERKS2_
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt10__distanceIPcENSt15iterator_traitsIT_E15difference_typeES2_S2_St26random_access_iterator_tag
  leave
  ret
.LC10:
  .string "basic_string::_M_construct null not valid"
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPcEEvT_S7_St20forward_iterator_tag:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZN9__gnu_cxx17__is_null_pointerIcEEbPT_
  test al, al
  je .L411
  mov rax, QWORD PTR [rbp-32]
  cmp rax, QWORD PTR [rbp-40]
  je .L411
  mov eax, 1
  jmp .L412
.L411:
  mov eax, 0
.L412:
  test al, al
  je .L413
  mov edi, OFFSET FLAT:.LC10
  call _ZSt19__throw_logic_errorPKc
.L413:
  mov rdx, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rbp-32]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt8distanceIPcENSt15iterator_traitsIT_E15difference_typeES2_S2_
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-8]
  cmp rax, 15
  jbe .L414
  lea rcx, [rbp-8]
  mov rax, QWORD PTR [rbp-24]
  mov edx, 0
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm
  mov rdx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc
  mov rdx, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm
.L414:
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rcx, rax
  mov rdx, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rbp-32]
  mov rsi, rax
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_S_copy_charsEPcS5_S5_
  mov rdx, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm
  nop
  leave
  ret
_ZNSt16allocator_traitsISaIcEE8max_sizeERKS0_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNK9__gnu_cxx13new_allocatorIcE8max_sizeEv
  leave
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_moveEPcPKcm:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  cmp QWORD PTR [rbp-24], 1
  jne .L418
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt11char_traitsIcE6assignERcRKc
  jmp .L420
.L418:
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt11char_traitsIcE4moveEPcPKcm
.L420:
  nop
  leave
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_S_assignEPcmc:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov eax, edx
  mov BYTE PTR [rbp-20], al
  cmp QWORD PTR [rbp-16], 1
  jne .L422
  lea rdx, [rbp-20]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt11char_traitsIcE6assignERcRKc
  jmp .L424
.L422:
  movzx eax, BYTE PTR [rbp-20]
  movsx edx, al
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt11char_traitsIcE6assignEPcmc
.L424:
  nop
  leave
  ret
_ZSt4moveIRSt6localeEONSt16remove_referenceIT_E4typeEOS3_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt8__detail8_ScannerIcED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 200
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev
  nop
  leave
  ret
_ZNSt12__shared_ptrINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 8
  mov rdi, rax
  call _ZNSt14__shared_countILN9__gnu_cxx12_Lock_policyE2EED1Ev
  nop
  leave
  ret
_ZNSt10shared_ptrINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12__shared_ptrINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EED2Ev
  nop
  leave
  ret
_ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EED1Ev
  nop
  leave
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 304
  mov rdi, rax
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEED1Ev
  mov rax, QWORD PTR [rbp-8]
  add rax, 272
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev
  mov rax, QWORD PTR [rbp-8]
  add rax, 256
  mov rdi, rax
  call _ZNSt10shared_ptrINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEED1Ev
  mov rax, QWORD PTR [rbp-8]
  add rax, 8
  mov rdi, rax
  call _ZNSt8__detail8_ScannerIcED1Ev
  nop
  leave
  ret
_ZNSt8__detail13__compile_nfaINSt7__cxx1112regex_traitsIcEEPKcEENSt9enable_ifIXsrNS_27__is_contiguous_normal_iterIT0_EE5valueESt10shared_ptrIKNS_4_NFAIT_EEEE4typeES8_S8_RKNSC_11locale_typeENSt15regex_constants18syntax_option_typeE:
  push rbp
  mov rbp, rsp
  sub rsp, 464
  mov QWORD PTR [rbp-424], rdi
  mov QWORD PTR [rbp-432], rsi
  mov QWORD PTR [rbp-440], rdx
  mov QWORD PTR [rbp-448], rcx
  mov DWORD PTR [rbp-452], r8d
  mov rax, QWORD PTR [rbp-440]
  sub rax, QWORD PTR [rbp-432]
  mov QWORD PTR [rbp-8], rax
  cmp QWORD PTR [rbp-8], 0
  je .L433
  mov rax, QWORD PTR [rbp-432]
  mov rdi, rax
  call _ZSt11__addressofIKcEPT_RS1_
  jmp .L434
.L433:
  mov eax, 0
.L434:
  mov QWORD PTR [rbp-16], rax
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  lea rdi, [rdx+rax]
  mov ecx, DWORD PTR [rbp-452]
  mov rdx, QWORD PTR [rbp-448]
  mov rsi, QWORD PTR [rbp-16]
  lea rax, [rbp-416]
  mov r8d, ecx
  mov rcx, rdx
  mov rdx, rdi
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEEC1EPKcS6_RKSt6localeNSt15regex_constants18syntax_option_typeE
  mov rax, QWORD PTR [rbp-424]
  lea rdx, [rbp-416]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE10_M_get_nfaEv
  lea rax, [rbp-416]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEED1Ev
  mov rax, QWORD PTR [rbp-424]
  leave
  ret
_ZN9__gnu_cxx13new_allocatorINSt7__cxx119sub_matchIPKcEEEC2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  nop
  pop rbp
  ret
_ZN9__gnu_cxx13new_allocatorINSt7__cxx119sub_matchIPKcEEED2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  nop
  pop rbp
  ret
_ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEC2ERKS5_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EEC2ERKS5_
  nop
  leave
  ret
_ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt8_DestroyIPNSt7__cxx119sub_matchIPKcEES4_EvT_S6_RSaIT0_E
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EED2Ev
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE11_State_infoISt17integral_constantIbLb0EESt6vectorIS5_S6_EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 24
  mov rdi, rax
  call _ZNSt10unique_ptrIA_bSt14default_deleteIS0_EED1Ev
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EED1Ev
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 96
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE11_State_infoISt17integral_constantIbLb0EESt6vectorIS5_S6_EED1Ev
  mov rax, QWORD PTR [rbp-8]
  add rax, 72
  mov rdi, rax
  call _ZNSt6vectorISt4pairIPKciESaIS3_EED1Ev
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EED1Ev
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 72
  mov rdi, rax
  call _ZNSt6vectorISt4pairIPKciESaIS3_EED1Ev
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EED1Ev
  nop
  leave
  ret
_ZNSt8__detail17__regex_algo_implIPKcSaINSt7__cxx119sub_matchIS2_EEEcNS3_12regex_traitsIcEELNS_20_RegexExecutorPolicyE0ELb1EEEbT_SA_RNS3_13match_resultsISA_T0_EERKNS3_11basic_regexIT1_T2_EENSt15regex_constants15match_flag_typeE:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 328
  mov QWORD PTR [rbp-296], rdi
  mov QWORD PTR [rbp-304], rsi
  mov QWORD PTR [rbp-312], rdx
  mov QWORD PTR [rbp-320], rcx
  mov DWORD PTR [rbp-324], r8d
  mov rax, QWORD PTR [rbp-320]
  add rax, 16
  mov esi, 0
  mov rdi, rax
  call _ZSteqIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEEbRKSt10shared_ptrIT_EDn
  test al, al
  je .L444
  mov eax, 0
  jmp .L463
.L444:
  mov rax, QWORD PTR [rbp-312]
  mov QWORD PTR [rbp-32], rax
  mov rax, QWORD PTR [rbp-312]
  mov rdx, QWORD PTR [rbp-296]
  mov QWORD PTR [rax+24], rdx
  mov rax, QWORD PTR [rbp-320]
  add rax, 16
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNKSt8__detail9_NFA_base12_M_sub_countEv
  mov edx, eax
  mov rax, QWORD PTR [rbp-312]
  mov esi, edx
  mov rdi, rax
  call _ZNSt7__cxx1113match_resultsIPKcSaINS_9sub_matchIS2_EEEE9_M_resizeEj
  mov rax, QWORD PTR [rbp-32]
  mov QWORD PTR [rbp-40], rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE5beginEv
  mov QWORD PTR [rbp-104], rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE3endEv
  mov QWORD PTR [rbp-112], rax
.L447:
  lea rdx, [rbp-112]
  lea rax, [rbp-104]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxxneIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEEbRKNS_17__normal_iteratorIT_T0_EESF_
  test al, al
  je .L446
  lea rax, [rbp-104]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEdeEv
  mov QWORD PTR [rbp-48], rax
  mov rax, QWORD PTR [rbp-48]
  mov BYTE PTR [rax+16], 0
  lea rax, [rbp-104]
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEppEv
  jmp .L447
.L446:
  mov rax, QWORD PTR [rbp-320]
  mov rdi, rax
  call _ZNKSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEE5flagsEv
  mov esi, 1024
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  je .L448
  mov eax, 1
  jmp .L449
.L448:
  mov eax, 0
.L449:
  test al, al
  je .L450
  mov rcx, QWORD PTR [rbp-312]
  mov r8d, DWORD PTR [rbp-324]
  mov rdi, QWORD PTR [rbp-320]
  mov rdx, QWORD PTR [rbp-304]
  mov rsi, QWORD PTR [rbp-296]
  lea rax, [rbp-288]
  mov r9d, r8d
  mov r8, rdi
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EEC1ES2_S2_RSt6vectorIS5_S6_ERKNS3_11basic_regexIcS8_EENSt15regex_constants15match_flag_typeE
  lea rax, [rbp-288]
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE8_M_matchEv
  mov BYTE PTR [rbp-17], al
  lea rax, [rbp-288]
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EED1Ev
  jmp .L451
.L450:
  mov rcx, QWORD PTR [rbp-312]
  mov r8d, DWORD PTR [rbp-324]
  mov rdi, QWORD PTR [rbp-320]
  mov rdx, QWORD PTR [rbp-304]
  mov rsi, QWORD PTR [rbp-296]
  lea rax, [rbp-288]
  mov r9d, r8d
  mov r8, rdi
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EEC1ES2_S2_RSt6vectorIS5_S6_ERKNS3_11basic_regexIcS8_EENSt15regex_constants15match_flag_typeE
  lea rax, [rbp-288]
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE8_M_matchEv
  mov BYTE PTR [rbp-17], al
  lea rax, [rbp-288]
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EED1Ev
.L451:
  cmp BYTE PTR [rbp-17], 0
  je .L452
  mov rax, QWORD PTR [rbp-32]
  mov QWORD PTR [rbp-56], rax
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE5beginEv
  mov QWORD PTR [rbp-120], rax
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE3endEv
  mov QWORD PTR [rbp-128], rax
.L455:
  lea rdx, [rbp-128]
  lea rax, [rbp-120]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxxneIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEEbRKNS_17__normal_iteratorIT_T0_EESF_
  test al, al
  je .L453
  lea rax, [rbp-120]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEdeEv
  mov QWORD PTR [rbp-64], rax
  mov rax, QWORD PTR [rbp-64]
  movzx eax, BYTE PTR [rax+16]
  xor eax, 1
  test al, al
  je .L454
  mov rax, QWORD PTR [rbp-64]
  mov rdx, QWORD PTR [rbp-304]
  mov QWORD PTR [rax+8], rdx
  mov rax, QWORD PTR [rbp-64]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-64]
  mov QWORD PTR [rax], rdx
.L454:
  lea rax, [rbp-120]
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEppEv
  jmp .L455
.L453:
  mov rax, QWORD PTR [rbp-312]
  mov rdi, rax
  call _ZNSt7__cxx1113match_resultsIPKcSaINS_9sub_matchIS2_EEEE9_M_prefixEv
  mov QWORD PTR [rbp-72], rax
  mov rax, QWORD PTR [rbp-312]
  mov rdi, rax
  call _ZNSt7__cxx1113match_resultsIPKcSaINS_9sub_matchIS2_EEEE9_M_suffixEv
  mov QWORD PTR [rbp-80], rax
  mov rax, QWORD PTR [rbp-72]
  mov BYTE PTR [rax+16], 0
  mov rax, QWORD PTR [rbp-72]
  mov rdx, QWORD PTR [rbp-296]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-72]
  mov rdx, QWORD PTR [rbp-296]
  mov QWORD PTR [rax+8], rdx
  mov rax, QWORD PTR [rbp-80]
  mov BYTE PTR [rax+16], 0
  mov rax, QWORD PTR [rbp-80]
  mov rdx, QWORD PTR [rbp-304]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-80]
  mov rdx, QWORD PTR [rbp-304]
  mov QWORD PTR [rax+8], rdx
  jmp .L456
.L452:
  mov rax, QWORD PTR [rbp-312]
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1113match_resultsIPKcSaINS_9sub_matchIS2_EEEE9_M_resizeEj
  mov rax, QWORD PTR [rbp-32]
  mov QWORD PTR [rbp-88], rax
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE5beginEv
  mov QWORD PTR [rbp-136], rax
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE3endEv
  mov QWORD PTR [rbp-144], rax
.L458:
  lea rdx, [rbp-144]
  lea rax, [rbp-136]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxxneIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEEbRKNS_17__normal_iteratorIT_T0_EESF_
  test al, al
  je .L456
  lea rax, [rbp-136]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEdeEv
  mov QWORD PTR [rbp-96], rax
  mov rax, QWORD PTR [rbp-96]
  mov BYTE PTR [rax+16], 0
  mov rax, QWORD PTR [rbp-96]
  mov rdx, QWORD PTR [rbp-304]
  mov QWORD PTR [rax+8], rdx
  mov rax, QWORD PTR [rbp-96]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-96]
  mov QWORD PTR [rax], rdx
  lea rax, [rbp-136]
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEppEv
  jmp .L458
.L456:
  movzx eax, BYTE PTR [rbp-17]
  jmp .L463
  mov rbx, rax
  lea rax, [rbp-288]
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
  mov rbx, rax
  lea rax, [rbp-288]
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L463:
  add rsp, 328
  pop rbx
  pop rbp
  ret
_ZSt11__addressofIcEPT_RS0_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt19__iterator_categoryIPKcENSt15iterator_traitsIT_E17iterator_categoryERKS3_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  pop rbp
  ret
_ZSt10__distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_St26random_access_iterator_tag:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  sub rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZN9__gnu_cxx13new_allocatorImEC2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  nop
  pop rbp
  ret
_ZN9__gnu_cxx13new_allocatorImE10deallocateEPmm:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZdlPv
  leave
  ret
_ZSt9addressofIKcEPT_RS1_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZSt11__addressofIKcEPT_RS1_
  leave
  ret
_ZN9__gnu_cxx17__is_null_pointerIcEEbPT_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  cmp QWORD PTR [rbp-8], 0
  sete al
  pop rbp
  ret
_ZSt19__iterator_categoryIPcENSt15iterator_traitsIT_E17iterator_categoryERKS2_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  pop rbp
  ret
_ZSt10__distanceIPcENSt15iterator_traitsIT_E15difference_typeES2_S2_St26random_access_iterator_tag:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  sub rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_S_copy_charsEPcS5_S5_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-24]
  sub rax, QWORD PTR [rbp-16]
  mov rdx, rax
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_copyEPcPKcm
  nop
  leave
  ret
_ZSt11__addressofIKcEPT_RS1_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEEC2EPKcS6_RKSt6localeNSt15regex_constants18syntax_option_typeE:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 120
  mov QWORD PTR [rbp-88], rdi
  mov QWORD PTR [rbp-96], rsi
  mov QWORD PTR [rbp-104], rdx
  mov QWORD PTR [rbp-112], rcx
  mov DWORD PTR [rbp-116], r8d
  mov esi, 32
  mov edi, 16
  call _ZNSt15regex_constantsorENS_18syntax_option_typeES0_
  mov esi, 64
  mov edi, eax
  call _ZNSt15regex_constantsorENS_18syntax_option_typeES0_
  mov esi, 256
  mov edi, eax
  call _ZNSt15regex_constantsorENS_18syntax_option_typeES0_
  mov esi, 512
  mov edi, eax
  call _ZNSt15regex_constantsorENS_18syntax_option_typeES0_
  mov esi, 128
  mov edi, eax
  call _ZNSt15regex_constantsorENS_18syntax_option_typeES0_
  mov edx, eax
  mov eax, DWORD PTR [rbp-116]
  mov esi, edx
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  jne .L485
  mov eax, DWORD PTR [rbp-116]
  mov esi, 16
  mov edi, eax
  call _ZNSt15regex_constantsorENS_18syntax_option_typeES0_
  jmp .L486
.L485:
  mov eax, DWORD PTR [rbp-116]
.L486:
  mov rdx, QWORD PTR [rbp-88]
  mov DWORD PTR [rdx], eax
  mov rax, QWORD PTR [rbp-88]
  lea rbx, [rax+8]
  mov rdx, QWORD PTR [rbp-112]
  lea rax, [rbp-56]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6localeC1ERKS_
  mov rax, QWORD PTR [rbp-88]
  mov ecx, DWORD PTR [rax]
  lea rsi, [rbp-56]
  mov rdx, QWORD PTR [rbp-104]
  mov rax, QWORD PTR [rbp-96]
  mov r8, rsi
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt8__detail8_ScannerIcEC1EPKcS3_NSt15regex_constants18syntax_option_typeESt6locale
  lea rax, [rbp-56]
  mov rdi, rax
  call _ZNSt6localeD1Ev
  mov rdx, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rbp-88]
  lea rcx, [rax+256]
  mov rax, QWORD PTR [rbp-112]
  mov rsi, rax
  mov rdi, rcx
  call _ZSt11make_sharedINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEJRKSt6localeRNSt15regex_constants18syntax_option_typeEEESt10shared_ptrIT_EDpOT0_
  mov rax, QWORD PTR [rbp-88]
  add rax, 272
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1Ev
  mov rax, QWORD PTR [rbp-88]
  add rax, 304
  mov rdi, rax
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEEC1IS8_vEEv
  mov rax, QWORD PTR [rbp-88]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  lea rdx, [rax+80]
  mov rax, QWORD PTR [rbp-88]
  mov QWORD PTR [rax+384], rdx
  mov rax, QWORD PTR [rbp-112]
  mov rdi, rax
  call _ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale
  mov rdx, rax
  mov rax, QWORD PTR [rbp-88]
  mov QWORD PTR [rax+392], rdx
  mov rax, QWORD PTR [rbp-88]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNKSt8__detail9_NFA_base8_M_startEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-88]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-80]
  mov rdx, rbx
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  mov rax, QWORD PTR [rbp-88]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE23_M_insert_subexpr_beginEv
  mov rdx, rax
  lea rax, [rbp-80]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendEl
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_disjunctionEv
  mov rax, QWORD PTR [rbp-88]
  mov esi, 27
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  xor eax, 1
  test al, al
  je .L487
  mov edi, 5
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeE
.L487:
  lea rax, [rbp-48]
  mov rdx, QWORD PTR [rbp-88]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv
  lea rdx, [rbp-48]
  lea rax, [rbp-80]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendERKS4_
  mov rax, QWORD PTR [rbp-88]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE21_M_insert_subexpr_endEv
  mov rdx, rax
  lea rax, [rbp-80]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendEl
  mov rax, QWORD PTR [rbp-88]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE16_M_insert_acceptEv
  mov rdx, rax
  lea rax, [rbp-80]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendEl
  mov rax, QWORD PTR [rbp-88]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE18_M_eliminate_dummyEv
  jmp .L496
  mov rbx, rax
  lea rax, [rbp-56]
  mov rdi, rax
  call _ZNSt6localeD1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
  mov rbx, rax
  mov rax, QWORD PTR [rbp-88]
  add rax, 304
  mov rdi, rax
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEED1Ev
  jmp .L490
  mov rbx, rax
.L490:
  mov rax, QWORD PTR [rbp-88]
  add rax, 272
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev
  mov rax, QWORD PTR [rbp-88]
  add rax, 256
  mov rdi, rax
  call _ZNSt10shared_ptrINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEED1Ev
  jmp .L491
  mov rbx, rax
.L491:
  mov rax, QWORD PTR [rbp-88]
  add rax, 8
  mov rdi, rax
  call _ZNSt8__detail8_ScannerIcED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L496:
  add rsp, 120
  pop rbx
  pop rbp
  ret
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EED2Ev:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 88
  mov QWORD PTR [rbp-88], rdi
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE19_M_get_Tp_allocatorEv
  mov rbx, rax
  lea rax, [rbp-80]
  mov rdx, QWORD PTR [rbp-88]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE3endEv
  lea rax, [rbp-48]
  mov rdx, QWORD PTR [rbp-88]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE5beginEv
  lea rdx, [rbp-80]
  lea rsi, [rbp-48]
  mov rax, QWORD PTR [rbp-88]
  mov rcx, rbx
  mov rdi, rax
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE15_M_destroy_dataESt15_Deque_iteratorIS5_RS5_PS5_ESB_RKS6_
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EED2Ev
  nop
  add rsp, 88
  pop rbx
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE10_M_get_nfaEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  add rax, 256
  mov rdi, rax
  call _ZSt4moveIRSt10shared_ptrINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEEEONSt16remove_referenceIT_E4typeEOSA_
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt10shared_ptrIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEEC1IS5_vEEOS_IT_E
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE12_Vector_implD2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSaINSt7__cxx119sub_matchIPKcEEED2Ev
  nop
  leave
  ret
_ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EEC2ERKS5_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE12_Vector_implC1ERKS5_
  nop
  leave
  ret
_ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  sub rdx, rax
  mov rax, rdx
  sar rax, 3
  mov rdx, rax
  movabs rax, -6148914691236517205
  imul rax, rdx
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE13_M_deallocateEPS4_m
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE12_Vector_implD1Ev
  nop
  leave
  ret
_ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt8_DestroyIPNSt7__cxx119sub_matchIPKcEES4_EvT_S6_RSaIT0_E:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt8_DestroyIPNSt7__cxx119sub_matchIPKcEEEvT_S6_
  nop
  leave
  ret
_ZSteqIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEEbRKSt10shared_ptrIT_EDn:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt12__shared_ptrIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EEcvbEv
  xor eax, 1
  leave
  ret
_ZNKSt19__shared_ptr_accessIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EE6_M_getEv
  leave
  ret
_ZNSt7__cxx1113match_resultsIPKcSaINS_9sub_matchIS2_EEEE9_M_resizeEj:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov DWORD PTR [rbp-12], esi
  mov rax, QWORD PTR [rbp-8]
  mov edx, DWORD PTR [rbp-12]
  add edx, 3
  mov edx, edx
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE6resizeEm
  nop
  leave
  ret
_ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE5beginEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEC1ERKS6_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE3endEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  lea rdx, [rax+8]
  lea rax, [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEC1ERKS6_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZN9__gnu_cxxneIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEEbRKNS_17__normal_iteratorIT_T0_EESF_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEE4baseEv
  mov rbx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEE4baseEv
  mov rax, QWORD PTR [rax]
  cmp rbx, rax
  setne al
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZN9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEppEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  lea rdx, [rax+24]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNK9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEdeEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  pop rbp
  ret
_ZNKSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEE5flagsEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax]
  pop rbp
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EEC2ES2_S2_RSt6vectorIS5_S6_ERKNS3_11basic_regexIcS8_EENSt15regex_constants15match_flag_typeE:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 64
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  mov QWORD PTR [rbp-56], rdx
  mov QWORD PTR [rbp-64], rcx
  mov QWORD PTR [rbp-72], r8
  mov DWORD PTR [rbp-76], r9d
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEC1Ev
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rbp-48]
  mov QWORD PTR [rax+32], rdx
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rbp-56]
  mov QWORD PTR [rax+40], rdx
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rbp-72]
  mov QWORD PTR [rax+48], rdx
  mov rax, QWORD PTR [rbp-72]
  add rax, 16
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-40]
  mov QWORD PTR [rax+56], rdx
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rbp-64]
  mov QWORD PTR [rax+64], rdx
  mov rax, QWORD PTR [rbp-40]
  lea rbx, [rax+72]
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaISt4pairIPKciEEC1Ev
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+56]
  add rax, 56
  mov rdi, rax
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EE4sizeEv
  mov rcx, rax
  lea rax, [rbp-17]
  mov rdx, rax
  mov rsi, rcx
  mov rdi, rbx
  call _ZNSt6vectorISt4pairIPKciESaIS3_EEC1EmRKS4_
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaISt4pairIPKciEED1Ev
  mov rax, QWORD PTR [rbp-40]
  lea rbx, [rax+96]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+56]
  add rax, 56
  mov rdi, rax
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EE4sizeEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+56]
  mov rdi, rax
  call _ZNKSt8__detail9_NFA_base8_M_startEv
  mov rdx, r12
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE11_State_infoISt17integral_constantIbLb0EESt6vectorIS5_S6_EEC1Elm
  mov eax, DWORD PTR [rbp-76]
  mov esi, 128
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  test eax, eax
  je .L524
  mov edi, 4
  call _ZNSt15regex_constantscoENS_15match_flag_typeE
  mov ebx, eax
  mov edi, 1
  call _ZNSt15regex_constantscoENS_15match_flag_typeE
  mov edx, eax
  mov eax, DWORD PTR [rbp-76]
  mov esi, edx
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  mov esi, ebx
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  jmp .L525
.L524:
  mov eax, DWORD PTR [rbp-76]
.L525:
  mov rdx, QWORD PTR [rbp-40]
  mov DWORD PTR [rdx+136], eax
  jmp .L531
  mov rbx, rax
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaISt4pairIPKciEED1Ev
  jmp .L527
  mov rbx, rax
  mov rax, QWORD PTR [rbp-40]
  add rax, 72
  mov rdi, rax
  call _ZNSt6vectorISt4pairIPKciESaIS3_EED1Ev
.L527:
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L531:
  add rsp, 64
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt6vectorISt4pairIPKciESaIS3_EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIPKciESaIS3_EE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt8_DestroyIPSt4pairIPKciES3_EvT_S5_RSaIT0_E
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIPKciESaIS3_EED2Ev
  nop
  leave
  ret
_ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt8_DestroyIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEES9_EvT_SB_RSaIT0_E
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EED2Ev
  nop
  leave
  ret
_ZNSt10unique_ptrIA_bSt14default_deleteIS0_EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt15__uniq_ptr_implIbSt14default_deleteIA_bEE6_M_ptrEv
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  test rax, rax
  je .L535
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt10unique_ptrIA_bSt14default_deleteIS0_EE11get_deleterEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt14default_deleteIA_bEclIbEENSt9enable_ifIXsrSt14is_convertibleIPA_T_PS0_E5valueEvE4typeEPS5_
.L535:
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], 0
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE8_M_matchEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+32]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+24], rdx
  mov rax, QWORD PTR [rbp-8]
  mov esi, 0
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE7_M_mainENS9_11_Match_modeE
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EEC2ES2_S2_RSt6vectorIS5_S6_ERKNS3_11basic_regexIcS8_EENSt15regex_constants15match_flag_typeE:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 64
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  mov QWORD PTR [rbp-56], rdx
  mov QWORD PTR [rbp-64], rcx
  mov QWORD PTR [rbp-72], r8
  mov DWORD PTR [rbp-76], r9d
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEC1Ev
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rbp-48]
  mov QWORD PTR [rax+32], rdx
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rbp-56]
  mov QWORD PTR [rax+40], rdx
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rbp-72]
  mov QWORD PTR [rax+48], rdx
  mov rax, QWORD PTR [rbp-72]
  add rax, 16
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-40]
  mov QWORD PTR [rax+56], rdx
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rbp-64]
  mov QWORD PTR [rax+64], rdx
  mov rax, QWORD PTR [rbp-40]
  lea rbx, [rax+72]
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaISt4pairIPKciEEC1Ev
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+56]
  add rax, 56
  mov rdi, rax
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EE4sizeEv
  mov rcx, rax
  lea rax, [rbp-17]
  mov rdx, rax
  mov rsi, rcx
  mov rdi, rbx
  call _ZNSt6vectorISt4pairIPKciESaIS3_EEC1EmRKS4_
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaISt4pairIPKciEED1Ev
  mov rax, QWORD PTR [rbp-40]
  lea rbx, [rax+96]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+56]
  add rax, 56
  mov rdi, rax
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EE4sizeEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+56]
  mov rdi, rax
  call _ZNKSt8__detail9_NFA_base8_M_startEv
  mov rdx, r12
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE11_State_infoISt17integral_constantIbLb1EESt6vectorIS5_S6_EEC1Elm
  mov eax, DWORD PTR [rbp-76]
  mov esi, 128
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  test eax, eax
  je .L539
  mov edi, 4
  call _ZNSt15regex_constantscoENS_15match_flag_typeE
  mov ebx, eax
  mov edi, 1
  call _ZNSt15regex_constantscoENS_15match_flag_typeE
  mov edx, eax
  mov eax, DWORD PTR [rbp-76]
  mov esi, edx
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  mov esi, ebx
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  jmp .L540
.L539:
  mov eax, DWORD PTR [rbp-76]
.L540:
  mov rdx, QWORD PTR [rbp-40]
  mov DWORD PTR [rdx+112], eax
  jmp .L543
  mov rbx, rax
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaISt4pairIPKciEED1Ev
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L543:
  add rsp, 64
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE8_M_matchEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+32]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+24], rdx
  mov rax, QWORD PTR [rbp-8]
  mov esi, 0
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE7_M_mainENS9_11_Match_modeE
  leave
  ret
_ZNSt7__cxx1113match_resultsIPKcSaINS_9sub_matchIS2_EEEE9_M_prefixEv:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov rbx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv
  sub rax, 2
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEixEm
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZNSt7__cxx1113match_resultsIPKcSaINS_9sub_matchIS2_EEEE9_M_suffixEv:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov rbx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv
  sub rax, 1
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEixEm
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEixEm:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-16]
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 3
  add rax, rcx
  pop rbp
  ret
_ZSt11make_sharedINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEJRKSt6localeRNSt15regex_constants18syntax_option_typeEEESt10shared_ptrIT_EDpOT0_:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 48
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  mov QWORD PTR [rbp-56], rdx
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZSt7forwardIRNSt15regex_constants18syntax_option_typeEEOT_RNSt16remove_referenceIS3_E4typeE
  mov r12, rax
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZSt7forwardIRKSt6localeEOT_RNSt16remove_referenceIS3_E4typeE
  mov rbx, rax
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEEC1Ev
  mov rax, QWORD PTR [rbp-40]
  lea rsi, [rbp-17]
  mov rcx, r12
  mov rdx, rbx
  mov rdi, rax
  call _ZSt15allocate_sharedINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_EJRKSt6localeRNSt15regex_constants18syntax_option_typeEEESt10shared_ptrIT_ERKT0_DpOT1_
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEED1Ev
  jmp .L556
  mov rbx, rax
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L556:
  mov rax, QWORD PTR [rbp-40]
  add rsp, 48
  pop rbx
  pop r12
  pop rbp
  ret
_ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EE6_M_getEv
  leave
  ret
_ZNSt8__detail8_ScannerIcEC2EPKcS3_NSt15regex_constants18syntax_option_typeESt6locale:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 56
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov DWORD PTR [rbp-44], ecx
  mov QWORD PTR [rbp-56], r8
  mov rax, QWORD PTR [rbp-24]
  mov edx, DWORD PTR [rbp-44]
  mov esi, edx
  mov rdi, rax
  call _ZNSt8__detail12_ScannerBaseC2ENSt15regex_constants18syntax_option_typeE
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rbp-32]
  mov QWORD PTR [rax+176], rdx
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rbp-40]
  mov QWORD PTR [rax+184], rdx
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZSt9use_facetIKSt5ctypeIcEERKT_RKSt6locale
  mov rdx, rax
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+192], rdx
  mov rax, QWORD PTR [rbp-24]
  add rax, 200
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1Ev
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt8__detail12_ScannerBase10_M_is_ecmaEv
  test al, al
  je .L560
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+232], OFFSET FLAT:_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+240], 0
  jmp .L561
.L560:
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+232], OFFSET FLAT:_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+240], 0
.L561:
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt8__detail8_ScannerIcE10_M_advanceEv
  jmp .L564
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  add rax, 200
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L564:
  add rsp, 56
  pop rbx
  pop rbp
  ret
_ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEEC2IS8_vEEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EEC1Ev
  nop
  leave
  ret
_ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EE6_M_getEv
  leave
  ret
_ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC2ERNS_4_NFAIS3_EEl:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+8], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+16], rdx
  nop
  pop rbp
  ret
_ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE23_M_insert_subexpr_beginEv:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 136
  mov QWORD PTR [rbp-136], rdi
  mov rax, QWORD PTR [rbp-136]
  mov rax, QWORD PTR [rax+40]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-136]
  mov QWORD PTR [rdx+40], rcx
  mov QWORD PTR [rbp-72], rax
  mov rax, QWORD PTR [rbp-136]
  lea rdx, [rbp-72]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorImSaImEE9push_backERKm
  lea rax, [rbp-128]
  mov esi, 8
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1ENS_7_OpcodeE
  mov rax, QWORD PTR [rbp-72]
  mov QWORD PTR [rbp-112], rax
  lea rax, [rbp-128]
  mov rdi, rax
  call _ZSt4moveIRNSt8__detail6_StateIcEEEONSt16remove_referenceIT_E4typeEOS5_
  mov rdx, rax
  lea rax, [rbp-64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1EOS1_
  lea rdx, [rbp-64]
  mov rax, QWORD PTR [rbp-136]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_stateENS_6_StateIcEE
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  lea rax, [rbp-128]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  jmp .L575
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  jmp .L572
  mov rbx, rax
.L572:
  lea rax, [rbp-128]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L575:
  add rsp, 136
  pop rbx
  pop rbp
  ret
_ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendEl:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rbx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+16]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rax+8], rbx
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rbp-32]
  mov QWORD PTR [rax+16], rdx
  nop
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_disjunctionEv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 112
  mov QWORD PTR [rbp-120], rdi
  mov rax, QWORD PTR [rbp-120]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_alternativeEv
.L579:
  mov rax, QWORD PTR [rbp-120]
  mov esi, 19
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L580
  lea rax, [rbp-80]
  mov rdx, QWORD PTR [rbp-120]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv
  mov rax, QWORD PTR [rbp-120]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_alternativeEv
  lea rax, [rbp-112]
  mov rdx, QWORD PTR [rbp-120]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv
  mov rax, QWORD PTR [rbp-120]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_dummyEv
  mov QWORD PTR [rbp-24], rax
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-80]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendEl
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-112]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendEl
  mov rax, QWORD PTR [rbp-120]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-120]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  mov rdx, QWORD PTR [rbp-72]
  mov rax, QWORD PTR [rbp-104]
  mov ecx, 0
  mov rsi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE13_M_insert_altEllb
  mov r12, rax
  mov rax, QWORD PTR [rbp-120]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rsi, rax
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-48]
  mov rcx, rdx
  mov rdx, r12
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEll
  lea rax, [rbp-48]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  jmp .L579
.L580:
  nop
  add rsp, 112
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov DWORD PTR [rbp-12], esi
  mov rax, QWORD PTR [rbp-8]
  add rax, 8
  mov rdi, rax
  call _ZNKSt8__detail8_ScannerIcE12_M_get_tokenEv
  cmp DWORD PTR [rbp-12], eax
  sete al
  test al, al
  je .L582
  mov rax, QWORD PTR [rbp-8]
  add rax, 8
  mov rdi, rax
  call _ZNKSt8__detail8_ScannerIcE12_M_get_valueB5cxx11Ev
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  add rax, 272
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSERKS4_
  mov rax, QWORD PTR [rbp-8]
  add rax, 8
  mov rdi, rax
  call _ZNSt8__detail8_ScannerIcE10_M_advanceEv
  mov eax, 1
  jmp .L583
.L582:
  mov eax, 0
.L583:
  leave
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  add rax, 304
  mov rdi, rax
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE3topEv
  mov rsi, rax
  mov rcx, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rsi]
  mov rdx, QWORD PTR [rsi+8]
  mov QWORD PTR [rcx], rax
  mov QWORD PTR [rcx+8], rdx
  mov rax, QWORD PTR [rsi+16]
  mov QWORD PTR [rcx+16], rax
  mov rax, QWORD PTR [rbp-16]
  add rax, 304
  mov rdi, rax
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE3popEv
  nop
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendERKS4_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-32]
  mov rbx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+16]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rax+8], rbx
  mov rax, QWORD PTR [rbp-32]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+16], rdx
  nop
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE21_M_insert_subexpr_endEv:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 120
  mov QWORD PTR [rbp-120], rdi
  lea rax, [rbp-112]
  mov esi, 9
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1ENS_7_OpcodeE
  mov rax, QWORD PTR [rbp-120]
  mov rdi, rax
  call _ZNSt6vectorImSaImEE4backEv
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-96], rax
  mov rax, QWORD PTR [rbp-120]
  mov rdi, rax
  call _ZNSt6vectorImSaImEE8pop_backEv
  lea rax, [rbp-112]
  mov rdi, rax
  call _ZSt4moveIRNSt8__detail6_StateIcEEEONSt16remove_referenceIT_E4typeEOS5_
  mov rdx, rax
  lea rax, [rbp-64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1EOS1_
  lea rdx, [rbp-64]
  mov rax, QWORD PTR [rbp-120]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_stateENS_6_StateIcEE
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  lea rax, [rbp-112]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  jmp .L593
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  jmp .L590
  mov rbx, rax
.L590:
  lea rax, [rbp-112]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L593:
  add rsp, 120
  pop rbx
  pop rbp
  ret
_ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE16_M_insert_acceptEv:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 88
  mov QWORD PTR [rbp-88], rdi
  lea rax, [rbp-80]
  mov esi, 12
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1ENS_7_OpcodeE
  lea rdx, [rbp-80]
  mov rax, QWORD PTR [rbp-88]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_stateENS_6_StateIcEE
  mov QWORD PTR [rbp-24], rax
  lea rax, [rbp-80]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, QWORD PTR [rbp-24]
  jmp .L598
  mov rbx, rax
  lea rax, [rbp-80]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L598:
  add rsp, 88
  pop rbx
  pop rbp
  ret
_ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE18_M_eliminate_dummyEv:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-40], rdi
  mov rax, QWORD PTR [rbp-40]
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-8]
  add rax, 56
  mov rdi, rax
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EE5beginEv
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-8]
  add rax, 56
  mov rdi, rax
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EE3endEv
  mov QWORD PTR [rbp-32], rax
.L609:
  lea rdx, [rbp-32]
  lea rax, [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxxneIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEEEbRKNS_17__normal_iteratorIT_T0_EESD_
  test al, al
  je .L610
  lea rax, [rbp-24]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEEdeEv
  mov QWORD PTR [rbp-16], rax
.L604:
  mov rax, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rax+8]
  test rax, rax
  js .L601
  mov rax, QWORD PTR [rbp-40]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rax+8]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov rdi, rax
  call _ZNKSt8__detail6_StateIcE9_M_opcodeEv
  cmp eax, 10
  jne .L601
  mov eax, 1
  jmp .L602
.L601:
  mov eax, 0
.L602:
  test al, al
  je .L603
  mov rax, QWORD PTR [rbp-40]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rax+8]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-16]
  mov QWORD PTR [rax+8], rdx
  jmp .L604
.L603:
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZNSt8__detail11_State_base10_M_has_altEv
  test al, al
  je .L605
.L608:
  mov rax, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rax+16]
  test rax, rax
  js .L606
  mov rax, QWORD PTR [rbp-40]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rax+16]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov rdi, rax
  call _ZNKSt8__detail6_StateIcE9_M_opcodeEv
  cmp eax, 10
  jne .L606
  mov eax, 1
  jmp .L607
.L606:
  mov eax, 0
.L607:
  test al, al
  je .L605
  mov rax, QWORD PTR [rbp-40]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rax+16]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-16]
  mov QWORD PTR [rax+16], rdx
  jmp .L608
.L605:
  lea rax, [rbp-24]
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEEppEv
  jmp .L609
.L610:
  nop
  leave
  ret
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE11_Deque_implD2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSaINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEED2Ev
  nop
  leave
  ret
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  test rax, rax
  je .L613
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+72]
  lea rdx, [rax+8]
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax+40]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE16_M_destroy_nodesEPPS5_S9_
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE17_M_deallocate_mapEPPS5_m
.L613:
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE11_Deque_implD1Ev
  nop
  leave
  ret
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE5beginEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  lea rdx, [rax+16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_EC1ERKS8_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE3endEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  lea rdx, [rax+48]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_EC1ERKS8_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE19_M_get_Tp_allocatorEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE15_M_destroy_dataESt15_Deque_iteratorIS5_RS5_PS5_ESB_RKS6_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-72], rdi
  mov QWORD PTR [rbp-80], rsi
  mov QWORD PTR [rbp-88], rdx
  mov QWORD PTR [rbp-96], rcx
  nop
  pop rbp
  ret
_ZSt4moveIRSt10shared_ptrINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEEEONSt16remove_referenceIT_E4typeEOSA_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt10shared_ptrIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEEC2IS5_vEEOS_IT_E:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rbx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt4moveIRSt10shared_ptrINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEEEONSt16remove_referenceIT_E4typeEOSA_
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt12__shared_ptrIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EEC2IS5_vEEOS_IT_LS8_2EE
  nop
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE12_Vector_implC2ERKS5_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSaINSt7__cxx119sub_matchIPKcEEEC2ERKS4_
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+8], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+16], 0
  nop
  leave
  ret
_ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE13_M_deallocateEPS4_m:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  cmp QWORD PTR [rbp-16], 0
  je .L627
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaINSt7__cxx119sub_matchIPKcEEEE10deallocateERS5_PS4_m
.L627:
  nop
  leave
  ret
_ZSt8_DestroyIPNSt7__cxx119sub_matchIPKcEEEvT_S6_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Destroy_auxILb1EE9__destroyIPNSt7__cxx119sub_matchIPKcEEEEvT_S8_
  nop
  leave
  ret
_ZNKSt12__shared_ptrIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EEcvbEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  test rax, rax
  setne al
  pop rbp
  ret
_ZNKSt19__shared_ptr_accessIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EE6_M_getEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt12__shared_ptrIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EE3getEv
  leave
  ret
_ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE6resizeEm:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv
  cmp QWORD PTR [rbp-16], rax
  seta al
  test al, al
  je .L634
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-16]
  sub rax, rdx
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE17_M_default_appendEm
  jmp .L636
.L634:
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv
  cmp QWORD PTR [rbp-16], rax
  setb al
  test al, al
  je .L636
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-16]
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 3
  lea rdx, [rcx+rax]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE15_M_erase_at_endEPS4_
.L636:
  nop
  leave
  ret
_ZN9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEC2ERKS6_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  nop
  pop rbp
  ret
_ZNK9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEE4baseEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNKSt19__shared_ptr_accessIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EE6_M_getEv
  leave
  ret
_ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EE4sizeEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  sub rdx, rax
  mov rax, rdx
  sar rax, 4
  mov rdx, rax
  movabs rax, -6148914691236517205
  imul rax, rdx
  pop rbp
  ret
_ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EEC2Ev
  nop
  leave
  ret
_ZNSaISt4pairIPKciEEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorISt4pairIPKciEEC2Ev
  nop
  leave
  ret
_ZNSaISt4pairIPKciEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorISt4pairIPKciEED2Ev
  nop
  leave
  ret
_ZNSt6vectorISt4pairIPKciESaIS3_EEC2EmRKS4_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rbp-40]
  mov rcx, QWORD PTR [rbp-32]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIPKciESaIS3_EEC2EmRKS4_
  mov rdx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorISt4pairIPKciESaIS3_EE21_M_default_initializeEm
  jmp .L650
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIPKciESaIS3_EED2Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L650:
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE11_State_infoISt17integral_constantIbLb0EESt6vectorIS5_S6_EEC2Elm:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EEC1Ev
  mov rax, QWORD PTR [rbp-24]
  lea r12, [rax+24]
  mov rbx, QWORD PTR [rbp-40]
  mov rdi, rbx
  call _Znam
  mov rcx, rax
  mov rdx, rcx
  lea rax, [rbx-1]
.L653:
  test rax, rax
  js .L652
  mov BYTE PTR [rdx], 0
  add rdx, 1
  sub rax, 1
  jmp .L653
.L652:
  mov rsi, rcx
  mov rdi, r12
  call _ZNSt10unique_ptrIA_bSt14default_deleteIS0_EEC1IPbS2_vbEET_
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rbp-32]
  mov QWORD PTR [rax+32], rdx
  jmp .L656
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L656:
  add rsp, 32
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt12_Vector_baseISt4pairIPKciESaIS3_EE12_Vector_implD2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSaISt4pairIPKciEED2Ev
  nop
  leave
  ret
_ZNSt12_Vector_baseISt4pairIPKciESaIS3_EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  sub rdx, rax
  mov rax, rdx
  sar rax, 4
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIPKciESaIS3_EE13_M_deallocateEPS3_m
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIPKciESaIS3_EE12_Vector_implD1Ev
  nop
  leave
  ret
_ZNSt12_Vector_baseISt4pairIPKciESaIS3_EE19_M_get_Tp_allocatorEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt8_DestroyIPSt4pairIPKciES3_EvT_S5_RSaIT0_E:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt8_DestroyIPSt4pairIPKciEEvT_S5_
  nop
  leave
  ret
_ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE12_Vector_implD2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSaISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS5_EEEED2Ev
  nop
  leave
  ret
_ZN9__gnu_cxx14__alloc_traitsISaINSt7__cxx119sub_matchIPKcEEES5_E15_S_always_equalEv:
  push rbp
  mov rbp, rsp
  mov eax, 1
  pop rbp
  ret
_ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  sub rdx, rax
  mov rax, rdx
  sar rax, 5
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE13_M_deallocateEPS9_m
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE12_Vector_implD1Ev
  nop
  leave
  ret
_ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE19_M_get_Tp_allocatorEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt8_DestroyIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEES9_EvT_SB_RSaIT0_E:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt8_DestroyIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEEvT_SB_
  nop
  leave
  ret
_ZNSt15__uniq_ptr_implIbSt14default_deleteIA_bEE6_M_ptrEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZSt3getILm0EJPbSt14default_deleteIA_bEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_
  leave
  ret
_ZNSt10unique_ptrIA_bSt14default_deleteIS0_EE11get_deleterEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt15__uniq_ptr_implIbSt14default_deleteIA_bEE10_M_deleterEv
  leave
  ret
_ZNKSt14default_deleteIA_bEclIbEENSt9enable_ifIXsrSt14is_convertibleIPA_T_PS0_E5valueEvE4typeEPS5_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  cmp QWORD PTR [rbp-16], 0
  je .L675
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZdaPv
.L675:
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE7_M_mainENS9_11_Match_modeE:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov BYTE PTR [rbp-28], al
  movzx edx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, edx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE16_M_main_dispatchENS9_11_Match_modeESt17integral_constantIbLb0EE
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE20_M_search_from_firstEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+32]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+24], rdx
  mov rax, QWORD PTR [rbp-8]
  mov esi, 1
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE7_M_mainENS9_11_Match_modeE
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE11_State_infoISt17integral_constantIbLb1EESt6vectorIS5_S6_EEC2Elm:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov QWORD PTR [rax], rdx
  nop
  pop rbp
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE7_M_mainENS9_11_Match_modeE:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov BYTE PTR [rbp-28], al
  movzx edx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, edx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE16_M_main_dispatchENS9_11_Match_modeESt17integral_constantIbLb1EE
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE20_M_search_from_firstEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+32]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+24], rdx
  mov rax, QWORD PTR [rbp-8]
  mov esi, 1
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE7_M_mainENS9_11_Match_modeE
  leave
  ret
_ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  sub rdx, rax
  mov rax, rdx
  sar rax, 3
  mov rdx, rax
  movabs rax, -6148914691236517205
  imul rax, rdx
  pop rbp
  ret
_ZNSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EEC2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov edx, OFFSET FLAT:_ZTVSt16_Sp_counted_baseILN9__gnu_cxx12_Lock_policyE2EE+16
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-8]
  mov DWORD PTR [rax+8], 1
  mov rax, QWORD PTR [rbp-8]
  mov DWORD PTR [rax+12], 1
  nop
  pop rbp
  ret
_ZNSaINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEEC2Ev
  nop
  leave
  ret
_ZNSaINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEED2Ev
  nop
  leave
  ret
_ZSt7forwardIRKSt6localeEOT_RNSt16remove_referenceIS3_E4typeE:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt7forwardIRNSt15regex_constants18syntax_option_typeEEOT_RNSt16remove_referenceIS3_E4typeE:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt15allocate_sharedINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_EJRKSt6localeRNSt15regex_constants18syntax_option_typeEEESt10shared_ptrIT_ERKT0_DpOT1_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov QWORD PTR [rbp-48], rcx
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZSt7forwardIRNSt15regex_constants18syntax_option_typeEEOT_RNSt16remove_referenceIS3_E4typeE
  mov rbx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZSt7forwardIRKSt6localeEOT_RNSt16remove_referenceIS3_E4typeE
  mov rdx, rax
  mov rsi, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rcx, rbx
  mov rdi, rax
  call _ZNSt10shared_ptrINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEEC1ISaIS5_EJRKSt6localeRNSt15regex_constants18syntax_option_typeEEEESt20_Sp_alloc_shared_tagIT_EDpOT0_
  mov rax, QWORD PTR [rbp-24]
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EE6_M_getEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt12__shared_ptrINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EE3getEv
  leave
  ret
_ZSt9use_facetIKSt5ctypeIcEERKT_RKSt6locale:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov edi, OFFSET FLAT:_ZNSt5ctypeIcE2idE
  call _ZNKSt6locale2id5_M_idEv
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax]
  mov rax, QWORD PTR [rax+8]
  mov QWORD PTR [rbp-16], rax
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax]
  mov rax, QWORD PTR [rax+16]
  cmp QWORD PTR [rbp-8], rax
  jnb .L699
  mov rax, QWORD PTR [rbp-8]
  lea rdx, [0+rax*8]
  mov rax, QWORD PTR [rbp-16]
  add rax, rdx
  mov rax, QWORD PTR [rax]
  test rax, rax
  jne .L700
.L699:
  call _ZSt16__throw_bad_castv
.L700:
  mov rax, QWORD PTR [rbp-8]
  lea rdx, [0+rax*8]
  mov rax, QWORD PTR [rbp-16]
  add rax, rdx
  mov rax, QWORD PTR [rax]
  mov ecx, 0
  mov edx, OFFSET FLAT:_ZTISt5ctypeIcE
  mov esi, OFFSET FLAT:_ZTINSt6locale5facetE
  mov rdi, rax
  call __dynamic_cast
  test rax, rax
  jne .L704
  call __cxa_bad_cast
.L704:
  leave
  ret
.LC11:
  .string "Unexpected end of regex when escaping."
.LC12:
  .string "Unexpected end of regex when reading control code."
.LC13:
  .string "Unexpected end of regex when ascii character."
_ZNSt8__detail8_ScannerIcE18_M_eat_escape_ecmaEv:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-40], rdi
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  jne .L706
  mov esi, OFFSET FLAT:.LC11
  mov edi, 2
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L706:
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-40]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  mov BYTE PTR [rbp-21], al
  mov rbx, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+192]
  movsx ecx, BYTE PTR [rbp-21]
  mov edx, 0
  mov esi, ecx
  mov rdi, rax
  call _ZNKSt5ctypeIcE6narrowEcc
  movsx eax, al
  mov esi, eax
  mov rdi, rbx
  call _ZNSt8__detail12_ScannerBase14_M_find_escapeEc
  mov QWORD PTR [rbp-32], rax
  cmp QWORD PTR [rbp-32], 0
  je .L707
  cmp BYTE PTR [rbp-21], 98
  jne .L708
  mov rax, QWORD PTR [rbp-40]
  mov eax, DWORD PTR [rax+136]
  cmp eax, 2
  jne .L707
.L708:
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 1
  mov rax, QWORD PTR [rbp-40]
  lea rcx, [rax+200]
  mov rax, QWORD PTR [rbp-32]
  movzx eax, BYTE PTR [rax]
  movsx eax, al
  mov edx, eax
  mov esi, 1
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  jmp .L709
.L707:
  cmp BYTE PTR [rbp-21], 98
  jne .L710
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 24
  mov rax, QWORD PTR [rbp-40]
  add rax, 200
  mov edx, 112
  mov esi, 1
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  jmp .L731
.L710:
  cmp BYTE PTR [rbp-21], 66
  jne .L711
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 24
  mov rax, QWORD PTR [rbp-40]
  add rax, 200
  mov edx, 110
  mov esi, 1
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  jmp .L731
.L711:
  cmp BYTE PTR [rbp-21], 100
  je .L712
  cmp BYTE PTR [rbp-21], 68
  je .L712
  cmp BYTE PTR [rbp-21], 115
  je .L712
  cmp BYTE PTR [rbp-21], 83
  je .L712
  cmp BYTE PTR [rbp-21], 119
  je .L712
  cmp BYTE PTR [rbp-21], 87
  jne .L713
.L712:
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 14
  mov rax, QWORD PTR [rbp-40]
  lea rcx, [rax+200]
  movsx eax, BYTE PTR [rbp-21]
  mov edx, eax
  mov esi, 1
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  jmp .L709
.L713:
  cmp BYTE PTR [rbp-21], 99
  jne .L714
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  jne .L715
  mov esi, OFFSET FLAT:.LC12
  mov edi, 2
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L715:
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 1
  mov rax, QWORD PTR [rbp-40]
  lea rdi, [rax+200]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-40]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  movsx eax, al
  mov edx, eax
  mov esi, 1
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  jmp .L731
.L714:
  cmp BYTE PTR [rbp-21], 120
  je .L716
  cmp BYTE PTR [rbp-21], 117
  jne .L717
.L716:
  mov rax, QWORD PTR [rbp-40]
  add rax, 200
  mov rdx, -1
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5eraseEmm
  mov DWORD PTR [rbp-20], 0
.L725:
  cmp BYTE PTR [rbp-21], 120
  jne .L718
  mov eax, 2
  jmp .L719
.L718:
  mov eax, 4
.L719:
  cmp eax, DWORD PTR [rbp-20]
  jle .L720
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  je .L721
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+192]
  mov rdx, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rdx+176]
  movzx edx, BYTE PTR [rdx]
  movsx edx, dl
  mov esi, 4096
  mov rdi, rax
  call _ZNKSt5ctypeIcE2isEtc
  xor eax, 1
  test al, al
  je .L722
.L721:
  mov eax, 1
  jmp .L723
.L722:
  mov eax, 0
.L723:
  test al, al
  je .L724
  mov esi, OFFSET FLAT:.LC13
  mov edi, 2
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L724:
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-40]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  movsx eax, al
  mov rdx, QWORD PTR [rbp-40]
  add rdx, 200
  mov esi, eax
  mov rdi, rdx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEpLEc
  add DWORD PTR [rbp-20], 1
  jmp .L725
.L720:
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 3
  jmp .L709
.L717:
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+192]
  movsx edx, BYTE PTR [rbp-21]
  mov esi, 2048
  mov rdi, rax
  call _ZNKSt5ctypeIcE2isEtc
  test al, al
  je .L726
  mov rax, QWORD PTR [rbp-40]
  lea rcx, [rax+200]
  movsx eax, BYTE PTR [rbp-21]
  mov edx, eax
  mov esi, 1
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
.L730:
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  je .L727
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+192]
  mov rdx, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rdx+176]
  movzx edx, BYTE PTR [rdx]
  movsx edx, dl
  mov esi, 2048
  mov rdi, rax
  call _ZNKSt5ctypeIcE2isEtc
  test al, al
  je .L727
  mov eax, 1
  jmp .L728
.L727:
  mov eax, 0
.L728:
  test al, al
  je .L729
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-40]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  movsx eax, al
  mov rdx, QWORD PTR [rbp-40]
  add rdx, 200
  mov esi, eax
  mov rdi, rdx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEpLEc
  jmp .L730
.L729:
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 4
  jmp .L731
.L726:
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 1
  mov rax, QWORD PTR [rbp-40]
  lea rcx, [rax+200]
  movsx eax, BYTE PTR [rbp-21]
  mov edx, eax
  mov esi, 1
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  jmp .L731
.L709:
.L731:
  nop
  add rsp, 40
  pop rbx
  pop rbp
  ret
.LC14:
  .string "Unexpected escape character."
_ZNSt8__detail8_ScannerIcE19_M_eat_escape_posixEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  jne .L733
  mov esi, OFFSET FLAT:.LC11
  mov edi, 2
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L733:
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+176]
  movzx eax, BYTE PTR [rax]
  mov BYTE PTR [rbp-1], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+192]
  movsx ecx, BYTE PTR [rbp-1]
  mov edx, 0
  mov esi, ecx
  mov rdi, rax
  call _ZNKSt5ctypeIcE6narrowEcc
  movsx edx, al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+160]
  mov esi, edx
  mov rdi, rax
  call strchr
  mov QWORD PTR [rbp-16], rax
  cmp QWORD PTR [rbp-16], 0
  je .L734
  mov rax, QWORD PTR [rbp-16]
  movzx eax, BYTE PTR [rax]
  test al, al
  je .L734
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax+144], 1
  mov rax, QWORD PTR [rbp-24]
  lea rcx, [rax+200]
  movsx eax, BYTE PTR [rbp-1]
  mov edx, eax
  mov esi, 1
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  jmp .L735
.L734:
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt8__detail12_ScannerBase9_M_is_awkEv
  test al, al
  je .L736
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv
  jmp .L732
.L736:
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt8__detail12_ScannerBase11_M_is_basicEv
  test al, al
  je .L738
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+192]
  movsx edx, BYTE PTR [rbp-1]
  mov esi, 2048
  mov rdi, rax
  call _ZNKSt5ctypeIcE2isEtc
  test al, al
  je .L738
  cmp BYTE PTR [rbp-1], 48
  je .L738
  mov eax, 1
  jmp .L739
.L738:
  mov eax, 0
.L739:
  test al, al
  je .L740
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax+144], 4
  mov rax, QWORD PTR [rbp-24]
  lea rcx, [rax+200]
  movsx eax, BYTE PTR [rbp-1]
  mov edx, eax
  mov esi, 1
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  jmp .L735
.L740:
  mov esi, OFFSET FLAT:.LC14
  mov edi, 2
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L735:
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+176]
  lea rdx, [rax+1]
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+176], rdx
.L732:
  leave
  ret
_ZNSt8__detail8_ScannerIcE10_M_advanceEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  jne .L742
  mov rax, QWORD PTR [rbp-8]
  mov DWORD PTR [rax+144], 27
  jmp .L741
.L742:
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+136]
  test eax, eax
  jne .L744
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt8__detail8_ScannerIcE14_M_scan_normalEv
  jmp .L741
.L744:
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+136]
  cmp eax, 2
  jne .L745
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt8__detail8_ScannerIcE18_M_scan_in_bracketEv
  jmp .L741
.L745:
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+136]
  cmp eax, 1
  jne .L741
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv
.L741:
  leave
  ret
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EEC2Ev
  nop
  leave
  ret
_ZNSt6vectorImSaImEE9push_backERKm:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+16]
  cmp rdx, rax
  je .L748
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaImEE9constructImJRKmEEEvRS0_PT_DpOT0_
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+8]
  lea rdx, [rax+8]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+8], rdx
  jmp .L750
.L748:
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt6vectorImSaImEE3endEv
  mov rcx, rax
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_
.L750:
  nop
  leave
  ret
_ZNSt8__detail6_StateIcEC2ENS_7_OpcodeE:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov DWORD PTR [rbp-12], esi
  mov rax, QWORD PTR [rbp-8]
  mov edx, DWORD PTR [rbp-12]
  mov esi, edx
  mov rdi, rax
  call _ZNSt8__detail11_State_baseC2ENS_7_OpcodeE
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt8__detail6_StateIcE9_M_opcodeEv
  cmp eax, 11
  sete al
  test al, al
  je .L753
  mov rax, QWORD PTR [rbp-8]
  add rax, 16
  mov rdi, rax
  call _ZN9__gnu_cxx16__aligned_membufISt8functionIFbcEEE7_M_addrEv
  mov rsi, rax
  mov edi, 32
  call _ZnwmPv
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1Ev
.L753:
  nop
  leave
  ret
_ZNSt8functionIFbcEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt14_Function_baseD2Ev
  nop
  leave
  ret
_ZNSt8__detail6_StateIcED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt8__detail6_StateIcE9_M_opcodeEv
  cmp eax, 11
  sete al
  test al, al
  je .L757
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcE14_M_get_matcherEv
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
.L757:
  nop
  leave
  ret
.LC15:
  .string "Number of NFA states exceeds limit. Please use shorter regex string, or use smaller brace expression, or make _GLIBCXX_REGEX_STATE_LIMIT larger."
_ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_stateENS_6_StateIcEE:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  lea rbx, [rax+56]
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt4moveIRNSt8__detail6_StateIcEEEONSt16remove_referenceIT_E4typeEOS5_
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EE9push_backEOS2_
  mov rax, QWORD PTR [rbp-24]
  add rax, 56
  mov rdi, rax
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EE4sizeEv
  cmp rax, 100000
  seta al
  test al, al
  je .L759
  mov esi, OFFSET FLAT:.LC15
  mov edi, 9
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L759:
  mov rax, QWORD PTR [rbp-24]
  add rax, 56
  mov rdi, rax
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EE4sizeEv
  sub rax, 1
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZSt4moveIRNSt8__detail6_StateIcEEEONSt16remove_referenceIT_E4typeEOS5_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt8__detail6_StateIcEC2EOS1_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rbp-32]
  mov rcx, QWORD PTR [rdx]
  mov rbx, QWORD PTR [rdx+8]
  mov QWORD PTR [rax], rcx
  mov QWORD PTR [rax+8], rbx
  mov rcx, QWORD PTR [rdx+16]
  mov rbx, QWORD PTR [rdx+24]
  mov QWORD PTR [rax+16], rcx
  mov QWORD PTR [rax+24], rbx
  mov rcx, QWORD PTR [rdx+40]
  mov rdx, QWORD PTR [rdx+32]
  mov QWORD PTR [rax+32], rdx
  mov QWORD PTR [rax+40], rcx
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNKSt8__detail6_StateIcE9_M_opcodeEv
  cmp eax, 11
  sete al
  test al, al
  je .L765
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcE14_M_get_matcherEv
  mov rdi, rax
  call _ZSt4moveIRSt8functionIFbcEEEONSt16remove_referenceIT_E4typeEOS5_
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  add rax, 16
  mov rdi, rax
  call _ZN9__gnu_cxx16__aligned_membufISt8functionIFbcEEE7_M_addrEv
  mov rsi, rax
  mov edi, 32
  call _ZnwmPv
  mov rsi, rbx
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1EOS1_
.L765:
  nop
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-16]
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 4
  add rax, rcx
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_alternativeEv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 112
  mov QWORD PTR [rbp-120], rdi
  mov rax, QWORD PTR [rbp-120]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE7_M_termEv
  test al, al
  je .L769
  lea rax, [rbp-112]
  mov rdx, QWORD PTR [rbp-120]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv
  mov rax, QWORD PTR [rbp-120]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_alternativeEv
  lea rax, [rbp-80]
  mov rdx, QWORD PTR [rbp-120]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv
  lea rdx, [rbp-80]
  lea rax, [rbp-112]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendERKS4_
  mov rax, QWORD PTR [rbp-120]
  lea rdx, [rax+304]
  lea rax, [rbp-112]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushERKS5_
  jmp .L771
.L769:
  mov rax, QWORD PTR [rbp-120]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-120]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_dummyEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-120]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-48]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-48]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
.L771:
  nop
  add rsp, 112
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_dummyEv:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 72
  mov QWORD PTR [rbp-72], rdi
  lea rax, [rbp-64]
  mov esi, 10
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1ENS_7_OpcodeE
  lea rdx, [rbp-64]
  mov rax, QWORD PTR [rbp-72]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_stateENS_6_StateIcEE
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  jmp .L776
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L776:
  add rsp, 72
  pop rbx
  pop rbp
  ret
_ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE13_M_insert_altEllb:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 136
  mov QWORD PTR [rbp-120], rdi
  mov QWORD PTR [rbp-128], rsi
  mov QWORD PTR [rbp-136], rdx
  mov eax, ecx
  mov BYTE PTR [rbp-140], al
  lea rax, [rbp-112]
  mov esi, 1
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1ENS_7_OpcodeE
  mov rax, QWORD PTR [rbp-128]
  mov QWORD PTR [rbp-104], rax
  mov rax, QWORD PTR [rbp-136]
  mov QWORD PTR [rbp-96], rax
  lea rax, [rbp-112]
  mov rdi, rax
  call _ZSt4moveIRNSt8__detail6_StateIcEEEONSt16remove_referenceIT_E4typeEOS5_
  mov rdx, rax
  lea rax, [rbp-64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1EOS1_
  lea rdx, [rbp-64]
  mov rax, QWORD PTR [rbp-120]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_stateENS_6_StateIcEE
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  lea rax, [rbp-112]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  jmp .L781
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  lea rax, [rbp-112]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L781:
  add rsp, 136
  pop rbx
  pop rbp
  ret
_ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC2ERNS_4_NFAIS3_EEll:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov QWORD PTR [rbp-32], rcx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+8], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-32]
  mov QWORD PTR [rax+16], rdx
  nop
  pop rbp
  ret
_ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rbx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt4moveIRNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEONSt16remove_referenceIT_E4typeEOS8_
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE9push_backEOS5_
  nop
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZNKSt8__detail8_ScannerIcE12_M_get_tokenEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+144]
  pop rbp
  ret
_ZNKSt8__detail8_ScannerIcE12_M_get_valueB5cxx11Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 200
  pop rbp
  ret
_ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE3topEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE4backEv
  leave
  ret
_ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE3popEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE8pop_backEv
  nop
  leave
  ret
_ZNSt6vectorImSaImEE4backEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt6vectorImSaImEE3endEv
  mov QWORD PTR [rbp-8], rax
  lea rax, [rbp-8]
  mov esi, 1
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEmiEl
  mov QWORD PTR [rbp-16], rax
  lea rax, [rbp-16]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEdeEv
  nop
  leave
  ret
_ZNSt6vectorImSaImEE8pop_backEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+8]
  lea rdx, [rax-8]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+8], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaImEE7destroyImEEvRS0_PT_
  nop
  leave
  ret
_ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EE5beginEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEEC1ERKS4_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EE3endEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  lea rdx, [rax+8]
  lea rax, [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEEC1ERKS4_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZN9__gnu_cxxneIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEEEbRKNS_17__normal_iteratorIT_T0_EESD_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEE4baseEv
  mov rbx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEE4baseEv
  mov rax, QWORD PTR [rax]
  cmp rbx, rax
  setne al
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZN9__gnu_cxx17__normal_iteratorIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEEppEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  lea rdx, [rax+48]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNK9__gnu_cxx17__normal_iteratorIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEEdeEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  pop rbp
  ret
_ZNKSt8__detail6_StateIcE9_M_opcodeEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax]
  pop rbp
  ret
_ZNSaINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEED2Ev
  nop
  leave
  ret
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE16_M_destroy_nodesEPPS5_S9_:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov rax, QWORD PTR [rbp-32]
  mov QWORD PTR [rbp-8], rax
.L809:
  mov rax, QWORD PTR [rbp-8]
  cmp rax, QWORD PTR [rbp-40]
  jnb .L810
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE18_M_deallocate_nodeEPS5_
  add QWORD PTR [rbp-8], 8
  jmp .L809
.L810:
  nop
  leave
  ret
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE17_M_deallocate_mapEPPS5_m:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  lea rax, [rbp-1]
  mov rdx, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNKSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE20_M_get_map_allocatorEv
  mov rdx, QWORD PTR [rbp-40]
  mov rcx, QWORD PTR [rbp-32]
  lea rax, [rbp-1]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaIPNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEE10deallocateERS7_PS6_m
  lea rax, [rbp-1]
  mov rdi, rax
  call _ZNSaIPNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEED1Ev
  nop
  leave
  ret
_ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_EC2ERKS8_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+8], rdx
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+16], rdx
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+24], rdx
  nop
  pop rbp
  ret
_ZNSt12__shared_ptrIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EEC2IS5_vEEOS_IT_LS8_2EE:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-8]
  add rax, 8
  mov rdi, rax
  call _ZNSt14__shared_countILN9__gnu_cxx12_Lock_policyE2EEC1Ev
  mov rax, QWORD PTR [rbp-8]
  add rax, 8
  mov rdx, QWORD PTR [rbp-16]
  add rdx, 8
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt14__shared_countILN9__gnu_cxx12_Lock_policyE2EE7_M_swapERS2_
  mov rax, QWORD PTR [rbp-16]
  mov QWORD PTR [rax], 0
  nop
  leave
  ret
_ZNSaINSt7__cxx119sub_matchIPKcEEEC2ERKS4_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorINSt7__cxx119sub_matchIPKcEEEC2ERKS6_
  nop
  leave
  ret
_ZNSt16allocator_traitsISaINSt7__cxx119sub_matchIPKcEEEE10deallocateERS5_PS4_m:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorINSt7__cxx119sub_matchIPKcEEE10deallocateEPS5_m
  nop
  leave
  ret
_ZNSt12_Destroy_auxILb1EE9__destroyIPNSt7__cxx119sub_matchIPKcEEEEvT_S8_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  nop
  pop rbp
  ret
_ZNKSt12__shared_ptrIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EE3getEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  pop rbp
  ret
.LC16:
  .string "vector::_M_default_append"
_ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE17_M_default_appendEm:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 72
  mov QWORD PTR [rbp-72], rdi
  mov QWORD PTR [rbp-80], rsi
  cmp QWORD PTR [rbp-80], 0
  je .L832
  mov rax, QWORD PTR [rbp-72]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv
  mov QWORD PTR [rbp-32], rax
  mov rax, QWORD PTR [rbp-72]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-72]
  mov rax, QWORD PTR [rax+8]
  sub rdx, rax
  mov rax, rdx
  sar rax, 3
  mov rdx, rax
  movabs rax, -6148914691236517205
  imul rax, rdx
  mov QWORD PTR [rbp-40], rax
  mov rax, QWORD PTR [rbp-72]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE8max_sizeEv
  cmp QWORD PTR [rbp-32], rax
  ja .L821
  mov rax, QWORD PTR [rbp-72]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE8max_sizeEv
  sub rax, QWORD PTR [rbp-32]
  cmp QWORD PTR [rbp-40], rax
  jbe .L822
.L821:
  mov eax, 1
  jmp .L823
.L822:
  mov eax, 0
.L823:
  test al, al
  mov rax, QWORD PTR [rbp-40]
  cmp rax, QWORD PTR [rbp-80]
  jb .L825
  mov rax, QWORD PTR [rbp-72]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-72]
  mov rax, QWORD PTR [rax+8]
  mov rcx, QWORD PTR [rbp-80]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt27__uninitialized_default_n_aIPNSt7__cxx119sub_matchIPKcEEmS4_ET_S6_T0_RSaIT1_E
  mov rdx, rax
  mov rax, QWORD PTR [rbp-72]
  mov QWORD PTR [rax+8], rdx
  jmp .L832
.L825:
  mov rcx, QWORD PTR [rbp-80]
  mov rax, QWORD PTR [rbp-72]
  mov edx, OFFSET FLAT:.LC16
  mov rsi, rcx
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE12_M_check_lenEmS3_
  mov QWORD PTR [rbp-48], rax
  mov rax, QWORD PTR [rbp-72]
  mov rdx, QWORD PTR [rbp-48]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE11_M_allocateEm
  mov QWORD PTR [rbp-56], rax
  mov QWORD PTR [rbp-24], 0
  mov rax, QWORD PTR [rbp-72]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rsi, rax
  mov rdx, QWORD PTR [rbp-32]
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 3
  mov rdx, rax
  mov rax, QWORD PTR [rbp-56]
  lea rcx, [rdx+rax]
  mov rax, QWORD PTR [rbp-80]
  mov rdx, rsi
  mov rsi, rax
  mov rdi, rcx
  call _ZSt27__uninitialized_default_n_aIPNSt7__cxx119sub_matchIPKcEEmS4_ET_S6_T0_RSaIT1_E
  mov rdx, QWORD PTR [rbp-32]
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 3
  mov rdx, rax
  mov rax, QWORD PTR [rbp-56]
  add rax, rdx
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-72]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rcx, rax
  mov rax, QWORD PTR [rbp-72]
  mov rsi, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-72]
  mov rax, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZSt34__uninitialized_move_if_noexcept_aIPNSt7__cxx119sub_matchIPKcEES5_SaIS4_EET0_T_S8_S7_RT1_
  mov rax, QWORD PTR [rbp-72]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-72]
  mov rcx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-72]
  mov rax, QWORD PTR [rax]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt8_DestroyIPNSt7__cxx119sub_matchIPKcEES4_EvT_S6_RSaIT0_E
  mov rax, QWORD PTR [rbp-72]
  mov rdx, QWORD PTR [rbp-72]
  mov rcx, QWORD PTR [rdx+16]
  mov rdx, QWORD PTR [rbp-72]
  mov rdx, QWORD PTR [rdx]
  sub rcx, rdx
  mov rdx, rcx
  mov rcx, rdx
  sar rcx, 3
  movabs rdx, -6148914691236517205
  imul rdx, rcx
  mov rsi, rdx
  mov rdx, QWORD PTR [rbp-72]
  mov rcx, QWORD PTR [rdx]
  mov rdx, rsi
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE13_M_deallocateEPS4_m
  mov rax, QWORD PTR [rbp-72]
  mov rdx, QWORD PTR [rbp-56]
  mov QWORD PTR [rax], rdx
  mov rdx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-80]
  add rdx, rax
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 3
  mov rdx, rax
  mov rax, QWORD PTR [rbp-56]
  add rdx, rax
  mov rax, QWORD PTR [rbp-72]
  mov QWORD PTR [rax+8], rdx
  mov rdx, QWORD PTR [rbp-48]
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 3
  mov rdx, rax
  mov rax, QWORD PTR [rbp-56]
  add rdx, rax
  mov rax, QWORD PTR [rbp-72]
  mov QWORD PTR [rax+16], rdx
  jmp .L832
  mov rdi, rax
  call __cxa_begin_catch
  cmp QWORD PTR [rbp-24], 0
  je .L828
  mov rax, QWORD PTR [rbp-72]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rsi, rax
  mov rdx, QWORD PTR [rbp-80]
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 3
  mov rdx, rax
  mov rax, QWORD PTR [rbp-24]
  lea rcx, [rdx+rax]
  mov rax, QWORD PTR [rbp-24]
  mov rdx, rsi
  mov rsi, rcx
  mov rdi, rax
  call _ZSt8_DestroyIPNSt7__cxx119sub_matchIPKcEES4_EvT_S6_RSaIT0_E
.L828:
  mov rax, QWORD PTR [rbp-72]
  mov rdx, QWORD PTR [rbp-48]
  mov rcx, QWORD PTR [rbp-56]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE13_M_deallocateEPS4_m
  call __cxa_rethrow
  mov rbx, rax
  call __cxa_end_catch
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L832:
  nop
  add rsp, 72
  pop rbx
  pop rbp
  ret

_ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE15_M_erase_at_endEPS4_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+8]
  sub rax, QWORD PTR [rbp-32]
  sar rax, 3
  mov rdx, rax
  movabs rax, -6148914691236517205
  imul rax, rdx
  mov QWORD PTR [rbp-8], rax
  cmp QWORD PTR [rbp-8], 0
  je .L835
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-32]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt8_DestroyIPNSt7__cxx119sub_matchIPKcEES4_EvT_S6_RSaIT0_E
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rbp-32]
  mov QWORD PTR [rax+8], rdx
.L835:
  nop
  leave
  ret
_ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE12_Vector_implC1Ev
  nop
  leave
  ret
_ZN9__gnu_cxx13new_allocatorISt4pairIPKciEEC2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  nop
  pop rbp
  ret
_ZN9__gnu_cxx13new_allocatorISt4pairIPKciEED2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  nop
  pop rbp
  ret
_ZNSt12_Vector_baseISt4pairIPKciESaIS3_EEC2EmRKS4_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIPKciESaIS3_EE12_Vector_implC1ERKS4_
  mov rdx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIPKciESaIS3_EE17_M_create_storageEm
  jmp .L842
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIPKciESaIS3_EE12_Vector_implD1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L842:
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZNSt6vectorISt4pairIPKciESaIS3_EE21_M_default_initializeEm:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIPKciESaIS3_EE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  mov rcx, QWORD PTR [rbp-16]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt27__uninitialized_default_n_aIPSt4pairIPKciEmS3_ET_S5_T0_RSaIT1_E
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+8], rdx
  nop
  leave
  ret
_ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EEC2Ev
  nop
  leave
  ret
_ZNSt10unique_ptrIA_bSt14default_deleteIS0_EEC2IPbS2_vbEET_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt15__uniq_ptr_implIbSt14default_deleteIA_bEEC1EPb
  nop
  leave
  ret
_ZNSt12_Vector_baseISt4pairIPKciESaIS3_EE13_M_deallocateEPS3_m:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  cmp QWORD PTR [rbp-16], 0
  je .L848
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaISt4pairIPKciEEE10deallocateERS4_PS3_m
.L848:
  nop
  leave
  ret
_ZSt8_DestroyIPSt4pairIPKciEEvT_S5_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Destroy_auxILb1EE9__destroyIPSt4pairIPKciEEEvT_S7_
  nop
  leave
  ret
_ZNSaISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS5_EEEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEEED2Ev
  nop
  leave
  ret
_ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE13_M_deallocateEPS9_m:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  cmp QWORD PTR [rbp-16], 0
  je .L853
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEEE10deallocateERSA_PS9_m
.L853:
  nop
  leave
  ret
_ZSt8_DestroyIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEEvT_SB_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Destroy_auxILb0EE9__destroyIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS8_EEEEEvT_SD_
  nop
  leave
  ret
_ZSt3getILm0EJPbSt14default_deleteIA_bEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZSt12__get_helperILm0EPbJSt14default_deleteIA_bEEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE
  leave
  ret
_ZNSt15__uniq_ptr_implIbSt14default_deleteIA_bEE10_M_deleterEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZSt3getILm1EJPbSt14default_deleteIA_bEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE16_M_main_dispatchENS9_11_Match_modeESt17integral_constantIbLb0EE:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 104
  mov QWORD PTR [rbp-104], rdi
  mov eax, esi
  mov BYTE PTR [rbp-108], al
  mov rax, QWORD PTR [rbp-104]
  lea rcx, [rax+96]
  mov rax, QWORD PTR [rbp-104]
  mov rdx, QWORD PTR [rax+64]
  mov rax, QWORD PTR [rbp-104]
  mov rax, QWORD PTR [rax+128]
  mov rsi, rax
  mov rdi, rcx
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE11_State_infoISt17integral_constantIbLb0EESt6vectorIS5_S6_EE8_M_queueElRKSE_
  mov BYTE PTR [rbp-17], 0
.L867:
  mov rax, QWORD PTR [rbp-104]
  mov BYTE PTR [rax+140], 0
  mov rax, QWORD PTR [rbp-104]
  add rax, 96
  mov rdi, rax
  call _ZNKSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE5emptyEv
  test al, al
  jne .L873
  mov BYTE PTR [rbp-41], 0
  mov rax, QWORD PTR [rbp-104]
  mov rax, QWORD PTR [rax+56]
  add rax, 56
  mov rdi, rax
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EE4sizeEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-104]
  add rax, 120
  mov rdi, rax
  call _ZNKSt10unique_ptrIA_bSt14default_deleteIS0_EE3getEv
  mov rcx, rax
  lea rax, [rbp-41]
  mov rdx, rax
  mov rsi, rbx
  mov rdi, rcx
  call _ZSt6fill_nIPbmbET_S1_T0_RKT1_
  mov rax, QWORD PTR [rbp-104]
  add rax, 96
  mov rdi, rax
  call _ZSt4moveIRSt6vectorISt4pairIlS0_INSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EEEONSt16remove_referenceIT_E4typeEOSE_
  mov rdx, rax
  lea rax, [rbp-80]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EEC1EOSA_
  lea rax, [rbp-80]
  mov QWORD PTR [rbp-32], rax
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE5beginEv
  mov QWORD PTR [rbp-88], rax
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE3endEv
  mov QWORD PTR [rbp-96], rax
.L863:
  lea rdx, [rbp-96]
  lea rax, [rbp-88]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxxneIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEEbRKNS_17__normal_iteratorIT_T0_EESJ_
  test al, al
  je .L862
  lea rax, [rbp-88]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEdeEv
  mov QWORD PTR [rbp-40], rax
  mov rax, QWORD PTR [rbp-40]
  add rax, 8
  mov rdi, rax
  call _ZSt4moveIRSt6vectorINSt7__cxx119sub_matchIPKcEESaIS5_EEEONSt16remove_referenceIT_E4typeEOSA_
  mov rdx, rax
  mov rax, QWORD PTR [rbp-104]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEaSEOS6_
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax]
  movzx ecx, BYTE PTR [rbp-108]
  mov rax, QWORD PTR [rbp-104]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
  lea rax, [rbp-88]
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEppEv
  jmp .L863
.L862:
  cmp BYTE PTR [rbp-108], 1
  jne .L864
  mov rax, QWORD PTR [rbp-104]
  movzx eax, BYTE PTR [rax+140]
  or BYTE PTR [rbp-17], al
.L864:
  mov rax, QWORD PTR [rbp-104]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-104]
  mov rax, QWORD PTR [rax+40]
  cmp rdx, rax
  jne .L865
  mov ebx, 0
  jmp .L866
.L865:
  mov rax, QWORD PTR [rbp-104]
  mov rax, QWORD PTR [rax+24]
  lea rdx, [rax+1]
  mov rax, QWORD PTR [rbp-104]
  mov QWORD PTR [rax+24], rdx
  mov ebx, 1
.L866:
  lea rax, [rbp-80]
  mov rdi, rax
  call _ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EED1Ev
  cmp ebx, 1
  jne .L861
  jmp .L867
.L873:
  nop
.L861:
  cmp BYTE PTR [rbp-108], 0
  jne .L868
  mov rax, QWORD PTR [rbp-104]
  movzx eax, BYTE PTR [rax+140]
  mov BYTE PTR [rbp-17], al
.L868:
  mov rax, QWORD PTR [rbp-104]
  add rax, 96
  mov rdi, rax
  call _ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE5clearEv
  movzx eax, BYTE PTR [rbp-17]
  jmp .L872
  mov rbx, rax
  lea rax, [rbp-80]
  mov rdi, rax
  call _ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L872:
  add rsp, 104
  pop rbx
  pop rbp
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE16_M_main_dispatchENS9_11_Match_modeESt17integral_constantIbLb1EE:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov eax, esi
  mov BYTE PTR [rbp-12], al
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+116], 0
  mov rax, QWORD PTR [rbp-8]
  add rax, 96
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE11_State_infoISt17integral_constantIbLb1EESt6vectorIS5_S6_EE14_M_get_sol_posEv
  mov QWORD PTR [rax], 0
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+64]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEaSERKS6_
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+96]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-8]
  movzx eax, BYTE PTR [rax+116]
  leave
  ret
_ZN9__gnu_cxx13new_allocatorINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEEC2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  nop
  pop rbp
  ret
_ZN9__gnu_cxx13new_allocatorINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEED2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  nop
  pop rbp
  ret
_ZNSt10shared_ptrINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEEC2ISaIS5_EJRKSt6localeRNSt15regex_constants18syntax_option_typeEEEESt20_Sp_alloc_shared_tagIT_EDpOT0_:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov QWORD PTR [rbp-48], rcx
  mov rbx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZSt7forwardIRNSt15regex_constants18syntax_option_typeEEOT_RNSt16remove_referenceIS3_E4typeE
  mov r12, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZSt7forwardIRKSt6localeEOT_RNSt16remove_referenceIS3_E4typeE
  mov rdx, rax
  mov rax, QWORD PTR [rbp-32]
  mov rcx, r12
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt12__shared_ptrINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EEC2ISaIS5_EJRKSt6localeRNSt15regex_constants18syntax_option_typeEEEESt20_Sp_alloc_shared_tagIT_EDpOT0_
  nop
  add rsp, 32
  pop rbx
  pop r12
  pop rbp
  ret
_ZNKSt12__shared_ptrINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EE3getEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  pop rbp
  ret
.LC17:
  .string "basic_string::erase"
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5eraseEmm:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov edx, OFFSET FLAT:.LC17
  mov rsi, rcx
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8_M_checkEmPKc
  cmp QWORD PTR [rbp-24], -1
  jne .L882
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm
  jmp .L883
.L882:
  cmp QWORD PTR [rbp-24], 0
  je .L883
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8_M_limitEmm
  mov rdx, rax
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8_M_eraseEmm
.L883:
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSt8__detail8_ScannerIcE17_M_eat_escape_awkEv:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-40], rdi
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-40]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  mov BYTE PTR [rbp-21], al
  mov rbx, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+192]
  movsx ecx, BYTE PTR [rbp-21]
  mov edx, 0
  mov esi, ecx
  mov rdi, rax
  call _ZNKSt5ctypeIcE6narrowEcc
  movsx eax, al
  mov esi, eax
  mov rdi, rbx
  call _ZNSt8__detail12_ScannerBase14_M_find_escapeEc
  mov QWORD PTR [rbp-32], rax
  cmp QWORD PTR [rbp-32], 0
  je .L886
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 1
  mov rax, QWORD PTR [rbp-40]
  lea rcx, [rax+200]
  mov rax, QWORD PTR [rbp-32]
  movzx eax, BYTE PTR [rax]
  movsx eax, al
  mov edx, eax
  mov esi, 1
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  jmp .L885
.L886:
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+192]
  movsx edx, BYTE PTR [rbp-21]
  mov esi, 2048
  mov rdi, rax
  call _ZNKSt5ctypeIcE2isEtc
  test al, al
  je .L888
  cmp BYTE PTR [rbp-21], 56
  je .L888
  cmp BYTE PTR [rbp-21], 57
  je .L888
  mov eax, 1
  jmp .L889
.L888:
  mov eax, 0
.L889:
  test al, al
  je .L890
  mov rax, QWORD PTR [rbp-40]
  lea rcx, [rax+200]
  movsx eax, BYTE PTR [rbp-21]
  mov edx, eax
  mov esi, 1
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  mov DWORD PTR [rbp-20], 0
.L894:
  cmp DWORD PTR [rbp-20], 1
  jg .L891
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  je .L891
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+192]
  mov rdx, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rdx+176]
  movzx edx, BYTE PTR [rdx]
  movsx edx, dl
  mov esi, 2048
  mov rdi, rax
  call _ZNKSt5ctypeIcE2isEtc
  test al, al
  je .L891
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  movzx eax, BYTE PTR [rax]
  cmp al, 56
  je .L891
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  movzx eax, BYTE PTR [rax]
  cmp al, 57
  je .L891
  mov eax, 1
  jmp .L892
.L891:
  mov eax, 0
.L892:
  test al, al
  je .L893
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-40]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  movsx eax, al
  mov rdx, QWORD PTR [rbp-40]
  add rdx, 200
  mov esi, eax
  mov rdi, rdx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEpLEc
  add DWORD PTR [rbp-20], 1
  jmp .L894
.L893:
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 2
  jmp .L885
.L890:
  mov esi, OFFSET FLAT:.LC14
  mov edi, 2
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L885:
  add rsp, 40
  pop rbx
  pop rbp
  ret
.LC18:
  .string "Unexpected end of regex when in an open parenthesis."
.LC19:
  .string "Invalid special open parenthesis."
_ZNSt8__detail8_ScannerIcE14_M_scan_normalEv:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-40], rdi
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-40]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  mov BYTE PTR [rbp-1], al
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+192]
  movsx ecx, BYTE PTR [rbp-1]
  mov edx, 32
  mov esi, ecx
  mov rdi, rax
  call _ZNKSt5ctypeIcE6narrowEcc
  movsx edx, al
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+160]
  mov esi, edx
  mov rdi, rax
  call strchr
  test rax, rax
  sete al
  test al, al
  je .L896
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 1
  mov rax, QWORD PTR [rbp-40]
  lea rcx, [rax+200]
  movsx eax, BYTE PTR [rbp-1]
  mov edx, eax
  mov esi, 1
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  jmp .L895
.L896:
  cmp BYTE PTR [rbp-1], 92
  jne .L898
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  jne .L899
  mov esi, OFFSET FLAT:.LC11
  mov edi, 2
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L899:
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt8__detail12_ScannerBase11_M_is_basicEv
  xor eax, 1
  test al, al
  jne .L900
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  movzx eax, BYTE PTR [rax]
  cmp al, 40
  je .L901
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  movzx eax, BYTE PTR [rax]
  cmp al, 41
  je .L901
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  movzx eax, BYTE PTR [rax]
  cmp al, 123
  je .L901
.L900:
  mov eax, 1
  jmp .L902
.L901:
  mov eax, 0
.L902:
  test al, al
  je .L903
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+240]
  mov rdx, rax
  mov rax, QWORD PTR [rbp-40]
  add rdx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+232]
  and eax, 1
  test rax, rax
  je .L904
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+240]
  mov rcx, rax
  mov rax, QWORD PTR [rbp-40]
  add rax, rcx
  mov rcx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+232]
  sub rax, 1
  add rax, rcx
  mov rax, QWORD PTR [rax]
  jmp .L905
.L904:
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+232]
.L905:
  mov rdi, rdx
  call rax
  jmp .L895
.L903:
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-40]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  mov BYTE PTR [rbp-1], al
.L898:
  cmp BYTE PTR [rbp-1], 40
  jne .L906
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt8__detail12_ScannerBase10_M_is_ecmaEv
  test al, al
  je .L907
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  movzx eax, BYTE PTR [rax]
  cmp al, 63
  jne .L907
  mov eax, 1
  jmp .L908
.L907:
  mov eax, 0
.L908:
  test al, al
  je .L909
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  lea rdx, [rax+1]
  mov rax, QWORD PTR [rbp-40]
  mov QWORD PTR [rax+176], rdx
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  sete al
  test al, al
  je .L910
  mov esi, OFFSET FLAT:.LC18
  mov edi, 5
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L910:
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  movzx eax, BYTE PTR [rax]
  cmp al, 58
  jne .L911
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  lea rdx, [rax+1]
  mov rax, QWORD PTR [rbp-40]
  mov QWORD PTR [rax+176], rdx
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 6
  jmp .L895
.L911:
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  movzx eax, BYTE PTR [rax]
  cmp al, 61
  jne .L913
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  lea rdx, [rax+1]
  mov rax, QWORD PTR [rbp-40]
  mov QWORD PTR [rax+176], rdx
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 7
  mov rax, QWORD PTR [rbp-40]
  add rax, 200
  mov edx, 112
  mov esi, 1
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  jmp .L895
.L913:
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  movzx eax, BYTE PTR [rax]
  cmp al, 33
  jne .L914
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  lea rdx, [rax+1]
  mov rax, QWORD PTR [rbp-40]
  mov QWORD PTR [rax+176], rdx
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 7
  mov rax, QWORD PTR [rbp-40]
  add rax, 200
  mov edx, 110
  mov esi, 1
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  jmp .L895
.L914:
  mov esi, OFFSET FLAT:.LC19
  mov edi, 5
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
  jmp .L895
.L909:
  mov rax, QWORD PTR [rbp-40]
  mov eax, DWORD PTR [rax+140]
  mov esi, 2
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  setne al
  test al, al
  je .L916
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 6
  jmp .L895
.L916:
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 5
  jmp .L895
.L906:
  cmp BYTE PTR [rbp-1], 41
  jne .L917
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 8
  jmp .L895
.L917:
  cmp BYTE PTR [rbp-1], 91
  jne .L918
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+136], 2
  mov rax, QWORD PTR [rbp-40]
  mov BYTE PTR [rax+168], 1
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  je .L919
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  movzx eax, BYTE PTR [rax]
  cmp al, 94
  jne .L919
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 10
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+176]
  lea rdx, [rax+1]
  mov rax, QWORD PTR [rbp-40]
  mov QWORD PTR [rax+176], rdx
  jmp .L895
.L919:
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 9
  jmp .L895
.L918:
  cmp BYTE PTR [rbp-1], 123
  jne .L921
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+136], 1
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 12
  jmp .L895
.L921:
  cmp BYTE PTR [rbp-1], 93
  je .L922
  cmp BYTE PTR [rbp-1], 125
  je .L922
  mov rax, QWORD PTR [rbp-40]
  mov QWORD PTR [rbp-16], rax
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+192]
  movsx ecx, BYTE PTR [rbp-1]
  mov edx, 0
  mov esi, ecx
  mov rdi, rax
  call _ZNKSt5ctypeIcE6narrowEcc
  mov BYTE PTR [rbp-17], al
.L925:
  mov rax, QWORD PTR [rbp-16]
  movzx eax, BYTE PTR [rax]
  test al, al
  je .L926
  mov rax, QWORD PTR [rbp-16]
  movzx eax, BYTE PTR [rax]
  cmp BYTE PTR [rbp-17], al
  jne .L924
  mov rax, QWORD PTR [rbp-16]
  mov edx, DWORD PTR [rax+4]
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], edx
  jmp .L895
.L924:
  add QWORD PTR [rbp-16], 8
  jmp .L925
.L922:
  mov rax, QWORD PTR [rbp-40]
  mov DWORD PTR [rax+144], 1
  mov rax, QWORD PTR [rbp-40]
  lea rcx, [rax+200]
  movsx eax, BYTE PTR [rbp-1]
  mov edx, eax
  mov esi, 1
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  jmp .L895
.L926:
  nop
.L895:
  leave
  ret
.LC20:
  .string "Unexpected end of regex when in bracket expression."
.LC21:
  .string "Unexpected character class open bracket."
_ZNSt8__detail8_ScannerIcE18_M_scan_in_bracketEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  jne .L928
  mov esi, OFFSET FLAT:.LC20
  mov edi, 4
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L928:
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  mov BYTE PTR [rbp-1], al
  cmp BYTE PTR [rbp-1], 45
  jne .L929
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax+144], 28
  jmp .L930
.L929:
  cmp BYTE PTR [rbp-1], 91
  jne .L931
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  jne .L932
  mov esi, OFFSET FLAT:.LC21
  mov edi, 4
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L932:
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+176]
  movzx eax, BYTE PTR [rax]
  cmp al, 46
  jne .L933
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax+144], 16
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  movsx edx, al
  mov rax, QWORD PTR [rbp-24]
  mov esi, edx
  mov rdi, rax
  call _ZNSt8__detail8_ScannerIcE12_M_eat_classEc
  jmp .L930
.L933:
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+176]
  movzx eax, BYTE PTR [rax]
  cmp al, 58
  jne .L935
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax+144], 15
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  movsx edx, al
  mov rax, QWORD PTR [rbp-24]
  mov esi, edx
  mov rdi, rax
  call _ZNSt8__detail8_ScannerIcE12_M_eat_classEc
  jmp .L930
.L935:
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+176]
  movzx eax, BYTE PTR [rax]
  cmp al, 61
  jne .L936
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax+144], 17
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  movsx edx, al
  mov rax, QWORD PTR [rbp-24]
  mov esi, edx
  mov rdi, rax
  call _ZNSt8__detail8_ScannerIcE12_M_eat_classEc
  jmp .L930
.L936:
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax+144], 1
  mov rax, QWORD PTR [rbp-24]
  lea rcx, [rax+200]
  movsx eax, BYTE PTR [rbp-1]
  mov edx, eax
  mov esi, 1
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  jmp .L930
.L931:
  cmp BYTE PTR [rbp-1], 93
  jne .L937
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt8__detail12_ScannerBase10_M_is_ecmaEv
  test al, al
  jne .L938
  mov rax, QWORD PTR [rbp-24]
  movzx eax, BYTE PTR [rax+168]
  xor eax, 1
  test al, al
  je .L937
.L938:
  mov eax, 1
  jmp .L939
.L937:
  mov eax, 0
.L939:
  test al, al
  je .L940
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax+144], 11
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax+136], 0
  jmp .L930
.L940:
  cmp BYTE PTR [rbp-1], 92
  jne .L941
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt8__detail12_ScannerBase10_M_is_ecmaEv
  test al, al
  jne .L942
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt8__detail12_ScannerBase9_M_is_awkEv
  test al, al
  je .L941
.L942:
  mov eax, 1
  jmp .L943
.L941:
  mov eax, 0
.L943:
  test al, al
  je .L944
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+240]
  mov rdx, rax
  mov rax, QWORD PTR [rbp-24]
  add rdx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+232]
  and eax, 1
  test rax, rax
  je .L945
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+240]
  mov rcx, rax
  mov rax, QWORD PTR [rbp-24]
  add rax, rcx
  mov rcx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+232]
  sub rax, 1
  add rax, rcx
  mov rax, QWORD PTR [rax]
  jmp .L946
.L945:
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+232]
.L946:
  mov rdi, rdx
  call rax
  jmp .L930
.L944:
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax+144], 1
  mov rax, QWORD PTR [rbp-24]
  lea rcx, [rax+200]
  movsx eax, BYTE PTR [rbp-1]
  mov edx, eax
  mov esi, 1
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
.L930:
  mov rax, QWORD PTR [rbp-24]
  mov BYTE PTR [rax+168], 0
  nop
  leave
  ret
.LC22:
  .string "Unexpected end of regex when in brace expression."
.LC23:
  .string "Unexpected character in brace expression."
_ZNSt8__detail8_ScannerIcE16_M_scan_in_braceEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  jne .L948
  mov esi, OFFSET FLAT:.LC22
  mov edi, 6
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L948:
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  mov BYTE PTR [rbp-1], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+192]
  movsx edx, BYTE PTR [rbp-1]
  mov esi, 2048
  mov rdi, rax
  call _ZNKSt5ctypeIcE2isEtc
  test al, al
  je .L949
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax+144], 26
  mov rax, QWORD PTR [rbp-24]
  lea rcx, [rax+200]
  movsx eax, BYTE PTR [rbp-1]
  mov edx, eax
  mov esi, 1
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
.L953:
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  je .L950
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+192]
  mov rdx, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rdx+176]
  movzx edx, BYTE PTR [rdx]
  movsx edx, dl
  mov esi, 2048
  mov rdi, rax
  call _ZNKSt5ctypeIcE2isEtc
  test al, al
  je .L950
  mov eax, 1
  jmp .L951
.L950:
  mov eax, 0
.L951:
  test al, al
  je .L959
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  movsx eax, al
  mov rdx, QWORD PTR [rbp-24]
  add rdx, 200
  mov esi, eax
  mov rdi, rdx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEpLEc
  jmp .L953
.L949:
  cmp BYTE PTR [rbp-1], 44
  jne .L954
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax+144], 25
  jmp .L959
.L954:
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt8__detail12_ScannerBase11_M_is_basicEv
  test al, al
  je .L955
  cmp BYTE PTR [rbp-1], 92
  jne .L956
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  je .L956
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+176]
  movzx eax, BYTE PTR [rax]
  cmp al, 125
  jne .L956
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax+136], 0
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax+144], 13
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+176]
  lea rdx, [rax+1]
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+176], rdx
  jmp .L959
.L956:
  mov esi, OFFSET FLAT:.LC23
  mov edi, 7
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
  jmp .L959
.L955:
  cmp BYTE PTR [rbp-1], 125
  jne .L958
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax+136], 0
  mov rax, QWORD PTR [rbp-24]
  mov DWORD PTR [rax+144], 13
  jmp .L959
.L958:
  mov esi, OFFSET FLAT:.LC23
  mov edi, 7
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L959:
  nop
  leave
  ret
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EEC2Ev:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE11_Deque_implC1Ev
  mov rax, QWORD PTR [rbp-24]
  mov esi, 0
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE17_M_initialize_mapEm
  jmp .L963
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE11_Deque_implD1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L963:
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZNSt16allocator_traitsISaImEE9constructImJRKmEEEvRS0_PT_DpOT0_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt7forwardIRKmEOT_RNSt16remove_referenceIS2_E4typeE
  mov rdx, rax
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorImE9constructImJRKmEEEvPT_DpOT0_
  nop
  leave
  ret
_ZNSt6vectorImSaImEE3endEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  lea rdx, [rax+8]
  lea rax, [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEC1ERKS1_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
.LC24:
  .string "vector::_M_realloc_insert"
_ZNSt6vectorImSaImEE17_M_realloc_insertIJRKmEEEvN9__gnu_cxx17__normal_iteratorIPmS1_EEDpOT_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 104
  mov QWORD PTR [rbp-88], rdi
  mov QWORD PTR [rbp-96], rsi
  mov QWORD PTR [rbp-104], rdx
  mov rax, QWORD PTR [rbp-88]
  mov edx, OFFSET FLAT:.LC24
  mov esi, 1
  mov rdi, rax
  call _ZNKSt6vectorImSaImEE12_M_check_lenEmPKc
  mov QWORD PTR [rbp-32], rax
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-40], rax
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+8]
  mov QWORD PTR [rbp-48], rax
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt6vectorImSaImEE5beginEv
  mov QWORD PTR [rbp-72], rax
  lea rdx, [rbp-72]
  lea rax, [rbp-96]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxxmiIPmSt6vectorImSaImEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_
  mov QWORD PTR [rbp-56], rax
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-32]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Vector_baseImSaImEE11_M_allocateEm
  mov QWORD PTR [rbp-64], rax
  mov rax, QWORD PTR [rbp-64]
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-104]
  mov rdi, rax
  call _ZSt7forwardIRKmEOT_RNSt16remove_referenceIS2_E4typeE
  mov rsi, rax
  mov rax, QWORD PTR [rbp-56]
  lea rdx, [0+rax*8]
  mov rax, QWORD PTR [rbp-64]
  lea rcx, [rdx+rax]
  mov rax, QWORD PTR [rbp-88]
  mov rdx, rsi
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaImEE9constructImJRKmEEEvRS0_PT_DpOT0_
  mov QWORD PTR [rbp-24], 0
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt12_Vector_baseImSaImEE19_M_get_Tp_allocatorEv
  mov rbx, rax
  lea rax, [rbp-96]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEE4baseEv
  mov rsi, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-64]
  mov rax, QWORD PTR [rbp-40]
  mov rcx, rbx
  mov rdi, rax
  call _ZSt34__uninitialized_move_if_noexcept_aIPmS0_SaImEET0_T_S3_S2_RT1_
  mov QWORD PTR [rbp-24], rax
  add QWORD PTR [rbp-24], 8
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt12_Vector_baseImSaImEE19_M_get_Tp_allocatorEv
  mov rbx, rax
  lea rax, [rbp-96]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEE4baseEv
  mov rax, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-24]
  mov rsi, QWORD PTR [rbp-48]
  mov rcx, rbx
  mov rdi, rax
  call _ZSt34__uninitialized_move_if_noexcept_aIPmS0_SaImEET0_T_S3_S2_RT1_
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt12_Vector_baseImSaImEE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rcx, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt8_DestroyIPmmEvT_S1_RSaIT0_E
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rdx+16]
  sub rdx, QWORD PTR [rbp-40]
  sar rdx, 3
  mov rcx, QWORD PTR [rbp-40]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt12_Vector_baseImSaImEE13_M_deallocateEPmm
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-64]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+8], rdx
  mov rax, QWORD PTR [rbp-32]
  lea rdx, [0+rax*8]
  mov rax, QWORD PTR [rbp-64]
  add rdx, rax
  mov rax, QWORD PTR [rbp-88]
  mov QWORD PTR [rax+16], rdx
  jmp .L974
  mov rdi, rax
  call __cxa_begin_catch
  cmp QWORD PTR [rbp-24], 0
  jne .L969
  mov rax, QWORD PTR [rbp-56]
  lea rdx, [0+rax*8]
  mov rax, QWORD PTR [rbp-64]
  add rdx, rax
  mov rax, QWORD PTR [rbp-88]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaImEE7destroyImEEvRS0_PT_
  jmp .L970
.L969:
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt12_Vector_baseImSaImEE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rcx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-64]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt8_DestroyIPmmEvT_S1_RSaIT0_E
.L970:
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-32]
  mov rcx, QWORD PTR [rbp-64]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt12_Vector_baseImSaImEE13_M_deallocateEPmm
  call __cxa_rethrow
  mov rbx, rax
  call __cxa_end_catch
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L974:
  add rsp, 104
  pop rbx
  pop rbp
  ret

_ZN9__gnu_cxx16__aligned_membufISt8functionIFbcEEE7_M_addrEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt8functionIFbcEEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt14_Function_baseC2Ev
  nop
  leave
  ret
_ZNSt8__detail6_StateIcE14_M_get_matcherEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 16
  mov rdi, rax
  call _ZN9__gnu_cxx16__aligned_membufISt8functionIFbcEEE7_M_addrEv
  leave
  ret
_ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EE9push_backEOS2_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZSt4moveIRNSt8__detail6_StateIcEEEONSt16remove_referenceIT_E4typeEOS5_
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EE12emplace_backIJS2_EEERS2_DpOT_
  nop
  leave
  ret
_ZSt4moveIRSt8functionIFbcEEEONSt16remove_referenceIT_E4typeEOS5_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt8functionIFbcEEC2EOS1_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt14_Function_baseC2Ev
  mov rdx, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8functionIFbcEE4swapERS1_
  nop
  leave
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE7_M_termEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE12_M_assertionEv
  test al, al
  je .L985
  mov eax, 1
  jmp .L986
.L985:
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE7_M_atomEv
  test al, al
  je .L987
.L989:
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE13_M_quantifierEv
  test al, al
  je .L988
  jmp .L989
.L988:
  mov eax, 1
  jmp .L986
.L987:
  mov eax, 0
.L986:
  leave
  ret
_ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushERKS5_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE9push_backERKS5_
  nop
  leave
  ret
_ZSt4moveIRNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEONSt16remove_referenceIT_E4typeEOS8_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE9push_backEOS5_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZSt4moveIRNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEONSt16remove_referenceIT_E4typeEOS8_
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE12emplace_backIJS5_EEERS5_DpOT_
  nop
  leave
  ret
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE4backEv:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-40], rdi
  lea rax, [rbp-32]
  mov rdx, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE3endEv
  lea rax, [rbp-32]
  mov rdi, rax
  call _ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_EmmEv
  lea rax, [rbp-32]
  mov rdi, rax
  call _ZNKSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_EdeEv
  nop
  leave
  ret
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE8pop_backEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+48]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+56]
  cmp rdx, rax
  je .L997
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+48]
  lea rdx, [rax-24]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+48], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+48]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEE7destroyIS5_EEvRS6_PT_
  jmp .L999
.L997:
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE15_M_pop_back_auxEv
.L999:
  nop
  leave
  ret
_ZNK9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEmiEl:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-32]
  sal rdx, 3
  neg rdx
  add rax, rdx
  mov QWORD PTR [rbp-8], rax
  lea rdx, [rbp-8]
  lea rax, [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEC1ERKS1_
  mov rax, QWORD PTR [rbp-16]
  leave
  ret
_ZNK9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEdeEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  pop rbp
  ret
_ZNSt16allocator_traitsISaImEE7destroyImEEvRS0_PT_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorImE7destroyImEEvPT_
  nop
  leave
  ret
_ZN9__gnu_cxx17__normal_iteratorIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEEC2ERKS4_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  nop
  pop rbp
  ret
_ZNK9__gnu_cxx17__normal_iteratorIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEE4baseEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZN9__gnu_cxx13new_allocatorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEED2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  nop
  pop rbp
  ret
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE18_M_deallocate_nodeEPS5_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov edi, 24
  call _ZSt16__deque_buf_sizem
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rbp-16]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEE10deallocateERS6_PS5_m
  nop
  leave
  ret
_ZNKSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE20_M_get_map_allocatorEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZNKSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSaIPNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEC1IS4_EERKSaIT_E
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSaIPNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorIPNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEED2Ev
  nop
  leave
  ret
_ZNSt16allocator_traitsISaIPNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEE10deallocateERS7_PS6_m:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorIPNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEE10deallocateEPS7_m
  nop
  leave
  ret
_ZN9__gnu_cxx13new_allocatorINSt7__cxx119sub_matchIPKcEEEC2ERKS6_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  nop
  pop rbp
  ret
_ZN9__gnu_cxx13new_allocatorINSt7__cxx119sub_matchIPKcEEE10deallocateEPS5_m:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZdlPv
  leave
  ret
_ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE8max_sizeEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rdi, rax
  call _ZNSt16allocator_traitsISaINSt7__cxx119sub_matchIPKcEEEE8max_sizeERKS5_
  leave
  ret
_ZSt27__uninitialized_default_n_aIPNSt7__cxx119sub_matchIPKcEEmS4_ET_S6_T0_RSaIT1_E:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt25__uninitialized_default_nIPNSt7__cxx119sub_matchIPKcEEmET_S6_T0_
  leave
  ret
_ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE12_M_check_lenEmS3_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 56
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  mov QWORD PTR [rbp-56], rdx
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE8max_sizeEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv
  sub rbx, rax
  mov rdx, rbx
  mov rax, QWORD PTR [rbp-48]
  cmp rdx, rax
  setb al
  test al, al
  je .L1022
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZSt20__throw_length_errorPKc
.L1022:
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv
  mov QWORD PTR [rbp-32], rax
  lea rdx, [rbp-48]
  lea rax, [rbp-32]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt3maxImERKT_S2_S2_
  mov rax, QWORD PTR [rax]
  add rax, rbx
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv
  cmp QWORD PTR [rbp-24], rax
  jb .L1023
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE8max_sizeEv
  cmp QWORD PTR [rbp-24], rax
  jbe .L1024
.L1023:
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE8max_sizeEv
  jmp .L1025
.L1024:
  mov rax, QWORD PTR [rbp-24]
.L1025:
  add rsp, 56
  pop rbx
  pop rbp
  ret
_ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE11_M_allocateEm:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  cmp QWORD PTR [rbp-16], 0
  je .L1028
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaINSt7__cxx119sub_matchIPKcEEEE8allocateERS5_m
  jmp .L1030
.L1028:
  mov eax, 0
.L1030:
  leave
  ret
_ZSt34__uninitialized_move_if_noexcept_aIPNSt7__cxx119sub_matchIPKcEES5_SaIS4_EET0_T_S8_S7_RT1_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov QWORD PTR [rbp-48], rcx
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt32__make_move_if_noexcept_iteratorINSt7__cxx119sub_matchIPKcEESt13move_iteratorIPS4_EET0_PT_
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt32__make_move_if_noexcept_iteratorINSt7__cxx119sub_matchIPKcEESt13move_iteratorIPS4_EET0_PT_
  mov rdi, rax
  mov rdx, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rbp-40]
  mov rcx, rdx
  mov rdx, rax
  mov rsi, rbx
  call _ZSt22__uninitialized_copy_aISt13move_iteratorIPNSt7__cxx119sub_matchIPKcEEES6_S5_ET0_T_S9_S8_RSaIT1_E
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE12_Vector_implC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSaINSt7__cxx119sub_matchIPKcEEEC2Ev
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+8], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+16], 0
  nop
  leave
  ret
_ZNSt12_Vector_baseISt4pairIPKciESaIS3_EE12_Vector_implC2ERKS4_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSaISt4pairIPKciEEC2ERKS3_
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+8], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+16], 0
  nop
  leave
  ret
_ZNSt12_Vector_baseISt4pairIPKciESaIS3_EE17_M_create_storageEm:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIPKciESaIS3_EE11_M_allocateEm
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+8], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-16]
  sal rdx, 4
  add rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+16], rdx
  nop
  leave
  ret
_ZSt27__uninitialized_default_n_aIPSt4pairIPKciEmS3_ET_S5_T0_RSaIT1_E:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt25__uninitialized_default_nIPSt4pairIPKciEmET_S5_T0_
  leave
  ret
_ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE12_Vector_implC1Ev
  nop
  leave
  ret
_ZNSt15__uniq_ptr_implIbSt14default_deleteIA_bEEC2EPb:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt5tupleIJPbSt14default_deleteIA_bEEEC1IS0_S3_Lb1EEEv
  mov rbx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt15__uniq_ptr_implIbSt14default_deleteIA_bEE6_M_ptrEv
  mov QWORD PTR [rax], rbx
  nop
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZNSt16allocator_traitsISaISt4pairIPKciEEE10deallocateERS4_PS3_m:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorISt4pairIPKciEE10deallocateEPS4_m
  nop
  leave
  ret
_ZNSt12_Destroy_auxILb1EE9__destroyIPSt4pairIPKciEEEvT_S7_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  nop
  pop rbp
  ret
_ZN9__gnu_cxx13new_allocatorISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEEED2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  nop
  pop rbp
  ret
_ZNSt16allocator_traitsISaISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEEE10deallocateERSA_PS9_m:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEEE10deallocateEPSA_m
  nop
  leave
  ret
_ZNSt12_Destroy_auxILb0EE9__destroyIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS8_EEEEEvT_SD_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
.L1046:
  mov rax, QWORD PTR [rbp-8]
  cmp rax, QWORD PTR [rbp-16]
  je .L1047
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZSt11__addressofISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEEPT_RSA_
  mov rdi, rax
  call _ZSt8_DestroyISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEEvPT_
  add QWORD PTR [rbp-8], 32
  jmp .L1046
.L1047:
  nop
  leave
  ret
_ZSt12__get_helperILm0EPbJSt14default_deleteIA_bEEERT0_RSt11_Tuple_implIXT_EJS4_DpT1_EE:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt11_Tuple_implILm0EJPbSt14default_deleteIA_bEEE7_M_headERS4_
  leave
  ret
_ZSt3getILm1EJPbSt14default_deleteIA_bEEERNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERS8_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZSt12__get_helperILm1ESt14default_deleteIA_bEJEERT0_RSt11_Tuple_implIXT_EJS3_DpT1_EE
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE11_State_infoISt17integral_constantIbLb0EESt6vectorIS5_S6_EE8_M_queueElRKSE_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-24]
  lea rcx, [rbp-16]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE12emplace_backIJRlRKS7_EEERS8_DpOT_
  nop
  leave
  ret
_ZNKSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE5emptyEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE3endEv
  mov QWORD PTR [rbp-16], rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE5beginEv
  mov QWORD PTR [rbp-8], rax
  lea rdx, [rbp-16]
  lea rax, [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxxeqIPKSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEEbRKNS_17__normal_iteratorIT_T0_EESK_
  nop
  leave
  ret
_ZNKSt10unique_ptrIA_bSt14default_deleteIS0_EE3getEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt15__uniq_ptr_implIbSt14default_deleteIA_bEE6_M_ptrEv
  leave
  ret
_ZSt6fill_nIPbmbET_S1_T0_RKT1_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZSt12__niter_baseIPbET_S1_
  mov rcx, rax
  mov rdx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-16]
  mov rsi, rax
  mov rdi, rcx
  call _ZSt10__fill_n_aIPbmbEN9__gnu_cxx11__enable_ifIXsrSt11__is_scalarIT1_E7__valueET_E6__typeES6_T0_RKS4_
  leave
  ret
_ZSt4moveIRSt6vectorISt4pairIlS0_INSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EEEONSt16remove_referenceIT_E4typeEOSE_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EEC2EOSA_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rbx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt4moveIRSt6vectorISt4pairIlS0_INSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EEEONSt16remove_referenceIT_E4typeEOSE_
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EEC2EOSB_
  nop
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE5beginEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEC1ERKSB_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE3endEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  lea rdx, [rax+8]
  lea rax, [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEC1ERKSB_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZN9__gnu_cxxneIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEEbRKNS_17__normal_iteratorIT_T0_EESJ_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEE4baseEv
  mov rbx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEE4baseEv
  mov rax, QWORD PTR [rax]
  cmp rbx, rax
  setne al
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZN9__gnu_cxx17__normal_iteratorIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEppEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  lea rdx, [rax+32]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNK9__gnu_cxx17__normal_iteratorIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEdeEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  pop rbp
  ret
_ZSt4moveIRSt6vectorINSt7__cxx119sub_matchIPKcEESaIS5_EEEONSt16remove_referenceIT_E4typeEOSA_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEaSEOS6_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov BYTE PTR [rbp-1], 1
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt4moveIRSt6vectorINSt7__cxx119sub_matchIPKcEESaIS5_EEEONSt16remove_referenceIT_E4typeEOSA_
  mov rdx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE14_M_move_assignEOS6_St17integral_constantIbLb1EE
  mov rax, QWORD PTR [rbp-24]
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov eax, esi
  mov QWORD PTR [rbp-24], rdx
  mov BYTE PTR [rbp-12], al
  mov rax, QWORD PTR [rbp-8]
  lea rdx, [rax+96]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE11_State_infoISt17integral_constantIbLb0EESt6vectorIS5_S6_EE10_M_visitedEl
  test al, al
  jne .L1091
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov rdi, rax
  call _ZNKSt8__detail6_StateIcE9_M_opcodeEv
  cmp eax, 12
  ja .L1076
  mov eax, eax
  mov rax, QWORD PTR .L1080[0+rax*8]
  jmp rax
.L1080:
  .quad .L1076
  .quad .L1090
  .quad .L1089
  .quad .L1088
  .quad .L1087
  .quad .L1086
  .quad .L1085
  .quad .L1084
  .quad .L1083
  .quad .L1082
  .quad .L1076
  .quad .L1081
  .quad .L1079
.L1089:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE16_M_handle_repeatENS9_11_Match_modeEl
  jmp .L1076
.L1083:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE23_M_handle_subexpr_beginENS9_11_Match_modeEl
  jmp .L1076
.L1082:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE21_M_handle_subexpr_endENS9_11_Match_modeEl
  jmp .L1076
.L1087:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE30_M_handle_line_begin_assertionENS9_11_Match_modeEl
  jmp .L1076
.L1086:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE28_M_handle_line_end_assertionENS9_11_Match_modeEl
  jmp .L1076
.L1085:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE23_M_handle_word_boundaryENS9_11_Match_modeEl
  jmp .L1076
.L1084:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE27_M_handle_subexpr_lookaheadENS9_11_Match_modeEl
  jmp .L1076
.L1081:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE15_M_handle_matchENS9_11_Match_modeEl
  jmp .L1076
.L1088:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE17_M_handle_backrefENS9_11_Match_modeEl
  jmp .L1076
.L1079:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE16_M_handle_acceptENS9_11_Match_modeEl
  jmp .L1076
.L1090:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE21_M_handle_alternativeENS9_11_Match_modeEl
  nop
  jmp .L1076
.L1091:
  nop
.L1076:
  leave
  ret
_ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE5clearEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE15_M_erase_at_endEPS8_
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE11_State_infoISt17integral_constantIbLb1EESt6vectorIS5_S6_EE14_M_get_sol_posEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 8
  pop rbp
  ret
_ZN9__gnu_cxx14__alloc_traitsISaINSt7__cxx119sub_matchIPKcEEES5_E27_S_propagate_on_copy_assignEv:
  push rbp
  mov rbp, rsp
  mov eax, 0
  pop rbp
  ret
_ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEaSERKS6_:
  push rbp
  mov rbp, rsp
  push r14
  push r13
  push r12
  push rbx
  sub rsp, 32
  mov QWORD PTR [rbp-56], rdi
  mov QWORD PTR [rbp-64], rsi
  mov rax, QWORD PTR [rbp-64]
  cmp rax, QWORD PTR [rbp-56]
  je .L1098
  call _ZN9__gnu_cxx14__alloc_traitsISaINSt7__cxx119sub_matchIPKcEEES5_E27_S_propagate_on_copy_assignEv
  test al, al
  je .L1099
  call _ZN9__gnu_cxx14__alloc_traitsISaINSt7__cxx119sub_matchIPKcEEES5_E15_S_always_equalEv
  xor eax, 1
  test al, al
  je .L1100
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZNKSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rsi, rbx
  mov rdi, rax
  call _ZStneINSt7__cxx119sub_matchIPKcEEEbRKSaIT_ES8_
  test al, al
  je .L1100
  mov eax, 1
  jmp .L1101
.L1100:
  mov eax, 0
.L1101:
  test al, al
  je .L1102
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE5clearEv
  mov rax, QWORD PTR [rbp-56]
  mov rdx, QWORD PTR [rbp-56]
  mov rcx, QWORD PTR [rdx+16]
  mov rdx, QWORD PTR [rbp-56]
  mov rdx, QWORD PTR [rdx]
  sub rcx, rdx
  mov rdx, rcx
  mov rcx, rdx
  sar rcx, 3
  movabs rdx, -6148914691236517205
  imul rdx, rcx
  mov rsi, rdx
  mov rdx, QWORD PTR [rbp-56]
  mov rcx, QWORD PTR [rdx]
  mov rdx, rsi
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE13_M_deallocateEPS4_m
  mov rax, QWORD PTR [rbp-56]
  mov QWORD PTR [rax], 0
  mov rax, QWORD PTR [rbp-56]
  mov QWORD PTR [rax+8], 0
  mov rax, QWORD PTR [rbp-56]
  mov QWORD PTR [rax+16], 0
.L1102:
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZNKSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rsi, rbx
  mov rdi, rax
  call _ZSt15__alloc_on_copyISaINSt7__cxx119sub_matchIPKcEEEEvRT_RKS6_
.L1099:
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv
  mov QWORD PTR [rbp-40], rax
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE8capacityEv
  cmp QWORD PTR [rbp-40], rax
  seta al
  test al, al
  je .L1103
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE3endEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE5beginEv
  mov rdx, rax
  mov rsi, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rbp-56]
  mov rcx, rbx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE20_M_allocate_and_copyIN9__gnu_cxx17__normal_iteratorIPKS4_S6_EEEEPS4_mT_SE_
  mov QWORD PTR [rbp-48], rax
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-56]
  mov rcx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-56]
  mov rax, QWORD PTR [rax]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt8_DestroyIPNSt7__cxx119sub_matchIPKcEES4_EvT_S6_RSaIT0_E
  mov rax, QWORD PTR [rbp-56]
  mov rdx, QWORD PTR [rbp-56]
  mov rcx, QWORD PTR [rdx+16]
  mov rdx, QWORD PTR [rbp-56]
  mov rdx, QWORD PTR [rdx]
  sub rcx, rdx
  mov rdx, rcx
  mov rcx, rdx
  sar rcx, 3
  movabs rdx, -6148914691236517205
  imul rdx, rcx
  mov rsi, rdx
  mov rdx, QWORD PTR [rbp-56]
  mov rcx, QWORD PTR [rdx]
  mov rdx, rsi
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE13_M_deallocateEPS4_m
  mov rax, QWORD PTR [rbp-56]
  mov rdx, QWORD PTR [rbp-48]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-56]
  mov rcx, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-40]
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 3
  lea rdx, [rcx+rax]
  mov rax, QWORD PTR [rbp-56]
  mov QWORD PTR [rax+16], rdx
  jmp .L1104
.L1103:
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv
  cmp QWORD PTR [rbp-40], rax
  setbe al
  test al, al
  je .L1105
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE3endEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE5beginEv
  mov r14, rax
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE3endEv
  mov r13, rax
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE5beginEv
  mov rdx, r14
  mov rsi, r13
  mov rdi, rax
  call _ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKNSt7__cxx119sub_matchIPKcEESt6vectorIS6_SaIS6_EEEENS1_IPS6_SB_EEET0_T_SG_SF_
  mov rdx, r12
  mov rsi, rbx
  mov rdi, rax
  call _ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS6_SaIS6_EEEES6_EvT_SC_RSaIT0_E
  jmp .L1104
.L1105:
  mov rax, QWORD PTR [rbp-56]
  mov rbx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-64]
  mov r12, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv
  mov rdx, rax
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 3
  lea rcx, [r12+rax]
  mov rax, QWORD PTR [rbp-64]
  mov rax, QWORD PTR [rax]
  mov rdx, rbx
  mov rsi, rcx
  mov rdi, rax
  call _ZSt4copyIPNSt7__cxx119sub_matchIPKcEES5_ET0_T_S7_S6_
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov r14, rax
  mov rax, QWORD PTR [rbp-56]
  mov r12, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-64]
  mov rbx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-64]
  mov r13, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv
  mov rdx, rax
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 3
  add rax, r13
  mov rcx, r14
  mov rdx, r12
  mov rsi, rbx
  mov rdi, rax
  call _ZSt22__uninitialized_copy_aIPNSt7__cxx119sub_matchIPKcEES5_S4_ET0_T_S7_S6_RSaIT1_E
.L1104:
  mov rax, QWORD PTR [rbp-56]
  mov rcx, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-40]
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 3
  lea rdx, [rcx+rax]
  mov rax, QWORD PTR [rbp-56]
  mov QWORD PTR [rax+8], rdx
.L1098:
  mov rax, QWORD PTR [rbp-56]
  add rsp, 32
  pop rbx
  pop r12
  pop r13
  pop r14
  pop rbp
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov eax, esi
  mov QWORD PTR [rbp-24], rdx
  mov BYTE PTR [rbp-12], al
  mov rax, QWORD PTR [rbp-8]
  lea rdx, [rax+96]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE11_State_infoISt17integral_constantIbLb1EESt6vectorIS5_S6_EE10_M_visitedEl
  test al, al
  jne .L1122
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov rdi, rax
  call _ZNKSt8__detail6_StateIcE9_M_opcodeEv
  cmp eax, 12
  ja .L1107
  mov eax, eax
  mov rax, QWORD PTR .L1111[0+rax*8]
  jmp rax
.L1111:
  .quad .L1107
  .quad .L1121
  .quad .L1120
  .quad .L1119
  .quad .L1118
  .quad .L1117
  .quad .L1116
  .quad .L1115
  .quad .L1114
  .quad .L1113
  .quad .L1107
  .quad .L1112
  .quad .L1110
.L1120:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE16_M_handle_repeatENS9_11_Match_modeEl
  jmp .L1107
.L1114:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE23_M_handle_subexpr_beginENS9_11_Match_modeEl
  jmp .L1107
.L1113:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE21_M_handle_subexpr_endENS9_11_Match_modeEl
  jmp .L1107
.L1118:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE30_M_handle_line_begin_assertionENS9_11_Match_modeEl
  jmp .L1107
.L1117:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE28_M_handle_line_end_assertionENS9_11_Match_modeEl
  jmp .L1107
.L1116:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE23_M_handle_word_boundaryENS9_11_Match_modeEl
  jmp .L1107
.L1115:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE27_M_handle_subexpr_lookaheadENS9_11_Match_modeEl
  jmp .L1107
.L1112:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE15_M_handle_matchENS9_11_Match_modeEl
  jmp .L1107
.L1119:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE17_M_handle_backrefENS9_11_Match_modeEl
  jmp .L1107
.L1110:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE16_M_handle_acceptENS9_11_Match_modeEl
  jmp .L1107
.L1121:
  mov rdx, QWORD PTR [rbp-24]
  movzx ecx, BYTE PTR [rbp-12]
  mov rax, QWORD PTR [rbp-8]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE21_M_handle_alternativeENS9_11_Match_modeEl
  nop
  jmp .L1107
.L1122:
  nop
.L1107:
  leave
  ret
_ZNSt12__shared_ptrINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EEC2ISaIS5_EJRKSt6localeRNSt15regex_constants18syntax_option_typeEEEESt20_Sp_alloc_shared_tagIT_EDpOT0_:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov QWORD PTR [rbp-48], rcx
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax], 0
  mov rax, QWORD PTR [rbp-24]
  lea rbx, [rax+8]
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZSt7forwardIRNSt15regex_constants18syntax_option_typeEEOT_RNSt16remove_referenceIS3_E4typeE
  mov r12, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZSt7forwardIRKSt6localeEOT_RNSt16remove_referenceIS3_E4typeE
  mov rcx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rbp-32]
  mov r8, r12
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt14__shared_countILN9__gnu_cxx12_Lock_policyE2EEC1INSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS9_EJRKSt6localeRNSt15regex_constants18syntax_option_typeEEEERPT_St20_Sp_alloc_shared_tagIT0_EDpOT1_
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12__shared_ptrINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EE31_M_enable_shared_from_this_withIS5_S5_EENSt9enable_ifIXntsrNS8_15__has_esft_baseIT0_vEE5valueEvE4typeEPT_
  nop
  add rsp, 32
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8_M_eraseEmm:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 56
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  mov QWORD PTR [rbp-56], rdx
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv
  sub rax, QWORD PTR [rbp-48]
  sub rax, QWORD PTR [rbp-56]
  mov QWORD PTR [rbp-24], rax
  cmp QWORD PTR [rbp-24], 0
  je .L1125
  cmp QWORD PTR [rbp-56], 0
  je .L1125
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rcx, rax
  mov rdx, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rbp-56]
  add rax, rdx
  lea rbx, [rcx+rax]
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-48]
  lea rcx, [rdx+rax]
  mov rax, QWORD PTR [rbp-24]
  mov rdx, rax
  mov rsi, rbx
  mov rdi, rcx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_moveEPcPKcm
.L1125:
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv
  sub rax, QWORD PTR [rbp-56]
  mov rdx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm
  nop
  add rsp, 56
  pop rbx
  pop rbp
  ret
.LC25:
  .string "Unexpected end of character class."
_ZNSt8__detail8_ScannerIcE12_M_eat_classEc:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov eax, esi
  mov BYTE PTR [rbp-12], al
  mov rax, QWORD PTR [rbp-8]
  add rax, 200
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5clearEv
.L1128:
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  je .L1127
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+176]
  movzx eax, BYTE PTR [rax]
  cmp BYTE PTR [rbp-12], al
  je .L1127
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-8]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  movsx eax, al
  mov rdx, QWORD PTR [rbp-8]
  add rdx, 200
  mov esi, eax
  mov rdi, rdx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEpLEc
  jmp .L1128
.L1127:
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  je .L1129
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-8]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  cmp BYTE PTR [rbp-12], al
  jne .L1129
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+176]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+184]
  cmp rdx, rax
  je .L1129
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+176]
  lea rcx, [rax+1]
  mov rdx, QWORD PTR [rbp-8]
  mov QWORD PTR [rdx+176], rcx
  movzx eax, BYTE PTR [rax]
  cmp al, 93
  je .L1130
.L1129:
  mov eax, 1
  jmp .L1131
.L1130:
  mov eax, 0
.L1131:
  test al, al
  je .L1135
  cmp BYTE PTR [rbp-12], 58
  jne .L1133
  mov esi, OFFSET FLAT:.LC25
  mov edi, 1
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
  jmp .L1135
.L1133:
  mov esi, OFFSET FLAT:.LC25
  mov edi, 0
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L1135:
  nop
  leave
  ret
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE11_Deque_implC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSaINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEC2Ev
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+8], 0
  mov rax, QWORD PTR [rbp-8]
  add rax, 16
  mov rdi, rax
  call _ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_EC1Ev
  mov rax, QWORD PTR [rbp-8]
  add rax, 48
  mov rdi, rax
  call _ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_EC1Ev
  nop
  leave
  ret
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE17_M_initialize_mapEm:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 72
  mov QWORD PTR [rbp-72], rdi
  mov QWORD PTR [rbp-80], rsi
  mov edi, 24
  call _ZSt16__deque_buf_sizem
  mov rbx, rax
  mov rax, QWORD PTR [rbp-80]
  mov edx, 0
  div rbx
  add rax, 1
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-24]
  add rax, 2
  mov QWORD PTR [rbp-56], rax
  mov QWORD PTR [rbp-48], 8
  lea rdx, [rbp-56]
  lea rax, [rbp-48]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt3maxImERKT_S2_S2_
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-72]
  mov QWORD PTR [rax+8], rdx
  mov rax, QWORD PTR [rbp-72]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-72]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE15_M_allocate_mapEm
  mov rdx, rax
  mov rax, QWORD PTR [rbp-72]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-72]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-72]
  mov rax, QWORD PTR [rax+8]
  sub rax, QWORD PTR [rbp-24]
  shr rax
  sal rax, 3
  add rax, rdx
  mov QWORD PTR [rbp-32], rax
  mov rax, QWORD PTR [rbp-24]
  lea rdx, [0+rax*8]
  mov rax, QWORD PTR [rbp-32]
  add rax, rdx
  mov QWORD PTR [rbp-40], rax
  mov rdx, QWORD PTR [rbp-40]
  mov rcx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-72]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE15_M_create_nodesEPPS5_S9_
  mov rax, QWORD PTR [rbp-72]
  lea rdx, [rax+16]
  mov rax, QWORD PTR [rbp-32]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_E11_M_set_nodeEPS7_
  mov rax, QWORD PTR [rbp-72]
  add rax, 48
  mov rdx, QWORD PTR [rbp-40]
  sub rdx, 8
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_E11_M_set_nodeEPS7_
  mov rax, QWORD PTR [rbp-72]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-72]
  mov QWORD PTR [rax+16], rdx
  mov rax, QWORD PTR [rbp-72]
  mov rbx, QWORD PTR [rax+56]
  mov edi, 24
  call _ZSt16__deque_buf_sizem
  mov rcx, rax
  mov rax, QWORD PTR [rbp-80]
  mov edx, 0
  div rcx
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 3
  lea rdx, [rbx+rax]
  mov rax, QWORD PTR [rbp-72]
  mov QWORD PTR [rax+48], rdx
  jmp .L1142
  mov rdi, rax
  call __cxa_begin_catch
  mov rax, QWORD PTR [rbp-72]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-72]
  mov rcx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-72]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE17_M_deallocate_mapEPPS5_m
  mov rax, QWORD PTR [rbp-72]
  mov QWORD PTR [rax], 0
  mov rax, QWORD PTR [rbp-72]
  mov QWORD PTR [rax+8], 0
  call __cxa_rethrow
  mov rbx, rax
  call __cxa_end_catch
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1142:
  add rsp, 72
  pop rbx
  pop rbp
  ret

_ZSt7forwardIRKmEOT_RNSt16remove_referenceIS2_E4typeE:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZN9__gnu_cxx13new_allocatorImE9constructImJRKmEEEvPT_DpOT0_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZSt7forwardIRKmEOT_RNSt16remove_referenceIS2_E4typeE
  mov rbx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-32]
  mov rsi, rax
  mov edi, 8
  call _ZnwmPv
  mov QWORD PTR [rax], rbx
  nop
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEC2ERKS1_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  nop
  pop rbp
  ret
_ZNKSt6vectorImSaImEE12_M_check_lenEmPKc:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 56
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  mov QWORD PTR [rbp-56], rdx
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt6vectorImSaImEE8max_sizeEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt6vectorImSaImEE4sizeEv
  sub rbx, rax
  mov rdx, rbx
  mov rax, QWORD PTR [rbp-48]
  cmp rdx, rax
  setb al
  test al, al
  je .L1148
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZSt20__throw_length_errorPKc
.L1148:
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt6vectorImSaImEE4sizeEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt6vectorImSaImEE4sizeEv
  mov QWORD PTR [rbp-32], rax
  lea rdx, [rbp-48]
  lea rax, [rbp-32]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt3maxImERKT_S2_S2_
  mov rax, QWORD PTR [rax]
  add rax, rbx
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt6vectorImSaImEE4sizeEv
  cmp QWORD PTR [rbp-24], rax
  jb .L1149
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt6vectorImSaImEE8max_sizeEv
  cmp QWORD PTR [rbp-24], rax
  jbe .L1150
.L1149:
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNKSt6vectorImSaImEE8max_sizeEv
  jmp .L1151
.L1150:
  mov rax, QWORD PTR [rbp-24]
.L1151:
  add rsp, 56
  pop rbx
  pop rbp
  ret
_ZNSt6vectorImSaImEE5beginEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEC1ERKS1_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZN9__gnu_cxxmiIPmSt6vectorImSaImEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEE4baseEv
  mov rbx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEE4baseEv
  mov rax, QWORD PTR [rax]
  sub rbx, rax
  mov rax, rbx
  sar rax, 3
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZNSt12_Vector_baseImSaImEE11_M_allocateEm:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  cmp QWORD PTR [rbp-16], 0
  je .L1158
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaImEE8allocateERS0_m
  jmp .L1160
.L1158:
  mov eax, 0
.L1160:
  leave
  ret
_ZNK9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEE4baseEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt34__uninitialized_move_if_noexcept_aIPmS0_SaImEET0_T_S3_S2_RT1_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov QWORD PTR [rbp-48], rcx
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt32__make_move_if_noexcept_iteratorImSt13move_iteratorIPmEET0_PT_
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt32__make_move_if_noexcept_iteratorImSt13move_iteratorIPmEET0_PT_
  mov rdi, rax
  mov rdx, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rbp-40]
  mov rcx, rdx
  mov rdx, rax
  mov rsi, rbx
  call _ZSt22__uninitialized_copy_aISt13move_iteratorIPmES1_mET0_T_S4_S3_RSaIT1_E
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EE12emplace_backIJS2_EEERS2_DpOT_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+16]
  cmp rdx, rax
  je .L1166
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt7forwardINSt8__detail6_StateIcEEEOT_RNSt16remove_referenceIS3_E4typeE
  mov rdx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaINSt8__detail6_StateIcEEEE9constructIS2_JS2_EEEvRS3_PT_DpOT0_
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+8]
  lea rdx, [rax+48]
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+8], rdx
  jmp .L1167
.L1166:
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt7forwardINSt8__detail6_StateIcEEEOT_RNSt16remove_referenceIS3_E4typeE
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EE3endEv
  mov rcx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdx, rbx
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EE17_M_realloc_insertIJS2_EEEvN9__gnu_cxx17__normal_iteratorIPS2_S4_EEDpOT_
.L1167:
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EE4backEv
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZNSt8functionIFbcEE4swapERS1_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt4swapISt9_Any_dataENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS5_ESt18is_move_assignableIS5_EEE5valueEvE4typeERS5_SF_
  mov rax, QWORD PTR [rbp-16]
  lea rdx, [rax+16]
  mov rax, QWORD PTR [rbp-8]
  add rax, 16
  mov rsi, rdx
  mov rdi, rax
  call _ZSt4swapIPFbRSt9_Any_dataRKS0_St18_Manager_operationEENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISB_ESt18is_move_assignableISB_EEE5valueEvE4typeERSB_SL_
  mov rax, QWORD PTR [rbp-16]
  lea rdx, [rax+24]
  mov rax, QWORD PTR [rbp-8]
  add rax, 24
  mov rsi, rdx
  mov rdi, rax
  call _ZSt4swapIPFbRKSt9_Any_dataOcEENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISA_ESt18is_move_assignableISA_EEE5valueEvE4typeERSA_SK_
  nop
  leave
  ret
.LC26:
  .string "Parenthesis is not closed."
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE12_M_assertionEv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 176
  mov QWORD PTR [rbp-184], rdi
  mov rax, QWORD PTR [rbp-184]
  mov esi, 22
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1171
  mov rax, QWORD PTR [rbp-184]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-184]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE20_M_insert_line_beginEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-184]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-144]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-144]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  jmp .L1172
.L1171:
  mov rax, QWORD PTR [rbp-184]
  mov esi, 23
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1173
  mov rax, QWORD PTR [rbp-184]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-184]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE18_M_insert_line_endEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-184]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-112]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-112]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  jmp .L1172
.L1173:
  mov rax, QWORD PTR [rbp-184]
  mov esi, 24
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1174
  mov rax, QWORD PTR [rbp-184]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-184]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-184]
  add rax, 272
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEixEm
  movzx eax, BYTE PTR [rax]
  cmp al, 110
  sete al
  movzx eax, al
  mov esi, eax
  mov rdi, r12
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE20_M_insert_word_boundEb
  mov r12, rax
  mov rax, QWORD PTR [rbp-184]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-80]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-80]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  jmp .L1172
.L1174:
  mov rax, QWORD PTR [rbp-184]
  mov esi, 7
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1175
  mov rax, QWORD PTR [rbp-184]
  add rax, 272
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEixEm
  movzx eax, BYTE PTR [rax]
  cmp al, 110
  sete al
  mov BYTE PTR [rbp-17], al
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_disjunctionEv
  mov rax, QWORD PTR [rbp-184]
  mov esi, 8
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  xor eax, 1
  test al, al
  je .L1176
  mov esi, OFFSET FLAT:.LC26
  mov edi, 5
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L1176:
  lea rax, [rbp-176]
  mov rdx, QWORD PTR [rbp-184]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv
  mov rax, QWORD PTR [rbp-184]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE16_M_insert_acceptEv
  mov rdx, rax
  lea rax, [rbp-176]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendEl
  mov rax, QWORD PTR [rbp-184]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-184]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rcx, rax
  movzx edx, BYTE PTR [rbp-17]
  mov rax, QWORD PTR [rbp-168]
  mov rsi, rax
  mov rdi, rcx
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE19_M_insert_lookaheadElb
  mov r12, rax
  mov rax, QWORD PTR [rbp-184]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-48]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-48]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  jmp .L1172
.L1175:
  mov eax, 0
  jmp .L1177
.L1172:
  mov eax, 1
.L1177:
  add rsp, 176
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE7_M_atomEv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 176
  mov QWORD PTR [rbp-184], rdi
  mov rax, QWORD PTR [rbp-184]
  mov esi, 0
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1179
  mov rax, QWORD PTR [rbp-184]
  mov eax, DWORD PTR [rax]
  mov esi, 16
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  sete al
  test al, al
  je .L1180
  mov rax, QWORD PTR [rbp-184]
  mov eax, DWORD PTR [rax]
  mov esi, 1
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  sete al
  test al, al
  je .L1181
  mov rax, QWORD PTR [rbp-184]
  mov eax, DWORD PTR [rax]
  mov esi, 8
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  sete al
  test al, al
  je .L1182
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE27_M_insert_any_matcher_posixILb0ELb0EEEvv
  jmp .L1191
.L1182:
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE27_M_insert_any_matcher_posixILb0ELb1EEEvv
  jmp .L1191
.L1181:
  mov rax, QWORD PTR [rbp-184]
  mov eax, DWORD PTR [rax]
  mov esi, 8
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  sete al
  test al, al
  je .L1185
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE27_M_insert_any_matcher_posixILb1ELb0EEEvv
  jmp .L1191
.L1185:
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE27_M_insert_any_matcher_posixILb1ELb1EEEvv
  jmp .L1191
.L1180:
  mov rax, QWORD PTR [rbp-184]
  mov eax, DWORD PTR [rax]
  mov esi, 1
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  sete al
  test al, al
  je .L1187
  mov rax, QWORD PTR [rbp-184]
  mov eax, DWORD PTR [rax]
  mov esi, 8
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  sete al
  test al, al
  je .L1188
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE26_M_insert_any_matcher_ecmaILb0ELb0EEEvv
  jmp .L1191
.L1188:
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE26_M_insert_any_matcher_ecmaILb0ELb1EEEvv
  jmp .L1191
.L1187:
  mov rax, QWORD PTR [rbp-184]
  mov eax, DWORD PTR [rax]
  mov esi, 8
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  sete al
  test al, al
  je .L1190
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE26_M_insert_any_matcher_ecmaILb1ELb0EEEvv
  jmp .L1191
.L1190:
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE26_M_insert_any_matcher_ecmaILb1ELb1EEEvv
  jmp .L1191
.L1179:
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE11_M_try_charEv
  test al, al
  je .L1192
  mov rax, QWORD PTR [rbp-184]
  mov eax, DWORD PTR [rax]
  mov esi, 1
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  sete al
  test al, al
  je .L1193
  mov rax, QWORD PTR [rbp-184]
  mov eax, DWORD PTR [rax]
  mov esi, 8
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  sete al
  test al, al
  je .L1194
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE22_M_insert_char_matcherILb0ELb0EEEvv
  jmp .L1191
.L1194:
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE22_M_insert_char_matcherILb0ELb1EEEvv
  jmp .L1191
.L1193:
  mov rax, QWORD PTR [rbp-184]
  mov eax, DWORD PTR [rax]
  mov esi, 8
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  sete al
  test al, al
  je .L1197
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE22_M_insert_char_matcherILb1ELb0EEEvv
  jmp .L1191
.L1197:
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE22_M_insert_char_matcherILb1ELb1EEEvv
  jmp .L1191
.L1192:
  mov rax, QWORD PTR [rbp-184]
  mov esi, 4
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1198
  mov rax, QWORD PTR [rbp-184]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-184]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-184]
  mov esi, 10
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE16_M_cur_int_valueEi
  cdqe
  mov rsi, rax
  mov rdi, r12
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_backrefEm
  mov r12, rax
  mov rax, QWORD PTR [rbp-184]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-112]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-112]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  jmp .L1191
.L1198:
  mov rax, QWORD PTR [rbp-184]
  mov esi, 14
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1199
  mov rax, QWORD PTR [rbp-184]
  mov eax, DWORD PTR [rax]
  mov esi, 1
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  sete al
  test al, al
  je .L1200
  mov rax, QWORD PTR [rbp-184]
  mov eax, DWORD PTR [rax]
  mov esi, 8
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  sete al
  test al, al
  je .L1201
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE33_M_insert_character_class_matcherILb0ELb0EEEvv
  jmp .L1191
.L1201:
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE33_M_insert_character_class_matcherILb0ELb1EEEvv
  jmp .L1191
.L1200:
  mov rax, QWORD PTR [rbp-184]
  mov eax, DWORD PTR [rax]
  mov esi, 8
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  sete al
  test al, al
  je .L1204
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE33_M_insert_character_class_matcherILb1ELb0EEEvv
  jmp .L1191
.L1204:
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE33_M_insert_character_class_matcherILb1ELb1EEEvv
  jmp .L1191
.L1199:
  mov rax, QWORD PTR [rbp-184]
  mov esi, 6
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1205
  mov rax, QWORD PTR [rbp-184]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_dummyEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-184]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-144]
  mov rdx, rbx
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_disjunctionEv
  mov rax, QWORD PTR [rbp-184]
  mov esi, 8
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  xor eax, 1
  test al, al
  je .L1206
  mov esi, OFFSET FLAT:.LC26
  mov edi, 5
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L1206:
  lea rax, [rbp-80]
  mov rdx, QWORD PTR [rbp-184]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv
  lea rdx, [rbp-80]
  lea rax, [rbp-144]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendERKS4_
  mov rax, QWORD PTR [rbp-184]
  lea rdx, [rax+304]
  lea rax, [rbp-144]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushERKS5_
  jmp .L1191
.L1205:
  mov rax, QWORD PTR [rbp-184]
  mov esi, 5
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1207
  mov rax, QWORD PTR [rbp-184]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE23_M_insert_subexpr_beginEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-184]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-176]
  mov rdx, rbx
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_disjunctionEv
  mov rax, QWORD PTR [rbp-184]
  mov esi, 8
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  xor eax, 1
  test al, al
  je .L1208
  mov esi, OFFSET FLAT:.LC26
  mov edi, 5
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L1208:
  lea rax, [rbp-48]
  mov rdx, QWORD PTR [rbp-184]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv
  lea rdx, [rbp-48]
  lea rax, [rbp-176]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendERKS4_
  mov rax, QWORD PTR [rbp-184]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE21_M_insert_subexpr_endEv
  mov rdx, rax
  lea rax, [rbp-176]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendEl
  mov rax, QWORD PTR [rbp-184]
  lea rdx, [rax+304]
  lea rax, [rbp-176]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushERKS5_
  jmp .L1191
.L1207:
  mov rax, QWORD PTR [rbp-184]
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE21_M_bracket_expressionEv
  xor eax, 1
  test al, al
  je .L1191
  mov eax, 0
  jmp .L1209
.L1191:
  mov eax, 1
.L1209:
  add rsp, 176
  pop rbx
  pop r12
  pop rbp
  ret
.LC27:
  .string "Nothing to repeat before a quantifier."
_ZZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE13_M_quantifierEvENKUlvE_clEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  add rax, 304
  mov rdi, rax
  call _ZNKSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE5emptyEv
  test al, al
  je .L1211
  mov esi, OFFSET FLAT:.LC27
  mov edi, 10
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L1211:
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+8]
  movzx eax, BYTE PTR [rax]
  test al, al
  je .L1212
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  mov esi, 18
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1212
  mov edx, 1
  jmp .L1213
.L1212:
  mov edx, 0
.L1213:
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+8]
  mov BYTE PTR [rax], dl
  nop
  leave
  ret
_ZNSt5stackIlSt5dequeIlSaIlEEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt5dequeIlSaIlEED1Ev
  nop
  leave
  ret
.LC28:
  .string "Unexpected token in brace expression."
.LC29:
  .string "Unexpected end of brace expression."
.LC30:
  .string "Invalid range in brace expression."
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE13_M_quantifierEv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 592
  mov QWORD PTR [rbp-600], rdi
  mov rax, QWORD PTR [rbp-600]
  mov eax, DWORD PTR [rax]
  mov esi, 16
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  setne al
  mov BYTE PTR [rbp-145], al
  mov rax, QWORD PTR [rbp-600]
  mov QWORD PTR [rbp-176], rax
  lea rax, [rbp-145]
  mov QWORD PTR [rbp-168], rax
  mov rax, QWORD PTR [rbp-600]
  mov esi, 20
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1216
  lea rax, [rbp-176]
  mov rdi, rax
  call _ZZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE13_M_quantifierEvENKUlvE_clEv
  lea rax, [rbp-208]
  mov rdx, QWORD PTR [rbp-600]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv
  mov rax, QWORD PTR [rbp-600]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  movzx eax, BYTE PTR [rbp-145]
  movzx edx, al
  mov rax, QWORD PTR [rbp-200]
  mov ecx, edx
  mov rdx, rax
  mov rsi, -1
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE16_M_insert_repeatEllb
  mov rbx, rax
  mov rax, QWORD PTR [rbp-600]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-240]
  mov rdx, rbx
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rdx, [rbp-240]
  lea rax, [rbp-208]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendERKS4_
  mov rax, QWORD PTR [rbp-600]
  lea rdx, [rax+304]
  lea rax, [rbp-240]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushERKS5_
  jmp .L1217
.L1216:
  mov rax, QWORD PTR [rbp-600]
  mov esi, 21
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1218
  lea rax, [rbp-176]
  mov rdi, rax
  call _ZZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE13_M_quantifierEvENKUlvE_clEv
  lea rax, [rbp-272]
  mov rdx, QWORD PTR [rbp-600]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv
  mov rax, QWORD PTR [rbp-600]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  movzx eax, BYTE PTR [rbp-145]
  movzx edx, al
  mov rax, QWORD PTR [rbp-264]
  mov ecx, edx
  mov rdx, rax
  mov rsi, -1
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE16_M_insert_repeatEllb
  mov rdx, rax
  lea rax, [rbp-272]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendEl
  mov rax, QWORD PTR [rbp-600]
  lea rdx, [rax+304]
  lea rax, [rbp-272]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushERKS5_
  jmp .L1217
.L1218:
  mov rax, QWORD PTR [rbp-600]
  mov esi, 18
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1219
  lea rax, [rbp-176]
  mov rdi, rax
  call _ZZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE13_M_quantifierEvENKUlvE_clEv
  lea rax, [rbp-304]
  mov rdx, QWORD PTR [rbp-600]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv
  mov rax, QWORD PTR [rbp-600]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_dummyEv
  mov QWORD PTR [rbp-56], rax
  mov rax, QWORD PTR [rbp-600]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  movzx eax, BYTE PTR [rbp-145]
  movzx edx, al
  mov rax, QWORD PTR [rbp-296]
  mov ecx, edx
  mov rdx, rax
  mov rsi, -1
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE16_M_insert_repeatEllb
  mov rbx, rax
  mov rax, QWORD PTR [rbp-600]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-336]
  mov rdx, rbx
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  mov rdx, QWORD PTR [rbp-56]
  lea rax, [rbp-304]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendEl
  mov rdx, QWORD PTR [rbp-56]
  lea rax, [rbp-336]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendEl
  mov rax, QWORD PTR [rbp-600]
  lea rdx, [rax+304]
  lea rax, [rbp-336]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushERKS5_
  jmp .L1217
.L1219:
  mov rax, QWORD PTR [rbp-600]
  mov esi, 12
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1220
  mov rax, QWORD PTR [rbp-600]
  add rax, 304
  mov rdi, rax
  call _ZNKSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE5emptyEv
  test al, al
  je .L1221
  mov esi, OFFSET FLAT:.LC27
  mov edi, 10
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L1221:
  mov rax, QWORD PTR [rbp-600]
  mov esi, 26
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  xor eax, 1
  test al, al
  je .L1222
  mov esi, OFFSET FLAT:.LC28
  mov edi, 7
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L1222:
  lea rax, [rbp-368]
  mov rdx, QWORD PTR [rbp-600]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE6_M_popEv
  mov rax, QWORD PTR [rbp-600]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_dummyEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-600]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-400]
  mov rdx, rbx
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  mov rax, QWORD PTR [rbp-600]
  mov esi, 10
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE16_M_cur_int_valueEi
  cdqe
  mov QWORD PTR [rbp-64], rax
  mov BYTE PTR [rbp-17], 0
  mov rax, QWORD PTR [rbp-600]
  mov esi, 25
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1223
  mov rax, QWORD PTR [rbp-600]
  mov esi, 26
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1224
  mov rax, QWORD PTR [rbp-600]
  mov esi, 10
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE16_M_cur_int_valueEi
  cdqe
  sub rax, QWORD PTR [rbp-64]
  mov QWORD PTR [rbp-32], rax
  jmp .L1226
.L1224:
  mov BYTE PTR [rbp-17], 1
  jmp .L1226
.L1223:
  mov QWORD PTR [rbp-32], 0
.L1226:
  mov rax, QWORD PTR [rbp-600]
  mov esi, 13
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  xor eax, 1
  test al, al
  je .L1227
  mov esi, OFFSET FLAT:.LC29
  mov edi, 6
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L1227:
  movzx eax, BYTE PTR [rbp-145]
  test al, al
  je .L1228
  mov rax, QWORD PTR [rbp-600]
  mov esi, 18
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1228
  mov eax, 1
  jmp .L1229
.L1228:
  mov eax, 0
.L1229:
  mov BYTE PTR [rbp-145], al
  mov QWORD PTR [rbp-40], 0
.L1231:
  mov rax, QWORD PTR [rbp-40]
  cmp rax, QWORD PTR [rbp-64]
  jge .L1230
  lea rax, [rbp-144]
  lea rdx, [rbp-368]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE8_M_cloneEv
  lea rdx, [rbp-144]
  lea rax, [rbp-400]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendERKS4_
  add QWORD PTR [rbp-40], 1
  jmp .L1231
.L1230:
  cmp BYTE PTR [rbp-17], 0
  je .L1232
  lea rax, [rbp-432]
  lea rdx, [rbp-368]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE8_M_cloneEv
  mov rax, QWORD PTR [rbp-600]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  movzx eax, BYTE PTR [rbp-145]
  movzx edx, al
  mov rax, QWORD PTR [rbp-424]
  mov ecx, edx
  mov rdx, rax
  mov rsi, -1
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE16_M_insert_repeatEllb
  mov rbx, rax
  mov rax, QWORD PTR [rbp-600]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-464]
  mov rdx, rbx
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rdx, [rbp-464]
  lea rax, [rbp-432]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendERKS4_
  lea rdx, [rbp-464]
  lea rax, [rbp-400]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendERKS4_
  jmp .L1233
.L1232:
  cmp QWORD PTR [rbp-32], 0
  jns .L1234
  mov esi, OFFSET FLAT:.LC30
  mov edi, 7
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L1234:
  mov rax, QWORD PTR [rbp-600]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_dummyEv
  mov QWORD PTR [rbp-72], rax
  lea rax, [rbp-592]
  mov rdi, rax
  call _ZNSt5stackIlSt5dequeIlSaIlEEEC1IS2_vEEv
  mov QWORD PTR [rbp-48], 0
.L1236:
  mov rax, QWORD PTR [rbp-48]
  cmp rax, QWORD PTR [rbp-32]
  jge .L1235
  lea rax, [rbp-496]
  lea rdx, [rbp-368]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE8_M_cloneEv
  mov rax, QWORD PTR [rbp-600]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov rdi, rax
  movzx eax, BYTE PTR [rbp-145]
  movzx ecx, al
  mov rax, QWORD PTR [rbp-488]
  mov rdx, QWORD PTR [rbp-72]
  mov rsi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE16_M_insert_repeatEllb
  mov QWORD PTR [rbp-504], rax
  lea rdx, [rbp-504]
  lea rax, [rbp-592]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt5stackIlSt5dequeIlSaIlEEE4pushERKl
  mov r12, QWORD PTR [rbp-480]
  mov rbx, QWORD PTR [rbp-504]
  mov rax, QWORD PTR [rbp-600]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rsi, rax
  lea rax, [rbp-112]
  mov rcx, r12
  mov rdx, rbx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEll
  lea rdx, [rbp-112]
  lea rax, [rbp-400]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendERKS4_
  add QWORD PTR [rbp-48], 1
  jmp .L1236
.L1235:
  mov rdx, QWORD PTR [rbp-72]
  lea rax, [rbp-400]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE9_M_appendEl
.L1238:
  lea rax, [rbp-592]
  mov rdi, rax
  call _ZNKSt5stackIlSt5dequeIlSaIlEEE5emptyEv
  xor eax, 1
  test al, al
  je .L1237
  mov rax, QWORD PTR [rbp-600]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  lea rbx, [rax+56]
  lea rax, [rbp-592]
  mov rdi, rax
  call _ZNSt5stackIlSt5dequeIlSaIlEEE3topEv
  mov rax, QWORD PTR [rax]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-80], rax
  lea rax, [rbp-592]
  mov rdi, rax
  call _ZNSt5stackIlSt5dequeIlSaIlEEE3popEv
  mov rax, QWORD PTR [rbp-80]
  lea rdx, [rax+16]
  mov rax, QWORD PTR [rbp-80]
  add rax, 8
  mov rsi, rdx
  mov rdi, rax
  call _ZSt4swapIlENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS4_ESt18is_move_assignableIS4_EEE5valueEvE4typeERS4_SE_
  jmp .L1238
.L1237:
  lea rax, [rbp-592]
  mov rdi, rax
  call _ZNSt5stackIlSt5dequeIlSaIlEEED1Ev
.L1233:
  mov rax, QWORD PTR [rbp-600]
  lea rdx, [rax+304]
  lea rax, [rbp-400]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushERKS5_
  jmp .L1217
.L1220:
  mov eax, 0
  jmp .L1243
.L1217:
  mov eax, 1
  jmp .L1243
  mov rbx, rax
  lea rax, [rbp-592]
  mov rdi, rax
  call _ZNSt5stackIlSt5dequeIlSaIlEEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1243:
  add rsp, 592
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE9push_backERKS5_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+48]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+64]
  sub rax, 24
  cmp rdx, rax
  je .L1245
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax+48]
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEE9constructIS5_JRKS5_EEEvRS6_PT_DpOT0_
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+48]
  lea rdx, [rax+24]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+48], rdx
  jmp .L1247
.L1245:
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE16_M_push_back_auxIJRKS5_EEEvDpOT_
.L1247:
  nop
  leave
  ret
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE12emplace_backIJS5_EEERS5_DpOT_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+48]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+64]
  sub rax, 24
  cmp rdx, rax
  je .L1249
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZSt7forwardINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEOT_RNSt16remove_referenceIS6_E4typeE
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax+48]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEE9constructIS5_JS5_EEEvRS6_PT_DpOT0_
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+48]
  lea rdx, [rax+24]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+48], rdx
  jmp .L1250
.L1249:
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZSt7forwardINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEOT_RNSt16remove_referenceIS6_E4typeE
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE16_M_push_back_auxIJS5_EEEvDpOT_
.L1250:
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE4backEv
  leave
  ret
_ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_EmmEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+8]
  cmp rdx, rax
  jne .L1253
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+24]
  lea rdx, [rax-8]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_E11_M_set_nodeEPS7_
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
.L1253:
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  lea rdx, [rax-24]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNKSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_EdeEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  pop rbp
  ret
_ZNSt16allocator_traitsISaINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEE7destroyIS5_EEvRS6_PT_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEE7destroyIS6_EEvPT_
  nop
  leave
  ret
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE15_M_pop_back_auxEv:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rdx+56]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE18_M_deallocate_nodeEPS5_
  mov rax, QWORD PTR [rbp-24]
  lea rdx, [rax+48]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+72]
  sub rax, 8
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_E11_M_set_nodeEPS7_
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+64]
  lea rdx, [rax-24]
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+48], rdx
  mov rax, QWORD PTR [rbp-24]
  mov rbx, QWORD PTR [rax+48]
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE19_M_get_Tp_allocatorEv
  mov rsi, rbx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEE7destroyIS5_EEvRS6_PT_
  nop
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZN9__gnu_cxx13new_allocatorImE7destroyImEEvPT_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  nop
  pop rbp
  ret
_ZNSt16allocator_traitsISaINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEE10deallocateERS6_PS5_m:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEE10deallocateEPS6_m
  nop
  leave
  ret
_ZNKSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE19_M_get_Tp_allocatorEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSaIPNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEC2IS4_EERKSaIT_E:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorIPNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEC2Ev
  nop
  leave
  ret
_ZN9__gnu_cxx13new_allocatorIPNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEED2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  nop
  pop rbp
  ret
_ZN9__gnu_cxx13new_allocatorIPNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEE10deallocateEPS7_m:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZdlPv
  leave
  ret
_ZNSt16allocator_traitsISaINSt7__cxx119sub_matchIPKcEEEE8max_sizeERKS5_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNK9__gnu_cxx13new_allocatorINSt7__cxx119sub_matchIPKcEEE8max_sizeEv
  leave
  ret
_ZNKSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt25__uninitialized_default_nIPNSt7__cxx119sub_matchIPKcEEmET_S6_T0_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov BYTE PTR [rbp-1], 1
  mov rdx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt27__uninitialized_default_n_1ILb0EE18__uninit_default_nIPNSt7__cxx119sub_matchIPKcEEmEET_S8_T0_
  leave
  ret
_ZNSt16allocator_traitsISaINSt7__cxx119sub_matchIPKcEEEE8allocateERS5_m:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov edx, 0
  mov rsi, rcx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorINSt7__cxx119sub_matchIPKcEEE8allocateEmPKv
  leave
  ret
_ZSt32__make_move_if_noexcept_iteratorINSt7__cxx119sub_matchIPKcEESt13move_iteratorIPS4_EET0_PT_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt13move_iteratorIPNSt7__cxx119sub_matchIPKcEEEC1ES5_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZSt22__uninitialized_copy_aISt13move_iteratorIPNSt7__cxx119sub_matchIPKcEEES6_S5_ET0_T_S9_S8_RSaIT1_E:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov QWORD PTR [rbp-32], rcx
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt18uninitialized_copyISt13move_iteratorIPNSt7__cxx119sub_matchIPKcEEES6_ET0_T_S9_S8_
  leave
  ret
_ZNSaISt4pairIPKciEEC2ERKS3_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorISt4pairIPKciEEC2ERKS5_
  nop
  leave
  ret
_ZNSt12_Vector_baseISt4pairIPKciESaIS3_EE11_M_allocateEm:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  cmp QWORD PTR [rbp-16], 0
  je .L1281
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaISt4pairIPKciEEE8allocateERS4_m
  jmp .L1283
.L1281:
  mov eax, 0
.L1283:
  leave
  ret
_ZSt25__uninitialized_default_nIPSt4pairIPKciEmET_S5_T0_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov BYTE PTR [rbp-1], 1
  mov rdx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt27__uninitialized_default_n_1ILb0EE18__uninit_default_nIPSt4pairIPKciEmEET_S7_T0_
  leave
  ret
_ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE12_Vector_implC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSaISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS5_EEEEC2Ev
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+8], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+16], 0
  nop
  leave
  ret
_ZNSt5tupleIJPbSt14default_deleteIA_bEEEC2IS0_S3_Lb1EEEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt11_Tuple_implILm0EJPbSt14default_deleteIA_bEEEC2Ev
  nop
  leave
  ret
_ZN9__gnu_cxx13new_allocatorISt4pairIPKciEE10deallocateEPS4_m:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZdlPv
  leave
  ret
_ZN9__gnu_cxx13new_allocatorISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEEE10deallocateEPSA_m:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZdlPv
  leave
  ret
_ZSt11__addressofISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEEPT_RSA_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS5_EEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 8
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EED1Ev
  nop
  leave
  ret
_ZSt8_DestroyISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEEvPT_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS5_EEED1Ev
  nop
  leave
  ret
_ZNSt11_Tuple_implILm0EJPbSt14default_deleteIA_bEEE7_M_headERS4_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt10_Head_baseILm0EPbLb0EE7_M_headERS1_
  leave
  ret
_ZSt12__get_helperILm1ESt14default_deleteIA_bEJEERT0_RSt11_Tuple_implIXT_EJS3_DpT1_EE:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt11_Tuple_implILm1EJSt14default_deleteIA_bEEE7_M_headERS3_
  leave
  ret
_ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE12emplace_backIJRlRKS7_EEERS8_DpOT_:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+16]
  cmp rdx, rax
  je .L1301
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZSt7forwardIRKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS5_EEEOT_RNSt16remove_referenceISA_E4typeE
  mov rbx, rax
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt7forwardIRlEOT_RNSt16remove_referenceIS1_E4typeE
  mov rdx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rsi, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-24]
  mov rcx, rbx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEEE9constructIS9_JRlRKS8_EEEvRSA_PT_DpOT0_
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+8]
  lea rdx, [rax+32]
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+8], rdx
  jmp .L1302
.L1301:
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZSt7forwardIRKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS5_EEEOT_RNSt16remove_referenceISA_E4typeE
  mov r12, rax
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt7forwardIRlEOT_RNSt16remove_referenceIS1_E4typeE
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE3endEv
  mov rsi, rax
  mov rax, QWORD PTR [rbp-24]
  mov rcx, r12
  mov rdx, rbx
  mov rdi, rax
  call _ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE17_M_realloc_insertIJRlRKS7_EEEvN9__gnu_cxx17__normal_iteratorIPS8_SA_EEDpOT_
.L1302:
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE4backEv
  add rsp, 32
  pop rbx
  pop r12
  pop rbp
  ret
_ZNKSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE5beginEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-8], rax
  lea rdx, [rbp-8]
  lea rax, [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPKSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEC1ERKSC_
  mov rax, QWORD PTR [rbp-16]
  leave
  ret
_ZNKSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE3endEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+8]
  mov QWORD PTR [rbp-8], rax
  lea rdx, [rbp-8]
  lea rax, [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPKSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEC1ERKSC_
  mov rax, QWORD PTR [rbp-16]
  leave
  ret
_ZN9__gnu_cxxeqIPKSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEEbRKNS_17__normal_iteratorIT_T0_EESK_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPKSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEE4baseEv
  mov rbx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPKSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEE4baseEv
  mov rax, QWORD PTR [rax]
  cmp rbx, rax
  sete al
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZNKSt15__uniq_ptr_implIbSt14default_deleteIA_bEE6_M_ptrEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZSt3getILm0EJPbSt14default_deleteIA_bEEERKNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERKS8_
  mov rax, QWORD PTR [rax]
  leave
  ret
_ZSt12__niter_baseIPbET_S1_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt10__fill_n_aIPbmbEN9__gnu_cxx11__enable_ifIXsrSt11__is_scalarIT1_E7__valueET_E6__typeES6_T0_RKS4_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov rax, QWORD PTR [rbp-40]
  movzx eax, BYTE PTR [rax]
  mov BYTE PTR [rbp-9], al
  mov rax, QWORD PTR [rbp-32]
  mov QWORD PTR [rbp-8], rax
.L1316:
  cmp QWORD PTR [rbp-8], 0
  je .L1315
  mov rax, QWORD PTR [rbp-24]
  movzx edx, BYTE PTR [rbp-9]
  mov BYTE PTR [rax], dl
  sub QWORD PTR [rbp-8], 1
  add QWORD PTR [rbp-24], 1
  jmp .L1316
.L1315:
  mov rax, QWORD PTR [rbp-24]
  pop rbp
  ret
_ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EEC2EOSB_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rbx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE19_M_get_Tp_allocatorEv
  mov rdi, rax
  call _ZSt4moveIRSaISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEEEONSt16remove_referenceIT_E4typeEOSD_
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE12_Vector_implC1EOSA_
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rbp-32]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE12_Vector_impl12_M_swap_dataERSC_
  nop
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZN9__gnu_cxx17__normal_iteratorIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEC2ERKSB_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  nop
  pop rbp
  ret
_ZNK9__gnu_cxx17__normal_iteratorIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEE4baseEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE14_M_move_assignEOS6_St17integral_constantIbLb1EE:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 56
  mov QWORD PTR [rbp-56], rdi
  mov QWORD PTR [rbp-64], rsi
  mov rdx, QWORD PTR [rbp-56]
  lea rax, [rbp-17]
  mov rsi, rdx
  mov rdi, rax
  call _ZNKSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE13get_allocatorEv
  lea rdx, [rbp-17]
  lea rax, [rbp-48]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEC1ERKS5_
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaINSt7__cxx119sub_matchIPKcEEED1Ev
  mov rax, QWORD PTR [rbp-56]
  lea rdx, [rbp-48]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE12_Vector_impl12_M_swap_dataERS7_
  mov rax, QWORD PTR [rbp-56]
  mov rdx, QWORD PTR [rbp-64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE12_Vector_impl12_M_swap_dataERS7_
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rbx, rax
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rsi, rbx
  mov rdi, rax
  call _ZSt15__alloc_on_moveISaINSt7__cxx119sub_matchIPKcEEEEvRT_S7_
  lea rax, [rbp-48]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EED1Ev
  nop
  add rsp, 56
  pop rbx
  pop rbp
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE11_State_infoISt17integral_constantIbLb0EESt6vectorIS5_S6_EE10_M_visitedEl:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  lea rdx, [rax+24]
  mov rax, QWORD PTR [rbp-16]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt10unique_ptrIA_bSt14default_deleteIS0_EEixEm
  movzx eax, BYTE PTR [rax]
  test al, al
  je .L1324
  mov eax, 1
  jmp .L1325
.L1324:
  mov rax, QWORD PTR [rbp-8]
  lea rdx, [rax+24]
  mov rax, QWORD PTR [rbp-16]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt10unique_ptrIA_bSt14default_deleteIS0_EEixEm
  mov BYTE PTR [rax], 1
  mov eax, 0
.L1325:
  leave
  ret
_ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rcx, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-16]
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 4
  add rax, rcx
  pop rbp
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE16_M_handle_repeatENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-28], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-8]
  movzx eax, BYTE PTR [rax+24]
  xor eax, 1
  test al, al
  je .L1329
  mov rdx, QWORD PTR [rbp-40]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE16_M_rep_once_moreENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
  jmp .L1331
.L1329:
  mov rax, QWORD PTR [rbp-24]
  movzx eax, BYTE PTR [rax+140]
  xor eax, 1
  test al, al
  je .L1331
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-24]
  movzx eax, BYTE PTR [rax+140]
  xor eax, 1
  test al, al
  je .L1331
  mov rdx, QWORD PTR [rbp-40]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE16_M_rep_once_moreENS9_11_Match_modeEl
.L1331:
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE23_M_handle_subexpr_beginENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 64
  mov QWORD PTR [rbp-40], rdi
  mov eax, esi
  mov QWORD PTR [rbp-56], rdx
  mov BYTE PTR [rbp-44], al
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rdx+16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEixEm
  mov QWORD PTR [rbp-16], rax
  mov rax, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-16]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-44]
  mov rax, QWORD PTR [rbp-40]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rax], rdx
  nop
  leave
  ret
_ZNSt7__cxx119sub_matchIPKcEaSERKS3_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt4pairIPKcS1_EaSERKS2_
  mov rax, QWORD PTR [rbp-16]
  movzx edx, BYTE PTR [rax+16]
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+16], dl
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE21_M_handle_subexpr_endENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 80
  mov QWORD PTR [rbp-56], rdi
  mov eax, esi
  mov QWORD PTR [rbp-72], rdx
  mov BYTE PTR [rbp-60], al
  mov rax, QWORD PTR [rbp-56]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-72]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-56]
  mov rdx, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rdx+16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEixEm
  mov QWORD PTR [rbp-16], rax
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rcx]
  mov rdx, QWORD PTR [rcx+8]
  mov QWORD PTR [rbp-48], rax
  mov QWORD PTR [rbp-40], rdx
  mov rax, QWORD PTR [rcx+16]
  mov QWORD PTR [rbp-32], rax
  mov rax, QWORD PTR [rbp-56]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-16]
  mov QWORD PTR [rax+8], rdx
  mov rax, QWORD PTR [rbp-16]
  mov BYTE PTR [rax+16], 1
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-60]
  mov rax, QWORD PTR [rbp-56]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
  lea rdx, [rbp-48]
  mov rax, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx119sub_matchIPKcEaSERKS3_
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE30_M_handle_line_begin_assertionENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-28], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE11_M_at_beginEv
  test al, al
  je .L1338
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
.L1338:
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE28_M_handle_line_end_assertionENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-28], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE9_M_at_endEv
  test al, al
  je .L1341
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
.L1341:
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE23_M_handle_word_boundaryENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-28], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE16_M_word_boundaryEv
  mov edx, eax
  mov rax, QWORD PTR [rbp-8]
  movzx eax, BYTE PTR [rax+24]
  xor eax, 1
  cmp dl, al
  sete al
  test al, al
  je .L1344
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
.L1344:
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE27_M_handle_subexpr_lookaheadENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-28], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE12_M_lookaheadEl
  mov edx, eax
  mov rax, QWORD PTR [rbp-8]
  movzx eax, BYTE PTR [rax+24]
  xor eax, 1
  cmp dl, al
  sete al
  test al, al
  je .L1347
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
.L1347:
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE15_M_handle_matchENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-28], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+40]
  cmp rdx, rax
  je .L1351
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+24]
  movzx eax, BYTE PTR [rax]
  movsx edx, al
  mov rax, QWORD PTR [rbp-8]
  mov esi, edx
  mov rdi, rax
  call _ZNKSt8__detail6_StateIcE10_M_matchesEc
  test al, al
  je .L1348
  mov rax, QWORD PTR [rbp-24]
  lea rcx, [rax+96]
  mov rdx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+8]
  mov rsi, rax
  mov rdi, rcx
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE11_State_infoISt17integral_constantIbLb0EESt6vectorIS5_S6_EE8_M_queueElRKSE_
  jmp .L1348
.L1351:
  nop
.L1348:
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE17_M_handle_backrefENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 104
  mov QWORD PTR [rbp-88], rdi
  mov eax, esi
  mov QWORD PTR [rbp-104], rdx
  mov BYTE PTR [rbp-92], al
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-104]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-40], rax
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rdx+16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEixEm
  mov QWORD PTR [rbp-48], rax
  mov rax, QWORD PTR [rbp-48]
  movzx eax, BYTE PTR [rax+16]
  xor eax, 1
  test al, al
  jne .L1359
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+24]
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-32], rax
.L1356:
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+40]
  cmp QWORD PTR [rbp-24], rax
  je .L1355
  mov rax, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rax+8]
  cmp QWORD PTR [rbp-32], rax
  je .L1355
  add QWORD PTR [rbp-24], 1
  add QWORD PTR [rbp-32], 1
  jmp .L1356
.L1355:
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+48]
  add rax, 16
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  lea rbx, [rax+80]
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+48]
  mov rdi, rax
  call _ZNKSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEE5flagsEv
  mov esi, 1
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  setne al
  movzx ecx, al
  lea rax, [rbp-80]
  mov rdx, rbx
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail16_Backref_matcherIPKcNSt7__cxx1112regex_traitsIcEEEC1EbRKS5_
  mov rax, QWORD PTR [rbp-88]
  mov rcx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-48]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-48]
  mov rsi, QWORD PTR [rax]
  mov rdi, QWORD PTR [rbp-24]
  lea rax, [rbp-80]
  mov r8, rdi
  mov rdi, rax
  call _ZNSt8__detail16_Backref_matcherIPKcNSt7__cxx1112regex_traitsIcEEE8_M_applyES2_S2_S2_S2_
  test al, al
  je .L1352
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+24]
  cmp QWORD PTR [rbp-24], rax
  je .L1357
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+24]
  mov QWORD PTR [rbp-56], rax
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+24], rdx
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-92]
  mov rax, QWORD PTR [rbp-88]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-56]
  mov QWORD PTR [rax+24], rdx
  jmp .L1352
.L1357:
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-92]
  mov rax, QWORD PTR [rbp-88]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
  jmp .L1352
.L1359:
  nop
.L1352:
  add rsp, 104
  pop rbx
  pop rbp
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE16_M_handle_acceptENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov eax, esi
  mov QWORD PTR [rbp-24], rdx
  mov BYTE PTR [rbp-12], al
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+32]
  cmp rdx, rax
  jne .L1361
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+136]
  mov esi, 32
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  test eax, eax
  je .L1361
  mov eax, 1
  jmp .L1362
.L1361:
  mov eax, 0
.L1362:
  test al, al
  jne .L1366
  cmp BYTE PTR [rbp-12], 1
  je .L1365
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+40]
  cmp rdx, rax
  jne .L1360
.L1365:
  mov rax, QWORD PTR [rbp-8]
  movzx eax, BYTE PTR [rax+140]
  xor eax, 1
  test al, al
  je .L1360
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+140], 1
  mov rdx, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEaSERKS6_
  jmp .L1360
.L1366:
  nop
.L1360:
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE21_M_handle_alternativeENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-28], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  mov eax, DWORD PTR [rax+24]
  mov esi, 16
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  setne al
  test al, al
  je .L1368
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-24]
  movzx eax, BYTE PTR [rax+140]
  xor eax, 1
  test al, al
  je .L1370
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
  jmp .L1370
.L1368:
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-24]
  movzx eax, BYTE PTR [rax+140]
  mov BYTE PTR [rbp-9], al
  mov rax, QWORD PTR [rbp-24]
  mov BYTE PTR [rax+140], 0
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-24]
  movzx eax, BYTE PTR [rax+140]
  or al, BYTE PTR [rbp-9]
  mov edx, eax
  mov rax, QWORD PTR [rbp-24]
  mov BYTE PTR [rax+140], dl
.L1370:
  nop
  leave
  ret
_ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE15_M_erase_at_endEPS8_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+8]
  sub rax, QWORD PTR [rbp-32]
  sar rax, 5
  mov QWORD PTR [rbp-8], rax
  cmp QWORD PTR [rbp-8], 0
  je .L1373
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-32]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt8_DestroyIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEES9_EvT_SB_RSaIT0_E
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rbp-32]
  mov QWORD PTR [rax+8], rdx
.L1373:
  nop
  leave
  ret
_ZStneINSt7__cxx119sub_matchIPKcEEEbRKSaIT_ES8_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov eax, 0
  pop rbp
  ret
_ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE5clearEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE15_M_erase_at_endEPS4_
  nop
  leave
  ret
_ZSt15__alloc_on_copyISaINSt7__cxx119sub_matchIPKcEEEEvRT_RKS6_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rdx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt18__do_alloc_on_copyISaINSt7__cxx119sub_matchIPKcEEEEvRT_RKS6_St17integral_constantIbLb0EE
  nop
  leave
  ret
_ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE8capacityEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  sub rdx, rax
  mov rax, rdx
  sar rax, 3
  mov rdx, rax
  movabs rax, -6148914691236517205
  imul rax, rdx
  pop rbp
  ret
_ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE5beginEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-8], rax
  lea rdx, [rbp-8]
  lea rax, [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPKNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEC1ERKS7_
  mov rax, QWORD PTR [rbp-16]
  leave
  ret
_ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE3endEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+8]
  mov QWORD PTR [rbp-8], rax
  lea rdx, [rbp-8]
  lea rax, [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPKNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEC1ERKS7_
  mov rax, QWORD PTR [rbp-16]
  leave
  ret
_ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE20_M_allocate_and_copyIN9__gnu_cxx17__normal_iteratorIPKS4_S6_EEEEPS4_mT_SE_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 56
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  mov QWORD PTR [rbp-56], rdx
  mov QWORD PTR [rbp-64], rcx
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rbp-48]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE11_M_allocateEm
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rcx, rax
  mov rdx, QWORD PTR [rbp-24]
  mov rsi, QWORD PTR [rbp-64]
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZSt22__uninitialized_copy_aIN9__gnu_cxx17__normal_iteratorIPKNSt7__cxx119sub_matchIPKcEESt6vectorIS6_SaIS6_EEEEPS6_S6_ET0_T_SF_SE_RSaIT1_E
  mov rax, QWORD PTR [rbp-24]
  jmp .L1390
  mov rdi, rax
  call __cxa_begin_catch
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rbp-48]
  mov rcx, QWORD PTR [rbp-24]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE13_M_deallocateEPS4_m
  call __cxa_rethrow
  mov rbx, rax
  call __cxa_end_catch
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1390:
  add rsp, 56
  pop rbx
  pop rbp
  ret

_ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKNSt7__cxx119sub_matchIPKcEESt6vectorIS6_SaIS6_EEEENS1_IPS6_SB_EEET0_T_SG_SF_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIPKNSt7__cxx119sub_matchIPKcEESt6vectorIS6_SaIS6_EEEEET_SD_
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIPKNSt7__cxx119sub_matchIPKcEESt6vectorIS6_SaIS6_EEEEET_SD_
  mov rcx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdx, rax
  mov rsi, rbx
  mov rdi, rcx
  call _ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKNSt7__cxx119sub_matchIPKcEESt6vectorIS6_SaIS6_EEEENS1_IPS6_SB_EEET1_T0_SG_SF_
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS6_SaIS6_EEEES6_EvT_SC_RSaIT0_E:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS6_SaIS6_EEEEEvT_SC_
  nop
  leave
  ret
_ZSt4copyIPNSt7__cxx119sub_matchIPKcEES5_ET0_T_S7_S6_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt12__miter_baseIPNSt7__cxx119sub_matchIPKcEEET_S6_
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt12__miter_baseIPNSt7__cxx119sub_matchIPKcEEET_S6_
  mov rcx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdx, rax
  mov rsi, rbx
  mov rdi, rcx
  call _ZSt14__copy_move_a2ILb0EPNSt7__cxx119sub_matchIPKcEES5_ET1_T0_S7_S6_
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZSt22__uninitialized_copy_aIPNSt7__cxx119sub_matchIPKcEES5_S4_ET0_T_S7_S6_RSaIT1_E:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov QWORD PTR [rbp-32], rcx
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt18uninitialized_copyIPNSt7__cxx119sub_matchIPKcEES5_ET0_T_S7_S6_
  leave
  ret
_ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE11_State_infoISt17integral_constantIbLb1EESt6vectorIS5_S6_EE10_M_visitedEl:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov eax, 0
  pop rbp
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE16_M_handle_repeatENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-28], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-8]
  movzx eax, BYTE PTR [rax+24]
  xor eax, 1
  test al, al
  je .L1401
  mov rdx, QWORD PTR [rbp-40]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE16_M_rep_once_moreENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-24]
  movzx eax, BYTE PTR [rax+116]
  xor eax, 1
  test al, al
  je .L1403
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
  jmp .L1403
.L1401:
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-24]
  movzx eax, BYTE PTR [rax+116]
  xor eax, 1
  test al, al
  je .L1403
  mov rdx, QWORD PTR [rbp-40]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE16_M_rep_once_moreENS9_11_Match_modeEl
.L1403:
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE23_M_handle_subexpr_beginENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 64
  mov QWORD PTR [rbp-40], rdi
  mov eax, esi
  mov QWORD PTR [rbp-56], rdx
  mov BYTE PTR [rbp-44], al
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rdx+16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEixEm
  mov QWORD PTR [rbp-16], rax
  mov rax, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-16]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-44]
  mov rax, QWORD PTR [rbp-40]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rax], rdx
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE21_M_handle_subexpr_endENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 80
  mov QWORD PTR [rbp-56], rdi
  mov eax, esi
  mov QWORD PTR [rbp-72], rdx
  mov BYTE PTR [rbp-60], al
  mov rax, QWORD PTR [rbp-56]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-72]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-56]
  mov rdx, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rdx+16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEixEm
  mov QWORD PTR [rbp-16], rax
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rcx]
  mov rdx, QWORD PTR [rcx+8]
  mov QWORD PTR [rbp-48], rax
  mov QWORD PTR [rbp-40], rdx
  mov rax, QWORD PTR [rcx+16]
  mov QWORD PTR [rbp-32], rax
  mov rax, QWORD PTR [rbp-56]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-16]
  mov QWORD PTR [rax+8], rdx
  mov rax, QWORD PTR [rbp-16]
  mov BYTE PTR [rax+16], 1
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-60]
  mov rax, QWORD PTR [rbp-56]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
  lea rdx, [rbp-48]
  mov rax, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt7__cxx119sub_matchIPKcEaSERKS3_
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE30_M_handle_line_begin_assertionENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-28], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE11_M_at_beginEv
  test al, al
  je .L1408
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
.L1408:
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE28_M_handle_line_end_assertionENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-28], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE9_M_at_endEv
  test al, al
  je .L1411
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
.L1411:
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE23_M_handle_word_boundaryENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-28], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE16_M_word_boundaryEv
  mov edx, eax
  mov rax, QWORD PTR [rbp-8]
  movzx eax, BYTE PTR [rax+24]
  xor eax, 1
  cmp dl, al
  sete al
  test al, al
  je .L1414
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
.L1414:
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE27_M_handle_subexpr_lookaheadENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-28], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE12_M_lookaheadEl
  mov edx, eax
  mov rax, QWORD PTR [rbp-8]
  movzx eax, BYTE PTR [rax+24]
  xor eax, 1
  cmp dl, al
  sete al
  test al, al
  je .L1417
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
.L1417:
  nop
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE15_M_handle_matchENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-28], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+40]
  cmp rdx, rax
  je .L1421
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+24]
  movzx eax, BYTE PTR [rax]
  movsx edx, al
  mov rax, QWORD PTR [rbp-8]
  mov esi, edx
  mov rdi, rax
  call _ZNKSt8__detail6_StateIcE10_M_matchesEc
  test al, al
  je .L1418
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+24]
  lea rdx, [rax+1]
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+24], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+24]
  lea rdx, [rax-1]
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+24], rdx
  jmp .L1418
.L1421:
  nop
.L1418:
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE17_M_handle_backrefENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 104
  mov QWORD PTR [rbp-88], rdi
  mov eax, esi
  mov QWORD PTR [rbp-104], rdx
  mov BYTE PTR [rbp-92], al
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-104]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-40], rax
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rdx+16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEixEm
  mov QWORD PTR [rbp-48], rax
  mov rax, QWORD PTR [rbp-48]
  movzx eax, BYTE PTR [rax+16]
  xor eax, 1
  test al, al
  jne .L1429
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+24]
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-32], rax
.L1426:
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+40]
  cmp QWORD PTR [rbp-24], rax
  je .L1425
  mov rax, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rax+8]
  cmp QWORD PTR [rbp-32], rax
  je .L1425
  add QWORD PTR [rbp-24], 1
  add QWORD PTR [rbp-32], 1
  jmp .L1426
.L1425:
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+48]
  add rax, 16
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessIKNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  lea rbx, [rax+80]
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+48]
  mov rdi, rax
  call _ZNKSt7__cxx1111basic_regexIcNS_12regex_traitsIcEEE5flagsEv
  mov esi, 1
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  setne al
  movzx ecx, al
  lea rax, [rbp-80]
  mov rdx, rbx
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail16_Backref_matcherIPKcNSt7__cxx1112regex_traitsIcEEEC1EbRKS5_
  mov rax, QWORD PTR [rbp-88]
  mov rcx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-48]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-48]
  mov rsi, QWORD PTR [rax]
  mov rdi, QWORD PTR [rbp-24]
  lea rax, [rbp-80]
  mov r8, rdi
  mov rdi, rax
  call _ZNSt8__detail16_Backref_matcherIPKcNSt7__cxx1112regex_traitsIcEEE8_M_applyES2_S2_S2_S2_
  test al, al
  je .L1422
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+24]
  cmp QWORD PTR [rbp-24], rax
  je .L1427
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+24]
  mov QWORD PTR [rbp-56], rax
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+24], rdx
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-92]
  mov rax, QWORD PTR [rbp-88]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-56]
  mov QWORD PTR [rax+24], rdx
  jmp .L1422
.L1427:
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-92]
  mov rax, QWORD PTR [rbp-88]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
  jmp .L1422
.L1429:
  nop
.L1422:
  add rsp, 104
  pop rbx
  pop rbp
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE16_M_handle_acceptENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-28], al
  cmp BYTE PTR [rbp-28], 0
  jne .L1431
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+40]
  cmp rdx, rax
  sete dl
  mov rax, QWORD PTR [rbp-24]
  mov BYTE PTR [rax+116], dl
  jmp .L1432
.L1431:
  mov rax, QWORD PTR [rbp-24]
  mov BYTE PTR [rax+116], 1
.L1432:
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+32]
  cmp rdx, rax
  jne .L1433
  mov rax, QWORD PTR [rbp-24]
  mov eax, DWORD PTR [rax+112]
  mov esi, 32
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  test eax, eax
  je .L1433
  mov eax, 1
  jmp .L1434
.L1433:
  mov eax, 0
.L1434:
  test al, al
  je .L1435
  mov rax, QWORD PTR [rbp-24]
  mov BYTE PTR [rax+116], 0
.L1435:
  mov rax, QWORD PTR [rbp-24]
  movzx eax, BYTE PTR [rax+116]
  test al, al
  je .L1430
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  mov eax, DWORD PTR [rax+24]
  mov esi, 16
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  setne al
  test al, al
  je .L1437
  mov rdx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEaSERKS6_
  jmp .L1430
.L1437:
  mov rax, QWORD PTR [rbp-24]
  add rax, 96
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE11_State_infoISt17integral_constantIbLb1EESt6vectorIS5_S6_EE14_M_get_sol_posEv
  mov rax, QWORD PTR [rax]
  test rax, rax
  je .L1438
  mov rax, QWORD PTR [rbp-24]
  add rax, 96
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE11_State_infoISt17integral_constantIbLb1EESt6vectorIS5_S6_EE14_M_get_sol_posEv
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+32]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+32]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_
  cmp rbx, rax
  jge .L1439
.L1438:
  mov eax, 1
  jmp .L1440
.L1439:
  mov eax, 0
.L1440:
  test al, al
  je .L1430
  mov rax, QWORD PTR [rbp-24]
  mov rbx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-24]
  add rax, 96
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE11_State_infoISt17integral_constantIbLb1EESt6vectorIS5_S6_EE14_M_get_sol_posEv
  mov QWORD PTR [rax], rbx
  mov rdx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEaSERKS6_
.L1430:
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE21_M_handle_alternativeENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-28], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+56]
  mov eax, DWORD PTR [rax+24]
  mov esi, 16
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  setne al
  test al, al
  je .L1442
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-24]
  movzx eax, BYTE PTR [rax+116]
  xor eax, 1
  test al, al
  je .L1444
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
  jmp .L1444
.L1442:
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-24]
  movzx eax, BYTE PTR [rax+116]
  mov BYTE PTR [rbp-9], al
  mov rax, QWORD PTR [rbp-24]
  mov BYTE PTR [rax+116], 0
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  movzx ecx, BYTE PTR [rbp-28]
  mov rax, QWORD PTR [rbp-24]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-24]
  movzx eax, BYTE PTR [rax+116]
  or al, BYTE PTR [rbp-9]
  mov edx, eax
  mov rax, QWORD PTR [rbp-24]
  mov BYTE PTR [rax+116], dl
.L1444:
  nop
  leave
  ret
_ZNSt14__shared_countILN9__gnu_cxx12_Lock_policyE2EEC2INSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS9_EJRKSt6localeRNSt15regex_constants18syntax_option_typeEEEERPT_St20_Sp_alloc_shared_tagIT0_EDpOT1_:
  push rbp
  mov rbp, rsp
  push r15
  push r14
  push r13
  push r12
  push rbx
  sub rsp, 104
  mov QWORD PTR [rbp-104], rdi
  mov QWORD PTR [rbp-112], rsi
  mov QWORD PTR [rbp-120], rdx
  mov QWORD PTR [rbp-128], rcx
  mov QWORD PTR [rbp-136], r8
  mov rdx, QWORD PTR [rbp-120]
  lea rax, [rbp-66]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSaISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EEEC1IS5_EERKSaIT_E
  lea rax, [rbp-96]
  lea rdx, [rbp-66]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt18__allocate_guardedISaISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS6_ELN9__gnu_cxx12_Lock_policyE2EEEESt15__allocated_ptrIT_ERSD_
  lea rax, [rbp-96]
  mov rdi, rax
  call _ZNSt15__allocated_ptrISaISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS6_ELN9__gnu_cxx12_Lock_policyE2EEEE3getEv
  mov QWORD PTR [rbp-56], rax
  mov rdx, QWORD PTR [rbp-120]
  lea rax, [rbp-65]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSaINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEEC1ERKS5_
  lea r13, [rbp-65]
  mov rax, QWORD PTR [rbp-128]
  mov rdi, rax
  call _ZSt7forwardIRKSt6localeEOT_RNSt16remove_referenceIS3_E4typeE
  mov r14, rax
  mov rax, QWORD PTR [rbp-136]
  mov rdi, rax
  call _ZSt7forwardIRNSt15regex_constants18syntax_option_typeEEOT_RNSt16remove_referenceIS3_E4typeE
  mov r15, rax
  mov r12, QWORD PTR [rbp-56]
  mov rsi, r12
  mov edi, 104
  call _ZnwmPv
  mov rbx, rax
  mov rcx, r15
  mov rdx, r14
  mov rsi, r13
  mov rdi, rbx
  call _ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EEC1IJRKSt6localeRNSt15regex_constants18syntax_option_typeEEEES6_DpOT_
  mov QWORD PTR [rbp-64], rbx
  lea rax, [rbp-65]
  mov rdi, rax
  call _ZNSaINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEED1Ev
  lea rax, [rbp-96]
  mov esi, 0
  mov rdi, rax
  call _ZNSt15__allocated_ptrISaISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS6_ELN9__gnu_cxx12_Lock_policyE2EEEEaSEDn
  mov rax, QWORD PTR [rbp-104]
  mov rdx, QWORD PTR [rbp-64]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZNSt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EE6_M_ptrEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-112]
  mov QWORD PTR [rax], rdx
  lea rax, [rbp-96]
  mov rdi, rax
  call _ZNSt15__allocated_ptrISaISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS6_ELN9__gnu_cxx12_Lock_policyE2EEEED1Ev
  lea rax, [rbp-66]
  mov rdi, rax
  call _ZNSaISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EEED1Ev
  jmp .L1450
  mov r13, rax
  mov rsi, r12
  mov rdi, rbx
  call _ZdlPvS_
  mov rbx, r13
  lea rax, [rbp-65]
  mov rdi, rax
  call _ZNSaINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEEED1Ev
  lea rax, [rbp-96]
  mov rdi, rax
  call _ZNSt15__allocated_ptrISaISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS6_ELN9__gnu_cxx12_Lock_policyE2EEEED1Ev
  jmp .L1447
  mov rbx, rax
.L1447:
  lea rax, [rbp-66]
  mov rdi, rax
  call _ZNSaISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1450:
  add rsp, 104
  pop rbx
  pop r12
  pop r13
  pop r14
  pop r15
  pop rbp
  ret
_ZNSt12__shared_ptrINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2EE31_M_enable_shared_from_this_withIS5_S5_EENSt9enable_ifIXntsrNS8_15__has_esft_baseIT0_vEE5valueEvE4typeEPT_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  nop
  pop rbp
  ret
_ZNSaINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEC2Ev
  nop
  leave
  ret
_ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_EC2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+8], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+16], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+24], 0
  nop
  pop rbp
  ret
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE15_M_allocate_mapEm:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  lea rax, [rbp-17]
  mov rdx, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZNKSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE20_M_get_map_allocatorEv
  mov rdx, QWORD PTR [rbp-48]
  lea rax, [rbp-17]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaIPNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEE8allocateERS7_m
  mov rbx, rax
  nop
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaIPNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEED1Ev
  mov rax, rbx
  jmp .L1458
  mov rbx, rax
  lea rax, [rbp-17]
  mov rdi, rax
  call _ZNSaIPNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1458:
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE15_M_create_nodesEPPS5_S9_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 56
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  mov QWORD PTR [rbp-56], rdx
  mov rax, QWORD PTR [rbp-48]
  mov QWORD PTR [rbp-24], rax
.L1461:
  mov rax, QWORD PTR [rbp-24]
  cmp rax, QWORD PTR [rbp-56]
  jnb .L1466
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE16_M_allocate_nodeEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax], rdx
  add QWORD PTR [rbp-24], 8
  jmp .L1461
  mov rdi, rax
  call __cxa_begin_catch
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE16_M_destroy_nodesEPPS5_S9_
  call __cxa_rethrow
  mov rbx, rax
  call __cxa_end_catch
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1466:
  nop
  add rsp, 56
  pop rbx
  pop rbp
  ret

_ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_E11_M_set_nodeEPS7_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rbp-32]
  mov QWORD PTR [rax+24], rdx
  mov rax, QWORD PTR [rbp-32]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+8], rdx
  mov rax, QWORD PTR [rbp-24]
  mov rbx, QWORD PTR [rax+8]
  call _ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_E14_S_buffer_sizeEv
  mov rdx, rax
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 3
  lea rdx, [rbx+rax]
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+16], rdx
  nop
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZNKSt6vectorImSaImEE8max_sizeEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt12_Vector_baseImSaImEE19_M_get_Tp_allocatorEv
  mov rdi, rax
  call _ZNSt16allocator_traitsISaImEE8max_sizeERKS0_
  leave
  ret
_ZNKSt6vectorImSaImEE4sizeEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax]
  sub rdx, rax
  mov rax, rdx
  sar rax, 3
  pop rbp
  ret
_ZNSt16allocator_traitsISaImEE8allocateERS0_m:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov edx, 0
  mov rsi, rcx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorImE8allocateEmPKv
  leave
  ret
_ZSt32__make_move_if_noexcept_iteratorImSt13move_iteratorIPmEET0_PT_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt13move_iteratorIPmEC1ES0_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZSt22__uninitialized_copy_aISt13move_iteratorIPmES1_mET0_T_S4_S3_RSaIT1_E:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov QWORD PTR [rbp-32], rcx
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt18uninitialized_copyISt13move_iteratorIPmES1_ET0_T_S4_S3_
  leave
  ret
_ZSt7forwardINSt8__detail6_StateIcEEEOT_RNSt16remove_referenceIS3_E4typeE:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt16allocator_traitsISaINSt8__detail6_StateIcEEEE9constructIS2_JS2_EEEvRS3_PT_DpOT0_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt7forwardINSt8__detail6_StateIcEEEOT_RNSt16remove_referenceIS3_E4typeE
  mov rdx, rax
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorINSt8__detail6_StateIcEEE9constructIS3_JS3_EEEvPT_DpOT0_
  nop
  leave
  ret
_ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EE17_M_realloc_insertIJS2_EEEvN9__gnu_cxx17__normal_iteratorIPS2_S4_EEDpOT_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 104
  mov QWORD PTR [rbp-88], rdi
  mov QWORD PTR [rbp-96], rsi
  mov QWORD PTR [rbp-104], rdx
  mov rax, QWORD PTR [rbp-88]
  mov edx, OFFSET FLAT:.LC24
  mov esi, 1
  mov rdi, rax
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EE12_M_check_lenEmPKc
  mov QWORD PTR [rbp-32], rax
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-40], rax
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+8]
  mov QWORD PTR [rbp-48], rax
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EE5beginEv
  mov QWORD PTR [rbp-72], rax
  lea rdx, [rbp-72]
  lea rax, [rbp-96]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxxmiIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKSB_SE_
  mov QWORD PTR [rbp-56], rax
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-32]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt8__detail6_StateIcEESaIS2_EE11_M_allocateEm
  mov QWORD PTR [rbp-64], rax
  mov rax, QWORD PTR [rbp-64]
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-104]
  mov rdi, rax
  call _ZSt7forwardINSt8__detail6_StateIcEEEOT_RNSt16remove_referenceIS3_E4typeE
  mov rsi, rax
  mov rdx, QWORD PTR [rbp-56]
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 4
  mov rdx, rax
  mov rax, QWORD PTR [rbp-64]
  lea rcx, [rdx+rax]
  mov rax, QWORD PTR [rbp-88]
  mov rdx, rsi
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaINSt8__detail6_StateIcEEEE9constructIS2_JS2_EEEvRS3_PT_DpOT0_
  mov QWORD PTR [rbp-24], 0
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt8__detail6_StateIcEESaIS2_EE19_M_get_Tp_allocatorEv
  mov rbx, rax
  lea rax, [rbp-96]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEE4baseEv
  mov rsi, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-64]
  mov rax, QWORD PTR [rbp-40]
  mov rcx, rbx
  mov rdi, rax
  call _ZSt34__uninitialized_move_if_noexcept_aIPNSt8__detail6_StateIcEES3_SaIS2_EET0_T_S6_S5_RT1_
  mov QWORD PTR [rbp-24], rax
  add QWORD PTR [rbp-24], 48
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt8__detail6_StateIcEESaIS2_EE19_M_get_Tp_allocatorEv
  mov rbx, rax
  lea rax, [rbp-96]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEE4baseEv
  mov rax, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-24]
  mov rsi, QWORD PTR [rbp-48]
  mov rcx, rbx
  mov rdi, rax
  call _ZSt34__uninitialized_move_if_noexcept_aIPNSt8__detail6_StateIcEES3_SaIS2_EET0_T_S6_S5_RT1_
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt8__detail6_StateIcEESaIS2_EE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rcx, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt8_DestroyIPNSt8__detail6_StateIcEES2_EvT_S4_RSaIT0_E
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rdx+16]
  sub rdx, QWORD PTR [rbp-40]
  mov rcx, rdx
  sar rcx, 4
  movabs rdx, -6148914691236517205
  imul rdx, rcx
  mov rcx, QWORD PTR [rbp-40]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt8__detail6_StateIcEESaIS2_EE13_M_deallocateEPS2_m
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-64]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+8], rdx
  mov rdx, QWORD PTR [rbp-32]
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 4
  mov rdx, rax
  mov rax, QWORD PTR [rbp-64]
  add rdx, rax
  mov rax, QWORD PTR [rbp-88]
  mov QWORD PTR [rax+16], rdx
  jmp .L1488
  mov rdi, rax
  call __cxa_begin_catch
  cmp QWORD PTR [rbp-24], 0
  jne .L1483
  mov rdx, QWORD PTR [rbp-56]
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 4
  mov rdx, rax
  mov rax, QWORD PTR [rbp-64]
  add rdx, rax
  mov rax, QWORD PTR [rbp-88]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaINSt8__detail6_StateIcEEEE7destroyIS2_EEvRS3_PT_
  jmp .L1484
.L1483:
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt8__detail6_StateIcEESaIS2_EE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rcx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-64]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt8_DestroyIPNSt8__detail6_StateIcEES2_EvT_S4_RSaIT0_E
.L1484:
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-32]
  mov rcx, QWORD PTR [rbp-64]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt12_Vector_baseINSt8__detail6_StateIcEESaIS2_EE13_M_deallocateEPS2_m
  call __cxa_rethrow
  mov rbx, rax
  call __cxa_end_catch
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1488:
  add rsp, 104
  pop rbx
  pop rbp
  ret

_ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EE4backEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EE3endEv
  mov QWORD PTR [rbp-8], rax
  lea rax, [rbp-8]
  mov esi, 1
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEEmiEl
  mov QWORD PTR [rbp-16], rax
  lea rax, [rbp-16]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPNSt8__detail6_StateIcEESt6vectorIS3_SaIS3_EEEdeEv
  nop
  leave
  ret
_ZSt4moveIRSt9_Any_dataEONSt16remove_referenceIT_E4typeEOS3_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt4swapISt9_Any_dataENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS5_ESt18is_move_assignableIS5_EEE5valueEvE4typeERS5_SF_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt4moveIRSt9_Any_dataEONSt16remove_referenceIT_E4typeEOS3_
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-16], rax
  mov QWORD PTR [rbp-8], rdx
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt4moveIRSt9_Any_dataEONSt16remove_referenceIT_E4typeEOS3_
  mov rcx, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rcx], rax
  mov QWORD PTR [rcx+8], rdx
  lea rax, [rbp-16]
  mov rdi, rax
  call _ZSt4moveIRSt9_Any_dataEONSt16remove_referenceIT_E4typeEOS3_
  mov rcx, QWORD PTR [rbp-32]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rcx], rax
  mov QWORD PTR [rcx+8], rdx
  nop
  leave
  ret
_ZSt4moveIRPFbRSt9_Any_dataRKS0_St18_Manager_operationEEONSt16remove_referenceIT_E4typeEOS9_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt4swapIPFbRSt9_Any_dataRKS0_St18_Manager_operationEENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISB_ESt18is_move_assignableISB_EEE5valueEvE4typeERSB_SL_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt4moveIRPFbRSt9_Any_dataRKS0_St18_Manager_operationEEONSt16remove_referenceIT_E4typeEOS9_
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt4moveIRPFbRSt9_Any_dataRKS0_St18_Manager_operationEEONSt16remove_referenceIT_E4typeEOS9_
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax], rdx
  lea rax, [rbp-8]
  mov rdi, rax
  call _ZSt4moveIRPFbRSt9_Any_dataRKS0_St18_Manager_operationEEONSt16remove_referenceIT_E4typeEOS9_
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-32]
  mov QWORD PTR [rax], rdx
  nop
  leave
  ret
_ZSt4moveIRPFbRKSt9_Any_dataOcEEONSt16remove_referenceIT_E4typeEOS8_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt4swapIPFbRKSt9_Any_dataOcEENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISA_ESt18is_move_assignableISA_EEE5valueEvE4typeERSA_SK_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt4moveIRPFbRKSt9_Any_dataOcEEONSt16remove_referenceIT_E4typeEOS8_
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt4moveIRPFbRKSt9_Any_dataOcEEONSt16remove_referenceIT_E4typeEOS8_
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax], rdx
  lea rax, [rbp-8]
  mov rdi, rax
  call _ZSt4moveIRPFbRKSt9_Any_dataOcEEONSt16remove_referenceIT_E4typeEOS8_
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-32]
  mov QWORD PTR [rax], rdx
  nop
  leave
  ret
_ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE20_M_insert_line_beginEv:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 72
  mov QWORD PTR [rbp-72], rdi
  lea rax, [rbp-64]
  mov esi, 4
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1ENS_7_OpcodeE
  lea rdx, [rbp-64]
  mov rax, QWORD PTR [rbp-72]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_stateENS_6_StateIcEE
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  jmp .L1504
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1504:
  add rsp, 72
  pop rbx
  pop rbp
  ret
_ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE18_M_insert_line_endEv:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 72
  mov QWORD PTR [rbp-72], rdi
  lea rax, [rbp-64]
  mov esi, 5
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1ENS_7_OpcodeE
  lea rdx, [rbp-64]
  mov rax, QWORD PTR [rbp-72]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_stateENS_6_StateIcEE
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  jmp .L1509
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1509:
  add rsp, 72
  pop rbx
  pop rbp
  ret
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEixEm:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-16]
  add rax, rdx
  leave
  ret
_ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE20_M_insert_word_boundEb:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 120
  mov QWORD PTR [rbp-120], rdi
  mov eax, esi
  mov BYTE PTR [rbp-124], al
  lea rax, [rbp-112]
  mov esi, 6
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1ENS_7_OpcodeE
  movzx eax, BYTE PTR [rbp-124]
  mov BYTE PTR [rbp-88], al
  lea rax, [rbp-112]
  mov rdi, rax
  call _ZSt4moveIRNSt8__detail6_StateIcEEEONSt16remove_referenceIT_E4typeEOS5_
  mov rdx, rax
  lea rax, [rbp-64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1EOS1_
  lea rdx, [rbp-64]
  mov rax, QWORD PTR [rbp-120]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_stateENS_6_StateIcEE
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  lea rax, [rbp-112]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  jmp .L1516
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  lea rax, [rbp-112]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1516:
  add rsp, 120
  pop rbx
  pop rbp
  ret
_ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE19_M_insert_lookaheadElb:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 136
  mov QWORD PTR [rbp-120], rdi
  mov QWORD PTR [rbp-128], rsi
  mov eax, edx
  mov BYTE PTR [rbp-132], al
  lea rax, [rbp-112]
  mov esi, 7
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1ENS_7_OpcodeE
  mov rax, QWORD PTR [rbp-128]
  mov QWORD PTR [rbp-96], rax
  movzx eax, BYTE PTR [rbp-132]
  mov BYTE PTR [rbp-88], al
  lea rax, [rbp-112]
  mov rdi, rax
  call _ZSt4moveIRNSt8__detail6_StateIcEEEONSt16remove_referenceIT_E4typeEOS5_
  mov rdx, rax
  lea rax, [rbp-64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1EOS1_
  lea rdx, [rbp-64]
  mov rax, QWORD PTR [rbp-120]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_stateENS_6_StateIcEE
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  lea rax, [rbp-112]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  jmp .L1521
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  lea rax, [rbp-112]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1521:
  add rsp, 136
  pop rbx
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE27_M_insert_any_matcher_posixILb0ELb0EEEvv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 96
  mov QWORD PTR [rbp-104], rdi
  mov rax, QWORD PTR [rbp-104]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-104]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-104]
  mov rdx, QWORD PTR [rax+384]
  lea rax, [rbp-17]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEC1ERKS3_
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1INSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb0EEEvvEET_
  lea rax, [rbp-64]
  mov rsi, rax
  mov rdi, r12
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_matcherESt8functionIFbcEE
  mov r12, rax
  mov rax, QWORD PTR [rbp-104]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-96]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-96]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  jmp .L1525
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1525:
  add rsp, 96
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE27_M_insert_any_matcher_posixILb0ELb1EEEvv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 96
  mov QWORD PTR [rbp-104], rdi
  mov rax, QWORD PTR [rbp-104]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-104]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-104]
  mov rdx, QWORD PTR [rax+384]
  lea rax, [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEC1ERKS3_
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1INSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0ELb1EEEvvEET_
  lea rax, [rbp-64]
  mov rsi, rax
  mov rdi, r12
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_matcherESt8functionIFbcEE
  mov r12, rax
  mov rax, QWORD PTR [rbp-104]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-96]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-96]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  jmp .L1529
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1529:
  add rsp, 96
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE27_M_insert_any_matcher_posixILb1ELb0EEEvv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 96
  mov QWORD PTR [rbp-104], rdi
  mov rax, QWORD PTR [rbp-104]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-104]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-104]
  mov rdx, QWORD PTR [rax+384]
  lea rax, [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEC1ERKS3_
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1INSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb0EEEvvEET_
  lea rax, [rbp-64]
  mov rsi, rax
  mov rdi, r12
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_matcherESt8functionIFbcEE
  mov r12, rax
  mov rax, QWORD PTR [rbp-104]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-96]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-96]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  jmp .L1533
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1533:
  add rsp, 96
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE27_M_insert_any_matcher_posixILb1ELb1EEEvv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 96
  mov QWORD PTR [rbp-104], rdi
  mov rax, QWORD PTR [rbp-104]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-104]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-104]
  mov rdx, QWORD PTR [rax+384]
  lea rax, [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEC1ERKS3_
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1INSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1ELb1EEEvvEET_
  lea rax, [rbp-64]
  mov rsi, rax
  mov rdi, r12
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_matcherESt8functionIFbcEE
  mov r12, rax
  mov rax, QWORD PTR [rbp-104]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-96]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-96]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  jmp .L1537
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1537:
  add rsp, 96
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE26_M_insert_any_matcher_ecmaILb0ELb0EEEvv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 96
  mov QWORD PTR [rbp-104], rdi
  mov rax, QWORD PTR [rbp-104]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-104]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-104]
  mov rdx, QWORD PTR [rax+384]
  lea rax, [rbp-17]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb0EEC1ERKS3_
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1INSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb0EEEvvEET_
  lea rax, [rbp-64]
  mov rsi, rax
  mov rdi, r12
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_matcherESt8functionIFbcEE
  mov r12, rax
  mov rax, QWORD PTR [rbp-104]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-96]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-96]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  jmp .L1541
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1541:
  add rsp, 96
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE26_M_insert_any_matcher_ecmaILb0ELb1EEEvv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 96
  mov QWORD PTR [rbp-104], rdi
  mov rax, QWORD PTR [rbp-104]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-104]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-104]
  mov rdx, QWORD PTR [rax+384]
  lea rax, [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EEC1ERKS3_
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1INSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0ELb1EEEvvEET_
  lea rax, [rbp-64]
  mov rsi, rax
  mov rdi, r12
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_matcherESt8functionIFbcEE
  mov r12, rax
  mov rax, QWORD PTR [rbp-104]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-96]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-96]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  jmp .L1545
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1545:
  add rsp, 96
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE26_M_insert_any_matcher_ecmaILb1ELb0EEEvv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 96
  mov QWORD PTR [rbp-104], rdi
  mov rax, QWORD PTR [rbp-104]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-104]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-104]
  mov rdx, QWORD PTR [rax+384]
  lea rax, [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEC1ERKS3_
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1INSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb0EEEvvEET_
  lea rax, [rbp-64]
  mov rsi, rax
  mov rdi, r12
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_matcherESt8functionIFbcEE
  mov r12, rax
  mov rax, QWORD PTR [rbp-104]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-96]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-96]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  jmp .L1549
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1549:
  add rsp, 96
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE26_M_insert_any_matcher_ecmaILb1ELb1EEEvv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 96
  mov QWORD PTR [rbp-104], rdi
  mov rax, QWORD PTR [rbp-104]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-104]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r12, rax
  mov rax, QWORD PTR [rbp-104]
  mov rdx, QWORD PTR [rax+384]
  lea rax, [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEC1ERKS3_
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1INSt8__detail11_AnyMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1ELb1EEEvvEET_
  lea rax, [rbp-64]
  mov rsi, rax
  mov rdi, r12
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_matcherESt8functionIFbcEE
  mov r12, rax
  mov rax, QWORD PTR [rbp-104]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-96]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-96]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  jmp .L1553
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1553:
  add rsp, 96
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE11_M_try_charEv:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-40], rdi
  mov BYTE PTR [rbp-17], 0
  mov rax, QWORD PTR [rbp-40]
  mov esi, 2
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1555
  mov BYTE PTR [rbp-17], 1
  mov rax, QWORD PTR [rbp-40]
  lea rbx, [rax+272]
  mov rax, QWORD PTR [rbp-40]
  mov esi, 8
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE16_M_cur_int_valueEi
  movsx eax, al
  mov edx, eax
  mov esi, 1
  mov rdi, rbx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  jmp .L1556
.L1555:
  mov rax, QWORD PTR [rbp-40]
  mov esi, 3
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1557
  mov BYTE PTR [rbp-17], 1
  mov rax, QWORD PTR [rbp-40]
  lea rbx, [rax+272]
  mov rax, QWORD PTR [rbp-40]
  mov esi, 16
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE16_M_cur_int_valueEi
  movsx eax, al
  mov edx, eax
  mov esi, 1
  mov rdi, rbx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignEmc
  jmp .L1556
.L1557:
  mov rax, QWORD PTR [rbp-40]
  mov esi, 1
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  test al, al
  je .L1556
  mov BYTE PTR [rbp-17], 1
.L1556:
  movzx eax, BYTE PTR [rbp-17]
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE22_M_insert_char_matcherILb0ELb0EEEvv:
  push rbp
  mov rbp, rsp
  push r13
  push r12
  push rbx
  sub rsp, 104
  mov QWORD PTR [rbp-120], rdi
  mov rax, QWORD PTR [rbp-120]
  lea r12, [rax+304]
  mov rax, QWORD PTR [rbp-120]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r13, rax
  mov rax, QWORD PTR [rbp-120]
  mov rbx, QWORD PTR [rax+384]
  mov rax, QWORD PTR [rbp-120]
  add rax, 272
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEixEm
  movzx eax, BYTE PTR [rax]
  movsx ecx, al
  lea rax, [rbp-34]
  mov rdx, rbx
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEC1EcRKS3_
  movzx edx, WORD PTR [rbp-34]
  lea rax, [rbp-80]
  mov esi, edx
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1INSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEvvEET_
  lea rax, [rbp-80]
  mov rsi, rax
  mov rdi, r13
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_matcherESt8functionIFbcEE
  mov rbx, rax
  mov rax, QWORD PTR [rbp-120]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-112]
  mov rdx, rbx
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-112]
  mov rsi, rax
  mov rdi, r12
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  lea rax, [rbp-80]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  jmp .L1562
  mov rbx, rax
  lea rax, [rbp-80]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1562:
  add rsp, 104
  pop rbx
  pop r12
  pop r13
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE22_M_insert_char_matcherILb0ELb1EEEvv:
  push rbp
  mov rbp, rsp
  push r13
  push r12
  push rbx
  sub rsp, 104
  mov QWORD PTR [rbp-120], rdi
  mov rax, QWORD PTR [rbp-120]
  lea r12, [rax+304]
  mov rax, QWORD PTR [rbp-120]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r13, rax
  mov rax, QWORD PTR [rbp-120]
  mov rbx, QWORD PTR [rax+384]
  mov rax, QWORD PTR [rbp-120]
  add rax, 272
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEixEm
  movzx eax, BYTE PTR [rax]
  movsx ecx, al
  lea rax, [rbp-48]
  mov rdx, rbx
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEC1EcRKS3_
  mov rcx, QWORD PTR [rbp-48]
  mov rdx, QWORD PTR [rbp-40]
  lea rax, [rbp-80]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1INSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEvvEET_
  lea rax, [rbp-80]
  mov rsi, rax
  mov rdi, r13
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_matcherESt8functionIFbcEE
  mov rbx, rax
  mov rax, QWORD PTR [rbp-120]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-112]
  mov rdx, rbx
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-112]
  mov rsi, rax
  mov rdi, r12
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  lea rax, [rbp-80]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  jmp .L1566
  mov rbx, rax
  lea rax, [rbp-80]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1566:
  add rsp, 104
  pop rbx
  pop r12
  pop r13
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE22_M_insert_char_matcherILb1ELb0EEEvv:
  push rbp
  mov rbp, rsp
  push r13
  push r12
  push rbx
  sub rsp, 104
  mov QWORD PTR [rbp-120], rdi
  mov rax, QWORD PTR [rbp-120]
  lea r12, [rax+304]
  mov rax, QWORD PTR [rbp-120]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r13, rax
  mov rax, QWORD PTR [rbp-120]
  mov rbx, QWORD PTR [rax+384]
  mov rax, QWORD PTR [rbp-120]
  add rax, 272
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEixEm
  movzx eax, BYTE PTR [rax]
  movsx ecx, al
  lea rax, [rbp-48]
  mov rdx, rbx
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEC1EcRKS3_
  mov rcx, QWORD PTR [rbp-48]
  mov rdx, QWORD PTR [rbp-40]
  lea rax, [rbp-80]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1INSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEvvEET_
  lea rax, [rbp-80]
  mov rsi, rax
  mov rdi, r13
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_matcherESt8functionIFbcEE
  mov rbx, rax
  mov rax, QWORD PTR [rbp-120]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-112]
  mov rdx, rbx
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-112]
  mov rsi, rax
  mov rdi, r12
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  lea rax, [rbp-80]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  jmp .L1570
  mov rbx, rax
  lea rax, [rbp-80]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1570:
  add rsp, 104
  pop rbx
  pop r12
  pop r13
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE22_M_insert_char_matcherILb1ELb1EEEvv:
  push rbp
  mov rbp, rsp
  push r13
  push r12
  push rbx
  sub rsp, 104
  mov QWORD PTR [rbp-120], rdi
  mov rax, QWORD PTR [rbp-120]
  lea r12, [rax+304]
  mov rax, QWORD PTR [rbp-120]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r13, rax
  mov rax, QWORD PTR [rbp-120]
  mov rbx, QWORD PTR [rax+384]
  mov rax, QWORD PTR [rbp-120]
  add rax, 272
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEixEm
  movzx eax, BYTE PTR [rax]
  movsx ecx, al
  lea rax, [rbp-48]
  mov rdx, rbx
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEC1EcRKS3_
  mov rcx, QWORD PTR [rbp-48]
  mov rdx, QWORD PTR [rbp-40]
  lea rax, [rbp-80]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1INSt8__detail12_CharMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEvvEET_
  lea rax, [rbp-80]
  mov rsi, rax
  mov rdi, r13
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_matcherESt8functionIFbcEE
  mov rbx, rax
  mov rax, QWORD PTR [rbp-120]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-112]
  mov rdx, rbx
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-112]
  mov rsi, rax
  mov rdi, r12
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  lea rax, [rbp-80]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  jmp .L1574
  mov rbx, rax
  lea rax, [rbp-80]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1574:
  add rsp, 104
  pop rbx
  pop r12
  pop r13
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE16_M_cur_int_valueEi:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 32
  mov QWORD PTR [rbp-40], rdi
  mov DWORD PTR [rbp-44], esi
  mov QWORD PTR [rbp-24], 0
  mov QWORD PTR [rbp-32], 0
.L1577:
  mov rax, QWORD PTR [rbp-40]
  add rax, 272
  mov rdi, rax
  call _ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv
  cmp QWORD PTR [rbp-32], rax
  setb al
  test al, al
  je .L1576
  mov eax, DWORD PTR [rbp-44]
  cdqe
  imul rax, QWORD PTR [rbp-24]
  mov r12, rax
  mov rax, QWORD PTR [rbp-40]
  mov rbx, QWORD PTR [rax+384]
  mov rax, QWORD PTR [rbp-40]
  lea rdx, [rax+272]
  mov rax, QWORD PTR [rbp-32]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEixEm
  movzx eax, BYTE PTR [rax]
  movsx eax, al
  mov edx, DWORD PTR [rbp-44]
  mov esi, eax
  mov rdi, rbx
  call _ZNKSt7__cxx1112regex_traitsIcE5valueEci
  cdqe
  add rax, r12
  mov QWORD PTR [rbp-24], rax
  add QWORD PTR [rbp-32], 1
  jmp .L1577
.L1576:
  mov rax, QWORD PTR [rbp-24]
  add rsp, 32
  pop rbx
  pop r12
  pop rbp
  ret
.LC31:
  .string "Unexpected back-reference in polynomial mode."
.LC32:
  .string "Back-reference index exceeds current sub-expression count."
.LC33:
  .string "Back-reference referred to an opened sub-expression."
_ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_backrefEm:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 152
  mov QWORD PTR [rbp-152], rdi
  mov QWORD PTR [rbp-160], rsi
  mov rax, QWORD PTR [rbp-152]
  mov eax, DWORD PTR [rax+24]
  mov esi, 1024
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  setne al
  test al, al
  je .L1580
  mov esi, OFFSET FLAT:.LC31
  mov edi, 11
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L1580:
  mov rax, QWORD PTR [rbp-152]
  mov rax, QWORD PTR [rax+40]
  cmp QWORD PTR [rbp-160], rax
  jb .L1581
  mov esi, OFFSET FLAT:.LC32
  mov edi, 3
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L1581:
  mov rax, QWORD PTR [rbp-152]
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt6vectorImSaImEE5beginEv
  mov QWORD PTR [rbp-136], rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt6vectorImSaImEE3endEv
  mov QWORD PTR [rbp-144], rax
.L1584:
  lea rdx, [rbp-144]
  lea rax, [rbp-136]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxxneIPmSt6vectorImSaImEEEEbRKNS_17__normal_iteratorIT_T0_EESA_
  test al, al
  je .L1582
  lea rax, [rbp-136]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEdeEv
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-32], rax
  mov rax, QWORD PTR [rbp-160]
  cmp rax, QWORD PTR [rbp-32]
  jne .L1583
  mov esi, OFFSET FLAT:.LC33
  mov edi, 3
  call _ZSt19__throw_regex_errorNSt15regex_constants10error_typeEPKc
.L1583:
  lea rax, [rbp-136]
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPmSt6vectorImSaImEEEppEv
  jmp .L1584
.L1582:
  mov rax, QWORD PTR [rbp-152]
  mov BYTE PTR [rax+48], 1
  lea rax, [rbp-128]
  mov esi, 3
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1ENS_7_OpcodeE
  mov rax, QWORD PTR [rbp-160]
  mov QWORD PTR [rbp-112], rax
  lea rax, [rbp-128]
  mov rdi, rax
  call _ZSt4moveIRNSt8__detail6_StateIcEEEONSt16remove_referenceIT_E4typeEOS5_
  mov rdx, rax
  lea rax, [rbp-80]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1EOS1_
  lea rdx, [rbp-80]
  mov rax, QWORD PTR [rbp-152]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_stateENS_6_StateIcEE
  mov rbx, rax
  lea rax, [rbp-80]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  lea rax, [rbp-128]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  jmp .L1588
  mov rbx, rax
  lea rax, [rbp-80]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  lea rax, [rbp-128]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1588:
  add rsp, 152
  pop rbx
  pop rbp
  ret
_ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 72
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx1112regex_traitsIcE10_RegexMaskESaIS3_EED1Ev
  mov rax, QWORD PTR [rbp-8]
  add rax, 48
  mov rdi, rax
  call _ZNSt6vectorISt4pairIccESaIS1_EED1Ev
  mov rax, QWORD PTR [rbp-8]
  add rax, 24
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EED1Ev
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt6vectorIcSaIcEED1Ev
  nop
  leave
  ret
_ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEC2EOS4_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorIcSaIcEEC1EOS1_
  mov rax, QWORD PTR [rbp-8]
  add rax, 24
  mov rdx, QWORD PTR [rbp-16]
  add rdx, 24
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EEC1EOS7_
  mov rax, QWORD PTR [rbp-8]
  add rax, 48
  mov rdx, QWORD PTR [rbp-16]
  add rdx, 48
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorISt4pairIccESaIS1_EEC1EOS3_
  mov rax, QWORD PTR [rbp-8]
  add rax, 72
  mov rdx, QWORD PTR [rbp-16]
  add rdx, 72
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx1112regex_traitsIcE10_RegexMaskESaIS3_EEC1EOS5_
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov edx, DWORD PTR [rdx+96]
  mov DWORD PTR [rax+96], edx
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax+104]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+104], rdx
  mov rax, QWORD PTR [rbp-16]
  movzx edx, BYTE PTR [rax+112]
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+112], dl
  mov rcx, QWORD PTR [rbp-8]
  mov rsi, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rsi+120]
  mov rdx, QWORD PTR [rsi+128]
  mov QWORD PTR [rcx+120], rax
  mov QWORD PTR [rcx+128], rdx
  mov rax, QWORD PTR [rsi+136]
  mov rdx, QWORD PTR [rsi+144]
  mov QWORD PTR [rcx+136], rax
  mov QWORD PTR [rcx+144], rdx
  nop
  leave
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE33_M_insert_character_class_matcherILb0ELb0EEEvv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 400
  mov QWORD PTR [rbp-408], rdi
  mov rax, QWORD PTR [rbp-408]
  mov rbx, QWORD PTR [rax+384]
  mov rax, QWORD PTR [rbp-408]
  mov r12, QWORD PTR [rax+392]
  mov rax, QWORD PTR [rbp-408]
  add rax, 272
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEixEm
  movzx eax, BYTE PTR [rax]
  movsx eax, al
  mov edx, eax
  mov esi, 256
  mov rdi, r12
  call _ZNKSt5ctypeIcE2isEtc
  movzx ecx, al
  lea rax, [rbp-400]
  mov rdx, rbx
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEC1EbRKS3_
  mov rax, QWORD PTR [rbp-408]
  lea rcx, [rax+272]
  lea rax, [rbp-400]
  mov edx, 0
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EE22_M_add_character_classERKNS1_12basic_stringIcSt11char_traitsIcESaIcEEEb
  lea rax, [rbp-400]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EE8_M_readyEv
  mov rax, QWORD PTR [rbp-408]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-408]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r12, rax
  lea rax, [rbp-400]
  mov rdi, rax
  call _ZSt4moveIRNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEEONSt16remove_referenceIT_E4typeEOS8_
  mov rdx, rax
  lea rax, [rbp-176]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEC1EOS4_
  lea rdx, [rbp-176]
  lea rax, [rbp-208]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1INSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EEEvvEET_
  lea rax, [rbp-208]
  mov rsi, rax
  mov rdi, r12
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_matcherESt8functionIFbcEE
  mov r12, rax
  mov rax, QWORD PTR [rbp-408]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-240]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-240]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  lea rax, [rbp-208]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  lea rax, [rbp-176]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EED1Ev
  lea rax, [rbp-400]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EED1Ev
  jmp .L1598
  mov rbx, rax
  lea rax, [rbp-208]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  jmp .L1593
  mov rbx, rax
.L1593:
  lea rax, [rbp-176]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EED1Ev
  jmp .L1594
  mov rbx, rax
.L1594:
  lea rax, [rbp-400]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb0EED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1598:
  add rsp, 400
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 72
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx1112regex_traitsIcE10_RegexMaskESaIS3_EED1Ev
  mov rax, QWORD PTR [rbp-8]
  add rax, 48
  mov rdi, rax
  call _ZNSt6vectorISt4pairINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES6_ESaIS7_EED1Ev
  mov rax, QWORD PTR [rbp-8]
  add rax, 24
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EED1Ev
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt6vectorIcSaIcEED1Ev
  nop
  leave
  ret
_ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEC2EOS4_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorIcSaIcEEC1EOS1_
  mov rax, QWORD PTR [rbp-8]
  add rax, 24
  mov rdx, QWORD PTR [rbp-16]
  add rdx, 24
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EEC1EOS7_
  mov rax, QWORD PTR [rbp-8]
  add rax, 48
  mov rdx, QWORD PTR [rbp-16]
  add rdx, 48
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorISt4pairINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES6_ESaIS7_EEC1EOS9_
  mov rax, QWORD PTR [rbp-8]
  add rax, 72
  mov rdx, QWORD PTR [rbp-16]
  add rdx, 72
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx1112regex_traitsIcE10_RegexMaskESaIS3_EEC1EOS5_
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov edx, DWORD PTR [rdx+96]
  mov DWORD PTR [rax+96], edx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rdx+104]
  mov QWORD PTR [rax+104], rdx
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax+112]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+112], rdx
  mov rax, QWORD PTR [rbp-16]
  movzx edx, BYTE PTR [rax+120]
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+120], dl
  mov rcx, QWORD PTR [rbp-8]
  mov rsi, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rsi+128]
  mov rdx, QWORD PTR [rsi+136]
  mov QWORD PTR [rcx+128], rax
  mov QWORD PTR [rcx+136], rdx
  mov rax, QWORD PTR [rsi+144]
  mov rdx, QWORD PTR [rsi+152]
  mov QWORD PTR [rcx+144], rax
  mov QWORD PTR [rcx+152], rdx
  nop
  leave
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE33_M_insert_character_class_matcherILb0ELb1EEEvv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 400
  mov QWORD PTR [rbp-408], rdi
  mov rax, QWORD PTR [rbp-408]
  mov rbx, QWORD PTR [rax+384]
  mov rax, QWORD PTR [rbp-408]
  mov r12, QWORD PTR [rax+392]
  mov rax, QWORD PTR [rbp-408]
  add rax, 272
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEixEm
  movzx eax, BYTE PTR [rax]
  movsx eax, al
  mov edx, eax
  mov esi, 256
  mov rdi, r12
  call _ZNKSt5ctypeIcE2isEtc
  movzx ecx, al
  lea rax, [rbp-400]
  mov rdx, rbx
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEC1EbRKS3_
  mov rax, QWORD PTR [rbp-408]
  lea rcx, [rax+272]
  lea rax, [rbp-400]
  mov edx, 0
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EE22_M_add_character_classERKNS1_12basic_stringIcSt11char_traitsIcESaIcEEEb
  lea rax, [rbp-400]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EE8_M_readyEv
  mov rax, QWORD PTR [rbp-408]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-408]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r12, rax
  lea rax, [rbp-400]
  mov rdi, rax
  call _ZSt4moveIRNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEEONSt16remove_referenceIT_E4typeEOS8_
  mov rdx, rax
  lea rax, [rbp-176]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEC1EOS4_
  lea rdx, [rbp-176]
  lea rax, [rbp-208]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1INSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EEEvvEET_
  lea rax, [rbp-208]
  mov rsi, rax
  mov rdi, r12
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_matcherESt8functionIFbcEE
  mov r12, rax
  mov rax, QWORD PTR [rbp-408]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-240]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-240]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  lea rax, [rbp-208]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  lea rax, [rbp-176]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EED1Ev
  lea rax, [rbp-400]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EED1Ev
  jmp .L1608
  mov rbx, rax
  lea rax, [rbp-208]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  jmp .L1603
  mov rbx, rax
.L1603:
  lea rax, [rbp-176]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EED1Ev
  jmp .L1604
  mov rbx, rax
.L1604:
  lea rax, [rbp-400]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb0ELb1EED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1608:
  add rsp, 400
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 72
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx1112regex_traitsIcE10_RegexMaskESaIS3_EED1Ev
  mov rax, QWORD PTR [rbp-8]
  add rax, 48
  mov rdi, rax
  call _ZNSt6vectorISt4pairIccESaIS1_EED1Ev
  mov rax, QWORD PTR [rbp-8]
  add rax, 24
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EED1Ev
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt6vectorIcSaIcEED1Ev
  nop
  leave
  ret
_ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEC2EOS4_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorIcSaIcEEC1EOS1_
  mov rax, QWORD PTR [rbp-8]
  add rax, 24
  mov rdx, QWORD PTR [rbp-16]
  add rdx, 24
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EEC1EOS7_
  mov rax, QWORD PTR [rbp-8]
  add rax, 48
  mov rdx, QWORD PTR [rbp-16]
  add rdx, 48
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorISt4pairIccESaIS1_EEC1EOS3_
  mov rax, QWORD PTR [rbp-8]
  add rax, 72
  mov rdx, QWORD PTR [rbp-16]
  add rdx, 72
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx1112regex_traitsIcE10_RegexMaskESaIS3_EEC1EOS5_
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov edx, DWORD PTR [rdx+96]
  mov DWORD PTR [rax+96], edx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rdx+104]
  mov QWORD PTR [rax+104], rdx
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax+112]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+112], rdx
  mov rax, QWORD PTR [rbp-16]
  movzx edx, BYTE PTR [rax+120]
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+120], dl
  mov rcx, QWORD PTR [rbp-8]
  mov rsi, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rsi+128]
  mov rdx, QWORD PTR [rsi+136]
  mov QWORD PTR [rcx+128], rax
  mov QWORD PTR [rcx+136], rdx
  mov rax, QWORD PTR [rsi+144]
  mov rdx, QWORD PTR [rsi+152]
  mov QWORD PTR [rcx+144], rax
  mov QWORD PTR [rcx+152], rdx
  nop
  leave
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE33_M_insert_character_class_matcherILb1ELb0EEEvv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 400
  mov QWORD PTR [rbp-408], rdi
  mov rax, QWORD PTR [rbp-408]
  mov rbx, QWORD PTR [rax+384]
  mov rax, QWORD PTR [rbp-408]
  mov r12, QWORD PTR [rax+392]
  mov rax, QWORD PTR [rbp-408]
  add rax, 272
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEixEm
  movzx eax, BYTE PTR [rax]
  movsx eax, al
  mov edx, eax
  mov esi, 256
  mov rdi, r12
  call _ZNKSt5ctypeIcE2isEtc
  movzx ecx, al
  lea rax, [rbp-400]
  mov rdx, rbx
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEC1EbRKS3_
  mov rax, QWORD PTR [rbp-408]
  lea rcx, [rax+272]
  lea rax, [rbp-400]
  mov edx, 0
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EE22_M_add_character_classERKNS1_12basic_stringIcSt11char_traitsIcESaIcEEEb
  lea rax, [rbp-400]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EE8_M_readyEv
  mov rax, QWORD PTR [rbp-408]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-408]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r12, rax
  lea rax, [rbp-400]
  mov rdi, rax
  call _ZSt4moveIRNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEEONSt16remove_referenceIT_E4typeEOS8_
  mov rdx, rax
  lea rax, [rbp-176]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEC1EOS4_
  lea rdx, [rbp-176]
  lea rax, [rbp-208]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1INSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EEEvvEET_
  lea rax, [rbp-208]
  mov rsi, rax
  mov rdi, r12
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_matcherESt8functionIFbcEE
  mov r12, rax
  mov rax, QWORD PTR [rbp-408]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-240]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-240]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  lea rax, [rbp-208]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  lea rax, [rbp-176]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EED1Ev
  lea rax, [rbp-400]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EED1Ev
  jmp .L1618
  mov rbx, rax
  lea rax, [rbp-208]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  jmp .L1613
  mov rbx, rax
.L1613:
  lea rax, [rbp-176]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EED1Ev
  jmp .L1614
  mov rbx, rax
.L1614:
  lea rax, [rbp-400]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb0EED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1618:
  add rsp, 400
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  add rax, 72
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx1112regex_traitsIcE10_RegexMaskESaIS3_EED1Ev
  mov rax, QWORD PTR [rbp-8]
  add rax, 48
  mov rdi, rax
  call _ZNSt6vectorISt4pairINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES6_ESaIS7_EED1Ev
  mov rax, QWORD PTR [rbp-8]
  add rax, 24
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EED1Ev
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt6vectorIcSaIcEED1Ev
  nop
  leave
  ret
_ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEC2EOS4_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorIcSaIcEEC1EOS1_
  mov rax, QWORD PTR [rbp-8]
  add rax, 24
  mov rdx, QWORD PTR [rbp-16]
  add rdx, 24
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EEC1EOS7_
  mov rax, QWORD PTR [rbp-8]
  add rax, 48
  mov rdx, QWORD PTR [rbp-16]
  add rdx, 48
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorISt4pairINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES6_ESaIS7_EEC1EOS9_
  mov rax, QWORD PTR [rbp-8]
  add rax, 72
  mov rdx, QWORD PTR [rbp-16]
  add rdx, 72
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx1112regex_traitsIcE10_RegexMaskESaIS3_EEC1EOS5_
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov edx, DWORD PTR [rdx+96]
  mov DWORD PTR [rax+96], edx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rdx+104]
  mov QWORD PTR [rax+104], rdx
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax+112]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+112], rdx
  mov rax, QWORD PTR [rbp-16]
  movzx edx, BYTE PTR [rax+120]
  mov rax, QWORD PTR [rbp-8]
  mov BYTE PTR [rax+120], dl
  mov rcx, QWORD PTR [rbp-8]
  mov rsi, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rsi+128]
  mov rdx, QWORD PTR [rsi+136]
  mov QWORD PTR [rcx+128], rax
  mov QWORD PTR [rcx+136], rdx
  mov rax, QWORD PTR [rsi+144]
  mov rdx, QWORD PTR [rsi+152]
  mov QWORD PTR [rcx+144], rax
  mov QWORD PTR [rcx+152], rdx
  nop
  leave
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE33_M_insert_character_class_matcherILb1ELb1EEEvv:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 400
  mov QWORD PTR [rbp-408], rdi
  mov rax, QWORD PTR [rbp-408]
  mov rbx, QWORD PTR [rax+384]
  mov rax, QWORD PTR [rbp-408]
  mov r12, QWORD PTR [rax+392]
  mov rax, QWORD PTR [rbp-408]
  add rax, 272
  mov esi, 0
  mov rdi, rax
  call _ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEixEm
  movzx eax, BYTE PTR [rax]
  movsx eax, al
  mov edx, eax
  mov esi, 256
  mov rdi, r12
  call _ZNKSt5ctypeIcE2isEtc
  movzx ecx, al
  lea rax, [rbp-400]
  mov rdx, rbx
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEC1EbRKS3_
  mov rax, QWORD PTR [rbp-408]
  lea rcx, [rax+272]
  lea rax, [rbp-400]
  mov edx, 0
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EE22_M_add_character_classERKNS1_12basic_stringIcSt11char_traitsIcESaIcEEEb
  lea rax, [rbp-400]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EE8_M_readyEv
  mov rax, QWORD PTR [rbp-408]
  lea rbx, [rax+304]
  mov rax, QWORD PTR [rbp-408]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEptEv
  mov r12, rax
  lea rax, [rbp-400]
  mov rdi, rax
  call _ZSt4moveIRNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEEONSt16remove_referenceIT_E4typeEOS8_
  mov rdx, rax
  lea rax, [rbp-176]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEC1EOS4_
  lea rdx, [rbp-176]
  lea rax, [rbp-208]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8functionIFbcEEC1INSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EEEvvEET_
  lea rax, [rbp-208]
  mov rsi, rax
  mov rdi, r12
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE17_M_insert_matcherESt8functionIFbcEE
  mov r12, rax
  mov rax, QWORD PTR [rbp-408]
  add rax, 256
  mov rdi, rax
  call _ZNKSt19__shared_ptr_accessINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEELN9__gnu_cxx12_Lock_policyE2ELb0ELb0EEdeEv
  mov rcx, rax
  lea rax, [rbp-240]
  mov rdx, r12
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEl
  lea rax, [rbp-240]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE4pushEOS5_
  lea rax, [rbp-208]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  lea rax, [rbp-176]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EED1Ev
  lea rax, [rbp-400]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EED1Ev
  jmp .L1628
  mov rbx, rax
  lea rax, [rbp-208]
  mov rdi, rax
  call _ZNSt8functionIFbcEED1Ev
  jmp .L1623
  mov rbx, rax
.L1623:
  lea rax, [rbp-176]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EED1Ev
  jmp .L1624
  mov rbx, rax
.L1624:
  lea rax, [rbp-400]
  mov rdi, rax
  call _ZNSt8__detail15_BracketMatcherINSt7__cxx1112regex_traitsIcEELb1ELb1EED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1628:
  add rsp, 400
  pop rbx
  pop r12
  pop rbp
  ret
_ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE21_M_bracket_expressionEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov esi, 10
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  mov BYTE PTR [rbp-1], al
  movzx eax, BYTE PTR [rbp-1]
  xor eax, 1
  test al, al
  je .L1630
  mov rax, QWORD PTR [rbp-24]
  mov esi, 9
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE14_M_match_tokenENS_12_ScannerBase7_TokenTE
  xor eax, 1
  test al, al
  je .L1630
  mov eax, 1
  jmp .L1631
.L1630:
  mov eax, 0
.L1631:
  test al, al
  je .L1632
  mov eax, 0
  jmp .L1633
.L1632:
  mov rax, QWORD PTR [rbp-24]
  mov eax, DWORD PTR [rax]
  mov esi, 1
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  sete al
  test al, al
  je .L1634
  mov rax, QWORD PTR [rbp-24]
  mov eax, DWORD PTR [rax]
  mov esi, 8
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  sete al
  test al, al
  je .L1635
  movzx edx, BYTE PTR [rbp-1]
  mov rax, QWORD PTR [rbp-24]
  mov esi, edx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE25_M_insert_bracket_matcherILb0ELb0EEEvb
  jmp .L1637
.L1635:
  movzx edx, BYTE PTR [rbp-1]
  mov rax, QWORD PTR [rbp-24]
  mov esi, edx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE25_M_insert_bracket_matcherILb0ELb1EEEvb
  jmp .L1637
.L1634:
  mov rax, QWORD PTR [rbp-24]
  mov eax, DWORD PTR [rax]
  mov esi, 8
  mov edi, eax
  call _ZNSt15regex_constantsanENS_18syntax_option_typeES0_
  test eax, eax
  sete al
  test al, al
  je .L1638
  movzx edx, BYTE PTR [rbp-1]
  mov rax, QWORD PTR [rbp-24]
  mov esi, edx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE25_M_insert_bracket_matcherILb1ELb0EEEvb
  jmp .L1637
.L1638:
  movzx edx, BYTE PTR [rbp-1]
  mov rax, QWORD PTR [rbp-24]
  mov esi, edx
  mov rdi, rax
  call _ZNSt8__detail9_CompilerINSt7__cxx1112regex_traitsIcEEE25_M_insert_bracket_matcherILb1ELb1EEEvb
.L1637:
  mov eax, 1
.L1633:
  leave
  ret
_ZNKSt5stackINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESt5dequeIS5_SaIS5_EEE5emptyEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE5emptyEv
  leave
  ret
_ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE16_M_insert_repeatEllb:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 136
  mov QWORD PTR [rbp-120], rdi
  mov QWORD PTR [rbp-128], rsi
  mov QWORD PTR [rbp-136], rdx
  mov eax, ecx
  mov BYTE PTR [rbp-140], al
  lea rax, [rbp-112]
  mov esi, 2
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1ENS_7_OpcodeE
  mov rax, QWORD PTR [rbp-128]
  mov QWORD PTR [rbp-104], rax
  mov rax, QWORD PTR [rbp-136]
  mov QWORD PTR [rbp-96], rax
  movzx eax, BYTE PTR [rbp-140]
  mov BYTE PTR [rbp-88], al
  lea rax, [rbp-112]
  mov rdi, rax
  call _ZSt4moveIRNSt8__detail6_StateIcEEEONSt16remove_referenceIT_E4typeEOS5_
  mov rdx, rax
  lea rax, [rbp-64]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1EOS1_
  lea rdx, [rbp-64]
  mov rax, QWORD PTR [rbp-120]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_stateENS_6_StateIcEE
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  lea rax, [rbp-112]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  jmp .L1645
  mov rbx, rax
  lea rax, [rbp-64]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  lea rax, [rbp-112]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1645:
  add rsp, 136
  pop rbx
  pop rbp
  ret
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE13_Rb_tree_implIS6_Lb1EED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSaISt13_Rb_tree_nodeISt4pairIKllEEED2Ev
  nop
  leave
  ret
_ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EE13_Rb_tree_implIS6_Lb1EEC1Ev
  nop
  leave
  ret
_ZNSt3mapIllSt4lessIlESaISt4pairIKllEEEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EEC1Ev
  nop
  leave
  ret
_ZNSt3mapIllSt4lessIlESaISt4pairIKllEEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt8_Rb_treeIlSt4pairIKllESt10_Select1stIS2_ESt4lessIlESaIS2_EED1Ev
  nop
  leave
  ret
_ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEE8_M_cloneEv:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 328
  mov QWORD PTR [rbp-328], rdi
  mov QWORD PTR [rbp-336], rsi
  lea rax, [rbp-144]
  mov rdi, rax
  call _ZNSt3mapIllSt4lessIlESaISt4pairIKllEEEC1Ev
  lea rax, [rbp-224]
  mov rdi, rax
  call _ZNSt5stackIlSt5dequeIlSaIlEEEC1IS2_vEEv
  mov rax, QWORD PTR [rbp-336]
  lea rdx, [rax+8]
  lea rax, [rbp-224]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt5stackIlSt5dequeIlSaIlEEE4pushERKl
.L1662:
  lea rax, [rbp-224]
  mov rdi, rax
  call _ZNKSt5stackIlSt5dequeIlSaIlEEE5emptyEv
  xor eax, 1
  test al, al
  je .L1651
  lea rax, [rbp-224]
  mov rdi, rax
  call _ZNSt5stackIlSt5dequeIlSaIlEEE3topEv
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-232], rax
  lea rax, [rbp-224]
  mov rdi, rax
  call _ZNSt5stackIlSt5dequeIlSaIlEEE3popEv
  mov rax, QWORD PTR [rbp-336]
  mov rax, QWORD PTR [rax]
  add rax, 56
  mov rdx, QWORD PTR [rbp-232]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov rdx, rax
  lea rax, [rbp-320]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1ERKS1_
  mov rax, QWORD PTR [rbp-336]
  mov rbx, QWORD PTR [rax]
  lea rax, [rbp-320]
  mov rdi, rax
  call _ZSt4moveIRNSt8__detail6_StateIcEEEONSt16remove_referenceIT_E4typeEOS5_
  mov rdx, rax
  lea rax, [rbp-96]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt8__detail6_StateIcEC1EOS1_
  lea rax, [rbp-96]
  mov rsi, rax
  mov rdi, rbx
  call _ZNSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEE15_M_insert_stateENS_6_StateIcEE
  mov QWORD PTR [rbp-24], rax
  lea rax, [rbp-96]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  mov rbx, QWORD PTR [rbp-24]
  lea rdx, [rbp-232]
  lea rax, [rbp-144]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt3mapIllSt4lessIlESaISt4pairIKllEEEixERS3_
  mov QWORD PTR [rax], rbx
  lea rax, [rbp-320]
  mov rdi, rax
  call _ZNSt8__detail11_State_base10_M_has_altEv
  test al, al
  je .L1652
  mov rax, QWORD PTR [rbp-304]
  cmp rax, -1
  je .L1653
  lea rax, [rbp-320]
  lea rdx, [rax+16]
  lea rax, [rbp-144]
  mov rsi, rdx
  mov rdi, rax
  call _ZNKSt3mapIllSt4lessIlESaISt4pairIKllEEE5countERS3_
  test rax, rax
  jne .L1653
  mov eax, 1
  jmp .L1654
.L1653:
  mov eax, 0
.L1654:
  test al, al
  je .L1652
  lea rax, [rbp-320]
  lea rdx, [rax+16]
  lea rax, [rbp-224]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt5stackIlSt5dequeIlSaIlEEE4pushERKl
.L1652:
  mov rax, QWORD PTR [rbp-336]
  mov rdx, QWORD PTR [rax+16]
  mov rax, QWORD PTR [rbp-232]
  cmp rdx, rax
  jne .L1655
  mov ebx, 0
  jmp .L1656
.L1655:
  mov rax, QWORD PTR [rbp-312]
  cmp rax, -1
  je .L1657
  lea rax, [rbp-320]
  lea rdx, [rax+8]
  lea rax, [rbp-144]
  mov rsi, rdx
  mov rdi, rax
  call _ZNKSt3mapIllSt4lessIlESaISt4pairIKllEEE5countERS3_
  test rax, rax
  jne .L1657
  mov eax, 1
  jmp .L1658
.L1657:
  mov eax, 0
.L1658:
  test al, al
  je .L1659
  lea rax, [rbp-320]
  lea rdx, [rax+8]
  lea rax, [rbp-224]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt5stackIlSt5dequeIlSaIlEEE4pushERKl
.L1659:
  mov ebx, 1
.L1656:
  lea rax, [rbp-320]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  cmp ebx, 1
  jmp .L1662
.L1651:
  lea rax, [rbp-144]
  mov QWORD PTR [rbp-32], rax
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNSt3mapIllSt4lessIlESaISt4pairIKllEEE5beginEv
  mov QWORD PTR [rbp-264], rax
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZNSt3mapIllSt4lessIlESaISt4pairIKllEEE3endEv
  mov QWORD PTR [rbp-272], rax
.L1666:
  lea rdx, [rbp-272]
  lea rax, [rbp-264]
  mov rsi, rdx
  mov rdi, rax
  call _ZNKSt17_Rb_tree_iteratorISt4pairIKllEEneERKS3_
  test al, al
  je .L1663
  lea rax, [rbp-264]
  mov rdi, rax
  call _ZNKSt17_Rb_tree_iteratorISt4pairIKllEEdeEv
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-256], rax
  mov QWORD PTR [rbp-248], rdx
  mov rax, QWORD PTR [rbp-248]
  mov QWORD PTR [rbp-40], rax
  mov rax, QWORD PTR [rbp-336]
  mov rax, QWORD PTR [rax]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-48], rax
  mov rax, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rax+8]
  cmp rax, -1
  je .L1664
  mov rax, QWORD PTR [rbp-48]
  lea rdx, [rax+8]
  lea rax, [rbp-144]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt3mapIllSt4lessIlESaISt4pairIKllEEEixERS3_
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-48]
  mov QWORD PTR [rax+8], rdx
.L1664:
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZNSt8__detail11_State_base10_M_has_altEv
  test al, al
  je .L1665
  mov rax, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rax+16]
  cmp rax, -1
  je .L1665
  mov rax, QWORD PTR [rbp-48]
  lea rdx, [rax+16]
  lea rax, [rbp-144]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt3mapIllSt4lessIlESaISt4pairIKllEEEixERS3_
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-48]
  mov QWORD PTR [rax+16], rdx
.L1665:
  lea rax, [rbp-264]
  mov rdi, rax
  call _ZNSt17_Rb_tree_iteratorISt4pairIKllEEppEv
  jmp .L1666
.L1663:
  mov rax, QWORD PTR [rbp-336]
  lea rdx, [rax+16]
  lea rax, [rbp-144]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt3mapIllSt4lessIlESaISt4pairIKllEEEixERS3_
  mov rbx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-336]
  lea rdx, [rax+8]
  lea rax, [rbp-144]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt3mapIllSt4lessIlESaISt4pairIKllEEEixERS3_
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-336]
  mov rsi, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-328]
  mov rcx, rbx
  mov rdi, rax
  call _ZNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEC1ERNS_4_NFAIS3_EEll
  lea rax, [rbp-224]
  mov rdi, rax
  call _ZNSt5stackIlSt5dequeIlSaIlEEED1Ev
  lea rax, [rbp-144]
  mov rdi, rax
  call _ZNSt3mapIllSt4lessIlESaISt4pairIKllEEED1Ev
  jmp .L1676
  mov rbx, rax
  lea rax, [rbp-96]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  jmp .L1669
  mov rbx, rax
.L1669:
  lea rax, [rbp-320]
  mov rdi, rax
  call _ZNSt8__detail6_StateIcED1Ev
  jmp .L1670
  mov rbx, rax
.L1670:
  lea rax, [rbp-224]
  mov rdi, rax
  call _ZNSt5stackIlSt5dequeIlSaIlEEED1Ev
  jmp .L1671
  mov rbx, rax
.L1671:
  lea rax, [rbp-144]
  mov rdi, rax
  call _ZNSt3mapIllSt4lessIlESaISt4pairIKllEEED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1676:
  mov rax, QWORD PTR [rbp-328]
  add rsp, 328
  pop rbx
  pop rbp
  ret
_ZNSt5stackIlSt5dequeIlSaIlEEEC2IS2_vEEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt5dequeIlSaIlEEC1Ev
  nop
  leave
  ret
_ZNSt5dequeIlSaIlEED2Ev:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 88
  mov QWORD PTR [rbp-88], rdi
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt11_Deque_baseIlSaIlEE19_M_get_Tp_allocatorEv
  mov rbx, rax
  lea rax, [rbp-80]
  mov rdx, QWORD PTR [rbp-88]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt5dequeIlSaIlEE3endEv
  lea rax, [rbp-48]
  mov rdx, QWORD PTR [rbp-88]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt5dequeIlSaIlEE5beginEv
  lea rdx, [rbp-80]
  lea rsi, [rbp-48]
  mov rax, QWORD PTR [rbp-88]
  mov rcx, rbx
  mov rdi, rax
  call _ZNSt5dequeIlSaIlEE15_M_destroy_dataESt15_Deque_iteratorIlRlPlES5_RKS0_
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt11_Deque_baseIlSaIlEED2Ev
  nop
  add rsp, 88
  pop rbx
  pop rbp
  ret
_ZNSt5stackIlSt5dequeIlSaIlEEE4pushERKl:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt5dequeIlSaIlEE9push_backERKl
  nop
  leave
  ret
_ZNKSt5stackIlSt5dequeIlSaIlEEE5emptyEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt5dequeIlSaIlEE5emptyEv
  leave
  ret
_ZNSt5stackIlSt5dequeIlSaIlEEE3topEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt5dequeIlSaIlEE4backEv
  leave
  ret
_ZNSt5stackIlSt5dequeIlSaIlEEE3popEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt5dequeIlSaIlEE8pop_backEv
  nop
  leave
  ret
_ZSt4moveIRlEONSt16remove_referenceIT_E4typeEOS2_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt4swapIlENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS4_ESt18is_move_assignableIS4_EEE5valueEvE4typeERS4_SE_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt4moveIRlEONSt16remove_referenceIT_E4typeEOS2_
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt4moveIRlEONSt16remove_referenceIT_E4typeEOS2_
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax], rdx
  lea rax, [rbp-8]
  mov rdi, rax
  call _ZSt4moveIRlEONSt16remove_referenceIT_E4typeEOS2_
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-32]
  mov QWORD PTR [rax], rdx
  nop
  leave
  ret
_ZNSt16allocator_traitsISaINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEE9constructIS5_JRKS5_EEEvRS6_PT_DpOT0_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt7forwardIRKNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEOT_RNSt16remove_referenceIS8_E4typeE
  mov rdx, rax
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEE9constructIS6_JRKS6_EEEvPT_DpOT0_
  nop
  leave
  ret
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE16_M_push_back_auxIJRKS5_EEEvDpOT_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov esi, 1
  mov rdi, rax
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE22_M_reserve_map_at_backEm
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rdx+72]
  lea rbx, [rdx+8]
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE16_M_allocate_nodeEv
  mov QWORD PTR [rbx], rax
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt7forwardIRKNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEOT_RNSt16remove_referenceIS8_E4typeE
  mov rdx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rax+48]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEE9constructIS5_JRKS5_EEEvRS6_PT_DpOT0_
  mov rax, QWORD PTR [rbp-24]
  lea rdx, [rax+48]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+72]
  add rax, 8
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_E11_M_set_nodeEPS7_
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax+56]
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+48], rdx
  nop
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZSt7forwardINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEOT_RNSt16remove_referenceIS6_E4typeE:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt16allocator_traitsISaINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEE9constructIS5_JS5_EEEvRS6_PT_DpOT0_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt7forwardINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEOT_RNSt16remove_referenceIS6_E4typeE
  mov rdx, rax
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEE9constructIS6_JS6_EEEvPT_DpOT0_
  nop
  leave
  ret
_ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE16_M_push_back_auxIJS5_EEEvDpOT_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rax, QWORD PTR [rbp-24]
  mov esi, 1
  mov rdi, rax
  call _ZNSt5dequeINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE22_M_reserve_map_at_backEm
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rdx+72]
  lea rbx, [rdx+8]
  mov rdi, rax
  call _ZNSt11_Deque_baseINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEESaIS5_EE16_M_allocate_nodeEv
  mov QWORD PTR [rbx], rax
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt7forwardINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEOT_RNSt16remove_referenceIS6_E4typeE
  mov rdx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rax+48]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEE9constructIS5_JS5_EEEvRS6_PT_DpOT0_
  mov rax, QWORD PTR [rbp-24]
  lea rdx, [rax+48]
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+72]
  add rax, 8
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt15_Deque_iteratorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEERS5_PS5_E11_M_set_nodeEPS7_
  mov rax, QWORD PTR [rbp-24]
  mov rdx, QWORD PTR [rax+56]
  mov rax, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+48], rdx
  nop
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZN9__gnu_cxx13new_allocatorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEE7destroyIS6_EEvPT_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  nop
  pop rbp
  ret
_ZN9__gnu_cxx13new_allocatorINSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEE10deallocateEPS6_m:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZdlPv
  leave
  ret
_ZN9__gnu_cxx13new_allocatorIPNSt8__detail9_StateSeqINSt7__cxx1112regex_traitsIcEEEEEC2Ev:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  nop
  pop rbp
  ret
_ZNK9__gnu_cxx13new_allocatorINSt7__cxx119sub_matchIPKcEEE8max_sizeEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  movabs rax, 768614336404564650
  pop rbp
  ret
_ZNSt27__uninitialized_default_n_1ILb0EE18__uninit_default_nIPNSt7__cxx119sub_matchIPKcEEmEET_S8_T0_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  mov rax, QWORD PTR [rbp-40]
  mov QWORD PTR [rbp-24], rax
.L1702:
  cmp QWORD PTR [rbp-48], 0
  je .L1701
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt11__addressofINSt7__cxx119sub_matchIPKcEEEPT_RS5_
  mov rdi, rax
  call _ZSt10_ConstructINSt7__cxx119sub_matchIPKcEEJEEvPT_DpOT0_
  sub QWORD PTR [rbp-48], 1
  add QWORD PTR [rbp-24], 24
  jmp .L1702
.L1701:
  mov rax, QWORD PTR [rbp-24]
  jmp .L1708
  mov rdi, rax
  call __cxa_begin_catch
  mov rdx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt8_DestroyIPNSt7__cxx119sub_matchIPKcEEEvT_S6_
  call __cxa_rethrow
  mov rbx, rax
  call __cxa_end_catch
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1708:
  add rsp, 40
  pop rbx
  pop rbp
  ret

_ZN9__gnu_cxx13new_allocatorINSt7__cxx119sub_matchIPKcEEE8allocateEmPKv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNK9__gnu_cxx13new_allocatorINSt7__cxx119sub_matchIPKcEEE8max_sizeEv
  cmp QWORD PTR [rbp-16], rax
  seta al
  test al, al
  je .L1712
  call _ZSt17__throw_bad_allocv
.L1712:
  mov rdx, QWORD PTR [rbp-16]
  mov rax, rdx
  add rax, rax
  add rax, rdx
  sal rax, 3
  mov rdi, rax
  call _Znwm
  nop
  leave
  ret
_ZNSt13move_iteratorIPNSt7__cxx119sub_matchIPKcEEEC2ES5_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-16]
  mov QWORD PTR [rax], rdx
  nop
  pop rbp
  ret
_ZSt18uninitialized_copyISt13move_iteratorIPNSt7__cxx119sub_matchIPKcEEES6_ET0_T_S9_S8_:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-1], 1
  mov rdx, QWORD PTR [rbp-40]
  mov rcx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt20__uninitialized_copyILb0EE13__uninit_copyISt13move_iteratorIPNSt7__cxx119sub_matchIPKcEEES8_EET0_T_SB_SA_
  leave
  ret
_ZN9__gnu_cxx13new_allocatorISt4pairIPKciEEC2ERKS5_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  nop
  pop rbp
  ret
_ZNSt16allocator_traitsISaISt4pairIPKciEEE8allocateERS4_m:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov edx, 0
  mov rsi, rcx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorISt4pairIPKciEE8allocateEmPKv
  leave
  ret
_ZNSt27__uninitialized_default_n_1ILb0EE18__uninit_default_nIPSt4pairIPKciEmEET_S7_T0_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  mov rax, QWORD PTR [rbp-40]
  mov QWORD PTR [rbp-24], rax
.L1721:
  cmp QWORD PTR [rbp-48], 0
  je .L1720
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt11__addressofISt4pairIPKciEEPT_RS4_
  mov rdi, rax
  call _ZSt10_ConstructISt4pairIPKciEJEEvPT_DpOT0_
  sub QWORD PTR [rbp-48], 1
  add QWORD PTR [rbp-24], 16
  jmp .L1721
.L1720:
  mov rax, QWORD PTR [rbp-24]
  jmp .L1727
  mov rdi, rax
  call __cxa_begin_catch
  mov rdx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt8_DestroyIPSt4pairIPKciEEvT_S5_
  call __cxa_rethrow
  mov rbx, rax
  call __cxa_end_catch
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1727:
  add rsp, 40
  pop rbx
  pop rbp
  ret

_ZNSaISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS5_EEEEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEEEC2Ev
  nop
  leave
  ret
_ZNSt11_Tuple_implILm0EJPbSt14default_deleteIA_bEEEC2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt11_Tuple_implILm1EJSt14default_deleteIA_bEEEC2Ev
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt10_Head_baseILm0EPbLb0EEC2Ev
  nop
  leave
  ret
_ZNSt10_Head_baseILm0EPbLb0EE7_M_headERS1_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt11_Tuple_implILm1EJSt14default_deleteIA_bEEE7_M_headERS3_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNSt10_Head_baseILm1ESt14default_deleteIA_bELb1EE7_M_headERS3_
  leave
  ret
_ZSt7forwardIRlEOT_RNSt16remove_referenceIS1_E4typeE:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt7forwardIRKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS5_EEEOT_RNSt16remove_referenceISA_E4typeE:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt16allocator_traitsISaISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEEE9constructIS9_JRlRKS8_EEEvRSA_PT_DpOT0_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 40
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov QWORD PTR [rbp-48], rcx
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZSt7forwardIRKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS5_EEEOT_RNSt16remove_referenceISA_E4typeE
  mov rbx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZSt7forwardIRlEOT_RNSt16remove_referenceIS1_E4typeE
  mov rdx, rax
  mov rsi, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rcx, rbx
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEEE9constructISA_JRlRKS9_EEEvPT_DpOT0_
  nop
  add rsp, 40
  pop rbx
  pop rbp
  ret
_ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE17_M_realloc_insertIJRlRKS7_EEEvN9__gnu_cxx17__normal_iteratorIPS8_SA_EEDpOT_:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 104
  mov QWORD PTR [rbp-88], rdi
  mov QWORD PTR [rbp-96], rsi
  mov QWORD PTR [rbp-104], rdx
  mov QWORD PTR [rbp-112], rcx
  mov rax, QWORD PTR [rbp-88]
  mov edx, OFFSET FLAT:.LC24
  mov esi, 1
  mov rdi, rax
  call _ZNKSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE12_M_check_lenEmS4_
  mov QWORD PTR [rbp-32], rax
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-40], rax
  mov rax, QWORD PTR [rbp-88]
  mov rax, QWORD PTR [rax+8]
  mov QWORD PTR [rbp-48], rax
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE5beginEv
  mov QWORD PTR [rbp-72], rax
  lea rdx, [rbp-72]
  lea rax, [rbp-96]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxxmiIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKSH_SK_
  mov QWORD PTR [rbp-56], rax
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-32]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE11_M_allocateEm
  mov QWORD PTR [rbp-64], rax
  mov rax, QWORD PTR [rbp-64]
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-112]
  mov rdi, rax
  call _ZSt7forwardIRKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS5_EEEOT_RNSt16remove_referenceISA_E4typeE
  mov rbx, rax
  mov rax, QWORD PTR [rbp-104]
  mov rdi, rax
  call _ZSt7forwardIRlEOT_RNSt16remove_referenceIS1_E4typeE
  mov rdi, rax
  mov rax, QWORD PTR [rbp-56]
  sal rax, 5
  mov rdx, rax
  mov rax, QWORD PTR [rbp-64]
  lea rsi, [rdx+rax]
  mov rax, QWORD PTR [rbp-88]
  mov rcx, rbx
  mov rdx, rdi
  mov rdi, rax
  call _ZNSt16allocator_traitsISaISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEEE9constructIS9_JRlRKS8_EEEvRSA_PT_DpOT0_
  mov QWORD PTR [rbp-24], 0
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE19_M_get_Tp_allocatorEv
  mov rbx, rax
  lea rax, [rbp-96]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEE4baseEv
  mov rsi, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-64]
  mov rax, QWORD PTR [rbp-40]
  mov rcx, rbx
  mov rdi, rax
  call _ZSt34__uninitialized_move_if_noexcept_aIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESA_SaIS9_EET0_T_SD_SC_RT1_
  mov QWORD PTR [rbp-24], rax
  add QWORD PTR [rbp-24], 32
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE19_M_get_Tp_allocatorEv
  mov rbx, rax
  lea rax, [rbp-96]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEE4baseEv
  mov rax, QWORD PTR [rax]
  mov rdx, QWORD PTR [rbp-24]
  mov rsi, QWORD PTR [rbp-48]
  mov rcx, rbx
  mov rdi, rax
  call _ZSt34__uninitialized_move_if_noexcept_aIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESA_SaIS9_EET0_T_SD_SC_RT1_
  mov QWORD PTR [rbp-24], rax
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rcx, QWORD PTR [rbp-48]
  mov rax, QWORD PTR [rbp-40]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt8_DestroyIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEES9_EvT_SB_RSaIT0_E
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rdx+16]
  sub rdx, QWORD PTR [rbp-40]
  sar rdx, 5
  mov rcx, QWORD PTR [rbp-40]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE13_M_deallocateEPS9_m
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-64]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+8], rdx
  mov rax, QWORD PTR [rbp-32]
  sal rax, 5
  mov rdx, rax
  mov rax, QWORD PTR [rbp-64]
  add rdx, rax
  mov rax, QWORD PTR [rbp-88]
  mov QWORD PTR [rax+16], rdx
  jmp .L1746
  mov rdi, rax
  call __cxa_begin_catch
  cmp QWORD PTR [rbp-24], 0
  jne .L1741
  mov rax, QWORD PTR [rbp-56]
  sal rax, 5
  mov rdx, rax
  mov rax, QWORD PTR [rbp-64]
  add rdx, rax
  mov rax, QWORD PTR [rbp-88]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt16allocator_traitsISaISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEEE7destroyIS9_EEvRSA_PT_
  jmp .L1742
.L1741:
  mov rax, QWORD PTR [rbp-88]
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rcx, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rbp-64]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt8_DestroyIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEES9_EvT_SB_RSaIT0_E
.L1742:
  mov rax, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-32]
  mov rcx, QWORD PTR [rbp-64]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE13_M_deallocateEPS9_m
  call __cxa_rethrow
  mov rbx, rax
  call __cxa_end_catch
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1746:
  add rsp, 104
  pop rbx
  pop rbp
  ret

_ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE4backEv:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZNSt6vectorISt4pairIlS_INSt7__cxx119sub_matchIPKcEESaIS5_EEESaIS8_EE3endEv
  mov QWORD PTR [rbp-8], rax
  lea rax, [rbp-8]
  mov esi, 1
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEmiEl
  mov QWORD PTR [rbp-16], rax
  lea rax, [rbp-16]
  mov rdi, rax
  call _ZNK9__gnu_cxx17__normal_iteratorIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEdeEv
  nop
  leave
  ret
_ZN9__gnu_cxx17__normal_iteratorIPKSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEEC2ERKSC_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  nop
  pop rbp
  ret
_ZNK9__gnu_cxx17__normal_iteratorIPKSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS7_EEES2_ISA_SaISA_EEE4baseEv:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt3getILm0EJPbSt14default_deleteIA_bEEERKNSt13tuple_elementIXT_ESt5tupleIJDpT0_EEE4typeERKS8_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZSt12__get_helperILm0EPbJSt14default_deleteIA_bEEERKT0_RKSt11_Tuple_implIXT_EJS4_DpT1_EE
  leave
  ret
_ZSt4moveIRSaISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEEEONSt16remove_referenceIT_E4typeEOSD_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE12_Vector_implC2EOSA_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZSt4moveIRSaISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEEEONSt16remove_referenceIT_E4typeEOSD_
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSaISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS5_EEEEC2ERKS9_
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+8], 0
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+16], 0
  nop
  leave
  ret
_ZNSt12_Vector_baseISt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEESaIS9_EE12_Vector_impl12_M_swap_dataERSC_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt4swapIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISF_ESt18is_move_assignableISF_EEE5valueEvE4typeERSF_SP_
  mov rax, QWORD PTR [rbp-16]
  lea rdx, [rax+8]
  mov rax, QWORD PTR [rbp-8]
  add rax, 8
  mov rsi, rdx
  mov rdi, rax
  call _ZSt4swapIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISF_ESt18is_move_assignableISF_EEE5valueEvE4typeERSF_SP_
  mov rax, QWORD PTR [rbp-16]
  lea rdx, [rax+16]
  mov rax, QWORD PTR [rbp-8]
  add rax, 16
  mov rsi, rdx
  mov rdi, rax
  call _ZSt4swapIPSt4pairIlSt6vectorINSt7__cxx119sub_matchIPKcEESaIS6_EEEENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISF_ESt18is_move_assignableISF_EEE5valueEvE4typeERSF_SP_
  nop
  leave
  ret
_ZNKSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE13get_allocatorEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov rdi, rax
  call _ZNKSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE19_M_get_Tp_allocatorEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSaINSt7__cxx119sub_matchIPKcEEEC1ERKS4_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSt12_Vector_baseINSt7__cxx119sub_matchIPKcEESaIS4_EE12_Vector_impl12_M_swap_dataERS7_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt4swapIPNSt7__cxx119sub_matchIPKcEEENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISA_ESt18is_move_assignableISA_EEE5valueEvE4typeERSA_SK_
  mov rax, QWORD PTR [rbp-16]
  lea rdx, [rax+8]
  mov rax, QWORD PTR [rbp-8]
  add rax, 8
  mov rsi, rdx
  mov rdi, rax
  call _ZSt4swapIPNSt7__cxx119sub_matchIPKcEEENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISA_ESt18is_move_assignableISA_EEE5valueEvE4typeERSA_SK_
  mov rax, QWORD PTR [rbp-16]
  lea rdx, [rax+16]
  mov rax, QWORD PTR [rbp-8]
  add rax, 16
  mov rsi, rdx
  mov rdi, rax
  call _ZSt4swapIPNSt7__cxx119sub_matchIPKcEEENSt9enable_ifIXsrSt6__and_IJSt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleISA_ESt18is_move_assignableISA_EEE5valueEvE4typeERSA_SK_
  nop
  leave
  ret
_ZSt15__alloc_on_moveISaINSt7__cxx119sub_matchIPKcEEEEvRT_S7_:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov rdx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZSt18__do_alloc_on_moveISaINSt7__cxx119sub_matchIPKcEEEEvRT_S7_St17integral_constantIbLb1EE
  nop
  leave
  ret
_ZNKSt10unique_ptrIA_bSt14default_deleteIS0_EEixEm:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt10unique_ptrIA_bSt14default_deleteIS0_EE3getEv
  mov rdx, rax
  mov rax, QWORD PTR [rbp-16]
  add rax, rdx
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE16_M_rep_once_moreENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 64
  mov QWORD PTR [rbp-40], rdi
  mov eax, esi
  mov QWORD PTR [rbp-56], rdx
  mov BYTE PTR [rbp-44], al
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-40]
  lea rdx, [rax+72]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt6vectorISt4pairIPKciESaIS3_EEixEm
  mov QWORD PTR [rbp-16], rax
  mov rax, QWORD PTR [rbp-16]
  mov eax, DWORD PTR [rax+8]
  test eax, eax
  je .L1765
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+24]
  cmp rdx, rax
  je .L1766
.L1765:
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-32], rax
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-16]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-16]
  mov DWORD PTR [rax+8], 1
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  movzx ecx, BYTE PTR [rbp-44]
  mov rax, QWORD PTR [rbp-40]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
  lea rdx, [rbp-32]
  mov rax, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt4pairIPKciEaSERKS2_
  nop
  jmp .L1768
.L1766:
  mov rax, QWORD PTR [rbp-16]
  mov eax, DWORD PTR [rax+8]
  cmp eax, 1
  jg .L1768
  mov rax, QWORD PTR [rbp-16]
  mov eax, DWORD PTR [rax+8]
  lea edx, [rax+1]
  mov rax, QWORD PTR [rbp-16]
  mov DWORD PTR [rax+8], edx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  movzx ecx, BYTE PTR [rbp-44]
  mov rax, QWORD PTR [rbp-40]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE6_M_dfsENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-16]
  mov eax, DWORD PTR [rax+8]
  lea edx, [rax-1]
  mov rax, QWORD PTR [rbp-16]
  mov DWORD PTR [rax+8], edx
.L1768:
  nop
  leave
  ret
_ZNSt4pairIPKcS1_EaSERKS2_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax+8], rdx
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE11_M_at_beginEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+32]
  cmp rdx, rax
  jne .L1772
  mov esi, 128
  mov edi, 1
  call _ZNSt15regex_constantsorENS_15match_flag_typeES0_
  mov edx, eax
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+136]
  mov esi, edx
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  test eax, eax
  jne .L1772
  mov eax, 1
  jmp .L1773
.L1772:
  mov eax, 0
.L1773:
  leave
  ret
_ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE9_M_at_endEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+40]
  cmp rdx, rax
  jne .L1776
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+136]
  mov esi, 2
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  test eax, eax
  jne .L1776
  mov eax, 1
  jmp .L1777
.L1776:
  mov eax, 0
.L1777:
  leave
  ret
_ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE16_M_word_boundaryEv:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-40], rdi
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+32]
  cmp rdx, rax
  jne .L1780
  mov rax, QWORD PTR [rbp-40]
  mov eax, DWORD PTR [rax+136]
  mov esi, 4
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  test eax, eax
  je .L1780
  mov eax, 1
  jmp .L1781
.L1780:
  mov eax, 0
.L1781:
  test al, al
  je .L1782
  mov eax, 0
  jmp .L1783
.L1782:
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+40]
  cmp rdx, rax
  jne .L1784
  mov rax, QWORD PTR [rbp-40]
  mov eax, DWORD PTR [rax+136]
  mov esi, 8
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  test eax, eax
  je .L1784
  mov eax, 1
  jmp .L1785
.L1784:
  mov eax, 0
.L1785:
  test al, al
  je .L1786
  mov eax, 0
  jmp .L1783
.L1786:
  mov BYTE PTR [rbp-1], 0
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+32]
  cmp rdx, rax
  jne .L1787
  mov rax, QWORD PTR [rbp-40]
  mov eax, DWORD PTR [rax+136]
  mov esi, 128
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  test eax, eax
  je .L1788
.L1787:
  mov eax, 1
  jmp .L1789
.L1788:
  mov eax, 0
.L1789:
  test al, al
  je .L1790
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+24]
  mov QWORD PTR [rbp-16], rax
  mov rax, QWORD PTR [rbp-16]
  mov esi, 1
  mov rdi, rax
  call _ZSt4prevIPKcET_S2_NSt15iterator_traitsIS2_E15difference_typeE
  movzx eax, BYTE PTR [rax]
  movsx edx, al
  mov rax, QWORD PTR [rbp-40]
  mov esi, edx
  mov rdi, rax
  call _ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE10_M_is_wordEc
  test al, al
  je .L1790
  mov BYTE PTR [rbp-1], 1
.L1790:
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+40]
  cmp rdx, rax
  je .L1792
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+24]
  movzx eax, BYTE PTR [rax]
  movsx edx, al
  mov rax, QWORD PTR [rbp-40]
  mov esi, edx
  mov rdi, rax
  call _ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE10_M_is_wordEc
  test al, al
  je .L1792
  mov eax, 1
  jmp .L1793
.L1792:
  mov eax, 0
.L1793:
  mov BYTE PTR [rbp-17], al
  movzx eax, BYTE PTR [rbp-1]
  cmp al, BYTE PTR [rbp-17]
  setne al
.L1783:
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE12_M_lookaheadEl:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 200
  mov QWORD PTR [rbp-200], rdi
  mov QWORD PTR [rbp-208], rsi
  mov rdx, QWORD PTR [rbp-200]
  lea rax, [rbp-48]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEC1ERKS6_
  mov rax, QWORD PTR [rbp-200]
  mov r8d, DWORD PTR [rax+136]
  mov rax, QWORD PTR [rbp-200]
  mov rdi, QWORD PTR [rax+48]
  mov rax, QWORD PTR [rbp-200]
  mov rdx, QWORD PTR [rax+40]
  mov rax, QWORD PTR [rbp-200]
  mov rsi, QWORD PTR [rax+24]
  lea rcx, [rbp-48]
  lea rax, [rbp-192]
  mov r9d, r8d
  mov r8, rdi
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EEC1ES2_S2_RSt6vectorIS5_S6_ERKNS3_11basic_regexIcS8_EENSt15regex_constants15match_flag_typeE
  mov rax, QWORD PTR [rbp-208]
  mov QWORD PTR [rbp-64], rax
  lea rax, [rbp-192]
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EE20_M_search_from_firstEv
  test al, al
  je .L1795
  mov QWORD PTR [rbp-24], 0
.L1798:
  lea rax, [rbp-48]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv
  cmp QWORD PTR [rbp-24], rax
  setb al
  test al, al
  je .L1796
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-48]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEixEm
  movzx eax, BYTE PTR [rax+16]
  test al, al
  je .L1797
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-48]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEixEm
  mov rbx, rax
  mov rax, QWORD PTR [rbp-200]
  mov rdx, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEixEm
  mov rsi, rbx
  mov rdi, rax
  call _ZNSt7__cxx119sub_matchIPKcEaSERKS3_
.L1797:
  add QWORD PTR [rbp-24], 1
  jmp .L1798
.L1796:
  mov ebx, 1
  jmp .L1799
.L1795:
  mov ebx, 0
.L1799:
  lea rax, [rbp-192]
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EED1Ev
  lea rax, [rbp-48]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EED1Ev
  mov eax, ebx
  jmp .L1805
  mov rbx, rax
  lea rax, [rbp-192]
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb0EED1Ev
  jmp .L1802
  mov rbx, rax
.L1802:
  lea rax, [rbp-48]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1805:
  add rsp, 200
  pop rbx
  pop rbp
  ret
_ZNKSt8__detail6_StateIcE10_M_matchesEc:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov eax, esi
  mov BYTE PTR [rbp-12], al
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZNKSt8__detail6_StateIcE14_M_get_matcherEv
  mov rdx, rax
  movsx eax, BYTE PTR [rbp-12]
  mov esi, eax
  mov rdi, rdx
  call _ZNKSt8functionIFbcEEclEc
  leave
  ret
_ZNSt8__detail16_Backref_matcherIPKcNSt7__cxx1112regex_traitsIcEEEC2EbRKS5_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov eax, esi
  mov QWORD PTR [rbp-24], rdx
  mov BYTE PTR [rbp-12], al
  mov rax, QWORD PTR [rbp-8]
  movzx edx, BYTE PTR [rbp-12]
  mov BYTE PTR [rax], dl
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rbp-24]
  mov QWORD PTR [rax+8], rdx
  nop
  pop rbp
  ret
_ZZNSt8__detail16_Backref_matcherIPKcNSt7__cxx1112regex_traitsIcEEE8_M_applyES2_S2_S2_S2_ENKUlccE_clEcc:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 24
  mov QWORD PTR [rbp-24], rdi
  mov eax, esi
  mov BYTE PTR [rbp-28], al
  mov eax, edx
  mov BYTE PTR [rbp-32], al
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+8]
  movsx edx, BYTE PTR [rbp-28]
  mov esi, edx
  mov rdi, rax
  call _ZNKSt5ctypeIcE7tolowerEc
  mov ebx, eax
  mov rax, QWORD PTR [rbp-24]
  mov rax, QWORD PTR [rax+8]
  movsx edx, BYTE PTR [rbp-32]
  mov esi, edx
  mov rdi, rax
  call _ZNKSt5ctypeIcE7tolowerEc
  cmp bl, al
  sete al
  add rsp, 24
  pop rbx
  pop rbp
  ret
_ZNSt8__detail16_Backref_matcherIPKcNSt7__cxx1112regex_traitsIcEEE8_M_applyES2_S2_S2_S2_:
  push rbp
  mov rbp, rsp
  push r13
  push r12
  push rbx
  sub rsp, 72
  mov QWORD PTR [rbp-56], rdi
  mov QWORD PTR [rbp-64], rsi
  mov QWORD PTR [rbp-72], rdx
  mov QWORD PTR [rbp-80], rcx
  mov QWORD PTR [rbp-88], r8
  mov rax, QWORD PTR [rbp-56]
  movzx eax, BYTE PTR [rax]
  xor eax, 1
  test al, al
  je .L1812
  mov rcx, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-80]
  mov rsi, QWORD PTR [rbp-72]
  mov rax, QWORD PTR [rbp-64]
  mov rdi, rax
  call _ZSt8__equal4IPKcS1_EbT_S2_T0_S3_
  jmp .L1816
.L1812:
  mov rax, QWORD PTR [rbp-56]
  mov rdx, QWORD PTR [rax+8]
  lea rax, [rbp-48]
  mov rsi, rdx
  mov rdi, rax
  call _ZNKSt7__cxx1112regex_traitsIcE6getlocEv
  lea rax, [rbp-48]
  mov rdi, rax
  call _ZSt9use_facetISt5ctypeIcEERKT_RKSt6locale
  mov QWORD PTR [rbp-40], rax
  lea rax, [rbp-48]
  mov rdi, rax
  call _ZNSt6localeD1Ev
  mov r12, QWORD PTR [rbp-56]
  mov r13, QWORD PTR [rbp-40]
  mov rcx, QWORD PTR [rbp-88]
  mov rdx, QWORD PTR [rbp-80]
  mov rsi, QWORD PTR [rbp-72]
  mov rax, QWORD PTR [rbp-64]
  mov r8, r12
  mov r9, r13
  mov rdi, rax
  call THIS_CALL_USED_TO_BE_ZSt8__equal4IPKcS1_ZNSt8__detail16_Backref_matcherIS1_NSt7__cxx1112regex_traitsIcEEE8_M_applyES1_S1_S1_S1_EUlccE_EbT_S9_T0_SA_T1_
  nop
  jmp .L1816
  mov rbx, rax
  lea rax, [rbp-48]
  mov rdi, rax
  call _ZNSt6localeD1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1816:
  add rsp, 72
  pop rbx
  pop r12
  pop r13
  pop rbp
  ret
_ZSt18__do_alloc_on_copyISaINSt7__cxx119sub_matchIPKcEEEEvRT_RKS6_St17integral_constantIbLb0EE:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  nop
  pop rbp
  ret
_ZN9__gnu_cxx17__normal_iteratorIPKNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEC2ERKS7_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-8]
  mov QWORD PTR [rax], rdx
  nop
  pop rbp
  ret
_ZSt22__uninitialized_copy_aIN9__gnu_cxx17__normal_iteratorIPKNSt7__cxx119sub_matchIPKcEESt6vectorIS6_SaIS6_EEEEPS6_S6_ET0_T_SF_SE_RSaIT1_E:
  push rbp
  mov rbp, rsp
  sub rsp, 32
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov QWORD PTR [rbp-24], rdx
  mov QWORD PTR [rbp-32], rcx
  mov rdx, QWORD PTR [rbp-24]
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZSt18uninitialized_copyIN9__gnu_cxx17__normal_iteratorIPKNSt7__cxx119sub_matchIPKcEESt6vectorIS6_SaIS6_EEEEPS6_ET0_T_SF_SE_
  leave
  ret
_ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIPKNSt7__cxx119sub_matchIPKcEESt6vectorIS6_SaIS6_EEEEET_SD_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKNSt7__cxx119sub_matchIPKcEESt6vectorIS6_SaIS6_EEEENS1_IPS6_SB_EEET1_T0_SG_SF_:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 48
  mov QWORD PTR [rbp-40], rdi
  mov QWORD PTR [rbp-48], rsi
  mov QWORD PTR [rbp-56], rdx
  mov rax, QWORD PTR [rbp-56]
  mov rdi, rax
  call _ZSt12__niter_baseIPNSt7__cxx119sub_matchIPKcEESt6vectorIS4_SaIS4_EEET_N9__gnu_cxx17__normal_iteratorIS9_T0_EE
  mov r12, rax
  mov rax, QWORD PTR [rbp-48]
  mov rdi, rax
  call _ZSt12__niter_baseIPKNSt7__cxx119sub_matchIPKcEESt6vectorIS4_SaIS4_EEET_N9__gnu_cxx17__normal_iteratorISA_T0_EE
  mov rbx, rax
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZSt12__niter_baseIPKNSt7__cxx119sub_matchIPKcEESt6vectorIS4_SaIS4_EEET_N9__gnu_cxx17__normal_iteratorISA_T0_EE
  mov rdx, r12
  mov rsi, rbx
  mov rdi, rax
  call _ZSt13__copy_move_aILb0EPKNSt7__cxx119sub_matchIPKcEEPS4_ET1_T0_S9_S8_
  mov QWORD PTR [rbp-24], rax
  lea rdx, [rbp-24]
  lea rax, [rbp-32]
  mov rsi, rdx
  mov rdi, rax
  call _ZN9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS5_SaIS5_EEEC1ERKS6_
  mov rax, QWORD PTR [rbp-32]
  add rsp, 48
  pop rbx
  pop r12
  pop rbp
  ret
_ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS6_SaIS6_EEEEEvT_SC_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rdx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt12_Destroy_auxILb1EE9__destroyIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx119sub_matchIPKcEESt6vectorIS8_SaIS8_EEEEEEvT_SE_
  nop
  leave
  ret
_ZSt12__miter_baseIPNSt7__cxx119sub_matchIPKcEEET_S6_:
  push rbp
  mov rbp, rsp
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  pop rbp
  ret
_ZSt14__copy_move_a2ILb0EPNSt7__cxx119sub_matchIPKcEES5_ET1_T0_S7_S6_:
  push rbp
  mov rbp, rsp
  push r12
  push rbx
  sub rsp, 32
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov rax, QWORD PTR [rbp-40]
  mov rdi, rax
  call _ZSt12__niter_baseIPNSt7__cxx119sub_matchIPKcEEET_S6_
  mov r12, rax
  mov rax, QWORD PTR [rbp-32]
  mov rdi, rax
  call _ZSt12__niter_baseIPNSt7__cxx119sub_matchIPKcEEET_S6_
  mov rbx, rax
  mov rax, QWORD PTR [rbp-24]
  mov rdi, rax
  call _ZSt12__niter_baseIPNSt7__cxx119sub_matchIPKcEEET_S6_
  mov rdx, r12
  mov rsi, rbx
  mov rdi, rax
  call _ZSt13__copy_move_aILb0EPNSt7__cxx119sub_matchIPKcEES5_ET1_T0_S7_S6_
  add rsp, 32
  pop rbx
  pop r12
  pop rbp
  ret
_ZSt18uninitialized_copyIPNSt7__cxx119sub_matchIPKcEES5_ET0_T_S7_S6_:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-24], rdi
  mov QWORD PTR [rbp-32], rsi
  mov QWORD PTR [rbp-40], rdx
  mov BYTE PTR [rbp-1], 1
  mov rdx, QWORD PTR [rbp-40]
  mov rcx, QWORD PTR [rbp-32]
  mov rax, QWORD PTR [rbp-24]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt20__uninitialized_copyILb0EE13__uninit_copyIPNSt7__cxx119sub_matchIPKcEES7_EET0_T_S9_S8_
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE16_M_rep_once_moreENS9_11_Match_modeEl:
  push rbp
  mov rbp, rsp
  sub rsp, 64
  mov QWORD PTR [rbp-40], rdi
  mov eax, esi
  mov QWORD PTR [rbp-56], rdx
  mov BYTE PTR [rbp-44], al
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+56]
  lea rdx, [rax+56]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rax
  mov rdi, rdx
  call _ZNKSt6vectorINSt8__detail6_StateIcEESaIS2_EEixEm
  mov QWORD PTR [rbp-8], rax
  mov rax, QWORD PTR [rbp-40]
  lea rdx, [rax+72]
  mov rax, QWORD PTR [rbp-56]
  mov rsi, rax
  mov rdi, rdx
  call _ZNSt6vectorISt4pairIPKciESaIS3_EEixEm
  mov QWORD PTR [rbp-16], rax
  mov rax, QWORD PTR [rbp-16]
  mov eax, DWORD PTR [rax+8]
  test eax, eax
  je .L1833
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+24]
  cmp rdx, rax
  je .L1834
.L1833:
  mov rax, QWORD PTR [rbp-16]
  mov rdx, QWORD PTR [rax+8]
  mov rax, QWORD PTR [rax]
  mov QWORD PTR [rbp-32], rax
  mov QWORD PTR [rbp-24], rdx
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-16]
  mov QWORD PTR [rax], rdx
  mov rax, QWORD PTR [rbp-16]
  mov DWORD PTR [rax+8], 1
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  movzx ecx, BYTE PTR [rbp-44]
  mov rax, QWORD PTR [rbp-40]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
  lea rdx, [rbp-32]
  mov rax, QWORD PTR [rbp-16]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt4pairIPKciEaSERKS2_
  nop
  jmp .L1836
.L1834:
  mov rax, QWORD PTR [rbp-16]
  mov eax, DWORD PTR [rax+8]
  cmp eax, 1
  jg .L1836
  mov rax, QWORD PTR [rbp-16]
  mov eax, DWORD PTR [rax+8]
  lea edx, [rax+1]
  mov rax, QWORD PTR [rbp-16]
  mov DWORD PTR [rax+8], edx
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+16]
  movzx ecx, BYTE PTR [rbp-44]
  mov rax, QWORD PTR [rbp-40]
  mov esi, ecx
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE6_M_dfsENS9_11_Match_modeEl
  mov rax, QWORD PTR [rbp-16]
  mov eax, DWORD PTR [rax+8]
  lea edx, [rax-1]
  mov rax, QWORD PTR [rbp-16]
  mov DWORD PTR [rax+8], edx
.L1836:
  nop
  leave
  ret
_ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE11_M_at_beginEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+32]
  cmp rdx, rax
  jne .L1838
  mov esi, 128
  mov edi, 1
  call _ZNSt15regex_constantsorENS_15match_flag_typeES0_
  mov edx, eax
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+112]
  mov esi, edx
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  test eax, eax
  jne .L1838
  mov eax, 1
  jmp .L1839
.L1838:
  mov eax, 0
.L1839:
  leave
  ret
_ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE9_M_at_endEv:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-8]
  mov rax, QWORD PTR [rax+40]
  cmp rdx, rax
  jne .L1842
  mov rax, QWORD PTR [rbp-8]
  mov eax, DWORD PTR [rax+112]
  mov esi, 2
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  test eax, eax
  jne .L1842
  mov eax, 1
  jmp .L1843
.L1842:
  mov eax, 0
.L1843:
  leave
  ret
_ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE16_M_word_boundaryEv:
  push rbp
  mov rbp, rsp
  sub rsp, 48
  mov QWORD PTR [rbp-40], rdi
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+32]
  cmp rdx, rax
  jne .L1846
  mov rax, QWORD PTR [rbp-40]
  mov eax, DWORD PTR [rax+112]
  mov esi, 4
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  test eax, eax
  je .L1846
  mov eax, 1
  jmp .L1847
.L1846:
  mov eax, 0
.L1847:
  test al, al
  je .L1848
  mov eax, 0
  jmp .L1849
.L1848:
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+40]
  cmp rdx, rax
  jne .L1850
  mov rax, QWORD PTR [rbp-40]
  mov eax, DWORD PTR [rax+112]
  mov esi, 8
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  test eax, eax
  je .L1850
  mov eax, 1
  jmp .L1851
.L1850:
  mov eax, 0
.L1851:
  test al, al
  je .L1852
  mov eax, 0
  jmp .L1849
.L1852:
  mov BYTE PTR [rbp-1], 0
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+32]
  cmp rdx, rax
  jne .L1853
  mov rax, QWORD PTR [rbp-40]
  mov eax, DWORD PTR [rax+112]
  mov esi, 128
  mov edi, eax
  call _ZNSt15regex_constantsanENS_15match_flag_typeES0_
  test eax, eax
  je .L1854
.L1853:
  mov eax, 1
  jmp .L1855
.L1854:
  mov eax, 0
.L1855:
  test al, al
  je .L1856
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+24]
  mov QWORD PTR [rbp-16], rax
  mov rax, QWORD PTR [rbp-16]
  mov esi, 1
  mov rdi, rax
  call _ZSt4prevIPKcET_S2_NSt15iterator_traitsIS2_E15difference_typeE
  movzx eax, BYTE PTR [rax]
  movsx edx, al
  mov rax, QWORD PTR [rbp-40]
  mov esi, edx
  mov rdi, rax
  call _ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE10_M_is_wordEc
  test al, al
  je .L1856
  mov BYTE PTR [rbp-1], 1
.L1856:
  mov rax, QWORD PTR [rbp-40]
  mov rdx, QWORD PTR [rax+24]
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+40]
  cmp rdx, rax
  je .L1858
  mov rax, QWORD PTR [rbp-40]
  mov rax, QWORD PTR [rax+24]
  movzx eax, BYTE PTR [rax]
  movsx edx, al
  mov rax, QWORD PTR [rbp-40]
  mov esi, edx
  mov rdi, rax
  call _ZNKSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE10_M_is_wordEc
  test al, al
  je .L1858
  mov eax, 1
  jmp .L1859
.L1858:
  mov eax, 0
.L1859:
  mov BYTE PTR [rbp-17], al
  movzx eax, BYTE PTR [rbp-1]
  cmp al, BYTE PTR [rbp-17]
  setne al
.L1849:
  leave
  ret
_ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE12_M_lookaheadEl:
  push rbp
  mov rbp, rsp
  push rbx
  sub rsp, 184
  mov QWORD PTR [rbp-184], rdi
  mov QWORD PTR [rbp-192], rsi
  mov rdx, QWORD PTR [rbp-184]
  lea rax, [rbp-48]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEC1ERKS6_
  mov rax, QWORD PTR [rbp-184]
  mov r8d, DWORD PTR [rax+112]
  mov rax, QWORD PTR [rbp-184]
  mov rdi, QWORD PTR [rax+48]
  mov rax, QWORD PTR [rbp-184]
  mov rdx, QWORD PTR [rax+40]
  mov rax, QWORD PTR [rbp-184]
  mov rsi, QWORD PTR [rax+24]
  lea rcx, [rbp-48]
  lea rax, [rbp-176]
  mov r9d, r8d
  mov r8, rdi
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EEC1ES2_S2_RSt6vectorIS5_S6_ERKNS3_11basic_regexIcS8_EENSt15regex_constants15match_flag_typeE
  mov rax, QWORD PTR [rbp-192]
  mov QWORD PTR [rbp-80], rax
  lea rax, [rbp-176]
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EE20_M_search_from_firstEv
  test al, al
  je .L1861
  mov QWORD PTR [rbp-24], 0
.L1864:
  lea rax, [rbp-48]
  mov rdi, rax
  call _ZNKSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EE4sizeEv
  cmp QWORD PTR [rbp-24], rax
  setb al
  test al, al
  je .L1862
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-48]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEixEm
  movzx eax, BYTE PTR [rax+16]
  test al, al
  je .L1863
  mov rdx, QWORD PTR [rbp-24]
  lea rax, [rbp-48]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEixEm
  mov rbx, rax
  mov rax, QWORD PTR [rbp-184]
  mov rdx, QWORD PTR [rbp-24]
  mov rsi, rdx
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EEixEm
  mov rsi, rbx
  mov rdi, rax
  call _ZNSt7__cxx119sub_matchIPKcEaSERKS3_
.L1863:
  add QWORD PTR [rbp-24], 1
  jmp .L1864
.L1862:
  mov ebx, 1
  jmp .L1865
.L1861:
  mov ebx, 0
.L1865:
  lea rax, [rbp-176]
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EED1Ev
  lea rax, [rbp-48]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EED1Ev
  mov eax, ebx
  jmp .L1871
  mov rbx, rax
  lea rax, [rbp-176]
  mov rdi, rax
  call _ZNSt8__detail9_ExecutorIPKcSaINSt7__cxx119sub_matchIS2_EEENS3_12regex_traitsIcEELb1EED1Ev
  jmp .L1868
  mov rbx, rax
.L1868:
  lea rax, [rbp-48]
  mov rdi, rax
  call _ZNSt6vectorINSt7__cxx119sub_matchIPKcEESaIS4_EED1Ev
  mov rax, rbx
  mov rdi, rax
  call _Unwind_Resume
.L1871:
  add rsp, 184
  pop rbx
  pop rbp
  ret
_ZNSaISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EEEC2IS5_EERKSaIT_E:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS7_ELNS_12_Lock_policyE2EEEC2Ev
  nop
  leave
  ret
_ZNSaISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS5_ELN9__gnu_cxx12_Lock_policyE2EEED2Ev:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov rax, QWORD PTR [rbp-8]
  mov rdi, rax
  call _ZN9__gnu_cxx13new_allocatorISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS7_ELNS_12_Lock_policyE2EEED2Ev
  nop
  leave
  ret
_ZSt18__allocate_guardedISaISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS6_ELN9__gnu_cxx12_Lock_policyE2EEEESt15__allocated_ptrIT_ERSD_:
  push rbp
  mov rbp, rsp
  sub rsp, 16
  mov QWORD PTR [rbp-8], rdi
  mov QWORD PTR [rbp-16], rsi
  mov rax, QWORD PTR [rbp-16]
  mov esi, 1
  mov rdi, rax
  call _ZNSt16allocator_traitsISaISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS6_ELN9__gnu_cxx12_Lock_policyE2EEEE8allocateERSB_m
  mov rdx, rax
  mov rcx, QWORD PTR [rbp-16]
  mov rax, QWORD PTR [rbp-8]
  mov rsi, rcx
  mov rdi, rax
  call _ZNSt15__allocated_ptrISaISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS6_ELN9__gnu_cxx12_Lock_policyE2EEEEC1ERSB_PSA_
  mov rax, QWORD PTR [rbp-8]
  leave
  ret
_ZNSt15__allocated_ptrISaISt23_Sp_counted_ptr_inplaceINSt8__detail4_NFAINSt7__cxx1112regex_traitsIcEEEESaIS6_ELN9__gnu_cxx12_Lock_policyE2EEEED2Ev:
