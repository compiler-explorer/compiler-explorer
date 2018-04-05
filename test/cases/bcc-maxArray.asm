        .386p
        ifdef ??version
        if ??version GT 500H
        .mmx
        endif
        endif
        model flat
        ifndef  ??version
        ?debug  macro
        endm
        endif
        ?debug  S "Z:/tmp/compiler-explorer-compiler11835-686-19j4wz8.s26r/example.cpp"
        ?debug  T "Z:/tmp/compiler-explorer-compiler11835-686-19j4wz8.s26r/example.cpp"
_TEXT   segment dword public use32 'CODE'
_TEXT   ends
_DATA   segment dword public use32 'DATA'
_DATA   ends
_BSS    segment dword public use32 'BSS'
_BSS    ends
DGROUP  group       _BSS,_DATA
_TEXT   segment dword public use32 'CODE'
@testFunction$qpii      segment virtual
        align   2
@@testFunction$qpii     proc      near
?live16385@0:
 ;      
 ;      int testFunction(int* input, int length) {
 ;      
        ?debug L 1
@1:
        push      ebp
        mov       ebp,esp
        push      ebx
 ;      
 ;        int sum = 0;
 ;      
        ?debug L 2
?live16385@16: ; EBX = length
        xor       ecx,ecx
        ?debug L 1
?live16385@32: ; 
        mov       ebx,dword ptr [ebp+12]
 ;      
 ;        for (int i = 0; i < length; ++i) {
 ;      
        ?debug L 3
?live16385@48: ; ECX = sum, EBX = length
@2:
        xor       edx,edx
        mov       eax,dword ptr [ebp+8]
        cmp       ebx,edx
        jle       short @4
 ;      
 ;          sum += input[i];
 ;      
        ?debug L 4
?live16385@64: ; EAX = @temp0, EDX = i, ECX = sum, EBX = length
@3:
@5:
        add       ecx,dword ptr [eax]
        ?debug L 3
@6:
@7:
        inc       edx
        add       eax,4
        cmp       ebx,edx
        jg        short @3
 ;      
 ;        }
 ;        return sum;
 ;      
        ?debug L 6
?live16385@96: ; ECX = sum
@4:
        mov       eax,ecx
 ;      
 ;      }
 ;      
        ?debug L 7
?live16385@112: ; 
@10:
@9:
        pop       ebx
        pop       ebp
        ret 
        ?debug L 0
@@testFunction$qpii     endp
@testFunction$qpii      ends
_TEXT   ends
_DATA   segment dword public use32 'DATA'
        align   4
_i      label       dword
        dd      0
_DATA   ends
_TEXT   segment dword public use32 'CODE'
_TEXT   ends
        public  _i
        ?debug  D "Z:/tmp/compiler-explorer-compiler11835-686-19j4wz8.s26r/example.cpp" 19589 38693
        end
        