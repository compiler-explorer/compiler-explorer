///////////////////////////////////////////////////////////////////////////////
//
// IAR ANSI C/C++ Compiler V8.32.3.193/W32 for ARM        09/Oct/2020  12:40:07
// Copyright 1999-2019 IAR Systems AB.
//
//    Cpu mode     =
//    Endian       =  little
//    Source file  =
//        C:\Users\NA_BUI~1\AppData\Local\Temp\compiler-explorer-compiler202099-14988-evjfs2.ood3c\example.cpp
//    Command line =
//        -lB
//        C:\Users\NA_BUI~1\AppData\Local\Temp\compiler-explorer-compiler202099-14988-evjfs2.ood3c\output.s
//        -o
//        C:\Users\NA_BUI~1\AppData\Local\Temp\compiler-explorer-compiler202099-14988-evjfs2.ood3c\output.s.obj
//        --enable_restrict -IC:\arm\inc -IC:\arm\inc\c -IC:\arm\inc\cpp
//        --dlib_config C:\arm\inc\c\DLib_Config_Full.h --c++ -e
//        --no_exceptions --no_rtti --no_static_destruction --cpu Cortex-M4
//        --fpu VFPv4_sp --endian little --cpu_mode thumb
//        C:\Users\NA_BUI~1\AppData\Local\Temp\compiler-explorer-compiler202099-14988-evjfs2.ood3c\example.cpp
//    Locale       =  C
//    List file    =
//        C:\Users\NA_BUI~1\AppData\Local\Temp\compiler-explorer-compiler202099-14988-evjfs2.ood3c\output.s
//
///////////////////////////////////////////////////////////////////////////////

        #define SHT_PROGBITS 0x1
        #define SHT_INIT_ARRAY 0xe

        EXTERN printf

        PUBLIC _ZN2NA10FixedStackIiLj16EE11IteratorEndEv
        PUBLIC _ZN2NA10FixedStackIiLj16EE13IteratorBeginEv
        PUBLIC _ZN2NA10FixedStackIiLj16EE4PushEi
        PUBLIC _ZN2NA10FixedStackIiLj16EEC1Ev
        PUBLIC _ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE3endEv
        PUBLIC _ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE5beginEv
        PUBLIC _ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratorC1EPi
        PUBLIC _ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratordeEv
        PUBLIC _ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratorppEv
        PUBLIC _ZNK2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratorneERKS4_
        PUBLIC _ZNSt5arrayIiLj16EE4dataEv
        PUBLIC _ZNSt5arrayIiLj16EEixEj
        PUBLIC desa
        PUBLIC main

// C:\Users\NA_BUI~1\AppData\Local\Temp\compiler-explorer-compiler202099-14988-evjfs2.ood3c\example.cpp
//    1 #include <array>
//    2
//    3 namespace NA {
//    4
//    5     template<class Derived, typename DataType>
//    6     class Iterator {
//    7     public:
//    8         Iterator() = default;
//    9
//   10         class iterator {
//   11         public:
//   12             iterator(DataType* ptr = nullptr) : ptr(ptr) {}
//   13             inline iterator operator++() {
//   14                 ++ptr;
//   15                 return *this;
//   16             }
//   17             inline bool operator!=(const iterator& other) const { return ptr != other.ptr; }
//   18             inline DataType& operator*() { return *ptr; }
//   19
//   20         private:
//   21             DataType* ptr;
//   22         };
//   23
//   24         class const_iterator {
//   25         public:
//   26             const_iterator(const DataType* ptr = nullptr) : ptr(ptr) {}
//   27             inline const_iterator operator++() {
//   28                 ++ptr;
//   29                 return *this;
//   30             }
//   31             inline bool operator!=(const iterator& other) const { return ptr != other.ptr; }
//   32             inline const DataType& operator*() const { return *ptr; }
//   33
//   34         private:
//   35             const DataType* ptr;
//   36         };
//   37
//   38         inline iterator begin() { return iterator((DataType*)((Derived*)this)->IteratorBegin()); }
//   39         inline iterator end() { return iterator((DataType*)((Derived*)this)->IteratorEnd()); }
//   40         inline const_iterator begin() const { return const_iterator((const DataType*)((Derived*)this)->IteratorBegin()); }
//   41         inline const_iterator end() const { return const_iterator((const DataType*)((Derived*)this)->IteratorEnd()); }
//   42     };
//   43
//   44 } // namespace NA
//   45
//   46
//   47 namespace NA {
//   48     template<typename T, unsigned int MAX_SIZE>
//   49     class FixedStack : public Iterator<FixedStack<T, MAX_SIZE>, T> {
//   50         friend class Iterator<FixedStack<T, MAX_SIZE>, T>;
//   51         inline constexpr auto IteratorBegin() { return m_Stack.data(); }
//   52         inline constexpr auto IteratorEnd() { return m_Stack.data() + m_Pos + 1; }
//   53
//   54     public:
//   55         FixedStack() : m_Pos(-1) {}
//   56
//   57         inline bool Push(T elem) {
//   58             if ((m_Pos + 1) < MAX_SIZE) {
//   59                 m_Pos++;
//   60                 m_Stack[m_Pos] = elem;
//   61                 return true;
//   62             }
//   63
//   64             return false;
//   65         }
//   66
//   67         inline bool Pop(T replaceLast) {
//   68             if (m_Pos != -1) {
//   69                 m_Stack[m_Pos--] = replaceLast;
//   70                 return true;
//   71             }
//   72
//   73             return false;
//   74         }
//   75
//   76         inline bool Pop() {
//   77             if (m_Pos != -1) {
//   78                 m_Pos--;
//   79                 return true;
//   80             }
//   81
//   82             return false;
//   83         }
//   84
//   85         inline void Clear(T fill) {
//   86             m_Pos = -1;
//   87             m_Stack.fill(fill);
//   88         }
//   89
//   90         inline void Clear() { m_Pos = -1; }
//   91
//   92         inline int FreeSpace() { return (MAX_SIZE - (m_Pos + 1)); }
//   93
//   94         inline T* DataPtr() const { return m_Stack.data(); }
//   95
//   96         inline int Size() const { return m_Pos + 1; }
//   97
//   98         inline bool Full() { return Size() == MAX_SIZE; }
//   99
//  100         inline T& Top(int back = 0) {
//  101              return m_Stack[m_Pos + back];
//  102         }
//  103
//  104         inline bool Empty() const { return m_Pos == -1; }
//  105
//  106         inline T& operator[](int index) { return m_Stack[index]; }
//  107         inline const T& operator[](int index) const { return m_Stack[index]; }
//  108
//  109     private:
//  110         int m_Pos;
//  111         std::array<T, MAX_SIZE> m_Stack;
//  112     };
//  113
//  114 } // namespace NA
//  115

        SECTION `.text`:CODE:NOROOT(1)
        THUMB
// static __intrinsic __interwork __softfp void __sti__routine()
__sti__routine:
        PUSH     {R7,LR}
//  116 NA::FixedStack<int, 16> desa;
        LDR.N    R0,??DataTable1_1
        BL       _ZN2NA10FixedStackIiLj16EEC1Ev
        POP      {R0,PC}          ;; return

        SECTION `.bss`:DATA:REORDER:NOROOT(2)
        DATA
desa:
        DS8 68

        SECTION `.text`:CODE:NOROOT(1)
        THUMB
main:
        PUSH     {R2-R4,LR}
        LDR.N    R4,??DataTable1_1
        MOVS     R1,#+1
        MOVS     R0,R4
        BL       _ZN2NA10FixedStackIiLj16EE4PushEi
        MOVS     R1,#+2
        MOVS     R0,R4
        BL       _ZN2NA10FixedStackIiLj16EE4PushEi
        MOVS     R1,#+3
        MOVS     R0,R4
        BL       _ZN2NA10FixedStackIiLj16EE4PushEi
        MOVS     R1,#+4
        MOVS     R0,R4
        BL       _ZN2NA10FixedStackIiLj16EE4PushEi
        MOVS     R0,R4
        BL       _ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE5beginEv
        STR      R0,[SP, #+0]
        MOVS     R0,R4
        BL       _ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE3endEv
        STR      R0,[SP, #+4]
        B.N      ??main_0
??main_1:
        MOV      R0,SP
        BL       _ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratordeEv
        LDR      R1,[R0, #+0]
        ADR.N    R0,??DataTable1  ;; "%d\n"
        BL       printf
        MOV      R0,SP
        BL       _ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratorppEv
??main_0:
        ADD      R1,SP,#+4
        MOV      R0,SP
        BL       _ZNK2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratorneERKS4_
        CMP      R0,#+0
        BNE.N    ??main_1
        POP      {R0,R1,R4,PC}    ;; return

        SECTION `.text`:CODE:NOROOT(2)
        SECTION_TYPE SHT_PROGBITS, 0
        DATA
??DataTable1:
        DATA8
        DC8      "%d\n"

        SECTION `.text`:CODE:NOROOT(2)
        SECTION_TYPE SHT_PROGBITS, 0
        DATA
??DataTable1_1:
        DATA32
        DC32     desa

        SECTION `.text`:CODE:REORDER:NOROOT(1)
        SECTION_GROUP _ZNSt5arrayIiLj16EEixEj
        THUMB
// __interwork __softfp int & std::array<int, 16U>::operator[](size_t)
_ZNSt5arrayIiLj16EEixEj:
        ADD      R0,R0,R1, LSL #+2
        BX       LR               ;; return

        SECTION `.text`:CODE:REORDER:NOROOT(1)
        SECTION_GROUP _ZNSt5arrayIiLj16EE4dataEv
        THUMB
// __interwork __softfp int *std::array<int, 16U>::data()
_ZNSt5arrayIiLj16EE4dataEv:
        BX       LR               ;; return

        SECTION `.text`:CODE:REORDER:NOROOT(1)
        SECTION_GROUP _ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratorC1EPi
        THUMB
// __code __interwork __softfp NA::Iterator<NA::FixedStack<int, 16U>, int>::iterator::iterator(int *)
_ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratorC1EPi:
        STR      R1,[R0, #+0]
        BX       LR               ;; return

        SECTION `.text`:CODE:REORDER:NOROOT(1)
        SECTION_GROUP _ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratorppEv
        THUMB
// __interwork __softfp NA::Iterator<NA::FixedStack<int, 16U>, int>::iterator NA::Iterator<NA::FixedStack<int, 16U>, int>::iterator::operator++()
_ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratorppEv:
        LDR      R1,[R0, #+0]
        ADDS     R1,R1,#+4
        STR      R1,[R0, #+0]
        LDR      R0,[R0, #+0]
        BX       LR               ;; return

        SECTION `.text`:CODE:REORDER:NOROOT(1)
        SECTION_GROUP _ZNK2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratorneERKS4_
        THUMB
// __interwork __softfp bool NA::Iterator<NA::FixedStack<int, 16U>, int>::iterator::operator!=(NA::Iterator<NA::FixedStack<int, 16U>, int>::iterator const &) const
_ZNK2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratorneERKS4_:
        LDR      R0,[R0, #+0]
        LDR      R1,[R1, #+0]
        CMP      R0,R1
        BEQ.N    `??operator!=_0`
        MOVS     R0,#+1
        B.N      `??operator!=_1`
`??operator!=_0`:
        MOVS     R0,#+0
`??operator!=_1`:
        UXTB     R0,R0            ;; ZeroExt  R0,R0,#+24,#+24
        BX       LR               ;; return

        SECTION `.text`:CODE:REORDER:NOROOT(1)
        SECTION_GROUP _ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratordeEv
        THUMB
// __interwork __softfp int &NA::Iterator<NA::FixedStack<int, 16U>, int>::iterator::operator*()
_ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratordeEv:
        LDR      R0,[R0, #+0]
        BX       LR               ;; return

        SECTION `.text`:CODE:REORDER:NOROOT(1)
        SECTION_GROUP _ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE5beginEv
        THUMB
// __interwork __softfp NA::Iterator<NA::FixedStack<int, 16U>, int>::iterator NA::Iterator<NA::FixedStack<int, 16U>, int>::begin()
_ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE5beginEv:
        PUSH     {R7,LR}
        BL       _ZN2NA10FixedStackIiLj16EE13IteratorBeginEv
        MOVS     R1,R0
        MOV      R0,SP
        BL       _ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratorC1EPi
        LDR      R0,[R0, #+0]
        POP      {R1,PC}          ;; return

        SECTION `.text`:CODE:REORDER:NOROOT(1)
        SECTION_GROUP _ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE3endEv
        THUMB
// __interwork __softfp NA::Iterator<NA::FixedStack<int, 16U>, int>::iterator NA::Iterator<NA::FixedStack<int, 16U>, int>::end()
_ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE3endEv:
        PUSH     {R7,LR}
        BL       _ZN2NA10FixedStackIiLj16EE11IteratorEndEv
        MOVS     R1,R0
        MOV      R0,SP
        BL       _ZN2NA8IteratorINS_10FixedStackIiLj16EEEiE8iteratorC1EPi
        LDR      R0,[R0, #+0]
        POP      {R1,PC}          ;; return

        SECTION `.text`:CODE:REORDER:NOROOT(1)
        SECTION_GROUP _ZN2NA10FixedStackIiLj16EE13IteratorBeginEv
        THUMB
// __interwork __softfp int *NA::FixedStack<int, 16U>::IteratorBegin()
_ZN2NA10FixedStackIiLj16EE13IteratorBeginEv:
        PUSH     {R7,LR}
        ADDS     R0,R0,#+4
        BL       _ZNSt5arrayIiLj16EE4dataEv
        POP      {R1,PC}          ;; return

        SECTION `.text`:CODE:REORDER:NOROOT(1)
        SECTION_GROUP _ZN2NA10FixedStackIiLj16EE11IteratorEndEv
        THUMB
// __interwork __softfp int *NA::FixedStack<int, 16U>::IteratorEnd()
_ZN2NA10FixedStackIiLj16EE11IteratorEndEv:
        PUSH     {R4,LR}
        MOVS     R4,R0
        ADDS     R0,R4,#+4
        BL       _ZNSt5arrayIiLj16EE4dataEv
        LDR      R1,[R4, #+0]
        ADD      R0,R0,R1, LSL #+2
        ADDS     R0,R0,#+4
        POP      {R4,PC}          ;; return

        SECTION `.text`:CODE:REORDER:NOROOT(1)
        SECTION_GROUP _ZN2NA10FixedStackIiLj16EEC1Ev
        THUMB
// __code __interwork __softfp NA::FixedStack<int, 16U>::FixedStack()
_ZN2NA10FixedStackIiLj16EEC1Ev:
        MOVS     R1,#-1
        STR      R1,[R0, #+0]
        BX       LR               ;; return

        SECTION `.text`:CODE:REORDER:NOROOT(1)
        SECTION_GROUP _ZN2NA10FixedStackIiLj16EE4PushEi
        THUMB
// __interwork __softfp bool NA::FixedStack<int, 16U>::Push(int)
_ZN2NA10FixedStackIiLj16EE4PushEi:
        PUSH     {R4,LR}
        MOVS     R4,R1
        LDR      R1,[R0, #+0]
        ADDS     R1,R1,#+1
        CMP      R1,#+16
        BCS.N    ??Push_0
        LDR      R1,[R0, #+0]
        ADDS     R1,R1,#+1
        STR      R1,[R0, #+0]
        LDR      R1,[R0, #+0]
        ADDS     R0,R0,#+4
        BL       _ZNSt5arrayIiLj16EEixEj
        STR      R4,[R0, #+0]
        MOVS     R0,#+1
        B.N      ??Push_1
??Push_0:
        MOVS     R0,#+0
??Push_1:
        POP      {R4,PC}          ;; return

        SECTION `.init_array`:CODE:ROOT(2)
        SECTION_TYPE SHT_INIT_ARRAY, 0
        DATA
        DC32    RELOC_ARM_TARGET1 __sti__routine

        SECTION `.iar_vfe_header`:DATA:NOALLOC:NOROOT(2)
        SECTION_TYPE SHT_PROGBITS, 0
        DATA
        DC32 0

        SECTION `.rodata`:CONST:REORDER:NOROOT(2)
        DATA
        DC8 "%d\012"

        END
//  117
//  118 void main() {
//  119
//  120
//  121     desa.Push(1);
//  122     desa.Push(2);
//  123     desa.Push(3);
//  124     desa.Push(4);
//  125
//  126     for(auto d : desa) {
//  127         printf("%d\n", d);
//  128     }
//  129 }
//
//  68 bytes in section .bss
//   4 bytes in section .init_array
//   4 bytes in section .rodata
// 260 bytes in section .text
//
// 110 bytes of CODE  memory (+ 154 bytes shared)
//   4 bytes of CONST memory
//  68 bytes of DATA  memory
//
//Errors: none
//Warnings: none
