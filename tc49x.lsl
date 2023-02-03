//****************************************************************************
//**                                                                         *
//**  FILE        :  tc49x.lsl                                               *
//**                                                                         *
//**  DESCRIPTION :  LSL description: Infineon TC49X                         *
//**                 This 3rd Generation AURIX derivative has:               *
//**                            7 TriCore cores                              *
//**                            10 MCS cores                                 *
//**                            1 XC800 cores                                *
//**                                                                         *
//**  Copyright 2020-2021 TASKING BV                                         *
//**                                                                         *
//**  Configuration macros:                                                  *
//**    __NO_VTC                                                             *
//**      When this macro is defined, the virtual core vtc is not available, *
//**      so a separate link task is needed for each TriCore core            *
//**      (default: undefined).                                              *
//**    USTACK_TC0                                                           *
//**      Size of the global stack for core 0                                *
//**      (default: 16k for core 0, 1k for other cores).                     *
//**    ISTACK_TC0                                                           *
//**      Interrupt stack size for core 0 (default: 1k).                     *
//**    USTACK_TC1                                                           *
//**      Size of the global stack for core 1                                *
//**      (default: 16k for core 0, 1k for other cores).                     *
//**    ISTACK_TC1                                                           *
//**      Interrupt stack size for core 1 (default: 1k).                     *
//**    USTACK_TC2                                                           *
//**      Size of the global stack for core 2                                *
//**      (default: 16k for core 0, 1k for other cores).                     *
//**    ISTACK_TC2                                                           *
//**      Interrupt stack size for core 2 (default: 1k).                     *
//**    USTACK_TC3                                                           *
//**      Size of the global stack for core 3                                *
//**      (default: 16k for core 0, 1k for other cores).                     *
//**    ISTACK_TC3                                                           *
//**      Interrupt stack size for core 3 (default: 1k).                     *
//**    USTACK_TC4                                                           *
//**      Size of the global stack for core 4                                *
//**      (default: 16k for core 0, 1k for other cores).                     *
//**    ISTACK_TC4                                                           *
//**      Interrupt stack size for core 4 (default: 1k).                     *
//**    USTACK_TC5                                                           *
//**      Size of the global stack for core 5                                *
//**      (default: 16k for core 0, 1k for other cores).                     *
//**    ISTACK_TC5                                                           *
//**      Interrupt stack size for core 5 (default: 1k).                     *
//**    USTACK_TC6                                                           *
//**      Size of the global stack for core 6                                *
//**      (default: 16k for core 0, 1k for other cores).                     *
//**    ISTACK_TC6                                                           *
//**      Interrupt stack size for core 6 (default: 1k).                     *
//**    HEAP                                                                 *
//**      Size of the heap, available when macro __NO_VTC is not defined     *
//**      (default: 16k)                                                     *
//**    HEAP_TC0                                                             *
//**      Size of the heap for core 0, available when macro __NO_VTC is      *
//**      defined (default: 16k)                                             *
//**    HEAP_TC1                                                             *
//**      Size of the heap for core 1, available when macro __NO_VTC is      *
//**      defined (default: 16k)                                             *
//**    HEAP_TC2                                                             *
//**      Size of the heap for core 2, available when macro __NO_VTC is      *
//**      defined (default: 16k)                                             *
//**    HEAP_TC3                                                             *
//**      Size of the heap for core 3, available when macro __NO_VTC is      *
//**      defined (default: 16k)                                             *
//**    HEAP_TC4                                                             *
//**      Size of the heap for core 4, available when macro __NO_VTC is      *
//**      defined (default: 16k)                                             *
//**    HEAP_TC5                                                             *
//**      Size of the heap for core 5, available when macro __NO_VTC is      *
//**      defined (default: 16k)                                             *
//**    HEAP_TC6                                                             *
//**      Size of the heap for core 6, available when macro __NO_VTC is      *
//**      defined (default: 16k)                                             *
//**    A0_START                                                             *
//**      Fixed address of the A0-addressable segment for all cores          *
//**      (default: not set).                                                *
//**    A1_START                                                             *
//**      Fixed address of the A1-addressable segment for all cores          *
//**      (default: not set).                                                *
//**    A8_START                                                             *
//**      Fixed address of the A8-addressable segment for all cores          *
//**      (default: not set).                                                *
//**    A9_START                                                             *
//**      Fixed address of the A9-addressable segment for all cores          *
//**      (default: not set).                                                *
//**    __TC0_A0_START                                                       *
//**      Fixed address of the A0-addressable segment for core 0,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC0_A1_START                                                       *
//**      Fixed address of the A1-addressable segment for core 0,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC0_A8_START                                                       *
//**      Fixed address of the A8-addressable segment for core 0,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC0_A9_START                                                       *
//**      Fixed address of the A9-addressable segment for core 0,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC1_A0_START                                                       *
//**      Fixed address of the A0-addressable segment for core 1,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC1_A1_START                                                       *
//**      Fixed address of the A1-addressable segment for core 1,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC1_A8_START                                                       *
//**      Fixed address of the A8-addressable segment for core 1,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC1_A9_START                                                       *
//**      Fixed address of the A9-addressable segment for core 1,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC2_A0_START                                                       *
//**      Fixed address of the A0-addressable segment for core 2,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC2_A1_START                                                       *
//**      Fixed address of the A1-addressable segment for core 2,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC2_A8_START                                                       *
//**      Fixed address of the A8-addressable segment for core 2,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC2_A9_START                                                       *
//**      Fixed address of the A9-addressable segment for core 2,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC3_A0_START                                                       *
//**      Fixed address of the A0-addressable segment for core 3,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC3_A1_START                                                       *
//**      Fixed address of the A1-addressable segment for core 3,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC3_A8_START                                                       *
//**      Fixed address of the A8-addressable segment for core 3,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC3_A9_START                                                       *
//**      Fixed address of the A9-addressable segment for core 3,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC4_A0_START                                                       *
//**      Fixed address of the A0-addressable segment for core 4,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC4_A1_START                                                       *
//**      Fixed address of the A1-addressable segment for core 4,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC4_A8_START                                                       *
//**      Fixed address of the A8-addressable segment for core 4,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC4_A9_START                                                       *
//**      Fixed address of the A9-addressable segment for core 4,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC5_A0_START                                                       *
//**      Fixed address of the A0-addressable segment for core 5,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC5_A1_START                                                       *
//**      Fixed address of the A1-addressable segment for core 5,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC5_A8_START                                                       *
//**      Fixed address of the A8-addressable segment for core 5,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC5_A9_START                                                       *
//**      Fixed address of the A9-addressable segment for core 5,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC6_A0_START                                                       *
//**      Fixed address of the A0-addressable segment for core 6,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC6_A1_START                                                       *
//**      Fixed address of the A1-addressable segment for core 6,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC6_A8_START                                                       *
//**      Fixed address of the A8-addressable segment for core 6,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**    __TC6_A9_START                                                       *
//**      Fixed address of the A9-addressable segment for core 6,            *
//**      available when macro __NO_VTC is defined (default: not set).       *
//**                                                                         *
//****************************************************************************

#ifdef __NO_VTC
#define __CORE0 tc0
#else
#define __CORE0 vtc
#endif

#include "tc1v1_8.lsl"
#define GTM_BASE_ADDR           0xf0100000
#define GTM_MCS_COPYTABLE_SPACE __CORE0:linear
#define GTM_CPU_ENDIANNESS little

#ifdef    __REDEFINE_ON_CHIP_ITEMS
#define GTM_REDEFINE_ON_CHIP_ITEMS
#endif  // __REDEFINE_ON_CHIP_ITEMS

#ifndef GTM_MCS_RAM0_SIZE
# define GTM_MCS_RAM0_SIZE              0x2000
#endif

#include "gtm41_10.lsl"

#define HAS_CSM_MEM_MIRROR

#include "vppu_tc49x.lsl"


#ifndef __VSTACK_XDATA
#define __VSTACK_XDATA 1k
#endif

#include "arch_scr3g.lsl"

#ifndef XC800INIT_FILL
#define XC800INIT_FILL          0
#endif

#define SCR_BASE_ADDR           0xf0240000

#ifndef    __REDEFINE_ON_CHIP_ITEMS

// Specify a multi-core processor environment (mpe)

processor mpe
{
        derivative = tc49x;
}
#endif  // __REDEFINE_ON_CHIP_ITEMS

#ifndef CSA_TC0
#define CSA_TC0         64                      /* number of context blocks tc0 */
#endif
#ifndef CSA_TC1
#define CSA_TC1         64                      /* number of context blocks tc1 */
#endif
#ifndef CSA_TC2
#define CSA_TC2         64                      /* number of context blocks tc2 */
#endif
#ifndef CSA_TC3
#define CSA_TC3         64                      /* number of context blocks tc3 */
#endif
#ifndef CSA_TC4
#define CSA_TC4         64                      /* number of context blocks tc4 */
#endif
#ifndef CSA_TC5
#define CSA_TC5         64                      /* number of context blocks tc5 */
#endif
#ifndef CSA_TC6
#define CSA_TC6         64                      /* number of context blocks tc6 */
#endif

#ifndef CSA_START_TC0
#define CSA_START_TC0   0xd0004000              /* start address of CSA tc0 */
#endif
#ifndef CSA_START_TC1
#define CSA_START_TC1   0xd0004000              /* start address of CSA tc1 */
#endif
#ifndef CSA_START_TC2
#define CSA_START_TC2   0xd0004000              /* start address of CSA tc2 */
#endif
#ifndef CSA_START_TC3
#define CSA_START_TC3   0xd0004000              /* start address of CSA tc3 */
#endif
#ifndef CSA_START_TC4
#define CSA_START_TC4   0xd0004000              /* start address of CSA tc4 */
#endif
#ifndef CSA_START_TC5
#define CSA_START_TC5   0xd0004000              /* start address of CSA tc5 */
#endif
#ifndef CSA_START_TC6
#define CSA_START_TC6   0xd0004000              /* start address of CSA tc6 */
#endif

#ifndef USTACK_TC0
#define USTACK_TC0      16k                     /* user stack size tc0 */
#endif

#ifndef USTACK_TC1
#define USTACK_TC1      1k                      /* user stack size tc1 */
#endif
#ifndef USTACK_TC2
#define USTACK_TC2      1k                      /* user stack size tc2 */
#endif
#ifndef USTACK_TC3
#define USTACK_TC3      1k                      /* user stack size tc3 */
#endif
#ifndef USTACK_TC4
#define USTACK_TC4      1k                      /* user stack size tc4 */
#endif
#ifndef USTACK_TC5
#define USTACK_TC5      1k                      /* user stack size tc5 */
#endif
#ifndef USTACK_TC6
#define USTACK_TC6      1k                      /* user stack size tc6 */
#endif

#ifndef ISTACK_TC0
#define ISTACK_TC0      1k                      /* interrupt stack size tc0 */
#endif
#ifndef ISTACK_TC1
#define ISTACK_TC1      1k                      /* interrupt stack size tc1 */
#endif
#ifndef ISTACK_TC2
#define ISTACK_TC2      1k                      /* interrupt stack size tc2 */
#endif
#ifndef ISTACK_TC3
#define ISTACK_TC3      1k                      /* interrupt stack size tc3 */
#endif
#ifndef ISTACK_TC4
#define ISTACK_TC4      1k                      /* interrupt stack size tc4 */
#endif
#ifndef ISTACK_TC5
#define ISTACK_TC5      1k                      /* interrupt stack size tc5 */
#endif
#ifndef ISTACK_TC6
#define ISTACK_TC6      1k                      /* interrupt stack size tc6 */
#endif

#ifndef HEAP
#define HEAP            16k                     /* heap size */
#endif

#ifndef HEAP_TC0
#define HEAP_TC0        16k                     /* heap size tc0 */
#endif
#ifndef HEAP_TC1
#define HEAP_TC1        16k                     /* heap size tc1 */
#endif
#ifndef HEAP_TC2
#define HEAP_TC2        16k                     /* heap size tc2 */
#endif
#ifndef HEAP_TC3
#define HEAP_TC3        16k                     /* heap size tc3 */
#endif
#ifndef HEAP_TC4
#define HEAP_TC4        16k                     /* heap size tc4 */
#endif
#ifndef HEAP_TC5
#define HEAP_TC5        16k                     /* heap size tc5 */
#endif
#ifndef HEAP_TC6
#define HEAP_TC6        16k                     /* heap size tc6 */
#endif

#ifndef INTTAB0_ENTRY_SIZE
#define INTTAB0_ENTRY_SIZE      32
#endif
#if (INTTAB0_ENTRY_SIZE) == 32
#define INTTAB0_VECTOR_PREFIX   ".inttab0.intvec."
#elif (INTTAB0_ENTRY_SIZE) == 8
#define INTTAB0_VECTOR_PREFIX   ".inttab0.intvec8."
#else
#error unsupported vector table entry size for inttab0
#endif
#ifndef INTTAB0VM0_ENTRY_SIZE
#define INTTAB0VM0_ENTRY_SIZE   32
#endif
#if (INTTAB0VM0_ENTRY_SIZE) == 32
#define INTTAB0VM0_VECTOR_PREFIX        ".inttab0.vm0.intvec."
#elif (INTTAB0VM0_ENTRY_SIZE) == 8
#define INTTAB0VM0_VECTOR_PREFIX        ".inttab0.vm0.intvec8."
#else
#error unsupported vector table entry size for inttab0vm0
#endif
#ifndef INTTAB0VM1_ENTRY_SIZE
#define INTTAB0VM1_ENTRY_SIZE   32
#endif
#if (INTTAB0VM1_ENTRY_SIZE) == 32
#define INTTAB0VM1_VECTOR_PREFIX        ".inttab0.vm1.intvec."
#elif (INTTAB0VM1_ENTRY_SIZE) == 8
#define INTTAB0VM1_VECTOR_PREFIX        ".inttab0.vm1.intvec8."
#else
#error unsupported vector table entry size for inttab0vm1
#endif
#ifndef INTTAB0VM2_ENTRY_SIZE
#define INTTAB0VM2_ENTRY_SIZE   32
#endif
#if (INTTAB0VM2_ENTRY_SIZE) == 32
#define INTTAB0VM2_VECTOR_PREFIX        ".inttab0.vm2.intvec."
#elif (INTTAB0VM2_ENTRY_SIZE) == 8
#define INTTAB0VM2_VECTOR_PREFIX        ".inttab0.vm2.intvec8."
#else
#error unsupported vector table entry size for inttab0vm2
#endif
#ifndef INTTAB0VM3_ENTRY_SIZE
#define INTTAB0VM3_ENTRY_SIZE   32
#endif
#if (INTTAB0VM3_ENTRY_SIZE) == 32
#define INTTAB0VM3_VECTOR_PREFIX        ".inttab0.vm3.intvec."
#elif (INTTAB0VM3_ENTRY_SIZE) == 8
#define INTTAB0VM3_VECTOR_PREFIX        ".inttab0.vm3.intvec8."
#else
#error unsupported vector table entry size for inttab0vm3
#endif
#ifndef INTTAB0VM4_ENTRY_SIZE
#define INTTAB0VM4_ENTRY_SIZE   32
#endif
#if (INTTAB0VM4_ENTRY_SIZE) == 32
#define INTTAB0VM4_VECTOR_PREFIX        ".inttab0.vm4.intvec."
#elif (INTTAB0VM4_ENTRY_SIZE) == 8
#define INTTAB0VM4_VECTOR_PREFIX        ".inttab0.vm4.intvec8."
#else
#error unsupported vector table entry size for inttab0vm4
#endif
#ifndef INTTAB0VM5_ENTRY_SIZE
#define INTTAB0VM5_ENTRY_SIZE   32
#endif
#if (INTTAB0VM5_ENTRY_SIZE) == 32
#define INTTAB0VM5_VECTOR_PREFIX        ".inttab0.vm5.intvec."
#elif (INTTAB0VM5_ENTRY_SIZE) == 8
#define INTTAB0VM5_VECTOR_PREFIX        ".inttab0.vm5.intvec8."
#else
#error unsupported vector table entry size for inttab0vm5
#endif
#ifndef INTTAB0VM6_ENTRY_SIZE
#define INTTAB0VM6_ENTRY_SIZE   32
#endif
#if (INTTAB0VM6_ENTRY_SIZE) == 32
#define INTTAB0VM6_VECTOR_PREFIX        ".inttab0.vm6.intvec."
#elif (INTTAB0VM6_ENTRY_SIZE) == 8
#define INTTAB0VM6_VECTOR_PREFIX        ".inttab0.vm6.intvec8."
#else
#error unsupported vector table entry size for inttab0vm6
#endif
#ifndef INTTAB0VM7_ENTRY_SIZE
#define INTTAB0VM7_ENTRY_SIZE   32
#endif
#if (INTTAB0VM7_ENTRY_SIZE) == 32
#define INTTAB0VM7_VECTOR_PREFIX        ".inttab0.vm7.intvec."
#elif (INTTAB0VM7_ENTRY_SIZE) == 8
#define INTTAB0VM7_VECTOR_PREFIX        ".inttab0.vm7.intvec8."
#else
#error unsupported vector table entry size for inttab0vm7
#endif

#ifndef INTTAB1_ENTRY_SIZE
#define INTTAB1_ENTRY_SIZE      32
#endif
#if (INTTAB1_ENTRY_SIZE) == 32
#define INTTAB1_VECTOR_PREFIX   ".inttab1.intvec."
#elif (INTTAB1_ENTRY_SIZE) == 8
#define INTTAB1_VECTOR_PREFIX   ".inttab1.intvec8."
#else
#error unsupported vector table entry size for inttab1
#endif
#ifndef INTTAB1VM0_ENTRY_SIZE
#define INTTAB1VM0_ENTRY_SIZE   32
#endif
#if (INTTAB1VM0_ENTRY_SIZE) == 32
#define INTTAB1VM0_VECTOR_PREFIX        ".inttab1.vm0.intvec."
#elif (INTTAB1VM0_ENTRY_SIZE) == 8
#define INTTAB1VM0_VECTOR_PREFIX        ".inttab1.vm0.intvec8."
#else
#error unsupported vector table entry size for inttab1vm0
#endif
#ifndef INTTAB1VM1_ENTRY_SIZE
#define INTTAB1VM1_ENTRY_SIZE   32
#endif
#if (INTTAB1VM1_ENTRY_SIZE) == 32
#define INTTAB1VM1_VECTOR_PREFIX        ".inttab1.vm1.intvec."
#elif (INTTAB1VM1_ENTRY_SIZE) == 8
#define INTTAB1VM1_VECTOR_PREFIX        ".inttab1.vm1.intvec8."
#else
#error unsupported vector table entry size for inttab1vm1
#endif
#ifndef INTTAB1VM2_ENTRY_SIZE
#define INTTAB1VM2_ENTRY_SIZE   32
#endif
#if (INTTAB1VM2_ENTRY_SIZE) == 32
#define INTTAB1VM2_VECTOR_PREFIX        ".inttab1.vm2.intvec."
#elif (INTTAB1VM2_ENTRY_SIZE) == 8
#define INTTAB1VM2_VECTOR_PREFIX        ".inttab1.vm2.intvec8."
#else
#error unsupported vector table entry size for inttab1vm2
#endif
#ifndef INTTAB1VM3_ENTRY_SIZE
#define INTTAB1VM3_ENTRY_SIZE   32
#endif
#if (INTTAB1VM3_ENTRY_SIZE) == 32
#define INTTAB1VM3_VECTOR_PREFIX        ".inttab1.vm3.intvec."
#elif (INTTAB1VM3_ENTRY_SIZE) == 8
#define INTTAB1VM3_VECTOR_PREFIX        ".inttab1.vm3.intvec8."
#else
#error unsupported vector table entry size for inttab1vm3
#endif
#ifndef INTTAB1VM4_ENTRY_SIZE
#define INTTAB1VM4_ENTRY_SIZE   32
#endif
#if (INTTAB1VM4_ENTRY_SIZE) == 32
#define INTTAB1VM4_VECTOR_PREFIX        ".inttab1.vm4.intvec."
#elif (INTTAB1VM4_ENTRY_SIZE) == 8
#define INTTAB1VM4_VECTOR_PREFIX        ".inttab1.vm4.intvec8."
#else
#error unsupported vector table entry size for inttab1vm4
#endif
#ifndef INTTAB1VM5_ENTRY_SIZE
#define INTTAB1VM5_ENTRY_SIZE   32
#endif
#if (INTTAB1VM5_ENTRY_SIZE) == 32
#define INTTAB1VM5_VECTOR_PREFIX        ".inttab1.vm5.intvec."
#elif (INTTAB1VM5_ENTRY_SIZE) == 8
#define INTTAB1VM5_VECTOR_PREFIX        ".inttab1.vm5.intvec8."
#else
#error unsupported vector table entry size for inttab1vm5
#endif
#ifndef INTTAB1VM6_ENTRY_SIZE
#define INTTAB1VM6_ENTRY_SIZE   32
#endif
#if (INTTAB1VM6_ENTRY_SIZE) == 32
#define INTTAB1VM6_VECTOR_PREFIX        ".inttab1.vm6.intvec."
#elif (INTTAB1VM6_ENTRY_SIZE) == 8
#define INTTAB1VM6_VECTOR_PREFIX        ".inttab1.vm6.intvec8."
#else
#error unsupported vector table entry size for inttab1vm6
#endif
#ifndef INTTAB1VM7_ENTRY_SIZE
#define INTTAB1VM7_ENTRY_SIZE   32
#endif
#if (INTTAB1VM7_ENTRY_SIZE) == 32
#define INTTAB1VM7_VECTOR_PREFIX        ".inttab1.vm7.intvec."
#elif (INTTAB1VM7_ENTRY_SIZE) == 8
#define INTTAB1VM7_VECTOR_PREFIX        ".inttab1.vm7.intvec8."
#else
#error unsupported vector table entry size for inttab1vm7
#endif

#ifndef INTTAB2_ENTRY_SIZE
#define INTTAB2_ENTRY_SIZE      32
#endif
#if (INTTAB2_ENTRY_SIZE) == 32
#define INTTAB2_VECTOR_PREFIX   ".inttab2.intvec."
#elif (INTTAB2_ENTRY_SIZE) == 8
#define INTTAB2_VECTOR_PREFIX   ".inttab2.intvec8."
#else
#error unsupported vector table entry size for inttab2
#endif
#ifndef INTTAB2VM0_ENTRY_SIZE
#define INTTAB2VM0_ENTRY_SIZE   32
#endif
#if (INTTAB2VM0_ENTRY_SIZE) == 32
#define INTTAB2VM0_VECTOR_PREFIX        ".inttab2.vm0.intvec."
#elif (INTTAB2VM0_ENTRY_SIZE) == 8
#define INTTAB2VM0_VECTOR_PREFIX        ".inttab2.vm0.intvec8."
#else
#error unsupported vector table entry size for inttab2vm0
#endif
#ifndef INTTAB2VM1_ENTRY_SIZE
#define INTTAB2VM1_ENTRY_SIZE   32
#endif
#if (INTTAB2VM1_ENTRY_SIZE) == 32
#define INTTAB2VM1_VECTOR_PREFIX        ".inttab2.vm1.intvec."
#elif (INTTAB2VM1_ENTRY_SIZE) == 8
#define INTTAB2VM1_VECTOR_PREFIX        ".inttab2.vm1.intvec8."
#else
#error unsupported vector table entry size for inttab2vm1
#endif
#ifndef INTTAB2VM2_ENTRY_SIZE
#define INTTAB2VM2_ENTRY_SIZE   32
#endif
#if (INTTAB2VM2_ENTRY_SIZE) == 32
#define INTTAB2VM2_VECTOR_PREFIX        ".inttab2.vm2.intvec."
#elif (INTTAB2VM2_ENTRY_SIZE) == 8
#define INTTAB2VM2_VECTOR_PREFIX        ".inttab2.vm2.intvec8."
#else
#error unsupported vector table entry size for inttab2vm2
#endif
#ifndef INTTAB2VM3_ENTRY_SIZE
#define INTTAB2VM3_ENTRY_SIZE   32
#endif
#if (INTTAB2VM3_ENTRY_SIZE) == 32
#define INTTAB2VM3_VECTOR_PREFIX        ".inttab2.vm3.intvec."
#elif (INTTAB2VM3_ENTRY_SIZE) == 8
#define INTTAB2VM3_VECTOR_PREFIX        ".inttab2.vm3.intvec8."
#else
#error unsupported vector table entry size for inttab2vm3
#endif
#ifndef INTTAB2VM4_ENTRY_SIZE
#define INTTAB2VM4_ENTRY_SIZE   32
#endif
#if (INTTAB2VM4_ENTRY_SIZE) == 32
#define INTTAB2VM4_VECTOR_PREFIX        ".inttab2.vm4.intvec."
#elif (INTTAB2VM4_ENTRY_SIZE) == 8
#define INTTAB2VM4_VECTOR_PREFIX        ".inttab2.vm4.intvec8."
#else
#error unsupported vector table entry size for inttab2vm4
#endif
#ifndef INTTAB2VM5_ENTRY_SIZE
#define INTTAB2VM5_ENTRY_SIZE   32
#endif
#if (INTTAB2VM5_ENTRY_SIZE) == 32
#define INTTAB2VM5_VECTOR_PREFIX        ".inttab2.vm5.intvec."
#elif (INTTAB2VM5_ENTRY_SIZE) == 8
#define INTTAB2VM5_VECTOR_PREFIX        ".inttab2.vm5.intvec8."
#else
#error unsupported vector table entry size for inttab2vm5
#endif
#ifndef INTTAB2VM6_ENTRY_SIZE
#define INTTAB2VM6_ENTRY_SIZE   32
#endif
#if (INTTAB2VM6_ENTRY_SIZE) == 32
#define INTTAB2VM6_VECTOR_PREFIX        ".inttab2.vm6.intvec."
#elif (INTTAB2VM6_ENTRY_SIZE) == 8
#define INTTAB2VM6_VECTOR_PREFIX        ".inttab2.vm6.intvec8."
#else
#error unsupported vector table entry size for inttab2vm6
#endif
#ifndef INTTAB2VM7_ENTRY_SIZE
#define INTTAB2VM7_ENTRY_SIZE   32
#endif
#if (INTTAB2VM7_ENTRY_SIZE) == 32
#define INTTAB2VM7_VECTOR_PREFIX        ".inttab2.vm7.intvec."
#elif (INTTAB2VM7_ENTRY_SIZE) == 8
#define INTTAB2VM7_VECTOR_PREFIX        ".inttab2.vm7.intvec8."
#else
#error unsupported vector table entry size for inttab2vm7
#endif

#ifndef INTTAB3_ENTRY_SIZE
#define INTTAB3_ENTRY_SIZE      32
#endif
#if (INTTAB3_ENTRY_SIZE) == 32
#define INTTAB3_VECTOR_PREFIX   ".inttab3.intvec."
#elif (INTTAB3_ENTRY_SIZE) == 8
#define INTTAB3_VECTOR_PREFIX   ".inttab3.intvec8."
#else
#error unsupported vector table entry size for inttab3
#endif
#ifndef INTTAB3VM0_ENTRY_SIZE
#define INTTAB3VM0_ENTRY_SIZE   32
#endif
#if (INTTAB3VM0_ENTRY_SIZE) == 32
#define INTTAB3VM0_VECTOR_PREFIX        ".inttab3.vm0.intvec."
#elif (INTTAB3VM0_ENTRY_SIZE) == 8
#define INTTAB3VM0_VECTOR_PREFIX        ".inttab3.vm0.intvec8."
#else
#error unsupported vector table entry size for inttab3vm0
#endif
#ifndef INTTAB3VM1_ENTRY_SIZE
#define INTTAB3VM1_ENTRY_SIZE   32
#endif
#if (INTTAB3VM1_ENTRY_SIZE) == 32
#define INTTAB3VM1_VECTOR_PREFIX        ".inttab3.vm1.intvec."
#elif (INTTAB3VM1_ENTRY_SIZE) == 8
#define INTTAB3VM1_VECTOR_PREFIX        ".inttab3.vm1.intvec8."
#else
#error unsupported vector table entry size for inttab3vm1
#endif
#ifndef INTTAB3VM2_ENTRY_SIZE
#define INTTAB3VM2_ENTRY_SIZE   32
#endif
#if (INTTAB3VM2_ENTRY_SIZE) == 32
#define INTTAB3VM2_VECTOR_PREFIX        ".inttab3.vm2.intvec."
#elif (INTTAB3VM2_ENTRY_SIZE) == 8
#define INTTAB3VM2_VECTOR_PREFIX        ".inttab3.vm2.intvec8."
#else
#error unsupported vector table entry size for inttab3vm2
#endif
#ifndef INTTAB3VM3_ENTRY_SIZE
#define INTTAB3VM3_ENTRY_SIZE   32
#endif
#if (INTTAB3VM3_ENTRY_SIZE) == 32
#define INTTAB3VM3_VECTOR_PREFIX        ".inttab3.vm3.intvec."
#elif (INTTAB3VM3_ENTRY_SIZE) == 8
#define INTTAB3VM3_VECTOR_PREFIX        ".inttab3.vm3.intvec8."
#else
#error unsupported vector table entry size for inttab3vm3
#endif
#ifndef INTTAB3VM4_ENTRY_SIZE
#define INTTAB3VM4_ENTRY_SIZE   32
#endif
#if (INTTAB3VM4_ENTRY_SIZE) == 32
#define INTTAB3VM4_VECTOR_PREFIX        ".inttab3.vm4.intvec."
#elif (INTTAB3VM4_ENTRY_SIZE) == 8
#define INTTAB3VM4_VECTOR_PREFIX        ".inttab3.vm4.intvec8."
#else
#error unsupported vector table entry size for inttab3vm4
#endif
#ifndef INTTAB3VM5_ENTRY_SIZE
#define INTTAB3VM5_ENTRY_SIZE   32
#endif
#if (INTTAB3VM5_ENTRY_SIZE) == 32
#define INTTAB3VM5_VECTOR_PREFIX        ".inttab3.vm5.intvec."
#elif (INTTAB3VM5_ENTRY_SIZE) == 8
#define INTTAB3VM5_VECTOR_PREFIX        ".inttab3.vm5.intvec8."
#else
#error unsupported vector table entry size for inttab3vm5
#endif
#ifndef INTTAB3VM6_ENTRY_SIZE
#define INTTAB3VM6_ENTRY_SIZE   32
#endif
#if (INTTAB3VM6_ENTRY_SIZE) == 32
#define INTTAB3VM6_VECTOR_PREFIX        ".inttab3.vm6.intvec."
#elif (INTTAB3VM6_ENTRY_SIZE) == 8
#define INTTAB3VM6_VECTOR_PREFIX        ".inttab3.vm6.intvec8."
#else
#error unsupported vector table entry size for inttab3vm6
#endif
#ifndef INTTAB3VM7_ENTRY_SIZE
#define INTTAB3VM7_ENTRY_SIZE   32
#endif
#if (INTTAB3VM7_ENTRY_SIZE) == 32
#define INTTAB3VM7_VECTOR_PREFIX        ".inttab3.vm7.intvec."
#elif (INTTAB3VM7_ENTRY_SIZE) == 8
#define INTTAB3VM7_VECTOR_PREFIX        ".inttab3.vm7.intvec8."
#else
#error unsupported vector table entry size for inttab3vm7
#endif

#ifndef INTTAB4_ENTRY_SIZE
#define INTTAB4_ENTRY_SIZE      32
#endif
#if (INTTAB4_ENTRY_SIZE) == 32
#define INTTAB4_VECTOR_PREFIX   ".inttab4.intvec."
#elif (INTTAB4_ENTRY_SIZE) == 8
#define INTTAB4_VECTOR_PREFIX   ".inttab4.intvec8."
#else
#error unsupported vector table entry size for inttab4
#endif
#ifndef INTTAB4VM0_ENTRY_SIZE
#define INTTAB4VM0_ENTRY_SIZE   32
#endif
#if (INTTAB4VM0_ENTRY_SIZE) == 32
#define INTTAB4VM0_VECTOR_PREFIX        ".inttab4.vm0.intvec."
#elif (INTTAB4VM0_ENTRY_SIZE) == 8
#define INTTAB4VM0_VECTOR_PREFIX        ".inttab4.vm0.intvec8."
#else
#error unsupported vector table entry size for inttab4vm0
#endif
#ifndef INTTAB4VM1_ENTRY_SIZE
#define INTTAB4VM1_ENTRY_SIZE   32
#endif
#if (INTTAB4VM1_ENTRY_SIZE) == 32
#define INTTAB4VM1_VECTOR_PREFIX        ".inttab4.vm1.intvec."
#elif (INTTAB4VM1_ENTRY_SIZE) == 8
#define INTTAB4VM1_VECTOR_PREFIX        ".inttab4.vm1.intvec8."
#else
#error unsupported vector table entry size for inttab4vm1
#endif
#ifndef INTTAB4VM2_ENTRY_SIZE
#define INTTAB4VM2_ENTRY_SIZE   32
#endif
#if (INTTAB4VM2_ENTRY_SIZE) == 32
#define INTTAB4VM2_VECTOR_PREFIX        ".inttab4.vm2.intvec."
#elif (INTTAB4VM2_ENTRY_SIZE) == 8
#define INTTAB4VM2_VECTOR_PREFIX        ".inttab4.vm2.intvec8."
#else
#error unsupported vector table entry size for inttab4vm2
#endif
#ifndef INTTAB4VM3_ENTRY_SIZE
#define INTTAB4VM3_ENTRY_SIZE   32
#endif
#if (INTTAB4VM3_ENTRY_SIZE) == 32
#define INTTAB4VM3_VECTOR_PREFIX        ".inttab4.vm3.intvec."
#elif (INTTAB4VM3_ENTRY_SIZE) == 8
#define INTTAB4VM3_VECTOR_PREFIX        ".inttab4.vm3.intvec8."
#else
#error unsupported vector table entry size for inttab4vm3
#endif
#ifndef INTTAB4VM4_ENTRY_SIZE
#define INTTAB4VM4_ENTRY_SIZE   32
#endif
#if (INTTAB4VM4_ENTRY_SIZE) == 32
#define INTTAB4VM4_VECTOR_PREFIX        ".inttab4.vm4.intvec."
#elif (INTTAB4VM4_ENTRY_SIZE) == 8
#define INTTAB4VM4_VECTOR_PREFIX        ".inttab4.vm4.intvec8."
#else
#error unsupported vector table entry size for inttab4vm4
#endif
#ifndef INTTAB4VM5_ENTRY_SIZE
#define INTTAB4VM5_ENTRY_SIZE   32
#endif
#if (INTTAB4VM5_ENTRY_SIZE) == 32
#define INTTAB4VM5_VECTOR_PREFIX        ".inttab4.vm5.intvec."
#elif (INTTAB4VM5_ENTRY_SIZE) == 8
#define INTTAB4VM5_VECTOR_PREFIX        ".inttab4.vm5.intvec8."
#else
#error unsupported vector table entry size for inttab4vm5
#endif
#ifndef INTTAB4VM6_ENTRY_SIZE
#define INTTAB4VM6_ENTRY_SIZE   32
#endif
#if (INTTAB4VM6_ENTRY_SIZE) == 32
#define INTTAB4VM6_VECTOR_PREFIX        ".inttab4.vm6.intvec."
#elif (INTTAB4VM6_ENTRY_SIZE) == 8
#define INTTAB4VM6_VECTOR_PREFIX        ".inttab4.vm6.intvec8."
#else
#error unsupported vector table entry size for inttab4vm6
#endif
#ifndef INTTAB4VM7_ENTRY_SIZE
#define INTTAB4VM7_ENTRY_SIZE   32
#endif
#if (INTTAB4VM7_ENTRY_SIZE) == 32
#define INTTAB4VM7_VECTOR_PREFIX        ".inttab4.vm7.intvec."
#elif (INTTAB4VM7_ENTRY_SIZE) == 8
#define INTTAB4VM7_VECTOR_PREFIX        ".inttab4.vm7.intvec8."
#else
#error unsupported vector table entry size for inttab4vm7
#endif

#ifndef INTTAB5_ENTRY_SIZE
#define INTTAB5_ENTRY_SIZE      32
#endif
#if (INTTAB5_ENTRY_SIZE) == 32
#define INTTAB5_VECTOR_PREFIX   ".inttab5.intvec."
#elif (INTTAB5_ENTRY_SIZE) == 8
#define INTTAB5_VECTOR_PREFIX   ".inttab5.intvec8."
#else
#error unsupported vector table entry size for inttab5
#endif
#ifndef INTTAB5VM0_ENTRY_SIZE
#define INTTAB5VM0_ENTRY_SIZE   32
#endif
#if (INTTAB5VM0_ENTRY_SIZE) == 32
#define INTTAB5VM0_VECTOR_PREFIX        ".inttab5.vm0.intvec."
#elif (INTTAB5VM0_ENTRY_SIZE) == 8
#define INTTAB5VM0_VECTOR_PREFIX        ".inttab5.vm0.intvec8."
#else
#error unsupported vector table entry size for inttab5vm0
#endif
#ifndef INTTAB5VM1_ENTRY_SIZE
#define INTTAB5VM1_ENTRY_SIZE   32
#endif
#if (INTTAB5VM1_ENTRY_SIZE) == 32
#define INTTAB5VM1_VECTOR_PREFIX        ".inttab5.vm1.intvec."
#elif (INTTAB5VM1_ENTRY_SIZE) == 8
#define INTTAB5VM1_VECTOR_PREFIX        ".inttab5.vm1.intvec8."
#else
#error unsupported vector table entry size for inttab5vm1
#endif
#ifndef INTTAB5VM2_ENTRY_SIZE
#define INTTAB5VM2_ENTRY_SIZE   32
#endif
#if (INTTAB5VM2_ENTRY_SIZE) == 32
#define INTTAB5VM2_VECTOR_PREFIX        ".inttab5.vm2.intvec."
#elif (INTTAB5VM2_ENTRY_SIZE) == 8
#define INTTAB5VM2_VECTOR_PREFIX        ".inttab5.vm2.intvec8."
#else
#error unsupported vector table entry size for inttab5vm2
#endif
#ifndef INTTAB5VM3_ENTRY_SIZE
#define INTTAB5VM3_ENTRY_SIZE   32
#endif
#if (INTTAB5VM3_ENTRY_SIZE) == 32
#define INTTAB5VM3_VECTOR_PREFIX        ".inttab5.vm3.intvec."
#elif (INTTAB5VM3_ENTRY_SIZE) == 8
#define INTTAB5VM3_VECTOR_PREFIX        ".inttab5.vm3.intvec8."
#else
#error unsupported vector table entry size for inttab5vm3
#endif
#ifndef INTTAB5VM4_ENTRY_SIZE
#define INTTAB5VM4_ENTRY_SIZE   32
#endif
#if (INTTAB5VM4_ENTRY_SIZE) == 32
#define INTTAB5VM4_VECTOR_PREFIX        ".inttab5.vm4.intvec."
#elif (INTTAB5VM4_ENTRY_SIZE) == 8
#define INTTAB5VM4_VECTOR_PREFIX        ".inttab5.vm4.intvec8."
#else
#error unsupported vector table entry size for inttab5vm4
#endif
#ifndef INTTAB5VM5_ENTRY_SIZE
#define INTTAB5VM5_ENTRY_SIZE   32
#endif
#if (INTTAB5VM5_ENTRY_SIZE) == 32
#define INTTAB5VM5_VECTOR_PREFIX        ".inttab5.vm5.intvec."
#elif (INTTAB5VM5_ENTRY_SIZE) == 8
#define INTTAB5VM5_VECTOR_PREFIX        ".inttab5.vm5.intvec8."
#else
#error unsupported vector table entry size for inttab5vm5
#endif
#ifndef INTTAB5VM6_ENTRY_SIZE
#define INTTAB5VM6_ENTRY_SIZE   32
#endif
#if (INTTAB5VM6_ENTRY_SIZE) == 32
#define INTTAB5VM6_VECTOR_PREFIX        ".inttab5.vm6.intvec."
#elif (INTTAB5VM6_ENTRY_SIZE) == 8
#define INTTAB5VM6_VECTOR_PREFIX        ".inttab5.vm6.intvec8."
#else
#error unsupported vector table entry size for inttab5vm6
#endif
#ifndef INTTAB5VM7_ENTRY_SIZE
#define INTTAB5VM7_ENTRY_SIZE   32
#endif
#if (INTTAB5VM7_ENTRY_SIZE) == 32
#define INTTAB5VM7_VECTOR_PREFIX        ".inttab5.vm7.intvec."
#elif (INTTAB5VM7_ENTRY_SIZE) == 8
#define INTTAB5VM7_VECTOR_PREFIX        ".inttab5.vm7.intvec8."
#else
#error unsupported vector table entry size for inttab5vm7
#endif

#ifndef INTTAB6_ENTRY_SIZE
#define INTTAB6_ENTRY_SIZE      32
#endif
#if (INTTAB6_ENTRY_SIZE) == 32
#define INTTAB6_VECTOR_PREFIX   ".inttab6.intvec."
#elif (INTTAB6_ENTRY_SIZE) == 8
#define INTTAB6_VECTOR_PREFIX   ".inttab6.intvec8."
#else
#error unsupported vector table entry size for inttab6
#endif
#ifndef INTTAB6VM0_ENTRY_SIZE
#define INTTAB6VM0_ENTRY_SIZE   32
#endif
#if (INTTAB6VM0_ENTRY_SIZE) == 32
#define INTTAB6VM0_VECTOR_PREFIX        ".inttab6.vm0.intvec."
#elif (INTTAB6VM0_ENTRY_SIZE) == 8
#define INTTAB6VM0_VECTOR_PREFIX        ".inttab6.vm0.intvec8."
#else
#error unsupported vector table entry size for inttab6vm0
#endif
#ifndef INTTAB6VM1_ENTRY_SIZE
#define INTTAB6VM1_ENTRY_SIZE   32
#endif
#if (INTTAB6VM1_ENTRY_SIZE) == 32
#define INTTAB6VM1_VECTOR_PREFIX        ".inttab6.vm1.intvec."
#elif (INTTAB6VM1_ENTRY_SIZE) == 8
#define INTTAB6VM1_VECTOR_PREFIX        ".inttab6.vm1.intvec8."
#else
#error unsupported vector table entry size for inttab6vm1
#endif
#ifndef INTTAB6VM2_ENTRY_SIZE
#define INTTAB6VM2_ENTRY_SIZE   32
#endif
#if (INTTAB6VM2_ENTRY_SIZE) == 32
#define INTTAB6VM2_VECTOR_PREFIX        ".inttab6.vm2.intvec."
#elif (INTTAB6VM2_ENTRY_SIZE) == 8
#define INTTAB6VM2_VECTOR_PREFIX        ".inttab6.vm2.intvec8."
#else
#error unsupported vector table entry size for inttab6vm2
#endif
#ifndef INTTAB6VM3_ENTRY_SIZE
#define INTTAB6VM3_ENTRY_SIZE   32
#endif
#if (INTTAB6VM3_ENTRY_SIZE) == 32
#define INTTAB6VM3_VECTOR_PREFIX        ".inttab6.vm3.intvec."
#elif (INTTAB6VM3_ENTRY_SIZE) == 8
#define INTTAB6VM3_VECTOR_PREFIX        ".inttab6.vm3.intvec8."
#else
#error unsupported vector table entry size for inttab6vm3
#endif
#ifndef INTTAB6VM4_ENTRY_SIZE
#define INTTAB6VM4_ENTRY_SIZE   32
#endif
#if (INTTAB6VM4_ENTRY_SIZE) == 32
#define INTTAB6VM4_VECTOR_PREFIX        ".inttab6.vm4.intvec."
#elif (INTTAB6VM4_ENTRY_SIZE) == 8
#define INTTAB6VM4_VECTOR_PREFIX        ".inttab6.vm4.intvec8."
#else
#error unsupported vector table entry size for inttab6vm4
#endif
#ifndef INTTAB6VM5_ENTRY_SIZE
#define INTTAB6VM5_ENTRY_SIZE   32
#endif
#if (INTTAB6VM5_ENTRY_SIZE) == 32
#define INTTAB6VM5_VECTOR_PREFIX        ".inttab6.vm5.intvec."
#elif (INTTAB6VM5_ENTRY_SIZE) == 8
#define INTTAB6VM5_VECTOR_PREFIX        ".inttab6.vm5.intvec8."
#else
#error unsupported vector table entry size for inttab6vm5
#endif
#ifndef INTTAB6VM6_ENTRY_SIZE
#define INTTAB6VM6_ENTRY_SIZE   32
#endif
#if (INTTAB6VM6_ENTRY_SIZE) == 32
#define INTTAB6VM6_VECTOR_PREFIX        ".inttab6.vm6.intvec."
#elif (INTTAB6VM6_ENTRY_SIZE) == 8
#define INTTAB6VM6_VECTOR_PREFIX        ".inttab6.vm6.intvec8."
#else
#error unsupported vector table entry size for inttab6vm6
#endif
#ifndef INTTAB6VM7_ENTRY_SIZE
#define INTTAB6VM7_ENTRY_SIZE   32
#endif
#if (INTTAB6VM7_ENTRY_SIZE) == 32
#define INTTAB6VM7_VECTOR_PREFIX        ".inttab6.vm7.intvec."
#elif (INTTAB6VM7_ENTRY_SIZE) == 8
#define INTTAB6VM7_VECTOR_PREFIX        ".inttab6.vm7.intvec8."
#else
#error unsupported vector table entry size for inttab6vm7
#endif

/*
 * Define __MAX_CONCURRENT_HANDLERS to the number of interrupt and trap handlers
 * that should contribute to stack size estimation. The handlers that use the
 * most stack space are selected. Allowed values: 0 or higher. Default: not set.
 * Defining this macro adds a threads attribute to ustack, with the macro value
 * increased by one.
 */
#ifndef __MAX_CONCURRENT_HANDLERS
#define __USTACK0_THREADS_ATTRIBUTE
#define __USTACK1_THREADS_ATTRIBUTE
#define __USTACK2_THREADS_ATTRIBUTE
#define __USTACK3_THREADS_ATTRIBUTE
#define __USTACK4_THREADS_ATTRIBUTE
#define __USTACK5_THREADS_ATTRIBUTE
#define __USTACK6_THREADS_ATTRIBUTE
#else
#define __USTACK0_THREADS_ATTRIBUTE ,threads=(__MAX_CONCURRENT_HANDLERS)+1
#define __USTACK1_THREADS_ATTRIBUTE ,threads=(__MAX_CONCURRENT_HANDLERS)+1
#define __USTACK2_THREADS_ATTRIBUTE ,threads=(__MAX_CONCURRENT_HANDLERS)+1
#define __USTACK3_THREADS_ATTRIBUTE ,threads=(__MAX_CONCURRENT_HANDLERS)+1
#define __USTACK4_THREADS_ATTRIBUTE ,threads=(__MAX_CONCURRENT_HANDLERS)+1
#define __USTACK5_THREADS_ATTRIBUTE ,threads=(__MAX_CONCURRENT_HANDLERS)+1
#define __USTACK6_THREADS_ATTRIBUTE ,threads=(__MAX_CONCURRENT_HANDLERS)+1
#endif

/*
 * If a separate program is run on a specific core <n>, then the stack usage of this program
 * can be computed separately by defining LSL macro __USTACK<n>_ENTRY_POINTS to the name of the
 * symbol (between double quotes) that represents the main function for this program. Multiple
 * symbols can be specified by listing them between brackets, separated by commas.
 */
#ifdef __USTACK0_ENTRY_POINTS
#define __USTACK0_ENTRY_POINTS_ATTRIBUTE ,entry_points=__USTACK0_ENTRY_POINTS
#else
#ifdef __NO_VTC
#define __USTACK0_ENTRY_POINTS_ATTRIBUTE ,entry_points=["_START" /* , group trap_tab, group int_tab */ ]
#else
#define __USTACK0_ENTRY_POINTS_ATTRIBUTE ,entry_points=["_START" /* , group trap_tab_tc0, group int_tab_tc0 */ ]
#endif
#endif
#ifdef __ISTACK0_ENTRY_POINTS
#define __ISTACK0_ENTRY_POINTS_ATTRIBUTE ,entry_points=__ISTACK0_ENTRY_POINTS
#else
#define __ISTACK0_ENTRY_POINTS_ATTRIBUTE
#endif
#ifdef __USTACK1_ENTRY_POINTS
#define __USTACK1_ENTRY_POINTS_ATTRIBUTE ,entry_points=__USTACK1_ENTRY_POINTS
#else
#ifdef __NO_VTC
#define __USTACK1_ENTRY_POINTS_ATTRIBUTE ,entry_points=["__start_tc1_no_vtc" /* , group trap_tab, group int_tab */ ]
#else
#define __USTACK1_ENTRY_POINTS_ATTRIBUTE ,entry_points=["_start_tc1" /* , group trap_tab_tc1, group int_tab_tc1 */ ]
#endif
#endif
#ifdef __ISTACK1_ENTRY_POINTS
#define __ISTACK1_ENTRY_POINTS_ATTRIBUTE ,entry_points=__ISTACK1_ENTRY_POINTS
#else
#define __ISTACK1_ENTRY_POINTS_ATTRIBUTE
#endif
#ifdef __USTACK2_ENTRY_POINTS
#define __USTACK2_ENTRY_POINTS_ATTRIBUTE ,entry_points=__USTACK2_ENTRY_POINTS
#else
#ifdef __NO_VTC
#define __USTACK2_ENTRY_POINTS_ATTRIBUTE ,entry_points=["__start_tc2_no_vtc" /* , group trap_tab, group int_tab */ ]
#else
#define __USTACK2_ENTRY_POINTS_ATTRIBUTE ,entry_points=["_start_tc2" /* , group trap_tab_tc2, group int_tab_tc2 */ ]
#endif
#endif
#ifdef __ISTACK2_ENTRY_POINTS
#define __ISTACK2_ENTRY_POINTS_ATTRIBUTE ,entry_points=__ISTACK2_ENTRY_POINTS
#else
#define __ISTACK2_ENTRY_POINTS_ATTRIBUTE
#endif
#ifdef __USTACK3_ENTRY_POINTS
#define __USTACK3_ENTRY_POINTS_ATTRIBUTE ,entry_points=__USTACK3_ENTRY_POINTS
#else
#ifdef __NO_VTC
#define __USTACK3_ENTRY_POINTS_ATTRIBUTE ,entry_points=["__start_tc3_no_vtc" /* , group trap_tab, group int_tab */ ]
#else
#define __USTACK3_ENTRY_POINTS_ATTRIBUTE ,entry_points=["_start_tc3" /* , group trap_tab_tc3, group int_tab_tc3 */ ]
#endif
#endif
#ifdef __ISTACK3_ENTRY_POINTS
#define __ISTACK3_ENTRY_POINTS_ATTRIBUTE ,entry_points=__ISTACK3_ENTRY_POINTS
#else
#define __ISTACK3_ENTRY_POINTS_ATTRIBUTE
#endif
#ifdef __USTACK4_ENTRY_POINTS
#define __USTACK4_ENTRY_POINTS_ATTRIBUTE ,entry_points=__USTACK4_ENTRY_POINTS
#else
#ifdef __NO_VTC
#define __USTACK4_ENTRY_POINTS_ATTRIBUTE ,entry_points=["__start_tc4_no_vtc" /* , group trap_tab, group int_tab */ ]
#else
#define __USTACK4_ENTRY_POINTS_ATTRIBUTE ,entry_points=["_start_tc4" /* , group trap_tab_tc4, group int_tab_tc4 */ ]
#endif
#endif
#ifdef __ISTACK4_ENTRY_POINTS
#define __ISTACK4_ENTRY_POINTS_ATTRIBUTE ,entry_points=__ISTACK4_ENTRY_POINTS
#else
#define __ISTACK4_ENTRY_POINTS_ATTRIBUTE
#endif
#ifdef __USTACK5_ENTRY_POINTS
#define __USTACK5_ENTRY_POINTS_ATTRIBUTE ,entry_points=__USTACK5_ENTRY_POINTS
#else
#ifdef __NO_VTC
#define __USTACK5_ENTRY_POINTS_ATTRIBUTE ,entry_points=["__start_tc5_no_vtc" /* , group trap_tab, group int_tab */ ]
#else
#define __USTACK5_ENTRY_POINTS_ATTRIBUTE ,entry_points=["_start_tc5" /* , group trap_tab_tc5, group int_tab_tc5 */ ]
#endif
#endif
#ifdef __ISTACK5_ENTRY_POINTS
#define __ISTACK5_ENTRY_POINTS_ATTRIBUTE ,entry_points=__ISTACK5_ENTRY_POINTS
#else
#define __ISTACK5_ENTRY_POINTS_ATTRIBUTE
#endif
#ifdef __USTACK6_ENTRY_POINTS
#define __USTACK6_ENTRY_POINTS_ATTRIBUTE ,entry_points=__USTACK6_ENTRY_POINTS
#else
#ifdef __NO_VTC
#define __USTACK6_ENTRY_POINTS_ATTRIBUTE ,entry_points=["__start_tc6_no_vtc" /* , group trap_tab, group int_tab */ ]
#else
#define __USTACK6_ENTRY_POINTS_ATTRIBUTE ,entry_points=["_start_tc6" /* , group trap_tab_tc6, group int_tab_tc6 */ ]
#endif
#endif
#ifdef __ISTACK6_ENTRY_POINTS
#define __ISTACK6_ENTRY_POINTS_ATTRIBUTE ,entry_points=__ISTACK6_ENTRY_POINTS
#else
#define __ISTACK6_ENTRY_POINTS_ATTRIBUTE
#endif

#ifndef XVWBUF
#define XVWBUF          0                       /* buffer used by debugger */
#endif

#ifndef INTTAB
#define INTTAB          0xa00f0000              /* start address of interrupt table */
#endif
#ifndef TRAPTAB
#define TRAPTAB         (INTTAB + 0xe000)      /* start address of trap table */
#endif
#ifndef HVTRAPTAB
#define HVTRAPTAB       (TRAPTAB6 + 0x100)
#endif

#ifndef INTTAB0
#define INTTAB0         (INTTAB)
#endif
#ifndef INTTAB0VM0
#define INTTAB0VM0      (HVTRAPTAB6 + 0x100)
#endif
#ifndef INTTAB0VM1
#define INTTAB0VM1 (INTTAB0VM0 + 0x2000)
#endif
#ifndef INTTAB0VM2
#define INTTAB0VM2 (INTTAB0VM1 + 0x2000)
#endif
#ifndef INTTAB0VM3
#define INTTAB0VM3 (INTTAB0VM2 + 0x2000)
#endif
#ifndef INTTAB0VM4
#define INTTAB0VM4 (INTTAB0VM3 + 0x2000)
#endif
#ifndef INTTAB0VM5
#define INTTAB0VM5 (INTTAB0VM4 + 0x2000)
#endif
#ifndef INTTAB0VM6
#define INTTAB0VM6 (INTTAB0VM5 + 0x2000)
#endif
#ifndef INTTAB0VM7
#define INTTAB0VM7 (INTTAB0VM6 + 0x2000)
#endif
#ifndef INTTAB1
#define INTTAB1         (INTTAB0 + 0x2000)
#endif
#ifndef INTTAB1VM0
#define INTTAB1VM0      (INTTAB0VM7 + 0x2000)
#endif
#ifndef INTTAB1VM1
#define INTTAB1VM1 (INTTAB1VM0 + 0x2000)
#endif
#ifndef INTTAB1VM2
#define INTTAB1VM2 (INTTAB1VM1 + 0x2000)
#endif
#ifndef INTTAB1VM3
#define INTTAB1VM3 (INTTAB1VM2 + 0x2000)
#endif
#ifndef INTTAB1VM4
#define INTTAB1VM4 (INTTAB1VM3 + 0x2000)
#endif
#ifndef INTTAB1VM5
#define INTTAB1VM5 (INTTAB1VM4 + 0x2000)
#endif
#ifndef INTTAB1VM6
#define INTTAB1VM6 (INTTAB1VM5 + 0x2000)
#endif
#ifndef INTTAB1VM7
#define INTTAB1VM7 (INTTAB1VM6 + 0x2000)
#endif
#ifndef INTTAB2
#define INTTAB2         (INTTAB1 + 0x2000)
#endif
#ifndef INTTAB2VM0
#define INTTAB2VM0      (INTTAB1VM7 + 0x2000)
#endif
#ifndef INTTAB2VM1
#define INTTAB2VM1 (INTTAB2VM0 + 0x2000)
#endif
#ifndef INTTAB2VM2
#define INTTAB2VM2 (INTTAB2VM1 + 0x2000)
#endif
#ifndef INTTAB2VM3
#define INTTAB2VM3 (INTTAB2VM2 + 0x2000)
#endif
#ifndef INTTAB2VM4
#define INTTAB2VM4 (INTTAB2VM3 + 0x2000)
#endif
#ifndef INTTAB2VM5
#define INTTAB2VM5 (INTTAB2VM4 + 0x2000)
#endif
#ifndef INTTAB2VM6
#define INTTAB2VM6 (INTTAB2VM5 + 0x2000)
#endif
#ifndef INTTAB2VM7
#define INTTAB2VM7 (INTTAB2VM6 + 0x2000)
#endif
#ifndef INTTAB3
#define INTTAB3         (INTTAB2 + 0x2000)
#endif
#ifndef INTTAB3VM0
#define INTTAB3VM0      (INTTAB2VM7 + 0x2000)
#endif
#ifndef INTTAB3VM1
#define INTTAB3VM1 (INTTAB3VM0 + 0x2000)
#endif
#ifndef INTTAB3VM2
#define INTTAB3VM2 (INTTAB3VM1 + 0x2000)
#endif
#ifndef INTTAB3VM3
#define INTTAB3VM3 (INTTAB3VM2 + 0x2000)
#endif
#ifndef INTTAB3VM4
#define INTTAB3VM4 (INTTAB3VM3 + 0x2000)
#endif
#ifndef INTTAB3VM5
#define INTTAB3VM5 (INTTAB3VM4 + 0x2000)
#endif
#ifndef INTTAB3VM6
#define INTTAB3VM6 (INTTAB3VM5 + 0x2000)
#endif
#ifndef INTTAB3VM7
#define INTTAB3VM7 (INTTAB3VM6 + 0x2000)
#endif
#ifndef INTTAB4
#define INTTAB4         (INTTAB3 + 0x2000)
#endif
#ifndef INTTAB4VM0
#define INTTAB4VM0      (INTTAB3VM7 + 0x2000)
#endif
#ifndef INTTAB4VM1
#define INTTAB4VM1 (INTTAB4VM0 + 0x2000)
#endif
#ifndef INTTAB4VM2
#define INTTAB4VM2 (INTTAB4VM1 + 0x2000)
#endif
#ifndef INTTAB4VM3
#define INTTAB4VM3 (INTTAB4VM2 + 0x2000)
#endif
#ifndef INTTAB4VM4
#define INTTAB4VM4 (INTTAB4VM3 + 0x2000)
#endif
#ifndef INTTAB4VM5
#define INTTAB4VM5 (INTTAB4VM4 + 0x2000)
#endif
#ifndef INTTAB4VM6
#define INTTAB4VM6 (INTTAB4VM5 + 0x2000)
#endif
#ifndef INTTAB4VM7
#define INTTAB4VM7 (INTTAB4VM6 + 0x2000)
#endif
#ifndef INTTAB5
#define INTTAB5         (INTTAB4 + 0x2000)
#endif
#ifndef INTTAB5VM0
#define INTTAB5VM0      (INTTAB4VM7 + 0x2000)
#endif
#ifndef INTTAB5VM1
#define INTTAB5VM1 (INTTAB5VM0 + 0x2000)
#endif
#ifndef INTTAB5VM2
#define INTTAB5VM2 (INTTAB5VM1 + 0x2000)
#endif
#ifndef INTTAB5VM3
#define INTTAB5VM3 (INTTAB5VM2 + 0x2000)
#endif
#ifndef INTTAB5VM4
#define INTTAB5VM4 (INTTAB5VM3 + 0x2000)
#endif
#ifndef INTTAB5VM5
#define INTTAB5VM5 (INTTAB5VM4 + 0x2000)
#endif
#ifndef INTTAB5VM6
#define INTTAB5VM6 (INTTAB5VM5 + 0x2000)
#endif
#ifndef INTTAB5VM7
#define INTTAB5VM7 (INTTAB5VM6 + 0x2000)
#endif
#ifndef INTTAB6
#define INTTAB6         (INTTAB5 + 0x2000)
#endif
#ifndef INTTAB6VM0
#define INTTAB6VM0      (INTTAB5VM7 + 0x2000)
#endif
#ifndef INTTAB6VM1
#define INTTAB6VM1 (INTTAB6VM0 + 0x2000)
#endif
#ifndef INTTAB6VM2
#define INTTAB6VM2 (INTTAB6VM1 + 0x2000)
#endif
#ifndef INTTAB6VM3
#define INTTAB6VM3 (INTTAB6VM2 + 0x2000)
#endif
#ifndef INTTAB6VM4
#define INTTAB6VM4 (INTTAB6VM3 + 0x2000)
#endif
#ifndef INTTAB6VM5
#define INTTAB6VM5 (INTTAB6VM4 + 0x2000)
#endif
#ifndef INTTAB6VM6
#define INTTAB6VM6 (INTTAB6VM5 + 0x2000)
#endif
#ifndef INTTAB6VM7
#define INTTAB6VM7 (INTTAB6VM6 + 0x2000)
#endif
#ifndef TRAPTAB0
#define TRAPTAB0        (TRAPTAB)
#endif
#ifndef TRAPTAB1
#define TRAPTAB1        (TRAPTAB0 + 0x100)
#endif
#ifndef TRAPTAB2
#define TRAPTAB2        (TRAPTAB1 + 0x100)
#endif
#ifndef TRAPTAB3
#define TRAPTAB3        (TRAPTAB2 + 0x100)
#endif
#ifndef TRAPTAB4
#define TRAPTAB4        (TRAPTAB3 + 0x100)
#endif
#ifndef TRAPTAB5
#define TRAPTAB5        (TRAPTAB4 + 0x100)
#endif
#ifndef TRAPTAB6
#define TRAPTAB6        (TRAPTAB5 + 0x100)
#endif

#ifndef HVTRAPTAB0
#define HVTRAPTAB0      (HVTRAPTAB)
#endif
#ifndef HVTRAPTAB1
#define HVTRAPTAB1      (HVTRAPTAB0 + 0x100)
#endif
#ifndef HVTRAPTAB2
#define HVTRAPTAB2      (HVTRAPTAB1 + 0x100)
#endif
#ifndef HVTRAPTAB3
#define HVTRAPTAB3      (HVTRAPTAB2 + 0x100)
#endif
#ifndef HVTRAPTAB4
#define HVTRAPTAB4      (HVTRAPTAB3 + 0x100)
#endif
#ifndef HVTRAPTAB5
#define HVTRAPTAB5      (HVTRAPTAB4 + 0x100)
#endif
#ifndef HVTRAPTAB6
#define HVTRAPTAB6      (HVTRAPTAB5 + 0x100)
#endif

#ifndef RESET
#define RESET           0x80000000              /* internal flash start address tc0 */
#endif

// Eclipse generates Ax_START_ADDRESS,
// users may define Ax_START on the command line
#ifndef A0_START_ADDRESS
#ifdef  A0_START
#define A0_START_ADDRESS A0_START
#endif
#endif
#ifndef A1_START_ADDRESS
#ifdef  A1_START
#define A1_START_ADDRESS A1_START
#endif
#endif
#ifndef A8_START_ADDRESS
#ifdef  A8_START
#define A8_START_ADDRESS A8_START
#endif
#endif
#ifndef A9_START_ADDRESS
#ifdef  A9_START
#define A9_START_ADDRESS A9_START
#endif
#endif

#ifndef __NO_VTC
#ifdef A0_START_ADDRESS
#define __A0_START_ADDRESS A0_START_ADDRESS
#endif
#ifdef A1_START_ADDRESS
#define __A1_START_ADDRESS A1_START_ADDRESS
#endif
#ifdef A8_START_ADDRESS
#define __A8_START_ADDRESS A8_START_ADDRESS
#endif
#ifdef A9_START_ADDRESS
#define __A9_START_ADDRESS A9_START_ADDRESS
#endif
#endif



#define VEC_MEM_REGION          0xC0000000
#define VEC_MEM_REGION_SIZE     0x20000
#define CSM_MEM_REGION          0x92080000
#define CSM_MEM_MIRROR          0xB2080000
#define CSM_MEM_REGION_SIZE     0x40000

derivative tc49x
extends gtm41_10

, ppu_tc49x(CSM_MEM_REGION_SIZE, CSM_MEM_REGION, CSM_MEM_MIRROR, VEC_MEM_REGION_SIZE, VEC_MEM_REGION)
{
        core tc0
        {
                architecture = TC1V1.8;
#ifndef __NO_VTC
                space_id_offset = 100;  // add 100 to all space IDs in the architecture definition
                copytable_space = vtc:linear;   // use the copy table in the virtual core for 'bss' and initialized data sections
#endif
        }

        core tc1
        {
                architecture = TC1V1.8;
#ifndef __NO_VTC
                space_id_offset = 200;  // add 200 to all space IDs in the architecture definition
                copytable_space = vtc:linear;   // use the copy table in the virtual core for 'bss' and initialized data sections
#endif
        }

        core tc2
        {
                architecture = TC1V1.8;
#ifndef __NO_VTC
                space_id_offset = 300;  // add 300 to all space IDs in the architecture definition
                copytable_space = vtc:linear;   // use the copy table in the virtual core for 'bss' and initialized data sections
#endif
        }

        core tc3
        {
                architecture = TC1V1.8;
#ifndef __NO_VTC
                space_id_offset = 400;  // add 400 to all space IDs in the architecture definition
                copytable_space = vtc:linear;   // use the copy table in the virtual core for 'bss' and initialized data sections
#endif
        }

        core tc4
        {
                architecture = TC1V1.8;
#ifndef __NO_VTC
                space_id_offset = 500;  // add 500 to all space IDs in the architecture definition
                copytable_space = vtc:linear;   // use the copy table in the virtual core for 'bss' and initialized data sections
#endif
        }

        core tc5
        {
                architecture = TC1V1.8;
#ifndef __NO_VTC
                space_id_offset = 600;  // add 600 to all space IDs in the architecture definition
                copytable_space = vtc:linear;   // use the copy table in the virtual core for 'bss' and initialized data sections
#endif
        }

        core tc6
        {
                architecture = TC1V1.8;
#ifndef __NO_VTC
                space_id_offset = 700;  // add 700 to all space IDs in the architecture definition
                copytable_space = vtc:linear;   // use the copy table in the virtual core for 'bss' and initialized data sections
#endif
        }


#ifndef __NO_VTC
        core vtc
        {
                architecture = TC1V1.8;
                import tc0;                      // add all address spaces of core tc0 to core vtc for linking and locating
                import tc1;                      //                             tc1
                import tc2;                      //                             tc2
                import tc3;                      //                             tc3
                import tc4;                      //                             tc4
                import tc5;                      //                             tc5
                import tc6;                      //                             tc6
        }
#endif

        core xc800
        {
                architecture = scr3g;
        }

        bus sri
        {
                mau = 8;
                width = 32;

                // segments starting from 0x0
                map (dest=bus:tc0:fpi_bus, src_offset=0, dest_offset=0, size=0x10000000);
                map (dest=bus:tc1:fpi_bus, src_offset=0, dest_offset=0, size=0x10000000);
                map (dest=bus:tc2:fpi_bus, src_offset=0, dest_offset=0, size=0x10000000);
                map (dest=bus:tc3:fpi_bus, src_offset=0, dest_offset=0, size=0x10000000);
                map (dest=bus:tc4:fpi_bus, src_offset=0, dest_offset=0, size=0x10000000);
                map (dest=bus:tc5:fpi_bus, src_offset=0, dest_offset=0, size=0x10000000);
                map (dest=bus:tc6:fpi_bus, src_offset=0, dest_offset=0, size=0x10000000);
#ifndef __NO_VTC
                map (dest=bus:vtc:fpi_bus, src_offset=0, dest_offset=0, size=0x80000000);
#endif

                // local memory mapped to individual cores
                map (dest=bus:tc0:fpi_bus, src_offset=0x70000000, dest_offset=0x70000000, size=0x00100000, priority=3, exec_priority=0);
                map (dest=bus:tc0:fpi_bus, src_offset=0x70100000, dest_offset=0x70100000, size=0x0ff00000, exec_priority=3);
                map (dest=bus:tc0:fpi_bus, src_offset=0x10000000, dest_offset=0x10000000, size=0x60000000);
                map (dest=bus:tc1:fpi_bus, src_offset=0x60000000, dest_offset=0x60000000, size=0x00100000, priority=3, exec_priority=0);
                map (dest=bus:tc1:fpi_bus, src_offset=0x60100000, dest_offset=0x60100000, size=0x0ff00000, exec_priority=3);
                map (dest=bus:tc1:fpi_bus, src_offset=0x70000000, dest_offset=0x70000000, size=0x10000000);
                map (dest=bus:tc1:fpi_bus, src_offset=0x10000000, dest_offset=0x10000000, size=0x50000000);
                map (dest=bus:tc2:fpi_bus, src_offset=0x50000000, dest_offset=0x50000000, size=0x00100000, priority=3, exec_priority=0);
                map (dest=bus:tc2:fpi_bus, src_offset=0x50100000, dest_offset=0x50100000, size=0x0ff00000, exec_priority=3);
                map (dest=bus:tc2:fpi_bus, src_offset=0x60000000, dest_offset=0x60000000, size=0x20000000);
                map (dest=bus:tc2:fpi_bus, src_offset=0x10000000, dest_offset=0x10000000, size=0x40000000);
                map (dest=bus:tc3:fpi_bus, src_offset=0x40000000, dest_offset=0x40000000, size=0x00100000, priority=3, exec_priority=0);
                map (dest=bus:tc3:fpi_bus, src_offset=0x40100000, dest_offset=0x40100000, size=0x0ff00000, exec_priority=3);
                map (dest=bus:tc3:fpi_bus, src_offset=0x50000000, dest_offset=0x50000000, size=0x30000000);
                map (dest=bus:tc3:fpi_bus, src_offset=0x10000000, dest_offset=0x10000000, size=0x30000000);
                map (dest=bus:tc4:fpi_bus, src_offset=0x30000000, dest_offset=0x30000000, size=0x00100000, priority=3, exec_priority=0);
                map (dest=bus:tc4:fpi_bus, src_offset=0x30100000, dest_offset=0x30100000, size=0x0ff00000, exec_priority=3);
                map (dest=bus:tc4:fpi_bus, src_offset=0x40000000, dest_offset=0x40000000, size=0x40000000);
                map (dest=bus:tc4:fpi_bus, src_offset=0x10000000, dest_offset=0x10000000, size=0x20000000);
                map (dest=bus:tc5:fpi_bus, src_offset=0x20000000, dest_offset=0x20000000, size=0x00100000, priority=3, exec_priority=0);
                map (dest=bus:tc5:fpi_bus, src_offset=0x20100000, dest_offset=0x20100000, size=0x0ff00000, exec_priority=3);
                map (dest=bus:tc5:fpi_bus, src_offset=0x30000000, dest_offset=0x30000000, size=0x50000000);
                map (dest=bus:tc5:fpi_bus, src_offset=0x10000000, dest_offset=0x10000000, size=0x10000000);
                map (dest=bus:tc6:fpi_bus, src_offset=0x10000000, dest_offset=0x10000000, size=0x00100000, priority=3, exec_priority=0);
                map (dest=bus:tc6:fpi_bus, src_offset=0x10100000, dest_offset=0x10100000, size=0x0ff00000, exec_priority=3);
                map (dest=bus:tc6:fpi_bus, src_offset=0x20000000, dest_offset=0x20000000, size=0x60000000);

                // pflash
                map (dest=bus:tc0:fpi_bus, src_offset=0x80000000, dest_offset=0x80000000, size=0x400000, priority=2);
                map (dest=bus:tc0:fpi_bus, src_offset=0x80400000, dest_offset=0x80400000, size=0xfc00000);
                map (dest=bus:tc1:fpi_bus, src_offset=0x80000000, dest_offset=0x80000000, size=0x400000);
                map (dest=bus:tc1:fpi_bus, src_offset=0x80400000, dest_offset=0x80400000, size=0x400000, priority=2);
                map (dest=bus:tc1:fpi_bus, src_offset=0x80800000, dest_offset=0x80800000, size=0xf800000);
                map (dest=bus:tc2:fpi_bus, src_offset=0x80000000, dest_offset=0x80000000, size=0x800000);
                map (dest=bus:tc2:fpi_bus, src_offset=0x80800000, dest_offset=0x80800000, size=0x400000, priority=2);
                map (dest=bus:tc2:fpi_bus, src_offset=0x80c00000, dest_offset=0x80c00000, size=0xf400000);
                map (dest=bus:tc3:fpi_bus, src_offset=0x80000000, dest_offset=0x80000000, size=0xc00000);
                map (dest=bus:tc3:fpi_bus, src_offset=0x80c00000, dest_offset=0x80c00000, size=0x400000, priority=2);
                map (dest=bus:tc3:fpi_bus, src_offset=0x81000000, dest_offset=0x81000000, size=0xf000000);
                map (dest=bus:tc4:fpi_bus, src_offset=0x80000000, dest_offset=0x80000000, size=0x1000000);
                map (dest=bus:tc4:fpi_bus, src_offset=0x81000000, dest_offset=0x81000000, size=0x400000, priority=2);
                map (dest=bus:tc4:fpi_bus, src_offset=0x81400000, dest_offset=0x81400000, size=0xec00000);
                map (dest=bus:tc5:fpi_bus, src_offset=0x80000000, dest_offset=0x80000000, size=0x1400000);
                map (dest=bus:tc5:fpi_bus, src_offset=0x81400000, dest_offset=0x81400000, size=0x400000, priority=2);
                map (dest=bus:tc5:fpi_bus, src_offset=0x81800000, dest_offset=0x81800000, size=0xe800000);
                map (dest=bus:tc6:fpi_bus, src_offset=0x80000000, dest_offset=0x80000000, size=0x4000000);
                map (dest=bus:tc6:fpi_bus, src_offset=0x84000000, dest_offset=0x84000000, size=0x100000, priority=2);
                map (dest=bus:tc6:fpi_bus, src_offset=0x84100000, dest_offset=0x84100000, size=0xbf00000);
#ifndef __NO_VTC
                map (dest=bus:vtc:fpi_bus, src_offset=0x80000000, dest_offset=0x80000000, size=0x10000000);
#endif // __NO_VTC

                // dlmuram
                map (dest=bus:tc0:fpi_bus, src_offset=0x90000000, dest_offset=0x90000000, size=0x80000, priority=2);
                map (dest=bus:tc0:fpi_bus, src_offset=0x90080000, dest_offset=0x90080000, size=0x1a0000);
                map (dest=bus:tc1:fpi_bus, src_offset=0x90000000, dest_offset=0x90000000, size=0x80000);
                map (dest=bus:tc1:fpi_bus, src_offset=0x90080000, dest_offset=0x90080000, size=0x80000, priority=2);
                map (dest=bus:tc1:fpi_bus, src_offset=0x90100000, dest_offset=0x90100000, size=0x120000);
                map (dest=bus:tc2:fpi_bus, src_offset=0x90000000, dest_offset=0x90000000, size=0x100000);
                map (dest=bus:tc2:fpi_bus, src_offset=0x90100000, dest_offset=0x90100000, size=0x60000, priority=2);
                map (dest=bus:tc2:fpi_bus, src_offset=0x90160000, dest_offset=0x90160000, size=0xc0000);
                map (dest=bus:tc3:fpi_bus, src_offset=0x90000000, dest_offset=0x90000000, size=0x160000);
                map (dest=bus:tc3:fpi_bus, src_offset=0x90160000, dest_offset=0x90160000, size=0x40000, priority=2);
                map (dest=bus:tc3:fpi_bus, src_offset=0x901a0000, dest_offset=0x901a0000, size=0x80000);
                map (dest=bus:tc4:fpi_bus, src_offset=0x90000000, dest_offset=0x90000000, size=0x1a0000);
                map (dest=bus:tc4:fpi_bus, src_offset=0x901a0000, dest_offset=0x901a0000, size=0x40000, priority=2);
                map (dest=bus:tc4:fpi_bus, src_offset=0x901e0000, dest_offset=0x901e0000, size=0x40000);
                map (dest=bus:tc5:fpi_bus, src_offset=0x90000000, dest_offset=0x90000000, size=0x1e0000);
                map (dest=bus:tc5:fpi_bus, src_offset=0x901e0000, dest_offset=0x901e0000, size=0x40000, priority=2);
                map (dest=bus:tc5:fpi_bus, src_offset=0x90220000, dest_offset=0x90220000, size=0);
                map (dest=bus:tc6:fpi_bus, src_offset=0x90000000, dest_offset=0x90000000, size=0x220000);
                map (dest=bus:tc6:fpi_bus, src_offset=0x92000000, dest_offset=0x92000000, size=0x20000, priority=2);
#ifndef __NO_VTC
                map (dest=bus:vtc:fpi_bus, src_offset=0x90000000, dest_offset=0x90000000, size=0x220000, priority=1);
                map (dest=bus:vtc:fpi_bus, src_offset=0x92000000, dest_offset=0x92000000, size=0x20000, priority=1);
#endif // __NO_VTC

                // lmuram
                map (dest=bus:tc0:fpi_bus, src_offset=0x90300000, dest_offset=0x90300000, size=0x400000);
                map (dest=bus:tc1:fpi_bus, src_offset=0x90300000, dest_offset=0x90300000, size=0x400000);
                map (dest=bus:tc2:fpi_bus, src_offset=0x90300000, dest_offset=0x90300000, size=0x400000);
                map (dest=bus:tc3:fpi_bus, src_offset=0x90300000, dest_offset=0x90300000, size=0x400000);
                map (dest=bus:tc4:fpi_bus, src_offset=0x90300000, dest_offset=0x90300000, size=0x400000);
                map (dest=bus:tc5:fpi_bus, src_offset=0x90300000, dest_offset=0x90300000, size=0x400000);
                map (dest=bus:tc6:fpi_bus, src_offset=0x90300000, dest_offset=0x90300000, size=0x400000);
#ifndef __NO_VTC
                map (dest=bus:vtc:fpi_bus, src_offset=0x90300000, dest_offset=0x90300000, size=0x400000, priority=1);
#endif

                // remaining segment 9 areas
                map (dest=bus:tc0:fpi_bus, src_offset=0x92020000, dest_offset=0x92020000, size=0xdfe0000);
                map (dest=bus:tc1:fpi_bus, src_offset=0x92020000, dest_offset=0x92020000, size=0xdfe0000);
                map (dest=bus:tc2:fpi_bus, src_offset=0x92020000, dest_offset=0x92020000, size=0xdfe0000);
                map (dest=bus:tc3:fpi_bus, src_offset=0x92020000, dest_offset=0x92020000, size=0xdfe0000);
                map (dest=bus:tc4:fpi_bus, src_offset=0x92020000, dest_offset=0x92020000, size=0xdfe0000);
                map (dest=bus:tc5:fpi_bus, src_offset=0x92020000, dest_offset=0x92020000, size=0xdfe0000);
                map (dest=bus:tc6:fpi_bus, src_offset=0x92020000, dest_offset=0x92020000, size=0xdfe0000);
#ifndef __NO_VTC
                map (dest=bus:vtc:fpi_bus, src_offset=0x92020000, dest_offset=0x92020000, size=0xdfe0000);
#endif // __NO_VTC

                // pflash and dlmuram
                map (dest=bus:tc0:fpi_bus, src_offset=0xa0000000, dest_offset=0xa0000000, size=0x20000000);
                map (dest=bus:tc1:fpi_bus, src_offset=0xa0000000, dest_offset=0xa0000000, size=0x20000000);
                map (dest=bus:tc2:fpi_bus, src_offset=0xa0000000, dest_offset=0xa0000000, size=0x20000000);
                map (dest=bus:tc3:fpi_bus, src_offset=0xa0000000, dest_offset=0xa0000000, size=0x20000000);
                map (dest=bus:tc4:fpi_bus, src_offset=0xa0000000, dest_offset=0xa0000000, size=0x20000000);
                map (dest=bus:tc5:fpi_bus, src_offset=0xa0000000, dest_offset=0xa0000000, size=0x20000000);
                map (dest=bus:tc6:fpi_bus, src_offset=0xa0000000, dest_offset=0xa0000000, size=0x20000000);
#ifndef __NO_VTC
                map (dest=bus:vtc:fpi_bus, src_offset=0xa0000000, dest_offset=0xa0000000, size=0x20000000);
#endif // __NO_VTC

                // segments starting from 0xe
                map (dest=bus:tc0:fpi_bus, src_offset=0xe0000000, dest_offset=0xe0000000, size=0x20000000);
                map (dest=bus:tc1:fpi_bus, src_offset=0xe0000000, dest_offset=0xe0000000, size=0x20000000);
                map (dest=bus:tc2:fpi_bus, src_offset=0xe0000000, dest_offset=0xe0000000, size=0x20000000);
                map (dest=bus:tc3:fpi_bus, src_offset=0xe0000000, dest_offset=0xe0000000, size=0x20000000);
                map (dest=bus:tc4:fpi_bus, src_offset=0xe0000000, dest_offset=0xe0000000, size=0x20000000);
                map (dest=bus:tc5:fpi_bus, src_offset=0xe0000000, dest_offset=0xe0000000, size=0x20000000);
                map (dest=bus:tc6:fpi_bus, src_offset=0xe0000000, dest_offset=0xe0000000, size=0x20000000);
#ifndef __NO_VTC
                map (dest=bus:vtc:fpi_bus, src_offset=0xe0000000, dest_offset=0xe0000000, size=0x20000000);
#endif // __NO_VTC
                map (dest=bus:aei, src_offset=GTM_BASE_ADDR, dest_offset=0, size=0x100000000-GTM_BASE_ADDR);
                map (dest=bus:ppu_extern, src_offset=CSM_MEM_REGION, dest_offset=CSM_MEM_REGION, size=CSM_MEM_REGION_SIZE);
        }

#ifndef    __REDEFINE_ON_CHIP_ITEMS
#ifndef __CPP_RUN_TIME_ENTRY_FLAG
#define __CPP_RUN_TIME_ENTRY_FLAG mem:dspr0
#endif

#ifndef __NO_VTC
        section_layout :vtc:linear
        {
                group (ordered, run_addr=__CPP_RUN_TIME_ENTRY_FLAG)
                {
                        // C++ run-time variable "main_called" that ensures the global object constructors to execute exactly once.
                        // main_called is initialized so its name gets a data prefix: .sect '.data.__section_main_called'
                        select ".data.__section_main_called";
                        // C++ run-time variables to make destructors concurrent
                        select ".data.__section_dtor_finalizer";
                }
        }
#else
        section_layout :tc0:linear
        {
                group (ordered, run_addr=__CPP_RUN_TIME_ENTRY_FLAG)
                {
                        // C++ run-time variable "main_called" that ensures the global object constructors to execute exactly once.
                        // main_called is initialized so its name gets a data prefix: .sect '.data.__section_main_called'
                        select ".data.__section_main_called";
                        // C++ run-time variables to make destructors concurrent
                        select ".data.__section_dtor_finalizer";
                }
        }
        section_layout :tc1:linear
        {
                group (ordered, run_addr=__CPP_RUN_TIME_ENTRY_FLAG)
                {
                        // C++ run-time variable "main_called" that ensures the global object constructors to execute exactly once.
                        // main_called is initialized so its name gets a data prefix: .sect '.data.__section_main_called'
                        select ".data.__section_main_called";
                        // C++ run-time variables to make destructors concurrent
                        select ".data.__section_dtor_finalizer";
                }
        }
        section_layout :tc2:linear
        {
                group (ordered, run_addr=__CPP_RUN_TIME_ENTRY_FLAG)
                {
                        // C++ run-time variable "main_called" that ensures the global object constructors to execute exactly once.
                        // main_called is initialized so its name gets a data prefix: .sect '.data.__section_main_called'
                        select ".data.__section_main_called";
                        // C++ run-time variables to make destructors concurrent
                        select ".data.__section_dtor_finalizer";
                }
        }
        section_layout :tc3:linear
        {
                group (ordered, run_addr=__CPP_RUN_TIME_ENTRY_FLAG)
                {
                        // C++ run-time variable "main_called" that ensures the global object constructors to execute exactly once.
                        // main_called is initialized so its name gets a data prefix: .sect '.data.__section_main_called'
                        select ".data.__section_main_called";
                        // C++ run-time variables to make destructors concurrent
                        select ".data.__section_dtor_finalizer";
                }
        }
        section_layout :tc4:linear
        {
                group (ordered, run_addr=__CPP_RUN_TIME_ENTRY_FLAG)
                {
                        // C++ run-time variable "main_called" that ensures the global object constructors to execute exactly once.
                        // main_called is initialized so its name gets a data prefix: .sect '.data.__section_main_called'
                        select ".data.__section_main_called";
                        // C++ run-time variables to make destructors concurrent
                        select ".data.__section_dtor_finalizer";
                }
        }
        section_layout :tc5:linear
        {
                group (ordered, run_addr=__CPP_RUN_TIME_ENTRY_FLAG)
                {
                        // C++ run-time variable "main_called" that ensures the global object constructors to execute exactly once.
                        // main_called is initialized so its name gets a data prefix: .sect '.data.__section_main_called'
                        select ".data.__section_main_called";
                        // C++ run-time variables to make destructors concurrent
                        select ".data.__section_dtor_finalizer";
                }
        }
        section_layout :tc6:linear
        {
                group (ordered, run_addr=__CPP_RUN_TIME_ENTRY_FLAG)
                {
                        // C++ run-time variable "main_called" that ensures the global object constructors to execute exactly once.
                        // main_called is initialized so its name gets a data prefix: .sect '.data.__section_main_called'
                        select ".data.__section_main_called";
                        // C++ run-time variables to make destructors concurrent
                        select ".data.__section_dtor_finalizer";
                }
        }
#endif  // __NO_VTC
#endif  // __REDEFINE_ON_CHIP_ITEMS

#ifndef    __REDEFINE_ON_CHIP_ITEMS

        memory dspr0 // Data Scratch Pad Ram CPU0
        {
                mau = 8;
                size = 240k;
                type = ram;
                priority = 1;
                exec_priority = 0;
                map (dest=bus:tc0:fpi_bus, dest_offset=0xd0000000, size=240k, priority=1, exec_priority=0);
                map (dest=bus:sri, dest_offset=0x70000000, size=240k);
        }

        memory pspr0 // Program Scratch Pad Ram CPU0
        {
                mau = 8;
                size = 64k;
                type = ram;
                priority = 0;
                exec_priority = 1;
                map (dest=bus:tc0:fpi_bus, dest_offset=0xc0000000, size=64k, exec_priority=1);
                map (dest=bus:sri, dest_offset=0x70100000, size=64k);
        }

        memory dspr1 // Data Scratch Pad Ram CPU1
        {
                mau = 8;
                size = 240k;
                type = ram;
                priority = 1;
                exec_priority = 0;
                map (dest=bus:tc1:fpi_bus, dest_offset=0xd0000000, size=240k, priority=1, exec_priority=0);
                map (dest=bus:sri, dest_offset=0x60000000, size=240k);
        }

        memory pspr1 // Program Scratch Pad Ram CPU1
        {
                mau = 8;
                size = 64k;
                type = ram;
                priority = 0;
                exec_priority = 1;
                map (dest=bus:tc1:fpi_bus, dest_offset=0xc0000000, size=64k, exec_priority=1);
                map (dest=bus:sri, dest_offset=0x60100000, size=64k);
        }

        memory dspr2 // Data Scratch Pad Ram CPU2
        {
                mau = 8;
                size = 240k;
                type = ram;
                priority = 1;
                exec_priority = 0;
                map (dest=bus:tc2:fpi_bus, dest_offset=0xd0000000, size=240k, priority=1, exec_priority=0);
                map (dest=bus:sri, dest_offset=0x50000000, size=240k);
        }

        memory pspr2 // Program Scratch Pad Ram CPU2
        {
                mau = 8;
                size = 64k;
                type = ram;
                priority = 0;
                exec_priority = 1;
                map (dest=bus:tc2:fpi_bus, dest_offset=0xc0000000, size=64k, exec_priority=1);
                map (dest=bus:sri, dest_offset=0x50100000, size=64k);
        }

        memory dspr3 // Data Scratch Pad Ram CPU3
        {
                mau = 8;
                size = 112k;
                type = ram;
                priority = 1;
                exec_priority = 0;
                map (dest=bus:tc3:fpi_bus, dest_offset=0xd0000000, size=112k, priority=1, exec_priority=0);
                map (dest=bus:sri, dest_offset=0x40000000, size=112k);
        }

        memory pspr3 // Program Scratch Pad Ram CPU3
        {
                mau = 8;
                size = 64k;
                type = ram;
                priority = 0;
                exec_priority = 1;
                map (dest=bus:tc3:fpi_bus, dest_offset=0xc0000000, size=64k, exec_priority=1);
                map (dest=bus:sri, dest_offset=0x40100000, size=64k);
        }

        memory dspr4 // Data Scratch Pad Ram CPU4
        {
                mau = 8;
                size = 112k;
                type = ram;
                priority = 1;
                exec_priority = 0;
                map (dest=bus:tc4:fpi_bus, dest_offset=0xd0000000, size=112k, priority=1, exec_priority=0);
                map (dest=bus:sri, dest_offset=0x30000000, size=112k);
        }

        memory pspr4 // Program Scratch Pad Ram CPU4
        {
                mau = 8;
                size = 64k;
                type = ram;
                priority = 0;
                exec_priority = 1;
                map (dest=bus:tc4:fpi_bus, dest_offset=0xc0000000, size=64k, exec_priority=1);
                map (dest=bus:sri, dest_offset=0x30100000, size=64k);
        }

        memory dspr5 // Data Scratch Pad Ram CPU5
        {
                mau = 8;
                size = 112k;
                type = ram;
                priority = 1;
                exec_priority = 0;
                map (dest=bus:tc5:fpi_bus, dest_offset=0xd0000000, size=112k, priority=1, exec_priority=0);
                map (dest=bus:sri, dest_offset=0x20000000, size=112k);
        }

        memory pspr5 // Program Scratch Pad Ram CPU5
        {
                mau = 8;
                size = 64k;
                type = ram;
                priority = 0;
                exec_priority = 1;
                map (dest=bus:tc5:fpi_bus, dest_offset=0xc0000000, size=64k, exec_priority=1);
                map (dest=bus:sri, dest_offset=0x20100000, size=64k);
        }

        memory dspr6 // Data Scratch Pad Ram CPU6
        {
                mau = 8;
                size = 240k;
                type = ram;
                priority = 1;
                exec_priority = 0;
                map (dest=bus:tc6:fpi_bus, dest_offset=0xd0000000, size=240k, priority=1, exec_priority=0);
                map (dest=bus:sri, dest_offset=0x10000000, size=240k);
        }

        memory pspr6 // Program Scratch Pad Ram CPU6
        {
                mau = 8;
                size = 64k;
                type = ram;
                priority = 0;
                exec_priority = 1;
                map (dest=bus:tc6:fpi_bus, dest_offset=0xc0000000, size=64k, exec_priority=1);
                map (dest=bus:sri, dest_offset=0x10100000, size=64k);
        }

        memory pflash00 // Program Flash CPU0
        {
                mau = 8;
                size = 2M;
                type = rom;
                map     cached (dest=bus:sri, dest_offset=0x80000000,           size=2M, cached);
                map not_cached (dest=bus:sri, dest_offset=0xa0000000, reserved, size=2M);
        }

        memory pflash01 // Program Flash CPU0
        {
                mau = 8;
                size = 2M;
                type = rom;
                map     cached (dest=bus:sri, dest_offset=0x80200000,           size=2M, cached);
                map not_cached (dest=bus:sri, dest_offset=0xa0200000, reserved, size=2M);
        }

        memory pflash10 // Program Flash CPU1
        {
                mau = 8;
                size = 2M;
                type = rom;
                map     cached (dest=bus:sri, dest_offset=0x80400000,           size=2M, cached);
                map not_cached (dest=bus:sri, dest_offset=0xa0400000, reserved, size=2M);
        }

        memory pflash11 // Program Flash CPU1
        {
                mau = 8;
                size = 2M;
                type = rom;
                map     cached (dest=bus:sri, dest_offset=0x80600000,           size=2M, cached);
                map not_cached (dest=bus:sri, dest_offset=0xa0600000, reserved, size=2M);
        }

        memory pflash20 // Program Flash CPU2
        {
                mau = 8;
                size = 2M;
                type = rom;
                map     cached (dest=bus:sri, dest_offset=0x80800000,           size=2M, cached);
                map not_cached (dest=bus:sri, dest_offset=0xa0800000, reserved, size=2M);
        }

        memory pflash21 // Program Flash CPU2
        {
                mau = 8;
                size = 2M;
                type = rom;
                map     cached (dest=bus:sri, dest_offset=0x80a00000,           size=2M, cached);
                map not_cached (dest=bus:sri, dest_offset=0xa0a00000, reserved, size=2M);
        }

        memory pflash30 // Program Flash CPU3
        {
                mau = 8;
                size = 2M;
                type = rom;
                map     cached (dest=bus:sri, dest_offset=0x80c00000,           size=2M, cached);
                map not_cached (dest=bus:sri, dest_offset=0xa0c00000, reserved, size=2M);
        }

        memory pflash31 // Program Flash CPU3
        {
                mau = 8;
                size = 2M;
                type = rom;
                map     cached (dest=bus:sri, dest_offset=0x80e00000,           size=2M, cached);
                map not_cached (dest=bus:sri, dest_offset=0xa0e00000, reserved, size=2M);
        }

        memory pflash40 // Program Flash CPU4
        {
                mau = 8;
                size = 2M;
                type = rom;
                map     cached (dest=bus:sri, dest_offset=0x81000000,           size=2M, cached);
                map not_cached (dest=bus:sri, dest_offset=0xa1000000, reserved, size=2M);
        }

        memory pflash41 // Program Flash CPU4
        {
                mau = 8;
                size = 2M;
                type = rom;
                map     cached (dest=bus:sri, dest_offset=0x81200000,           size=2M, cached);
                map not_cached (dest=bus:sri, dest_offset=0xa1200000, reserved, size=2M);
        }

        memory pflash50 // Program Flash CPU5
        {
                mau = 8;
                size = 2M;
                type = rom;
                map     cached (dest=bus:sri, dest_offset=0x81400000,           size=2M, cached);
                map not_cached (dest=bus:sri, dest_offset=0xa1400000, reserved, size=2M);
        }

        memory pflash51 // Program Flash CPU5
        {
                mau = 8;
                size = 2M;
                type = rom;
                map     cached (dest=bus:sri, dest_offset=0x81600000,           size=2M, cached);
                map not_cached (dest=bus:sri, dest_offset=0xa1600000, reserved, size=2M);
        }

        memory pflashcs // Program Flash CPU6
        {
                mau = 8;
                size = 1M;
                type = rom;
                map     cached (dest=bus:sri, dest_offset=0x84000000,           size=1M, cached);
                map not_cached (dest=bus:sri, dest_offset=0xa4000000, reserved, size=1M);
        }

        memory brom
        {
                mau = 8;
                size = 64k;
                type = reserved rom;
                map     cached (dest=bus:sri, dest_offset=0x8fff0000,           size=64k, cached);
                map not_cached (dest=bus:sri, dest_offset=0xafff0000, reserved, size=64k);
        }

        memory eeprom0
        {
                mau = 8;
                size = 1024k;
                type = reserved nvram;
                map (dest=bus:sri, dest_offset=0xae000000, size=1024k);
        }

        memory ucb0
        {
                mau = 8;
                size = 80k;
                type = reserved nvram;
                map (dest=bus:sri, dest_offset=0xae400000, size=80k);
        }

        memory cfs0
        {
                mau = 8;
                size = 80k;
                type = reserved nvram;
                map (dest=bus:sri, dest_offset=0xae600000, size=80k);
        }

        memory eeprom1
        {
                mau = 8;
                size = 128k;
                type = reserved nvram;
                map (dest=bus:sri, dest_offset=0xae800000, size=128k);
        }

        memory ucb1
        {
                mau = 8;
                size = 52k;
                type = reserved nvram;
                map (dest=bus:sri, dest_offset=0xaec00000, size=52k);
        }

        memory cfs1
        {
                mau = 8;
                size = 44k;
                type = reserved nvram;
                map (dest=bus:sri, dest_offset=0xaee00000, size=44k);
        }

        memory dlmucpu0 // DLMU CPU0
        {
                mau = 8;
                size = 512k;
                type = ram;
                map     cached (dest=bus:sri, dest_offset=0x90000000,           size=512k, cached);
                map not_cached (dest=bus:sri, dest_offset=0xb0000000, reserved, size=512k);
        }

        memory dlmucpu1 // DLMU CPU1
        {
                mau = 8;
                size = 512k;
                type = ram;
                map     cached (dest=bus:sri, dest_offset=0x90080000,           size=512k, cached);
                map not_cached (dest=bus:sri, dest_offset=0xb0080000, reserved, size=512k);
        }

        memory dlmucpu2 // DLMU CPU2
        {
                mau = 8;
                size = 384k;
                type = ram;
                map     cached (dest=bus:sri, dest_offset=0x90100000,           size=384k, cached);
                map not_cached (dest=bus:sri, dest_offset=0xb0100000, reserved, size=384k);
        }

        memory dlmucpu3 // DLMU CPU3
        {
                mau = 8;
                size = 256k;
                type = ram;
                map     cached (dest=bus:sri, dest_offset=0x90160000,           size=256k, cached);
                map not_cached (dest=bus:sri, dest_offset=0xb0160000, reserved, size=256k);
        }

        memory dlmucpu4 // DLMU CPU4
        {
                mau = 8;
                size = 256k;
                type = ram;
                map     cached (dest=bus:sri, dest_offset=0x901a0000,           size=256k, cached);
                map not_cached (dest=bus:sri, dest_offset=0xb01a0000, reserved, size=256k);
        }

        memory dlmucpu5 // DLMU CPU5
        {
                mau = 8;
                size = 256k;
                type = ram;
                map     cached (dest=bus:sri, dest_offset=0x901e0000,           size=256k, cached);
                map not_cached (dest=bus:sri, dest_offset=0xb01e0000, reserved, size=256k);
        }

        memory dlmucpu6 // DLMU CPU6
        {
                mau = 8;
                size = 128k;
                type = ram;
                map     cached (dest=bus:sri, dest_offset=0x92000000,           size=128k, cached);
                map not_cached (dest=bus:sri, dest_offset=0xb2000000, reserved, size=128k);
        }

        memory lmuram0
        {
                mau = 8;
                size = 512k;
                type = ram;
                priority = 2;
                map     cached (dest=bus:sri, dest_offset=0x90300000,           size=512k, cached);
                map not_cached (dest=bus:sri, dest_offset=0xb0300000, reserved, size=512k);
        }

        memory lmuram1
        {
                mau = 8;
                size = 512k;
                type = ram;
                priority = 2;
                map     cached (dest=bus:sri, dest_offset=0x90380000,           size=512k, cached);
                map not_cached (dest=bus:sri, dest_offset=0xb0380000, reserved, size=512k);
        }

        memory lmuram2
        {
                mau = 8;
                size = 512k;
                type = ram;
                priority = 2;
                map     cached (dest=bus:sri, dest_offset=0x90400000,           size=512k, cached);
                map not_cached (dest=bus:sri, dest_offset=0xb0400000, reserved, size=512k);
        }

        memory lmuram3
        {
                mau = 8;
                size = 512k;
                type = ram;
                priority = 2;
                map     cached (dest=bus:sri, dest_offset=0x90480000,           size=512k, cached);
                map not_cached (dest=bus:sri, dest_offset=0xb0480000, reserved, size=512k);
        }

        memory lmuram4
        {
                mau = 8;
                size = 512k;
                type = ram;
                priority = 2;
                map     cached (dest=bus:sri, dest_offset=0x90500000,           size=512k, cached);
                map not_cached (dest=bus:sri, dest_offset=0xb0500000, reserved, size=512k);
        }

        memory lmuram5
        {
                mau = 8;
                size = 512k;
                type = ram;
                priority = 2;
                map     cached (dest=bus:sri, dest_offset=0x90580000,           size=512k, cached);
                map not_cached (dest=bus:sri, dest_offset=0xb0580000, reserved, size=512k);
        }

        memory lmuram6
        {
                mau = 8;
                size = 512k;
                type = ram;
                priority = 2;
                map     cached (dest=bus:sri, dest_offset=0x90600000,           size=512k, cached);
                map not_cached (dest=bus:sri, dest_offset=0xb0600000, reserved, size=512k);
        }

        memory lmuram7
        {
                mau = 8;
                size = 512k;
                type = ram;
                priority = 2;
                map     cached (dest=bus:sri, dest_offset=0x90680000,           size=512k, cached);
                map not_cached (dest=bus:sri, dest_offset=0xb0680000, reserved, size=512k);
        }

        memory scr_iram
        {
                mau = 8;
                type = ram;
                size = 256;
                map ( dest=bus:xc800:idata_bus, src_offset=0x0, dest_offset=0x0, size=256 );
        }

        memory scr_xram
        {
                mau = 8;
                type = nvram;
                size = 32k;
                map ( dest=bus:xc800:xdata_bus, src_offset=0x0, dest_offset=0x0, size=32k );
                map ( dest=bus:xc800:code_bus, src_offset=0x0, dest_offset=0x0, size=32k );
                map ( dest=bus:sri, dest_offset=SCR_BASE_ADDR, src_offset=0, size=32k, reserved );
        }

        memory scr_bootrom
        {
                mau = 8;
                type = reserved rom;
                size = 2k;
                map ( dest=bus:xc800:code_bus, src_offset=0x0, dest_offset=0xd000, size=2k );
        }

#endif  // __REDEFINE_ON_CHIP_ITEMS

#ifndef __NO_VTC
        section_setup :vtc:linear
        {
#ifndef __LINKONLY__
                start_address
                (
                        run_addr = (RESET),
                        symbol = "_START"
                );
#endif
        }
#else
        section_setup :tc0:linear
        {
#ifndef __LINKONLY__
                start_address
                (
                        run_addr = (RESET),
                        symbol = "_START"
                );
#endif
        }
        section_setup :tc1:linear
        {
#ifndef __LINKONLY__
                start_address
                (
                        symbol = "__start_tc1_no_vtc"
                );
#endif
        }
        section_setup :tc2:linear
        {
#ifndef __LINKONLY__
                start_address
                (
                        symbol = "__start_tc2_no_vtc"
                );
#endif
        }
        section_setup :tc3:linear
        {
#ifndef __LINKONLY__
                start_address
                (
                        symbol = "__start_tc3_no_vtc"
                );
#endif
        }
        section_setup :tc4:linear
        {
#ifndef __LINKONLY__
                start_address
                (
                        symbol = "__start_tc4_no_vtc"
                );
#endif
        }
        section_setup :tc5:linear
        {
#ifndef __LINKONLY__
                start_address
                (
                        symbol = "__start_tc5_no_vtc"
                );
#endif
        }
        section_setup :tc6:linear
        {
#ifndef __LINKONLY__
                start_address
                (
                        symbol = "__start_tc6_no_vtc"
                );
#endif
        }
#endif

#ifndef __NO_VTC
        section_setup :tc0:linear
        {
                stack "ustack_tc0"
                (
                        min_size = (USTACK_TC0)
                        ,fixed
                        ,align = 8
                        __USTACK0_THREADS_ATTRIBUTE
                        __USTACK0_ENTRY_POINTS_ATTRIBUTE
                );

                stack "istack_tc0"
                (
                        min_size = (ISTACK_TC0)
                        ,fixed
                        ,align = 8
                        __ISTACK0_ENTRY_POINTS_ATTRIBUTE
                );
        }

        section_setup :tc1:linear
        {
                stack "ustack_tc1"
                (
                        min_size = (USTACK_TC1)
                        ,fixed
                        ,align = 8
                        __USTACK1_THREADS_ATTRIBUTE
                        __USTACK1_ENTRY_POINTS_ATTRIBUTE
                );

                stack "istack_tc1"
                (
                        min_size = (ISTACK_TC1)
                        ,fixed
                        ,align = 8
                        __ISTACK1_ENTRY_POINTS_ATTRIBUTE
                );
        }

        section_setup :tc2:linear
        {
                stack "ustack_tc2"
                (
                        min_size = (USTACK_TC2)
                        ,fixed
                        ,align = 8
                        __USTACK2_THREADS_ATTRIBUTE
                        __USTACK2_ENTRY_POINTS_ATTRIBUTE
                );

                stack "istack_tc2"
                (
                        min_size = (ISTACK_TC2)
                        ,fixed
                        ,align = 8
                        __ISTACK2_ENTRY_POINTS_ATTRIBUTE
                );
        }

        section_setup :tc3:linear
        {
                stack "ustack_tc3"
                (
                        min_size = (USTACK_TC3)
                        ,fixed
                        ,align = 8
                        __USTACK3_THREADS_ATTRIBUTE
                        __USTACK3_ENTRY_POINTS_ATTRIBUTE
                );

                stack "istack_tc3"
                (
                        min_size = (ISTACK_TC3)
                        ,fixed
                        ,align = 8
                        __ISTACK3_ENTRY_POINTS_ATTRIBUTE
                );
        }

        section_setup :tc4:linear
        {
                stack "ustack_tc4"
                (
                        min_size = (USTACK_TC4)
                        ,fixed
                        ,align = 8
                        __USTACK4_THREADS_ATTRIBUTE
                        __USTACK4_ENTRY_POINTS_ATTRIBUTE
                );

                stack "istack_tc4"
                (
                        min_size = (ISTACK_TC4)
                        ,fixed
                        ,align = 8
                        __ISTACK4_ENTRY_POINTS_ATTRIBUTE
                );
        }

        section_setup :tc5:linear
        {
                stack "ustack_tc5"
                (
                        min_size = (USTACK_TC5)
                        ,fixed
                        ,align = 8
                        __USTACK5_THREADS_ATTRIBUTE
                        __USTACK5_ENTRY_POINTS_ATTRIBUTE
                );

                stack "istack_tc5"
                (
                        min_size = (ISTACK_TC5)
                        ,fixed
                        ,align = 8
                        __ISTACK5_ENTRY_POINTS_ATTRIBUTE
                );
        }

        section_setup :tc6:linear
        {
                stack "ustack_tc6"
                (
                        min_size = (USTACK_TC6)
                        ,fixed
                        ,align = 8
                        __USTACK6_THREADS_ATTRIBUTE
                        __USTACK6_ENTRY_POINTS_ATTRIBUTE
                );

                stack "istack_tc6"
                (
                        min_size = (ISTACK_TC6)
                        ,fixed
                        ,align = 8
                        __ISTACK6_ENTRY_POINTS_ATTRIBUTE
                );
        }

        section_setup :vtc:linear
        {
                heap "heap"
                (
                        min_size = (HEAP),
                        fixed,
                        align = 8
                );
        }
#else
        section_setup :tc0:linear
        {
                stack "ustack"
                (
                        min_size = (USTACK_TC0)
                        , fixed
                        , align = 8
                        __USTACK0_THREADS_ATTRIBUTE
                        __USTACK0_ENTRY_POINTS_ATTRIBUTE
                );

                stack "istack"
                (
                        min_size = (ISTACK_TC0)
                        ,fixed
                        ,align = 8
                        __ISTACK0_ENTRY_POINTS_ATTRIBUTE
                );
                heap "heap"
                (
                        min_size = (HEAP_TC0),
                        fixed,
                        align = 8
                );
        }

        section_setup :tc1:linear
        {
                stack "ustack"
                (
                        min_size = (USTACK_TC1)
                        , fixed
                        , align = 8
                        __USTACK1_THREADS_ATTRIBUTE
                        __USTACK1_ENTRY_POINTS_ATTRIBUTE
                );

                stack "istack"
                (
                        min_size = (ISTACK_TC1)
                        ,fixed
                        ,align = 8
                        __ISTACK1_ENTRY_POINTS_ATTRIBUTE
                );
                heap "heap"
                (
                        min_size = (HEAP_TC1),
                        fixed,
                        align = 8
                );
        }

        section_setup :tc2:linear
        {
                stack "ustack"
                (
                        min_size = (USTACK_TC2)
                        , fixed
                        , align = 8
                        __USTACK2_THREADS_ATTRIBUTE
                        __USTACK2_ENTRY_POINTS_ATTRIBUTE
                );

                stack "istack"
                (
                        min_size = (ISTACK_TC2)
                        ,fixed
                        ,align = 8
                        __ISTACK2_ENTRY_POINTS_ATTRIBUTE
                );
                heap "heap"
                (
                        min_size = (HEAP_TC2),
                        fixed,
                        align = 8
                );
        }

        section_setup :tc3:linear
        {
                stack "ustack"
                (
                        min_size = (USTACK_TC3)
                        , fixed
                        , align = 8
                        __USTACK3_THREADS_ATTRIBUTE
                        __USTACK3_ENTRY_POINTS_ATTRIBUTE
                );

                stack "istack"
                (
                        min_size = (ISTACK_TC3)
                        ,fixed
                        ,align = 8
                        __ISTACK3_ENTRY_POINTS_ATTRIBUTE
                );
                heap "heap"
                (
                        min_size = (HEAP_TC3),
                        fixed,
                        align = 8
                );
        }

        section_setup :tc4:linear
        {
                stack "ustack"
                (
                        min_size = (USTACK_TC4)
                        , fixed
                        , align = 8
                        __USTACK4_THREADS_ATTRIBUTE
                        __USTACK4_ENTRY_POINTS_ATTRIBUTE
                );

                stack "istack"
                (
                        min_size = (ISTACK_TC4)
                        ,fixed
                        ,align = 8
                        __ISTACK4_ENTRY_POINTS_ATTRIBUTE
                );
                heap "heap"
                (
                        min_size = (HEAP_TC4),
                        fixed,
                        align = 8
                );
        }

        section_setup :tc5:linear
        {
                stack "ustack"
                (
                        min_size = (USTACK_TC5)
                        , fixed
                        , align = 8
                        __USTACK5_THREADS_ATTRIBUTE
                        __USTACK5_ENTRY_POINTS_ATTRIBUTE
                );

                stack "istack"
                (
                        min_size = (ISTACK_TC5)
                        ,fixed
                        ,align = 8
                        __ISTACK5_ENTRY_POINTS_ATTRIBUTE
                );
                heap "heap"
                (
                        min_size = (HEAP_TC5),
                        fixed,
                        align = 8
                );
        }

        section_setup :tc6:linear
        {
                stack "ustack"
                (
                        min_size = (USTACK_TC6)
                        , fixed
                        , align = 8
                        __USTACK6_THREADS_ATTRIBUTE
                        __USTACK6_ENTRY_POINTS_ATTRIBUTE
                );

                stack "istack"
                (
                        min_size = (ISTACK_TC6)
                        ,fixed
                        ,align = 8
                        __ISTACK6_ENTRY_POINTS_ATTRIBUTE
                );
                heap "heap"
                (
                        min_size = (HEAP_TC6),
                        fixed,
                        align = 8
                );
        }

#endif
#ifndef __NO_VTC
        section_layout :tc0:linear
        {
                "_lc_ub_ustack" := "_lc_ub_ustack_tc0"; /* common cstart interface for first or single core */
                "_lc_ue_ustack" := "_lc_ue_ustack_tc0"; /* common cstart interface for first or single core */
                "_lc_ue_istack" := "_lc_ue_istack_tc0"; /* common cstart interface for first or single core */
        }
#endif

#ifndef __NO_VTC
        section_setup :vtc:linear
        {
                copytable
                (
                        align = 4,
                        dest = linear
                        ,
                        table
                        {
                                symbol = "_lc_ub_table_tc1";
                                space = :tc1:linear, :tc1:abs24, :tc1:abs18, :tc1:csa;
                        }
                        ,
                        table
                        {
                                symbol = "_lc_ub_table_tc2";
                                space = :tc2:linear, :tc2:abs24, :tc2:abs18, :tc2:csa;
                        }
                        ,
                        table
                        {
                                symbol = "_lc_ub_table_tc3";
                                space = :tc3:linear, :tc3:abs24, :tc3:abs18, :tc3:csa;
                        }
                        ,
                        table
                        {
                                symbol = "_lc_ub_table_tc4";
                                space = :tc4:linear, :tc4:abs24, :tc4:abs18, :tc4:csa;
                        }
                        ,
                        table
                        {
                                symbol = "_lc_ub_table_tc5";
                                space = :tc5:linear, :tc5:abs24, :tc5:abs18, :tc5:csa;
                        }
                        ,
                        table
                        {
                                symbol = "_lc_ub_table_tc6";
                                space = :tc6:linear, :tc6:abs24, :tc6:abs18, :tc6:csa;
                        }
                );

                mpu_data_table;
        }
        section_layout :tc0:csa
        {
                group  (ordered, align = 64, attributes=rw, run_addr=(CSA_START_TC0))
                        reserved "csa_tc0" (size = 64 * (CSA_TC0));
                "_lc_ub_csa_01" := "_lc_ub_csa_tc0"; /* common cstart interface for first or single core */
                "_lc_ue_csa_01" := "_lc_ue_csa_tc0"; /* common cstart interface for first or single core */
        }

        section_layout :tc1:csa
        {
                group  (ordered, align = 64, attributes=rw, run_addr=(CSA_START_TC1))
                        reserved "csa_tc1" (size = 64 * (CSA_TC1));
        }

        section_layout :tc2:csa
        {
                group  (ordered, align = 64, attributes=rw, run_addr=(CSA_START_TC2))
                        reserved "csa_tc2" (size = 64 * (CSA_TC2));
        }

        section_layout :tc3:csa
        {
                group  (ordered, align = 64, attributes=rw, run_addr=(CSA_START_TC3))
                        reserved "csa_tc3" (size = 64 * (CSA_TC3));
        }

        section_layout :tc4:csa
        {
                group  (ordered, align = 64, attributes=rw, run_addr=(CSA_START_TC4))
                        reserved "csa_tc4" (size = 64 * (CSA_TC4));
        }

        section_layout :tc5:csa
        {
                group  (ordered, align = 64, attributes=rw, run_addr=(CSA_START_TC5))
                        reserved "csa_tc5" (size = 64 * (CSA_TC5));
        }

        section_layout :tc6:csa
        {
                group  (ordered, align = 64, attributes=rw, run_addr=(CSA_START_TC6))
                        reserved "csa_tc6" (size = 64 * (CSA_TC6));
        }
#else
        section_setup :tc0:linear
        {
                copytable
                (
                        align = 4,
                        dest = linear
                );

                mpu_data_table;
        }
        section_setup :tc1:linear
        {
                copytable
                (
                        align = 4,
                        dest = linear
                );

                mpu_data_table;
        }
        section_setup :tc2:linear
        {
                copytable
                (
                        align = 4,
                        dest = linear
                );

                mpu_data_table;
        }
        section_setup :tc3:linear
        {
                copytable
                (
                        align = 4,
                        dest = linear
                );

                mpu_data_table;
        }
        section_setup :tc4:linear
        {
                copytable
                (
                        align = 4,
                        dest = linear
                );

                mpu_data_table;
        }
        section_setup :tc5:linear
        {
                copytable
                (
                        align = 4,
                        dest = linear
                );

                mpu_data_table;
        }
        section_setup :tc6:linear
        {
                copytable
                (
                        align = 4,
                        dest = linear
                );

                mpu_data_table;
        }

        section_layout :tc0:csa
        {
                group  (ordered, align = 64, attributes=rw, run_addr=(CSA_START_TC0))
                        reserved "csa_tc0" (size = 64 * (CSA_TC0));
                "_lc_ub_csa_01" := "_lc_ub_csa_tc0"; /* common cstart interface for first or single core */
                "_lc_ue_csa_01" := "_lc_ue_csa_tc0"; /* common cstart interface for first or single core */
        }

        section_layout :tc1:csa
        {
                group  (ordered, align = 64, attributes=rw, run_addr=(CSA_START_TC1))
                        reserved "csa_tc1" (size = 64 * (CSA_TC1));
                "_lc_ub_csa_01" := "_lc_ub_csa_tc1"; /* common cstart interface for first or single core */
                "_lc_ue_csa_01" := "_lc_ue_csa_tc1"; /* common cstart interface for first or single core */
        }

        section_layout :tc2:csa
        {
                group  (ordered, align = 64, attributes=rw, run_addr=(CSA_START_TC2))
                        reserved "csa_tc2" (size = 64 * (CSA_TC2));
                "_lc_ub_csa_01" := "_lc_ub_csa_tc2"; /* common cstart interface for first or single core */
                "_lc_ue_csa_01" := "_lc_ue_csa_tc2"; /* common cstart interface for first or single core */
        }

        section_layout :tc3:csa
        {
                group  (ordered, align = 64, attributes=rw, run_addr=(CSA_START_TC3))
                        reserved "csa_tc3" (size = 64 * (CSA_TC3));
                "_lc_ub_csa_01" := "_lc_ub_csa_tc3"; /* common cstart interface for first or single core */
                "_lc_ue_csa_01" := "_lc_ue_csa_tc3"; /* common cstart interface for first or single core */
        }

        section_layout :tc4:csa
        {
                group  (ordered, align = 64, attributes=rw, run_addr=(CSA_START_TC4))
                        reserved "csa_tc4" (size = 64 * (CSA_TC4));
                "_lc_ub_csa_01" := "_lc_ub_csa_tc4"; /* common cstart interface for first or single core */
                "_lc_ue_csa_01" := "_lc_ue_csa_tc4"; /* common cstart interface for first or single core */
        }

        section_layout :tc5:csa
        {
                group  (ordered, align = 64, attributes=rw, run_addr=(CSA_START_TC5))
                        reserved "csa_tc5" (size = 64 * (CSA_TC5));
                "_lc_ub_csa_01" := "_lc_ub_csa_tc5"; /* common cstart interface for first or single core */
                "_lc_ue_csa_01" := "_lc_ue_csa_tc5"; /* common cstart interface for first or single core */
        }

        section_layout :tc6:csa
        {
                group  (ordered, align = 64, attributes=rw, run_addr=(CSA_START_TC6))
                        reserved "csa_tc6" (size = 64 * (CSA_TC6));
                "_lc_ub_csa_01" := "_lc_ub_csa_tc6"; /* common cstart interface for first or single core */
                "_lc_ue_csa_01" := "_lc_ue_csa_tc6"; /* common cstart interface for first or single core */
        }
#endif

#ifndef __NO_VTC
        section_layout :vtc:linear
        {
#if (XVWBUF>0)
                group (align = 4) reserved "_xvwbuffer_" (size=XVWBUF, attributes=rw);
                "_lc_ub_xvwbuffer" = "_lc_ub__xvwbuffer_";
                "_lc_ue_xvwbuffer" = "_lc_ue__xvwbuffer_";
#endif

#ifndef __REDEFINE_BASE_ADDRESS_GROUPS
#include        "base_address_groups.lsl"
#endif

                "_SMALL_DATA_TC0" := "_SMALL_DATA_";
                "_SMALL_DATA_TC1" := "_SMALL_DATA_";
                "_SMALL_DATA_TC2" := "_SMALL_DATA_";
                "_SMALL_DATA_TC3" := "_SMALL_DATA_";
                "_SMALL_DATA_TC4" := "_SMALL_DATA_";
                "_SMALL_DATA_TC5" := "_SMALL_DATA_";
                "_SMALL_DATA_TC6" := "_SMALL_DATA_";

                "_LITERAL_DATA_TC0" := "_LITERAL_DATA_";
                "_LITERAL_DATA_TC1" := "_LITERAL_DATA_";
                "_LITERAL_DATA_TC2" := "_LITERAL_DATA_";
                "_LITERAL_DATA_TC3" := "_LITERAL_DATA_";
                "_LITERAL_DATA_TC4" := "_LITERAL_DATA_";
                "_LITERAL_DATA_TC5" := "_LITERAL_DATA_";
                "_LITERAL_DATA_TC6" := "_LITERAL_DATA_";

                "_A8_DATA_TC0" := "_A8_DATA_";
                "_A8_DATA_TC1" := "_A8_DATA_";
                "_A8_DATA_TC2" := "_A8_DATA_";
                "_A8_DATA_TC3" := "_A8_DATA_";
                "_A8_DATA_TC4" := "_A8_DATA_";
                "_A8_DATA_TC5" := "_A8_DATA_";
                "_A8_DATA_TC6" := "_A8_DATA_";

                "_A9_DATA_TC0" := "_A9_DATA_";
                "_A9_DATA_TC1" := "_A9_DATA_";
                "_A9_DATA_TC2" := "_A9_DATA_";
                "_A9_DATA_TC3" := "_A9_DATA_";
                "_A9_DATA_TC4" := "_A9_DATA_";
                "_A9_DATA_TC5" := "_A9_DATA_";
                "_A9_DATA_TC6" := "_A9_DATA_";
        }
#else // __NO_VTC
        section_layout :tc0:linear
        {
#if (XVWBUF>0)
                group (align = 4) reserved "_xvwbuffer_" (size=XVWBUF, attributes=rw);
                "_lc_ub_xvwbuffer" = "_lc_ub__xvwbuffer_";
                "_lc_ue_xvwbuffer" = "_lc_ue__xvwbuffer_";
#endif

#ifndef __REDEFINE_BASE_ADDRESS_GROUPS
#undef __A0_START_ADDRESS
#ifdef __TC0_A0_START
#define __A0_START_ADDRESS __TC0_A0_START
#else
#ifdef A0_START_ADDRESS
#define __A0_START_ADDRESS A0_START_ADDRESS
#endif
#endif
#undef __A1_START_ADDRESS
#ifdef __TC0_A1_START
#define __A1_START_ADDRESS __TC0_A1_START
#else
#ifdef A1_START_ADDRESS
#define __A1_START_ADDRESS A1_START_ADDRESS
#endif
#endif
#undef __A8_START_ADDRESS
#ifdef __TC0_A8_START
#define __A8_START_ADDRESS __TC0_A8_START
#else
#ifdef A8_START_ADDRESS
#define __A8_START_ADDRESS A8_START_ADDRESS
#endif
#endif
#undef __A9_START_ADDRESS
#ifdef __TC0_A9_START
#define __A9_START_ADDRESS __TC0_A9_START
#else
#ifdef A9_START_ADDRESS
#define __A9_START_ADDRESS A9_START_ADDRESS
#endif
#endif
#include        "base_address_groups.lsl"
#endif
        }
        section_layout :tc1:linear
        {
#if (XVWBUF>0)
                group (align = 4) reserved "_xvwbuffer_" (size=XVWBUF, attributes=rw);
                "_lc_ub_xvwbuffer" = "_lc_ub__xvwbuffer_";
                "_lc_ue_xvwbuffer" = "_lc_ue__xvwbuffer_";
#endif

#ifndef __REDEFINE_BASE_ADDRESS_GROUPS
#undef __A0_START_ADDRESS
#ifdef __TC1_A0_START
#define __A0_START_ADDRESS __TC1_A0_START
#else
#ifdef A0_START_ADDRESS
#define __A0_START_ADDRESS A0_START_ADDRESS
#endif
#endif
#undef __A1_START_ADDRESS
#ifdef __TC1_A1_START
#define __A1_START_ADDRESS __TC1_A1_START
#else
#ifdef A1_START_ADDRESS
#define __A1_START_ADDRESS A1_START_ADDRESS
#endif
#endif
#undef __A8_START_ADDRESS
#ifdef __TC1_A8_START
#define __A8_START_ADDRESS __TC1_A8_START
#else
#ifdef A8_START_ADDRESS
#define __A8_START_ADDRESS A8_START_ADDRESS
#endif
#endif
#undef __A9_START_ADDRESS
#ifdef __TC1_A9_START
#define __A9_START_ADDRESS __TC1_A9_START
#else
#ifdef A9_START_ADDRESS
#define __A9_START_ADDRESS A9_START_ADDRESS
#endif
#endif
#include        "base_address_groups.lsl"
#endif
        }
        section_layout :tc2:linear
        {
#if (XVWBUF>0)
                group (align = 4) reserved "_xvwbuffer_" (size=XVWBUF, attributes=rw);
                "_lc_ub_xvwbuffer" = "_lc_ub__xvwbuffer_";
                "_lc_ue_xvwbuffer" = "_lc_ue__xvwbuffer_";
#endif

#ifndef __REDEFINE_BASE_ADDRESS_GROUPS
#undef __A0_START_ADDRESS
#ifdef __TC2_A0_START
#define __A0_START_ADDRESS __TC2_A0_START
#else
#ifdef A0_START_ADDRESS
#define __A0_START_ADDRESS A0_START_ADDRESS
#endif
#endif
#undef __A1_START_ADDRESS
#ifdef __TC2_A1_START
#define __A1_START_ADDRESS __TC2_A1_START
#else
#ifdef A1_START_ADDRESS
#define __A1_START_ADDRESS A1_START_ADDRESS
#endif
#endif
#undef __A8_START_ADDRESS
#ifdef __TC2_A8_START
#define __A8_START_ADDRESS __TC2_A8_START
#else
#ifdef A8_START_ADDRESS
#define __A8_START_ADDRESS A8_START_ADDRESS
#endif
#endif
#undef __A9_START_ADDRESS
#ifdef __TC2_A9_START
#define __A9_START_ADDRESS __TC2_A9_START
#else
#ifdef A9_START_ADDRESS
#define __A9_START_ADDRESS A9_START_ADDRESS
#endif
#endif
#include        "base_address_groups.lsl"
#endif
        }
        section_layout :tc3:linear
        {
#if (XVWBUF>0)
                group (align = 4) reserved "_xvwbuffer_" (size=XVWBUF, attributes=rw);
                "_lc_ub_xvwbuffer" = "_lc_ub__xvwbuffer_";
                "_lc_ue_xvwbuffer" = "_lc_ue__xvwbuffer_";
#endif

#ifndef __REDEFINE_BASE_ADDRESS_GROUPS
#undef __A0_START_ADDRESS
#ifdef __TC3_A0_START
#define __A0_START_ADDRESS __TC3_A0_START
#else
#ifdef A0_START_ADDRESS
#define __A0_START_ADDRESS A0_START_ADDRESS
#endif
#endif
#undef __A1_START_ADDRESS
#ifdef __TC3_A1_START
#define __A1_START_ADDRESS __TC3_A1_START
#else
#ifdef A1_START_ADDRESS
#define __A1_START_ADDRESS A1_START_ADDRESS
#endif
#endif
#undef __A8_START_ADDRESS
#ifdef __TC3_A8_START
#define __A8_START_ADDRESS __TC3_A8_START
#else
#ifdef A8_START_ADDRESS
#define __A8_START_ADDRESS A8_START_ADDRESS
#endif
#endif
#undef __A9_START_ADDRESS
#ifdef __TC3_A9_START
#define __A9_START_ADDRESS __TC3_A9_START
#else
#ifdef A9_START_ADDRESS
#define __A9_START_ADDRESS A9_START_ADDRESS
#endif
#endif
#include        "base_address_groups.lsl"
#endif
        }
        section_layout :tc4:linear
        {
#if (XVWBUF>0)
                group (align = 4) reserved "_xvwbuffer_" (size=XVWBUF, attributes=rw);
                "_lc_ub_xvwbuffer" = "_lc_ub__xvwbuffer_";
                "_lc_ue_xvwbuffer" = "_lc_ue__xvwbuffer_";
#endif

#ifndef __REDEFINE_BASE_ADDRESS_GROUPS
#undef __A0_START_ADDRESS
#ifdef __TC4_A0_START
#define __A0_START_ADDRESS __TC4_A0_START
#else
#ifdef A0_START_ADDRESS
#define __A0_START_ADDRESS A0_START_ADDRESS
#endif
#endif
#undef __A1_START_ADDRESS
#ifdef __TC4_A1_START
#define __A1_START_ADDRESS __TC4_A1_START
#else
#ifdef A1_START_ADDRESS
#define __A1_START_ADDRESS A1_START_ADDRESS
#endif
#endif
#undef __A8_START_ADDRESS
#ifdef __TC4_A8_START
#define __A8_START_ADDRESS __TC4_A8_START
#else
#ifdef A8_START_ADDRESS
#define __A8_START_ADDRESS A8_START_ADDRESS
#endif
#endif
#undef __A9_START_ADDRESS
#ifdef __TC4_A9_START
#define __A9_START_ADDRESS __TC4_A9_START
#else
#ifdef A9_START_ADDRESS
#define __A9_START_ADDRESS A9_START_ADDRESS
#endif
#endif
#include        "base_address_groups.lsl"
#endif
        }
        section_layout :tc5:linear
        {
#if (XVWBUF>0)
                group (align = 4) reserved "_xvwbuffer_" (size=XVWBUF, attributes=rw);
                "_lc_ub_xvwbuffer" = "_lc_ub__xvwbuffer_";
                "_lc_ue_xvwbuffer" = "_lc_ue__xvwbuffer_";
#endif

#ifndef __REDEFINE_BASE_ADDRESS_GROUPS
#undef __A0_START_ADDRESS
#ifdef __TC5_A0_START
#define __A0_START_ADDRESS __TC5_A0_START
#else
#ifdef A0_START_ADDRESS
#define __A0_START_ADDRESS A0_START_ADDRESS
#endif
#endif
#undef __A1_START_ADDRESS
#ifdef __TC5_A1_START
#define __A1_START_ADDRESS __TC5_A1_START
#else
#ifdef A1_START_ADDRESS
#define __A1_START_ADDRESS A1_START_ADDRESS
#endif
#endif
#undef __A8_START_ADDRESS
#ifdef __TC5_A8_START
#define __A8_START_ADDRESS __TC5_A8_START
#else
#ifdef A8_START_ADDRESS
#define __A8_START_ADDRESS A8_START_ADDRESS
#endif
#endif
#undef __A9_START_ADDRESS
#ifdef __TC5_A9_START
#define __A9_START_ADDRESS __TC5_A9_START
#else
#ifdef A9_START_ADDRESS
#define __A9_START_ADDRESS A9_START_ADDRESS
#endif
#endif
#include        "base_address_groups.lsl"
#endif
        }
        section_layout :tc6:linear
        {
#if (XVWBUF>0)
                group (align = 4) reserved "_xvwbuffer_" (size=XVWBUF, attributes=rw);
                "_lc_ub_xvwbuffer" = "_lc_ub__xvwbuffer_";
                "_lc_ue_xvwbuffer" = "_lc_ue__xvwbuffer_";
#endif

#ifndef __REDEFINE_BASE_ADDRESS_GROUPS
#undef __A0_START_ADDRESS
#ifdef __TC6_A0_START
#define __A0_START_ADDRESS __TC6_A0_START
#else
#ifdef A0_START_ADDRESS
#define __A0_START_ADDRESS A0_START_ADDRESS
#endif
#endif
#undef __A1_START_ADDRESS
#ifdef __TC6_A1_START
#define __A1_START_ADDRESS __TC6_A1_START
#else
#ifdef A1_START_ADDRESS
#define __A1_START_ADDRESS A1_START_ADDRESS
#endif
#endif
#undef __A8_START_ADDRESS
#ifdef __TC6_A8_START
#define __A8_START_ADDRESS __TC6_A8_START
#else
#ifdef A8_START_ADDRESS
#define __A8_START_ADDRESS A8_START_ADDRESS
#endif
#endif
#undef __A9_START_ADDRESS
#ifdef __TC6_A9_START
#define __A9_START_ADDRESS __TC6_A9_START
#else
#ifdef A9_START_ADDRESS
#define __A9_START_ADDRESS A9_START_ADDRESS
#endif
#endif
#include        "base_address_groups.lsl"
#endif
        }
#endif // __NO_VTC

#ifndef __NO_VTC
        section_setup :vtc:linear
        {
                // interrupt vector tables
                vector_table "inttab0" (vector_size=(INTTAB0_ENTRY_SIZE), size=256, run_addr=(INTTAB0), vector_prefix=INTTAB0_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab0vm0" (vector_size=(INTTAB0VM0_ENTRY_SIZE), size=256, run_addr=(INTTAB0VM0), vector_prefix=INTTAB0VM0_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab0vm1" (vector_size=(INTTAB0VM1_ENTRY_SIZE), size=256, run_addr=(INTTAB0VM1), vector_prefix=INTTAB0VM1_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab0vm2" (vector_size=(INTTAB0VM2_ENTRY_SIZE), size=256, run_addr=(INTTAB0VM2), vector_prefix=INTTAB0VM2_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab0vm3" (vector_size=(INTTAB0VM3_ENTRY_SIZE), size=256, run_addr=(INTTAB0VM3), vector_prefix=INTTAB0VM3_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab0vm4" (vector_size=(INTTAB0VM4_ENTRY_SIZE), size=256, run_addr=(INTTAB0VM4), vector_prefix=INTTAB0VM4_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab0vm5" (vector_size=(INTTAB0VM5_ENTRY_SIZE), size=256, run_addr=(INTTAB0VM5), vector_prefix=INTTAB0VM5_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab0vm6" (vector_size=(INTTAB0VM6_ENTRY_SIZE), size=256, run_addr=(INTTAB0VM6), vector_prefix=INTTAB0VM6_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab0vm7" (vector_size=(INTTAB0VM7_ENTRY_SIZE), size=256, run_addr=(INTTAB0VM7), vector_prefix=INTTAB0VM7_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1" (vector_size=(INTTAB1_ENTRY_SIZE), size=256, run_addr=(INTTAB1), vector_prefix=INTTAB1_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1vm0" (vector_size=(INTTAB1VM0_ENTRY_SIZE), size=256, run_addr=(INTTAB1VM0), vector_prefix=INTTAB1VM0_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1vm1" (vector_size=(INTTAB1VM1_ENTRY_SIZE), size=256, run_addr=(INTTAB1VM1), vector_prefix=INTTAB1VM1_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1vm2" (vector_size=(INTTAB1VM2_ENTRY_SIZE), size=256, run_addr=(INTTAB1VM2), vector_prefix=INTTAB1VM2_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1vm3" (vector_size=(INTTAB1VM3_ENTRY_SIZE), size=256, run_addr=(INTTAB1VM3), vector_prefix=INTTAB1VM3_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1vm4" (vector_size=(INTTAB1VM4_ENTRY_SIZE), size=256, run_addr=(INTTAB1VM4), vector_prefix=INTTAB1VM4_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1vm5" (vector_size=(INTTAB1VM5_ENTRY_SIZE), size=256, run_addr=(INTTAB1VM5), vector_prefix=INTTAB1VM5_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1vm6" (vector_size=(INTTAB1VM6_ENTRY_SIZE), size=256, run_addr=(INTTAB1VM6), vector_prefix=INTTAB1VM6_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1vm7" (vector_size=(INTTAB1VM7_ENTRY_SIZE), size=256, run_addr=(INTTAB1VM7), vector_prefix=INTTAB1VM7_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2" (vector_size=(INTTAB2_ENTRY_SIZE), size=256, run_addr=(INTTAB2), vector_prefix=INTTAB2_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2vm0" (vector_size=(INTTAB2VM0_ENTRY_SIZE), size=256, run_addr=(INTTAB2VM0), vector_prefix=INTTAB2VM0_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2vm1" (vector_size=(INTTAB2VM1_ENTRY_SIZE), size=256, run_addr=(INTTAB2VM1), vector_prefix=INTTAB2VM1_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2vm2" (vector_size=(INTTAB2VM2_ENTRY_SIZE), size=256, run_addr=(INTTAB2VM2), vector_prefix=INTTAB2VM2_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2vm3" (vector_size=(INTTAB2VM3_ENTRY_SIZE), size=256, run_addr=(INTTAB2VM3), vector_prefix=INTTAB2VM3_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2vm4" (vector_size=(INTTAB2VM4_ENTRY_SIZE), size=256, run_addr=(INTTAB2VM4), vector_prefix=INTTAB2VM4_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2vm5" (vector_size=(INTTAB2VM5_ENTRY_SIZE), size=256, run_addr=(INTTAB2VM5), vector_prefix=INTTAB2VM5_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2vm6" (vector_size=(INTTAB2VM6_ENTRY_SIZE), size=256, run_addr=(INTTAB2VM6), vector_prefix=INTTAB2VM6_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2vm7" (vector_size=(INTTAB2VM7_ENTRY_SIZE), size=256, run_addr=(INTTAB2VM7), vector_prefix=INTTAB2VM7_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3" (vector_size=(INTTAB3_ENTRY_SIZE), size=256, run_addr=(INTTAB3), vector_prefix=INTTAB3_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3vm0" (vector_size=(INTTAB3VM0_ENTRY_SIZE), size=256, run_addr=(INTTAB3VM0), vector_prefix=INTTAB3VM0_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3vm1" (vector_size=(INTTAB3VM1_ENTRY_SIZE), size=256, run_addr=(INTTAB3VM1), vector_prefix=INTTAB3VM1_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3vm2" (vector_size=(INTTAB3VM2_ENTRY_SIZE), size=256, run_addr=(INTTAB3VM2), vector_prefix=INTTAB3VM2_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3vm3" (vector_size=(INTTAB3VM3_ENTRY_SIZE), size=256, run_addr=(INTTAB3VM3), vector_prefix=INTTAB3VM3_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3vm4" (vector_size=(INTTAB3VM4_ENTRY_SIZE), size=256, run_addr=(INTTAB3VM4), vector_prefix=INTTAB3VM4_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3vm5" (vector_size=(INTTAB3VM5_ENTRY_SIZE), size=256, run_addr=(INTTAB3VM5), vector_prefix=INTTAB3VM5_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3vm6" (vector_size=(INTTAB3VM6_ENTRY_SIZE), size=256, run_addr=(INTTAB3VM6), vector_prefix=INTTAB3VM6_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3vm7" (vector_size=(INTTAB3VM7_ENTRY_SIZE), size=256, run_addr=(INTTAB3VM7), vector_prefix=INTTAB3VM7_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4" (vector_size=(INTTAB4_ENTRY_SIZE), size=256, run_addr=(INTTAB4), vector_prefix=INTTAB4_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4vm0" (vector_size=(INTTAB4VM0_ENTRY_SIZE), size=256, run_addr=(INTTAB4VM0), vector_prefix=INTTAB4VM0_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4vm1" (vector_size=(INTTAB4VM1_ENTRY_SIZE), size=256, run_addr=(INTTAB4VM1), vector_prefix=INTTAB4VM1_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4vm2" (vector_size=(INTTAB4VM2_ENTRY_SIZE), size=256, run_addr=(INTTAB4VM2), vector_prefix=INTTAB4VM2_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4vm3" (vector_size=(INTTAB4VM3_ENTRY_SIZE), size=256, run_addr=(INTTAB4VM3), vector_prefix=INTTAB4VM3_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4vm4" (vector_size=(INTTAB4VM4_ENTRY_SIZE), size=256, run_addr=(INTTAB4VM4), vector_prefix=INTTAB4VM4_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4vm5" (vector_size=(INTTAB4VM5_ENTRY_SIZE), size=256, run_addr=(INTTAB4VM5), vector_prefix=INTTAB4VM5_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4vm6" (vector_size=(INTTAB4VM6_ENTRY_SIZE), size=256, run_addr=(INTTAB4VM6), vector_prefix=INTTAB4VM6_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4vm7" (vector_size=(INTTAB4VM7_ENTRY_SIZE), size=256, run_addr=(INTTAB4VM7), vector_prefix=INTTAB4VM7_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5" (vector_size=(INTTAB5_ENTRY_SIZE), size=256, run_addr=(INTTAB5), vector_prefix=INTTAB5_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5vm0" (vector_size=(INTTAB5VM0_ENTRY_SIZE), size=256, run_addr=(INTTAB5VM0), vector_prefix=INTTAB5VM0_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5vm1" (vector_size=(INTTAB5VM1_ENTRY_SIZE), size=256, run_addr=(INTTAB5VM1), vector_prefix=INTTAB5VM1_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5vm2" (vector_size=(INTTAB5VM2_ENTRY_SIZE), size=256, run_addr=(INTTAB5VM2), vector_prefix=INTTAB5VM2_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5vm3" (vector_size=(INTTAB5VM3_ENTRY_SIZE), size=256, run_addr=(INTTAB5VM3), vector_prefix=INTTAB5VM3_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5vm4" (vector_size=(INTTAB5VM4_ENTRY_SIZE), size=256, run_addr=(INTTAB5VM4), vector_prefix=INTTAB5VM4_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5vm5" (vector_size=(INTTAB5VM5_ENTRY_SIZE), size=256, run_addr=(INTTAB5VM5), vector_prefix=INTTAB5VM5_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5vm6" (vector_size=(INTTAB5VM6_ENTRY_SIZE), size=256, run_addr=(INTTAB5VM6), vector_prefix=INTTAB5VM6_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5vm7" (vector_size=(INTTAB5VM7_ENTRY_SIZE), size=256, run_addr=(INTTAB5VM7), vector_prefix=INTTAB5VM7_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6" (vector_size=(INTTAB6_ENTRY_SIZE), size=256, run_addr=(INTTAB6), vector_prefix=INTTAB6_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6vm0" (vector_size=(INTTAB6VM0_ENTRY_SIZE), size=256, run_addr=(INTTAB6VM0), vector_prefix=INTTAB6VM0_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6vm1" (vector_size=(INTTAB6VM1_ENTRY_SIZE), size=256, run_addr=(INTTAB6VM1), vector_prefix=INTTAB6VM1_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6vm2" (vector_size=(INTTAB6VM2_ENTRY_SIZE), size=256, run_addr=(INTTAB6VM2), vector_prefix=INTTAB6VM2_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6vm3" (vector_size=(INTTAB6VM3_ENTRY_SIZE), size=256, run_addr=(INTTAB6VM3), vector_prefix=INTTAB6VM3_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6vm4" (vector_size=(INTTAB6VM4_ENTRY_SIZE), size=256, run_addr=(INTTAB6VM4), vector_prefix=INTTAB6VM4_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6vm5" (vector_size=(INTTAB6VM5_ENTRY_SIZE), size=256, run_addr=(INTTAB6VM5), vector_prefix=INTTAB6VM5_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6vm6" (vector_size=(INTTAB6VM6_ENTRY_SIZE), size=256, run_addr=(INTTAB6VM6), vector_prefix=INTTAB6VM6_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6vm7" (vector_size=(INTTAB6VM7_ENTRY_SIZE), size=256, run_addr=(INTTAB6VM7), vector_prefix=INTTAB6VM7_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                // trap vector tables
                vector_table "traptab0" (vector_size=32, size=8, run_addr=(TRAPTAB0), vector_prefix=".traptab0.trapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "traptab1" (vector_size=32, size=8, run_addr=(TRAPTAB1), vector_prefix=".traptab1.trapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "traptab2" (vector_size=32, size=8, run_addr=(TRAPTAB2), vector_prefix=".traptab2.trapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "traptab3" (vector_size=32, size=8, run_addr=(TRAPTAB3), vector_prefix=".traptab3.trapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "traptab4" (vector_size=32, size=8, run_addr=(TRAPTAB4), vector_prefix=".traptab4.trapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "traptab5" (vector_size=32, size=8, run_addr=(TRAPTAB5), vector_prefix=".traptab5.trapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "traptab6" (vector_size=32, size=8, run_addr=(TRAPTAB6), vector_prefix=".traptab6.trapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                // hypervisor trap tables
                vector_table "hvtraptab0" (vector_size=32, size=6, run_addr=(HVTRAPTAB0), vector_prefix=".hvtraptab0.hvtrapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "hvtraptab1" (vector_size=32, size=6, run_addr=(HVTRAPTAB1), vector_prefix=".hvtraptab1.hvtrapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "hvtraptab2" (vector_size=32, size=6, run_addr=(HVTRAPTAB2), vector_prefix=".hvtraptab2.hvtrapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "hvtraptab3" (vector_size=32, size=6, run_addr=(HVTRAPTAB3), vector_prefix=".hvtraptab3.hvtrapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "hvtraptab4" (vector_size=32, size=6, run_addr=(HVTRAPTAB4), vector_prefix=".hvtraptab4.hvtrapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "hvtraptab5" (vector_size=32, size=6, run_addr=(HVTRAPTAB5), vector_prefix=".hvtraptab5.hvtrapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "hvtraptab6" (vector_size=32, size=6, run_addr=(HVTRAPTAB6), vector_prefix=".hvtraptab6.hvtrapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
        }
        section_layout :vtc:linear
        {
                "_lc_u_int_tab_tc0" = (INTTAB0);
                "_lc_u_int_tab_tc1" = (INTTAB1);
                "_lc_u_int_tab_tc2" = (INTTAB2);
                "_lc_u_int_tab_tc3" = (INTTAB3);
                "_lc_u_int_tab_tc4" = (INTTAB4);
                "_lc_u_int_tab_tc5" = (INTTAB5);
                "_lc_u_int_tab_tc6" = (INTTAB6);
                "_lc_u_int_tab" = "_lc_u_int_tab_tc0"; /* common cstart interface for first or single core */

                // interrupt vector tables
                "_lc_u_int_tab_tc0_vm0" = (INTTAB0VM0);
                "_lc_u_int_tab_tc0_vm1" = (INTTAB0VM1);
                "_lc_u_int_tab_tc0_vm2" = (INTTAB0VM2);
                "_lc_u_int_tab_tc0_vm3" = (INTTAB0VM3);
                "_lc_u_int_tab_tc0_vm4" = (INTTAB0VM4);
                "_lc_u_int_tab_tc0_vm5" = (INTTAB0VM5);
                "_lc_u_int_tab_tc0_vm6" = (INTTAB0VM6);
                "_lc_u_int_tab_tc0_vm7" = (INTTAB0VM7);
                "_lc_u_int_tab_tc1_vm0" = (INTTAB1VM0);
                "_lc_u_int_tab_tc1_vm1" = (INTTAB1VM1);
                "_lc_u_int_tab_tc1_vm2" = (INTTAB1VM2);
                "_lc_u_int_tab_tc1_vm3" = (INTTAB1VM3);
                "_lc_u_int_tab_tc1_vm4" = (INTTAB1VM4);
                "_lc_u_int_tab_tc1_vm5" = (INTTAB1VM5);
                "_lc_u_int_tab_tc1_vm6" = (INTTAB1VM6);
                "_lc_u_int_tab_tc1_vm7" = (INTTAB1VM7);
                "_lc_u_int_tab_tc2_vm0" = (INTTAB2VM0);
                "_lc_u_int_tab_tc2_vm1" = (INTTAB2VM1);
                "_lc_u_int_tab_tc2_vm2" = (INTTAB2VM2);
                "_lc_u_int_tab_tc2_vm3" = (INTTAB2VM3);
                "_lc_u_int_tab_tc2_vm4" = (INTTAB2VM4);
                "_lc_u_int_tab_tc2_vm5" = (INTTAB2VM5);
                "_lc_u_int_tab_tc2_vm6" = (INTTAB2VM6);
                "_lc_u_int_tab_tc2_vm7" = (INTTAB2VM7);
                "_lc_u_int_tab_tc3_vm0" = (INTTAB3VM0);
                "_lc_u_int_tab_tc3_vm1" = (INTTAB3VM1);
                "_lc_u_int_tab_tc3_vm2" = (INTTAB3VM2);
                "_lc_u_int_tab_tc3_vm3" = (INTTAB3VM3);
                "_lc_u_int_tab_tc3_vm4" = (INTTAB3VM4);
                "_lc_u_int_tab_tc3_vm5" = (INTTAB3VM5);
                "_lc_u_int_tab_tc3_vm6" = (INTTAB3VM6);
                "_lc_u_int_tab_tc3_vm7" = (INTTAB3VM7);
                "_lc_u_int_tab_tc4_vm0" = (INTTAB4VM0);
                "_lc_u_int_tab_tc4_vm1" = (INTTAB4VM1);
                "_lc_u_int_tab_tc4_vm2" = (INTTAB4VM2);
                "_lc_u_int_tab_tc4_vm3" = (INTTAB4VM3);
                "_lc_u_int_tab_tc4_vm4" = (INTTAB4VM4);
                "_lc_u_int_tab_tc4_vm5" = (INTTAB4VM5);
                "_lc_u_int_tab_tc4_vm6" = (INTTAB4VM6);
                "_lc_u_int_tab_tc4_vm7" = (INTTAB4VM7);
                "_lc_u_int_tab_tc5_vm0" = (INTTAB5VM0);
                "_lc_u_int_tab_tc5_vm1" = (INTTAB5VM1);
                "_lc_u_int_tab_tc5_vm2" = (INTTAB5VM2);
                "_lc_u_int_tab_tc5_vm3" = (INTTAB5VM3);
                "_lc_u_int_tab_tc5_vm4" = (INTTAB5VM4);
                "_lc_u_int_tab_tc5_vm5" = (INTTAB5VM5);
                "_lc_u_int_tab_tc5_vm6" = (INTTAB5VM6);
                "_lc_u_int_tab_tc5_vm7" = (INTTAB5VM7);
                "_lc_u_int_tab_tc6_vm0" = (INTTAB6VM0);
                "_lc_u_int_tab_tc6_vm1" = (INTTAB6VM1);
                "_lc_u_int_tab_tc6_vm2" = (INTTAB6VM2);
                "_lc_u_int_tab_tc6_vm3" = (INTTAB6VM3);
                "_lc_u_int_tab_tc6_vm4" = (INTTAB6VM4);
                "_lc_u_int_tab_tc6_vm5" = (INTTAB6VM5);
                "_lc_u_int_tab_tc6_vm6" = (INTTAB6VM6);
                "_lc_u_int_tab_tc6_vm7" = (INTTAB6VM7);

                "_lc_u_trap_tab_tc0" = (TRAPTAB0);
                "_lc_u_trap_tab_tc1" = (TRAPTAB1);
                "_lc_u_trap_tab_tc2" = (TRAPTAB2);
                "_lc_u_trap_tab_tc3" = (TRAPTAB3);
                "_lc_u_trap_tab_tc4" = (TRAPTAB4);
                "_lc_u_trap_tab_tc5" = (TRAPTAB5);
                "_lc_u_trap_tab_tc6" = (TRAPTAB6);
                "_lc_u_trap_tab" = "_lc_u_trap_tab_tc0"; /* common cstart interface for first or single core */

                "_lc_u_hvtrap_tab_tc0" = (HVTRAPTAB0);
                "_lc_u_hvtrap_tab_tc1" = (HVTRAPTAB1);
                "_lc_u_hvtrap_tab_tc2" = (HVTRAPTAB2);
                "_lc_u_hvtrap_tab_tc3" = (HVTRAPTAB3);
                "_lc_u_hvtrap_tab_tc4" = (HVTRAPTAB4);
                "_lc_u_hvtrap_tab_tc5" = (HVTRAPTAB5);
                "_lc_u_hvtrap_tab_tc6" = (HVTRAPTAB6);
        }
#else
        section_setup :tc0:linear
        {
                // interrupt vector table(s)
                vector_table "inttab0" (vector_size=(INTTAB0_ENTRY_SIZE), size=256, run_addr=(INTTAB0), vector_prefix=INTTAB0_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab0vm0" (vector_size=(INTTAB0VM0_ENTRY_SIZE), size=256, run_addr=(INTTAB0VM0), vector_prefix=INTTAB0VM0_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab0vm1" (vector_size=(INTTAB0VM1_ENTRY_SIZE), size=256, run_addr=(INTTAB0VM1), vector_prefix=INTTAB0VM1_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab0vm2" (vector_size=(INTTAB0VM2_ENTRY_SIZE), size=256, run_addr=(INTTAB0VM2), vector_prefix=INTTAB0VM2_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab0vm3" (vector_size=(INTTAB0VM3_ENTRY_SIZE), size=256, run_addr=(INTTAB0VM3), vector_prefix=INTTAB0VM3_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab0vm4" (vector_size=(INTTAB0VM4_ENTRY_SIZE), size=256, run_addr=(INTTAB0VM4), vector_prefix=INTTAB0VM4_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab0vm5" (vector_size=(INTTAB0VM5_ENTRY_SIZE), size=256, run_addr=(INTTAB0VM5), vector_prefix=INTTAB0VM5_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab0vm6" (vector_size=(INTTAB0VM6_ENTRY_SIZE), size=256, run_addr=(INTTAB0VM6), vector_prefix=INTTAB0VM6_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab0vm7" (vector_size=(INTTAB0VM7_ENTRY_SIZE), size=256, run_addr=(INTTAB0VM7), vector_prefix=INTTAB0VM7_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                // trap vector table
                vector_table "traptab0" (vector_size=32, size=8, run_addr=(TRAPTAB0), vector_prefix=".traptab0.trapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                // hypervisor trap table
                vector_table "hvtraptab0" (vector_size=32, size=6, run_addr=(HVTRAPTAB0), vector_prefix=".hvtraptab0.hvtrapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
        }
        section_layout :tc0:linear
        {
                "_lc_u_int_tab" = (INTTAB0);
                "_lc_u_int_tab_tc0" = (INTTAB0);

                "_lc_u_trap_tab" = (TRAPTAB0);
                "_lc_u_trap_tab_tc0" = (TRAPTAB0);

                "_lc_u_hvtrap_tab" = (HVTRAPTAB0);
                "_lc_u_hvtrap_tab_tc0" = (HVTRAPTAB0);
        }
        section_setup :tc1:linear
        {
                // interrupt vector table(s)
                vector_table "inttab1" (vector_size=(INTTAB1_ENTRY_SIZE), size=256, run_addr=(INTTAB1), vector_prefix=INTTAB1_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1vm0" (vector_size=(INTTAB1VM0_ENTRY_SIZE), size=256, run_addr=(INTTAB1VM0), vector_prefix=INTTAB1VM0_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1vm1" (vector_size=(INTTAB1VM1_ENTRY_SIZE), size=256, run_addr=(INTTAB1VM1), vector_prefix=INTTAB1VM1_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1vm2" (vector_size=(INTTAB1VM2_ENTRY_SIZE), size=256, run_addr=(INTTAB1VM2), vector_prefix=INTTAB1VM2_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1vm3" (vector_size=(INTTAB1VM3_ENTRY_SIZE), size=256, run_addr=(INTTAB1VM3), vector_prefix=INTTAB1VM3_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1vm4" (vector_size=(INTTAB1VM4_ENTRY_SIZE), size=256, run_addr=(INTTAB1VM4), vector_prefix=INTTAB1VM4_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1vm5" (vector_size=(INTTAB1VM5_ENTRY_SIZE), size=256, run_addr=(INTTAB1VM5), vector_prefix=INTTAB1VM5_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1vm6" (vector_size=(INTTAB1VM6_ENTRY_SIZE), size=256, run_addr=(INTTAB1VM6), vector_prefix=INTTAB1VM6_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab1vm7" (vector_size=(INTTAB1VM7_ENTRY_SIZE), size=256, run_addr=(INTTAB1VM7), vector_prefix=INTTAB1VM7_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                // trap vector table
                vector_table "traptab1" (vector_size=32, size=8, run_addr=(TRAPTAB1), vector_prefix=".traptab1.trapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                // hypervisor trap table
                vector_table "hvtraptab1" (vector_size=32, size=6, run_addr=(HVTRAPTAB1), vector_prefix=".hvtraptab1.hvtrapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
        }
        section_layout :tc1:linear
        {
                "_lc_u_int_tab" = (INTTAB1);
                "_lc_u_int_tab_tc1" = (INTTAB1);

                "_lc_u_trap_tab" = (TRAPTAB1);
                "_lc_u_trap_tab_tc1" = (TRAPTAB1);

                "_lc_u_hvtrap_tab" = (HVTRAPTAB1);
                "_lc_u_hvtrap_tab_tc1" = (HVTRAPTAB1);
        }
        section_setup :tc2:linear
        {
                // interrupt vector table(s)
                vector_table "inttab2" (vector_size=(INTTAB2_ENTRY_SIZE), size=256, run_addr=(INTTAB2), vector_prefix=INTTAB2_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2vm0" (vector_size=(INTTAB2VM0_ENTRY_SIZE), size=256, run_addr=(INTTAB2VM0), vector_prefix=INTTAB2VM0_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2vm1" (vector_size=(INTTAB2VM1_ENTRY_SIZE), size=256, run_addr=(INTTAB2VM1), vector_prefix=INTTAB2VM1_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2vm2" (vector_size=(INTTAB2VM2_ENTRY_SIZE), size=256, run_addr=(INTTAB2VM2), vector_prefix=INTTAB2VM2_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2vm3" (vector_size=(INTTAB2VM3_ENTRY_SIZE), size=256, run_addr=(INTTAB2VM3), vector_prefix=INTTAB2VM3_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2vm4" (vector_size=(INTTAB2VM4_ENTRY_SIZE), size=256, run_addr=(INTTAB2VM4), vector_prefix=INTTAB2VM4_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2vm5" (vector_size=(INTTAB2VM5_ENTRY_SIZE), size=256, run_addr=(INTTAB2VM5), vector_prefix=INTTAB2VM5_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2vm6" (vector_size=(INTTAB2VM6_ENTRY_SIZE), size=256, run_addr=(INTTAB2VM6), vector_prefix=INTTAB2VM6_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab2vm7" (vector_size=(INTTAB2VM7_ENTRY_SIZE), size=256, run_addr=(INTTAB2VM7), vector_prefix=INTTAB2VM7_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                // trap vector table
                vector_table "traptab2" (vector_size=32, size=8, run_addr=(TRAPTAB2), vector_prefix=".traptab2.trapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                // hypervisor trap table
                vector_table "hvtraptab2" (vector_size=32, size=6, run_addr=(HVTRAPTAB2), vector_prefix=".hvtraptab2.hvtrapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
        }
        section_layout :tc2:linear
        {
                "_lc_u_int_tab" = (INTTAB2);
                "_lc_u_int_tab_tc2" = (INTTAB2);

                "_lc_u_trap_tab" = (TRAPTAB2);
                "_lc_u_trap_tab_tc2" = (TRAPTAB2);

                "_lc_u_hvtrap_tab" = (HVTRAPTAB2);
                "_lc_u_hvtrap_tab_tc2" = (HVTRAPTAB2);
        }
        section_setup :tc3:linear
        {
                // interrupt vector table(s)
                vector_table "inttab3" (vector_size=(INTTAB3_ENTRY_SIZE), size=256, run_addr=(INTTAB3), vector_prefix=INTTAB3_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3vm0" (vector_size=(INTTAB3VM0_ENTRY_SIZE), size=256, run_addr=(INTTAB3VM0), vector_prefix=INTTAB3VM0_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3vm1" (vector_size=(INTTAB3VM1_ENTRY_SIZE), size=256, run_addr=(INTTAB3VM1), vector_prefix=INTTAB3VM1_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3vm2" (vector_size=(INTTAB3VM2_ENTRY_SIZE), size=256, run_addr=(INTTAB3VM2), vector_prefix=INTTAB3VM2_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3vm3" (vector_size=(INTTAB3VM3_ENTRY_SIZE), size=256, run_addr=(INTTAB3VM3), vector_prefix=INTTAB3VM3_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3vm4" (vector_size=(INTTAB3VM4_ENTRY_SIZE), size=256, run_addr=(INTTAB3VM4), vector_prefix=INTTAB3VM4_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3vm5" (vector_size=(INTTAB3VM5_ENTRY_SIZE), size=256, run_addr=(INTTAB3VM5), vector_prefix=INTTAB3VM5_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3vm6" (vector_size=(INTTAB3VM6_ENTRY_SIZE), size=256, run_addr=(INTTAB3VM6), vector_prefix=INTTAB3VM6_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab3vm7" (vector_size=(INTTAB3VM7_ENTRY_SIZE), size=256, run_addr=(INTTAB3VM7), vector_prefix=INTTAB3VM7_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                // trap vector table
                vector_table "traptab3" (vector_size=32, size=8, run_addr=(TRAPTAB3), vector_prefix=".traptab3.trapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                // hypervisor trap table
                vector_table "hvtraptab3" (vector_size=32, size=6, run_addr=(HVTRAPTAB3), vector_prefix=".hvtraptab3.hvtrapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
        }
        section_layout :tc3:linear
        {
                "_lc_u_int_tab" = (INTTAB3);
                "_lc_u_int_tab_tc3" = (INTTAB3);

                "_lc_u_trap_tab" = (TRAPTAB3);
                "_lc_u_trap_tab_tc3" = (TRAPTAB3);

                "_lc_u_hvtrap_tab" = (HVTRAPTAB3);
                "_lc_u_hvtrap_tab_tc3" = (HVTRAPTAB3);
        }
        section_setup :tc4:linear
        {
                // interrupt vector table(s)
                vector_table "inttab4" (vector_size=(INTTAB4_ENTRY_SIZE), size=256, run_addr=(INTTAB4), vector_prefix=INTTAB4_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4vm0" (vector_size=(INTTAB4VM0_ENTRY_SIZE), size=256, run_addr=(INTTAB4VM0), vector_prefix=INTTAB4VM0_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4vm1" (vector_size=(INTTAB4VM1_ENTRY_SIZE), size=256, run_addr=(INTTAB4VM1), vector_prefix=INTTAB4VM1_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4vm2" (vector_size=(INTTAB4VM2_ENTRY_SIZE), size=256, run_addr=(INTTAB4VM2), vector_prefix=INTTAB4VM2_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4vm3" (vector_size=(INTTAB4VM3_ENTRY_SIZE), size=256, run_addr=(INTTAB4VM3), vector_prefix=INTTAB4VM3_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4vm4" (vector_size=(INTTAB4VM4_ENTRY_SIZE), size=256, run_addr=(INTTAB4VM4), vector_prefix=INTTAB4VM4_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4vm5" (vector_size=(INTTAB4VM5_ENTRY_SIZE), size=256, run_addr=(INTTAB4VM5), vector_prefix=INTTAB4VM5_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4vm6" (vector_size=(INTTAB4VM6_ENTRY_SIZE), size=256, run_addr=(INTTAB4VM6), vector_prefix=INTTAB4VM6_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab4vm7" (vector_size=(INTTAB4VM7_ENTRY_SIZE), size=256, run_addr=(INTTAB4VM7), vector_prefix=INTTAB4VM7_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                // trap vector table
                vector_table "traptab4" (vector_size=32, size=8, run_addr=(TRAPTAB4), vector_prefix=".traptab4.trapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                // hypervisor trap table
                vector_table "hvtraptab4" (vector_size=32, size=6, run_addr=(HVTRAPTAB4), vector_prefix=".hvtraptab4.hvtrapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
        }
        section_layout :tc4:linear
        {
                "_lc_u_int_tab" = (INTTAB4);
                "_lc_u_int_tab_tc4" = (INTTAB4);

                "_lc_u_trap_tab" = (TRAPTAB4);
                "_lc_u_trap_tab_tc4" = (TRAPTAB4);

                "_lc_u_hvtrap_tab" = (HVTRAPTAB4);
                "_lc_u_hvtrap_tab_tc4" = (HVTRAPTAB4);
        }
        section_setup :tc5:linear
        {
                // interrupt vector table(s)
                vector_table "inttab5" (vector_size=(INTTAB5_ENTRY_SIZE), size=256, run_addr=(INTTAB5), vector_prefix=INTTAB5_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5vm0" (vector_size=(INTTAB5VM0_ENTRY_SIZE), size=256, run_addr=(INTTAB5VM0), vector_prefix=INTTAB5VM0_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5vm1" (vector_size=(INTTAB5VM1_ENTRY_SIZE), size=256, run_addr=(INTTAB5VM1), vector_prefix=INTTAB5VM1_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5vm2" (vector_size=(INTTAB5VM2_ENTRY_SIZE), size=256, run_addr=(INTTAB5VM2), vector_prefix=INTTAB5VM2_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5vm3" (vector_size=(INTTAB5VM3_ENTRY_SIZE), size=256, run_addr=(INTTAB5VM3), vector_prefix=INTTAB5VM3_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5vm4" (vector_size=(INTTAB5VM4_ENTRY_SIZE), size=256, run_addr=(INTTAB5VM4), vector_prefix=INTTAB5VM4_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5vm5" (vector_size=(INTTAB5VM5_ENTRY_SIZE), size=256, run_addr=(INTTAB5VM5), vector_prefix=INTTAB5VM5_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5vm6" (vector_size=(INTTAB5VM6_ENTRY_SIZE), size=256, run_addr=(INTTAB5VM6), vector_prefix=INTTAB5VM6_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab5vm7" (vector_size=(INTTAB5VM7_ENTRY_SIZE), size=256, run_addr=(INTTAB5VM7), vector_prefix=INTTAB5VM7_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                // trap vector table
                vector_table "traptab5" (vector_size=32, size=8, run_addr=(TRAPTAB5), vector_prefix=".traptab5.trapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                // hypervisor trap table
                vector_table "hvtraptab5" (vector_size=32, size=6, run_addr=(HVTRAPTAB5), vector_prefix=".hvtraptab5.hvtrapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
        }
        section_layout :tc5:linear
        {
                "_lc_u_int_tab" = (INTTAB5);
                "_lc_u_int_tab_tc5" = (INTTAB5);

                "_lc_u_trap_tab" = (TRAPTAB5);
                "_lc_u_trap_tab_tc5" = (TRAPTAB5);

                "_lc_u_hvtrap_tab" = (HVTRAPTAB5);
                "_lc_u_hvtrap_tab_tc5" = (HVTRAPTAB5);
        }
        section_setup :tc6:linear
        {
                // interrupt vector table(s)
                vector_table "inttab6" (vector_size=(INTTAB6_ENTRY_SIZE), size=256, run_addr=(INTTAB6), vector_prefix=INTTAB6_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6vm0" (vector_size=(INTTAB6VM0_ENTRY_SIZE), size=256, run_addr=(INTTAB6VM0), vector_prefix=INTTAB6VM0_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6vm1" (vector_size=(INTTAB6VM1_ENTRY_SIZE), size=256, run_addr=(INTTAB6VM1), vector_prefix=INTTAB6VM1_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6vm2" (vector_size=(INTTAB6VM2_ENTRY_SIZE), size=256, run_addr=(INTTAB6VM2), vector_prefix=INTTAB6VM2_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6vm3" (vector_size=(INTTAB6VM3_ENTRY_SIZE), size=256, run_addr=(INTTAB6VM3), vector_prefix=INTTAB6VM3_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6vm4" (vector_size=(INTTAB6VM4_ENTRY_SIZE), size=256, run_addr=(INTTAB6VM4), vector_prefix=INTTAB6VM4_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6vm5" (vector_size=(INTTAB6VM5_ENTRY_SIZE), size=256, run_addr=(INTTAB6VM5), vector_prefix=INTTAB6VM5_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6vm6" (vector_size=(INTTAB6VM6_ENTRY_SIZE), size=256, run_addr=(INTTAB6VM6), vector_prefix=INTTAB6VM6_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                vector_table "inttab6vm7" (vector_size=(INTTAB6VM7_ENTRY_SIZE), size=256, run_addr=(INTTAB6VM7), vector_prefix=INTTAB6VM7_VECTOR_PREFIX, template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                // trap vector table
                vector_table "traptab6" (vector_size=32, size=8, run_addr=(TRAPTAB6), vector_prefix=".traptab6.trapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
                // hypervisor trap table
                vector_table "hvtraptab6" (vector_size=32, size=6, run_addr=(HVTRAPTAB6), vector_prefix=".hvtraptab6.hvtrapvec.", template=".text.veneertemplate", template_symbol="_lc_veneertarget") {}
        }
        section_layout :tc6:linear
        {
                "_lc_u_int_tab" = (INTTAB6);
                "_lc_u_int_tab_tc6" = (INTTAB6);

                "_lc_u_trap_tab" = (TRAPTAB6);
                "_lc_u_trap_tab_tc6" = (TRAPTAB6);

                "_lc_u_hvtrap_tab" = (HVTRAPTAB6);
                "_lc_u_hvtrap_tab_tc6" = (HVTRAPTAB6);
        }
#endif // __NO_VTC

        section_setup :ppu:linear
        {
                modify input (attributes=-w)
                {
                        select ".(s|)(data|bss)(|.*)" (attributes=+w);
                }
        }

#ifndef NOXC800INIT
        section_layout :__CORE0:linear
        {
                group (ordered, align=4) memcopy ".rodata.xc800init" (memory = scr_xram, fill=XC800INIT_FILL);
        }
#endif
        section_layout :__CORE0:linear
        {
                group (ordered, align=4) memcopy ".rodata.ppuinit" (memory = csm, fill=0);
        }
        section_layout :__CORE0:linear
        {
#include        "tc1v1_8.bmhd.lsl"
        }

#if !defined(__DISABLE_SCR_BOOT_MAGIC)
        /*
         *      The last 8 bytes of SCR XRAM starting at address 0x07FF8 must contain
         *      4 pairs of bytes where each pair consists of 0x55 followed by 0xAA.
         *      The user code will not be executed and the SCR will enter an endless
         *      loop if the memory content does not match this data sequence. This
         *      feature is meant to avoid an unintentional entry into User Mode 1.
         *      When the 8 bytes match, the SCR boot code will trigger an interrupt
         *      to the TriCore by setting bit NMICON.SCRINTTC to 1 with a value of 0x80
         *      in the SCRINTEXCHG register. When the 8 bytes do not match, the same
         *      interrupt is triggered with a value of 0x81 in the SCRINTEXCHG register.
         */
        section_layout :xc800:xdata
        {
                group(ordered, run_addr=0x7ff8)
                {
                        struct "scr_boot_magic"
                        {
                                0x55:1; 0xaa:1;
                                0x55:1; 0xaa:1;
                                0x55:1; 0xaa:1;
                                0x55:1; 0xaa:1;
                        }
                }
        }
#endif
}

//
//      The following macros are required for extmem.lsl
//
#define HAS_ON_CHIP_FLASH       // This derivative has on-chip flash
#define HAS_NO_EXTERNAL_RAM     // Exclude xram_8_a

