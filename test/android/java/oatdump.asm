MAGIC:
oat
239

LOCATION:
classes.odex

CHECKSUM:
0x30300a3c

INSTRUCTION SET:
Arm64

INSTRUCTION SET FEATURES:
a53,crc,-lse,-fp16,-dotprod,-sve

DEX FILE COUNT:
1

EXECUTABLE OFFSET:
0x00001000

JNI DLSYM LOOKUP TRAMPOLINE OFFSET:
0x00000000

JNI DLSYM LOOKUP CRITICAL TRAMPOLINE OFFSET:
0x00000000

QUICK GENERIC JNI TRAMPOLINE OFFSET:
0x00000000

QUICK IMT CONFLICT TRAMPOLINE OFFSET:
0x00000000

QUICK RESOLUTION TRAMPOLINE OFFSET:
0x00000000

QUICK TO INTERPRETER BRIDGE OFFSET:
0x00000000

NTERP_TRAMPOLINE OFFSET:
0x00000000

KEY VALUE STORE:
apex-versions = /////
bootclasspath = /apex/com.android.art/core-oj.jar:/apex/com.android.art/core-libart.jar:/apex/com.android.art/okhttp.jar:/apex/com.android.art/bouncycastle.jar:/apex/com.android.art/javalib/apache-xml.jar
bootclasspath-checksums = d/c249736c:d/241a8e9b:d/d6810851:d/c6899d69:d/6ea0e235
classpath = PCL[]
compiler-filter = speed
concurrent-copying = true
debuggable = false
dex2oat-cmdline = /opt/compiler-explorer/dex2oat-latest/x86_64/bin/dex2oat64 --android-root=include --generate-debug-info --dex-location=/system/framework/classes.dex --dex-file=/tmp/compiler-explorer-compiler202409-1055822-1e3kxfv.la09/classes.dex --runtime-arg -Xbootclasspath:bootjars/core-oj.jar:bootjars/core-libart.jar:bootjars/okhttp.jar:bootjars/bouncycastle.jar:bootjars/apache-xml.jar --runtime-arg -Xbootclasspath-locations:/apex/com.android.art/core-oj.jar:/apex/com.android.art/core-libart.jar:/apex/com.android.art/okhttp.jar:/apex/com.android.art/bouncycastle.jar:/apex/com.android.art/javalib/apache-xml.jar --boot-image=/nonx/boot.art --oat-file=/tmp/compiler-explorer-compiler202409-1055822-1e3kxfv.la09/classes.odex --force-allow-oj-inlines --dump-cfg=/tmp/compiler-explorer-compiler202409-1055822-1e3kxfv.la09/classes.cfg --instruction-set=arm64 --compiler-filter=speed
native-debuggable = false
requires-image = false

SIZE:
4136

.data.bimg.rel.ro: empty.

.bss: empty.

Layout data
SectionTypeCode:LayoutTypeHot(0-0) LayoutTypeSometimesUsed(0-0) LayoutTypeStartupOnly(0-0) LayoutTypeUsedOnce(0-0) LayoutTypeUnused(0-0)
SectionTypeStrings:LayoutTypeHot(0-0) LayoutTypeSometimesUsed(0-0) LayoutTypeStartupOnly(0-0) LayoutTypeUsedOnce(0-0) LayoutTypeUnused(0-0)

.bss mapping for ArtMethod: empty.
.bss mapping for Class: empty.
.bss mapping for Public Class: empty.
.bss mapping for Package Class: empty.
.bss mapping for String: empty.
Dependencies of /system/framework/classes.dex:
  Dependencies of LSquare;:
OatDexFile:
location: /system/framework/classes.dex
checksum: 0xdc2b5a3a
dex-file: 0x00000040..0x000002d7
type-table: 0x000002ec..0x000002f3
0: LSquare;
0: LSquare; (offset=0x000005dc) (type_idx=1) (Verified) (AllCompiled)
  0: void Square.<init>() (dex_method_idx=0)
    DEX CODE:
      0x0000: 7010 0200 0000           	| invoke-direct {v0}, void java.lang.Object.<init>() // method@2
      0x0003: 0e00                     	| return-void
    OatMethodOffsets (offset=0x000005e4)
      code_offset: 0x00001010
    OatQuickMethodHeader (offset=0x0000100c)
      vmap_table: (offset=0x00000a24)
        CodeInfo CodeSize:4 FrameSize:0 CoreSpillMask:40000000 FpSpillMask:0 NumberOfDexRegisters:1
    QuickMethodFrameInfo
      frame_size_in_bytes: 0
      core_spill_mask: 0x40000000 (r30)
      fp_spill_mask: 0x00000000
      vr_stack_locations:
      	ins: v0[sp + #8]
      	method*: v1[sp + #0]
      	outs: v0[sp + #8]
    CODE: (code_offset=0x00001010 size=4)...
      0x00001010: d65f03c0	ret
  1: int Square.square(int) (dex_method_idx=1)
    DEX CODE:
      0x0000: 9200 0000                	| mul-int v0, v0, v0
      0x0002: 0f00                     	| return v0
    OatMethodOffsets (offset=0x000005e8)
      code_offset: 0x00001020
    OatQuickMethodHeader (offset=0x0000101c)
      vmap_table: (offset=0x00000a2c)
        CodeInfo CodeSize:8 FrameSize:0 CoreSpillMask:40000000 FpSpillMask:0 NumberOfDexRegisters:1
    QuickMethodFrameInfo
      frame_size_in_bytes: 0
      core_spill_mask: 0x40000000 (r30)
      fp_spill_mask: 0x00000000
      vr_stack_locations:
      	ins: v0[sp + #8]
      	method*: v1[sp + #0]
    CODE: (code_offset=0x00001020 size=8)...
      0x00001020: 1b017c20	mul w0, w1, w1
      0x00001024: d65f03c0	ret

OAT FILE STATS:
OatFile                                                   1      4.136KB  100.0%
  (other)                                                 1      4.100KB   99.1%
  CodeInfo                                                2      0.016KB    0.4%
    Header                                                2      0.015KB    0.4%
    (other)                                               2      0.001KB    0.0%
  Code                                                    2      0.012KB    0.3%
  QuickMethodHeader                                       2      0.008KB    0.2%
