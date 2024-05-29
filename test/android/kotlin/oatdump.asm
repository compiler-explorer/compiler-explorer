MAGIC:
oat
239

LOCATION:
classes.odex

CHECKSUM:
0xe11f04fb

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
dex2oat-cmdline = /opt/compiler-explorer/dex2oat-latest/x86_64/bin/dex2oat64 --android-root=include --generate-debug-info --dex-location=/system/framework/classes.dex --dex-file=/tmp/compiler-explorer-compiler202409-1055822-posq0d.3gtp/classes.dex --runtime-arg -Xbootclasspath:bootjars/core-oj.jar:bootjars/core-libart.jar:bootjars/okhttp.jar:bootjars/bouncycastle.jar:bootjars/apache-xml.jar --runtime-arg -Xbootclasspath-locations:/apex/com.android.art/core-oj.jar:/apex/com.android.art/core-libart.jar:/apex/com.android.art/okhttp.jar:/apex/com.android.art/bouncycastle.jar:/apex/com.android.art/javalib/apache-xml.jar --boot-image=/nonx/boot.art --oat-file=/tmp/compiler-explorer-compiler202409-1055822-posq0d.3gtp/classes.odex --force-allow-oj-inlines --dump-cfg=/tmp/compiler-explorer-compiler202409-1055822-posq0d.3gtp/classes.cfg --instruction-set=arm64 --compiler-filter=speed
native-debuggable = false
requires-image = false

SIZE:
4120

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
  Dependencies of LExampleKt;:
OatDexFile:
location: /system/framework/classes.dex
checksum: 0x05a766ed
dex-file: 0x00000040..0x00000343
type-table: 0x00000358..0x0000035f
0: LExampleKt;
0: LExampleKt; (offset=0x000005d8) (type_idx=1) (Verified) (AllCompiled)
  0: int ExampleKt.square(int) (dex_method_idx=0)
    DEX CODE:
      0x0000: 9200 0101                	| mul-int v0, v1, v1
      0x0002: 0f00                     	| return v0
    OatMethodOffsets (offset=0x000005e0)
      code_offset: 0x00001010
    OatQuickMethodHeader (offset=0x0000100c)
      vmap_table: (offset=0x00000a2c)
        CodeInfo CodeSize:8 FrameSize:0 CoreSpillMask:40000000 FpSpillMask:0 NumberOfDexRegisters:2
    QuickMethodFrameInfo
      frame_size_in_bytes: 0
      core_spill_mask: 0x40000000 (r30)
      fp_spill_mask: 0x00000000
      vr_stack_locations:
      	locals: v0[sp + #4294967280]
      	ins: v1[sp + #8]
      	method*: v2[sp + #0]
    CODE: (code_offset=0x00001010 size=8)...
      0x00001010: 1b017c20	mul w0, w1, w1
      0x00001014: d65f03c0	ret

OAT FILE STATS:
OatFile                                                   1      4.120KB  100.0%
  (other)                                                 1      4.100KB   99.5%
  Code                                                    1      0.008KB    0.2%
  CodeInfo                                                1      0.008KB    0.2%
    Header                                                1      0.007KB    0.2%
    (other)                                               1      0.001KB    0.0%
  QuickMethodHeader                                       1      0.004KB    0.1%
