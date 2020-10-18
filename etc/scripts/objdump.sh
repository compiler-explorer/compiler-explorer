#!/bin/sh

OBJDUMP=/opt/compiler-explorer/gcc-10.1.0/bin/objdump
ASMPARSER=/opt/compiler-explorer/asm-parser/build/bin/asm-parser

$OBJDUMP "$@" | $ASMPARSER -stdin -binary
