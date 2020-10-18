#!/bin/sh

OBJDUMP=/opt/compiler-explorer/gcc-10.1.0/bin/objdump
ASMPARSER=/cebin/asm-parser

$OBJDUMP "$@" | $ASMPARSER -stdin -binary
