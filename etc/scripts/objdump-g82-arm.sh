#!/bin/sh

OBJDUMP=/opt/compiler-explorer/arm/gcc-8.2.0/arm-unknown-linux-gnueabi/bin/arm-unknown-linux-gnueabi-objdump
ASMPARSER=/cebin/asm-parser

$OBJDUMP "$@" | $ASMPARSER -stdin -binary
