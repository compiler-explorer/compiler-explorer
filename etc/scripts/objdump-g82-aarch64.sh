#!/bin/sh

OBJDUMP=/opt/compiler-explorer/arm64/gcc-8.2.0/aarch64-unknown-linux-gnu/bin/aarch64-unknown-linux-gnu-objdump
ASMPARSER=/cebin/asm-parser

$OBJDUMP "$@" | $ASMPARSER -stdin -binary
