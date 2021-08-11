#/usr/bin/bash

JVMS_PATH=$(pwd)/vendor/jvms.html

[ -f "$JVMS_PATH" ] || curl https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html -o "$JVMS_PATH"

node docenizer-java.js > ../../lib/handlers/asm-docs-java.js
