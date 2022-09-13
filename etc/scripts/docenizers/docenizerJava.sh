#/usr/bin/bash

JVMS_PATH=$(pwd)/vendor/jvms.html

[ -f "$JVMS_PATH" ] || curl https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html -o "$JVMS_PATH"

$(pwd)/find-node ../../../.node-bin
$(cat ../../../.node-bin) docenizer-java.js > ../../../lib/asm-docs/generated/asm-docs-java.js
