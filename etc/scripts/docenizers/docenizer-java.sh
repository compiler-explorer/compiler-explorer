#/usr/bin/bash

JVMS_PATH=$(pwd)/vendor/jvms.html

[ -f "$JVMS_PATH" ] || curl https://docs.oracle.com/javase/specs/jvms/se18/html/jvms-6.html -o "$JVMS_PATH"

npx ts-node-esm docenizer-java.ts > ../../../lib/asm-docs/generated/asm-docs-java.ts
