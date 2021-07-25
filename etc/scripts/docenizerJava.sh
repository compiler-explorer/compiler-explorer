#/usr/bin/bash

curl https://docs.oracle.com/javase/specs/jvms/se16/html/jvms-6.html -o `pwd`/vendor/jvms.html

node docenizer-java.js > ../../lib/handlers/asm-docs-java.js
