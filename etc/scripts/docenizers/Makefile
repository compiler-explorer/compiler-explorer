all: \
 ../../../lib/asm-docs/generated/asm-docs-6502.ts \
 ../../../lib/asm-docs/generated/asm-docs-amd64.ts \
 ../../../lib/asm-docs/generated/asm-docs-arm32.ts \
 ../../../lib/asm-docs/generated/asm-docs-avr.ts \
 ../../../lib/asm-docs/generated/asm-docs-evm.ts \
 ../../../lib/asm-docs/generated/asm-docs-java.ts \
 ../../../lib/asm-docs/generated/asm-docs-llvm.ts \
 ../../../lib/asm-docs/generated/asm-docs-python.ts

../../../lib/asm-docs/generated/asm-docs-6502.ts:
	python3 docenizer-6502.py
../../../lib/asm-docs/generated/asm-docs-amd64.ts:
	python3 docenizer-amd64.py -o ../../../lib/asm-docs/generated/asm-docs-amd64.ts
../../../lib/asm-docs/generated/asm-docs-arm32.ts:
	python3 docenizer-arm32.py -o ../../../lib/asm-docs/generated/asm-docs-arm32.ts
../../../lib/asm-docs/generated/asm-docs-avr.ts:
	python3 docenizer-avr.py
../../../lib/asm-docs/generated/asm-docs-evm.ts:
	python3 docenizer-evm.py -o ../../../lib/asm-docs/generated/asm-docs-evm.ts
../../../lib/asm-docs/generated/asm-docs-java.ts:
	./docenizer-java.sh
../../../lib/asm-docs/generated/asm-docs-llvm.ts:
	./docenizer-llvm.sh
../../../lib/asm-docs/generated/asm-docs-python.ts:
	python3 docenizer-python.py -o ../../../lib/asm-docs/generated/asm-docs-python.ts
