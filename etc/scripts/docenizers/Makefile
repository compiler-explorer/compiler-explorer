GEN_DIR := ../../../lib/asm-docs/generated
all: \
 $(GEN_DIR)/asm-docs-6502.ts \
 $(GEN_DIR)/asm-docs-amd64.ts \
 $(GEN_DIR)/asm-docs-arm32.ts \
 $(GEN_DIR)/asm-docs-aarch64.ts \
 $(GEN_DIR)/asm-docs-avr.ts \
 $(GEN_DIR)/asm-docs-evm.ts \
 $(GEN_DIR)/asm-docs-java.ts \
 $(GEN_DIR)/asm-docs-llvm.ts \
 $(GEN_DIR)/asm-docs-ptx.ts \
 $(GEN_DIR)/asm-docs-sass.ts \
 $(GEN_DIR)/asm-docs-python.ts

$(GEN_DIR)/asm-docs-6502.ts: docenizer-6502.py
	python3 $<
$(GEN_DIR)/asm-docs-amd64.ts: docenizer-amd64.py
	python3 $< -o $@
$(GEN_DIR)/asm-docs-arm32.ts: docenizer-arm.py arm32.json
	python3 $< -o $@ -c arm32.json -d asm-docs-arm32 -i asm-docs-arm32
$(GEN_DIR)/asm-docs-aarch64.ts: docenizer-arm.py aarch64.json
	python3 $< -o $@ -c aarch64.json -d asm-docs-aarch64 -i asm-docs-aarch64
$(GEN_DIR)/asm-docs-avr.ts: docenizer-avr.py
	python3 $<
$(GEN_DIR)/asm-docs-evm.ts: docenizer-evm.py
	python3 $< -o $@
$(GEN_DIR)/asm-docs-java.ts: docenizer-java.sh docenizer-java.ts
	./$<
$(GEN_DIR)/asm-docs-llvm.ts: docenizer-llvm.sh docenizer-llvm.ts
	./$<
$(GEN_DIR)/asm-docs-ptx.ts: docenizer-ptx-sass.py
	python3 $< -o $@
$(GEN_DIR)/asm-docs-sass.ts: docenizer-ptx-sass.py
	python3 $< -o $@
$(GEN_DIR)/asm-docs-python.ts: docenizer-python.py
	python3 $< -o $@
