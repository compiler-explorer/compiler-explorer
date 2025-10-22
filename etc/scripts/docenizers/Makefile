GEN_DIR := ../../../lib/asm-docs/generated

# Use system uv if available, otherwise install locally
UV_SYSTEM:=$(shell command -v uv 2>/dev/null)
UV_BIN:=$(if $(UV_SYSTEM),$(UV_SYSTEM),$(CURDIR)/.uv/uv)
UV_VENV:=$(CURDIR)/.venv
UV_DEPS:=$(UV_VENV)/.deps

all: $(UV_DEPS) \
 $(GEN_DIR)/asm-docs-6502.ts \
 $(GEN_DIR)/asm-docs-65c816.ts \
 $(GEN_DIR)/asm-docs-amd64.ts \
 $(GEN_DIR)/asm-docs-arm32.ts \
 $(GEN_DIR)/asm-docs-aarch64.ts \
 $(GEN_DIR)/asm-docs-avr.ts \
 $(GEN_DIR)/asm-docs-evm.ts \
 $(GEN_DIR)/asm-docs-java.ts \
 $(GEN_DIR)/asm-docs-llvm.ts \
 $(GEN_DIR)/asm-docs-riscv64.ts \
 $(GEN_DIR)/asm-docs-ptx.ts \
 $(GEN_DIR)/asm-docs-sass.ts \
 $(GEN_DIR)/asm-docs-python.ts

# Install uv if not available system-wide
$(CURDIR)/.uv/uv:
	@echo "Installing uv..."
	@mkdir -p $(CURDIR)/.uv
	@curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 UV_INSTALL_DIR=$(CURDIR)/.uv sh -s

# Set up virtual environment and install dependencies
$(UV_DEPS): $(UV_BIN) pyproject.toml uv.lock
	$(UV_BIN) sync --no-install-project
	@touch $@

# Generate lock file if it doesn't exist
uv.lock: $(UV_BIN) pyproject.toml
	$(UV_BIN) lock

$(GEN_DIR)/asm-docs-6502.ts: docenizer-6502.py $(UV_DEPS)
	$(UV_BIN) run python3 $< -o $@ -c 6502 -m 65c02
$(GEN_DIR)/asm-docs-65c816.ts: docenizer-6502.py $(UV_DEPS)
	$(UV_BIN) run python3 $< -o $@ -c 65c816 -m 65c816
$(GEN_DIR)/asm-docs-amd64.ts: docenizer-amd64.py $(UV_DEPS)
	$(UV_BIN) run python3 $< -o $@
$(GEN_DIR)/asm-docs-arm32.ts: docenizer-arm.py arm32.json $(UV_DEPS)
	$(UV_BIN) run python3 $< -o $@ -c arm32.json -d asm-docs-arm32 -i asm-docs-arm32
$(GEN_DIR)/asm-docs-aarch64.ts: docenizer-arm.py aarch64.json $(UV_DEPS)
	$(UV_BIN) run python3 $< -o $@ -c aarch64.json -d asm-docs-aarch64 -i asm-docs-aarch64
$(GEN_DIR)/asm-docs-avr.ts: docenizer-avr.py $(UV_DEPS)
	$(UV_BIN) run python3 $<
$(GEN_DIR)/asm-docs-evm.ts: docenizer-evm.py $(UV_DEPS)
	$(UV_BIN) run python3 $< -o $@
$(GEN_DIR)/asm-docs-java.ts: docenizer-java.sh docenizer-java.ts
	./$<
$(GEN_DIR)/asm-docs-llvm.ts: docenizer-llvm.sh docenizer-llvm.ts
	./$<
$(GEN_DIR)/asm-docs-ptx.ts: docenizer-ptx-sass.py $(UV_DEPS)
	$(UV_BIN) run python3 $< -o $(GEN_DIR)
$(GEN_DIR)/asm-docs-sass.ts: docenizer-ptx-sass.py $(UV_DEPS)
	$(UV_BIN) run python3 $< -o $(GEN_DIR)
$(GEN_DIR)/asm-docs-python.ts: docenizer-python.py $(UV_DEPS)
	$(UV_BIN) run python3 $< -o $@
$(GEN_DIR)/asm-docs-riscv64.ts: docenizer-riscv64.py $(UV_DEPS)
	$(UV_BIN) run python3 $< -o $@

.PHONY: clean
clean:
	rm -rf $(UV_VENV) $(CURDIR)/.uv
