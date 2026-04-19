# MLRift compiler (built on KernRift)
# Usage:
#   make              - build the compiler (self-hosts from build/mlrc)
#   make test         - run test suite
#   make install      - install to ~/.local/bin/mlrc
#   make dist         - create distribution binaries for all platforms
#   make clean        - remove build artifacts
#   make bootstrap    - verify self-host fixed point

INSTALL_DIR ?= $(HOME)/.local/bin
DIST_DIR = dist

SRCS = src/lexer.kr src/ast.kr src/parser.kr src/codegen.kr \
       src/codegen_aarch64.kr src/ir.kr src/ir_aarch64.kr src/ir_hip.kr \
       src/format_macho.kr src/format_pe.kr src/format_hip.kr src/format_elf_dyn.kr \
       src/format_archive.kr src/format_android.kr src/bcj.kr src/analysis.kr src/living.kr \
       src/runtime.kr src/formatter.kr src/main.kr

.PHONY: all build test install dist clean bootstrap

all: build

build: build/mlrc

build/mlrc.kr: $(SRCS)
	@mkdir -p build
	cat $(SRCS) > build/mlrc.kr

# Self-compile. build/mlrc is committed as the bootstrap.
build/mlrc: build/mlrc.kr
	@if [ ! -x build/mlrc ]; then \
		echo "build/mlrc not found — clone must include the committed bootstrap."; \
		exit 1; \
	fi
	./build/mlrc --arch=x86_64 build/mlrc.kr -o build/mlrc.new
	mv build/mlrc.new build/mlrc
	chmod +x build/mlrc

# Run test suite
test: build/mlrc
	@echo "=== Running MLRift test suite ==="
	@echo '#!/bin/bash' > /tmp/mlrc-test && echo 'exec ./build/mlrc --arch=x86_64 "$$@"' >> /tmp/mlrc-test && chmod +x /tmp/mlrc-test
	@KRC=/tmp/mlrc-test bash tests/run_tests.sh || true

# Verify self-host fixed point (stage3 == stage4)
bootstrap: build/mlrc
	@echo "=== Bootstrap verification ==="
	@cp build/mlrc.kr /tmp/mlrc_bs_src.kr
	@./build/mlrc --arch=x86_64 /tmp/mlrc_bs_src.kr -o /tmp/mlrc3_bs 2>/dev/null
	@chmod +x /tmp/mlrc3_bs
	@/tmp/mlrc3_bs --arch=x86_64 /tmp/mlrc_bs_src.kr -o /tmp/mlrc4_bs 2>/dev/null
	@if diff /tmp/mlrc3_bs /tmp/mlrc4_bs >/dev/null 2>&1; then \
		echo "PASS: fixed point at $$(wc -c < /tmp/mlrc3_bs) bytes"; \
	else \
		echo "FAIL: stage3 != stage4"; exit 1; \
	fi
	@rm -f /tmp/mlrc_bs_src.kr /tmp/mlrc3_bs /tmp/mlrc4_bs

# Install as "mlrc" in INSTALL_DIR
install: build/mlrc
	@mkdir -p $(INSTALL_DIR)
	cp build/mlrc $(INSTALL_DIR)/mlrc
	chmod +x $(INSTALL_DIR)/mlrc
	@echo "Installed: $(INSTALL_DIR)/mlrc"
	@echo "Ensure $(INSTALL_DIR) is in your PATH"

# Distribution binaries
dist: build/mlrc
	@mkdir -p $(DIST_DIR)
	@echo "=== Building distribution ==="
	cp build/mlrc $(DIST_DIR)/mlrc-linux-x86_64
	chmod +x $(DIST_DIR)/mlrc-linux-x86_64
	@echo "  mlrc-linux-x86_64"
	./build/mlrc --arch=arm64 build/mlrc.kr -o $(DIST_DIR)/mlrc-linux-arm64 2>/dev/null
	chmod +x $(DIST_DIR)/mlrc-linux-arm64
	@echo "  mlrc-linux-arm64"
	./build/mlrc --arch=x86_64 --emit=pe build/mlrc.kr -o $(DIST_DIR)/mlrc-windows-x86_64.exe 2>/dev/null
	@echo "  mlrc-windows-x86_64.exe"
	./build/mlrc --arch=arm64 --emit=pe build/mlrc.kr -o $(DIST_DIR)/mlrc-windows-arm64.exe 2>/dev/null
	@echo "  mlrc-windows-arm64.exe"
	./build/mlrc build/mlrc.kr -o $(DIST_DIR)/mlrc.krbo 2>/dev/null
	@echo "  mlrc.krbo (fat binary, 8 slices)"
	cp build/mlrc.kr $(DIST_DIR)/mlrc-source.kr
	@echo "  mlrc-source.kr"
	@echo ""
	@ls -la $(DIST_DIR)/
	@echo "=== Distribution complete ==="

clean:
	rm -f build/mlrc.new build/mlrc.kr
	rm -rf $(DIST_DIR)
	rm -f a.out output.elf test_input.kr
	rm -f *.elf *.out
	@echo "Cleaned (build/mlrc preserved — committed bootstrap)."
