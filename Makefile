NODE_DIR?=/opt/compiler-explorer/node
YARN_DIR?=/opt/compiler-explorer/yarn
YARN_EXE:=$(shell env PATH=$(NODE_DIR)/bin:$(YARN_DIR)/bin:$(PATH) which yarn)
NODE:=$(shell env PATH=$(NODE_DIR)/bin:$(PATH) which node || env PATH=$(NODE_DIR)/bin:$(PATH) which nodejs)
YARN:=$(NODE) $(YARN_EXE).js
default: run

NODE_VERSION_USED:=8
NODE_VERSION:=$(shell $(NODE) --version)
NODE_MAJOR_VERSION:=$(shell echo $(NODE_VERSION) | cut -f1 -d. | sed 's/^v//g')
NODE_VERSION_TEST:=$(shell [ $(NODE_MAJOR_VERSION) -eq $(NODE_VERSION_USED) ] && echo true)
NODE_VERSION_TEST_FAIL:=$(shell [ $(NODE_MAJOR_VERSION) -lt $(NODE_VERSION_USED) ] && echo true)

ifneq ($(NODE_VERSION_TEST), true)
ifeq ($(NODE_VERSION_TEST_FAIL), true)
$(error Compiler Explorer needs node v$(NODE_VERSION_USED).x, but $(NODE_VERSION) was found. \
Visit https://nodejs.org/ for installation instructions \
To configure where we look for node, set NODE_DIR to its installation base)
else
$(warning Compiler Explorer needs node v$(NODE_VERSION_USED).x, but $(NODE_VERSION) was found. \
The higher node version might work but it has not been tested.)
endif
endif

.PHONY: clean run test run-amazon c-preload optional-haskell-support optional-d-support optional-rust-support
.PHONY: dist lint prereqs node_modules travis-dist
prereqs: optional-haskell-support optional-d-support optional-rust-support node_modules webpack c-preload
GDC?=gdc
DMD?=dmd
LDC?=ldc2
ifneq "" "$(shell which $(GDC) 2>/dev/null || which $(DMD) 2>/dev/null || which $(LDC) 2>/dev/null)"
optional-d-support:
	$(MAKE) -C d
else
optional-d-support:
	@echo "D language support disabled"
endif

GHC?=ghc
ifneq "" "$(shell which $(GHC) 2>/dev/null)"
optional-haskell-support:
	$(MAKE) -C haskell
else
optional-haskell-support:
	@echo "Haskell language support disabled"
endif

ifneq "" "$(shell which cargo)"
rust/bin/rustfilt: rust/src/main.rs rust/Cargo.lock rust/Cargo.toml
	cd rust && cargo build --release && cargo install --root . --force && cargo clean
optional-rust-support: rust/bin/rustfilt 
else
optional-rust-support:
	@echo "Rust language support disabled"
endif


NODE_MODULES=.npm-updated
$(NODE_MODULES): package.json
	$(YARN) install
	@touch $@

webpack:
	$(NODE) node_modules/webpack/bin/webpack.js 

lint: $(NODE_MODULES)
	$(NODE) ./node_modules/.bin/jshint --config etc/jshintrc.server app.js $(shell find lib -name '*.js')
	$(NODE) ./node_modules/.bin/jshint --config etc/jshintrc.client $(shell find static -name '*.js' -not -path 'static/dist/*' -not -path static/analytics.js -not -path 'static/vs/*')

node_modules: $(NODE_MODULES)
webpack: $(WEBPACK)

test: $(NODE_MODULES) lint
	$(MAKE) -C c-preload test
	@echo Tests pass

check: $(NODE_MODULES) lint
	$(NODE) ./node_modules/.bin/mocha --recursive

clean:
	rm -rf node_modules .npm-updated out static/dist static/vs
	$(MAKE) -C d clean
	$(MAKE) -C c-preload clean

run: export NODE_ENV=LOCAL
run: prereqs
	$(NODE) ./node_modules/.bin/supervisor -w app.js,lib,etc/config -e 'js|node|properties' --exec $(NODE) $(NODE_ARGS) -- ./app.js $(EXTRA_ARGS)

dev: export NODE_ENV=DEV
dev: prereqs
	 $(NODE) ./node_modules/.bin/supervisor -w app.js,lib,etc/config -e 'js|node|properties' --exec $(NODE) $(NODE_ARGS) -- ./app.js $(EXTRA_ARGS)
	
	

HASH := $(shell git rev-parse HEAD)
dist: prereqs
	rm -rf out/dist/
	mkdir -p out/dist
	mkdir -p out/dist/vs
	cp -r static/dist/ out/dist/
	cp -r static/vs/ out/dist/vs

travis-dist: dist
	tar --exclude './.travis-compilers' --exclude './.git' --exclude './static' -Jcf /tmp/ce-build.tar.xz . 
	rm -rf out/dist-bin
	mkdir -p out/dist-bin
	mv /tmp/ce-build.tar.xz out/dist-bin/${TRAVIS_BUILD_NUMBER}.tar.xz
	echo ${HASH} > out/dist-bin/${TRAVIS_BUILD_NUMBER}.txt

c-preload:
	$(MAKE) -C c-preload

install-git-hooks:
	ln -sf $(shell pwd)/etc/scripts/pre-commit .git/hooks/pre-commit
.PHONY: install-git-hooks

