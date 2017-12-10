NODE_DIR?=/opt/compiler-explorer/node
NPM:=$(shell env PATH=$(NODE_DIR)/bin:$(PATH) which npm)
NODE:=$(shell env PATH=$(NODE_DIR)/bin:$(PATH) which node || env PATH=$(NODE_DIR)/bin:$(PATH) which nodejs)
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
.PHONY: dist lint prereqs node_modules bower_modules travis-dist
prereqs: optional-haskell-support optional-d-support optional-rust-support node_modules c-preload bower_modules

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
ifneq "" "$(shell which $(GHC))"
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
	$(NPM) install
	@touch $@

BOWER_MODULES=.bower-updated
$(BOWER_MODULES): bower.json $(NODE_MODULES)
	if ! test -f "${BOWER_MODULES}"; then rm -rf static/ext; fi
	$(NODE) ./node_modules/bower/bin/bower install
	@touch $@
	# Workaround for lack of versioned monaco-editor in bower
	rm -rf static/ext/monaco-editor
	cp -r node_modules/monaco-editor static/ext/

lint: $(NODE_MODULES)
	$(NODE) ./node_modules/.bin/jshint --config etc/jshintrc.server app.js $(shell find lib -name '*.js')
	$(NODE) ./node_modules/.bin/jshint --config etc/jshintrc.client $(shell find static -name '*.js' -not -path 'static/ext/*' -not -path static/analytics.js)

LANG:=C++

node_modules: $(NODE_MODULES)
bower_modules: $(BOWER_MODULES)

test: $(NODE_MODULES) lint
	$(MAKE) -C c-preload test
	@echo Tests pass

check: $(NODE_MODULES) lint
	$(NODE) ./node_modules/.bin/mocha

clean:
	rm -rf bower_modules node_modules .npm-updated .bower-updated out static/ext
	$(MAKE) -C d clean
	$(MAKE) -C c-preload clean

run: prereqs
	$(NODE) ./node_modules/.bin/supervisor -w app.js,lib,etc/config -e 'js|node|properties' --exec $(NODE) $(NODE_ARGS) -- ./app.js --language $(LANG) $(EXTRA_ARGS)

HASH := $(shell git rev-parse HEAD)
dist: prereqs
	rm -rf out/dist
	$(NODE) ./node_modules/requirejs/bin/r.js -o app.build.js
	# Move all assets to a versioned directory
	echo $(HASH) > out/dist/git_hash
	mkdir -p out/dist/v/$(HASH)
	mv out/dist/main.js* out/dist/v/$(HASH)/
	mv out/dist/explorer.css out/dist/v/$(HASH)/
	mv out/dist/assets/ out/dist/v/$(HASH)/
	mv out/dist/themes/ out/dist/v/$(HASH)/
	# copy any external references into the directory too
	cp -r $(shell pwd)/out/dist/ext out/dist/v/$(HASH)/ext
	# uglify requirejs itself
	$(NODE) ./node_modules/.bin/uglifyjs out/dist/v/$(HASH)/ext/requirejs/require.js \
	    -c \
	    --output out/dist/v/$(HASH)/ext/requirejs/require.js \
	    --source-map out/dist/v/$(HASH)/ext/requirejs/require.js.map \
	    --source-map-url require.js.map \
	    --source-map-root //v/$(HASH)/ext/requirejs \
	    --prefix 6

travis-dist: dist
	tar --exclude './.travis-compilers' --exclude './.git' --exclude './static' --exclude './out/dist/ext' -Jcf /tmp/ce-build.tar.xz . 
	rm -rf out/dist-bin
	mkdir -p out/dist-bin
	mv /tmp/ce-build.tar.xz out/dist-bin/${TRAVIS_BUILD_NUMBER}.tar.xz
	echo ${HASH} > out/dist-bin/${TRAVIS_BUILD_NUMBER}.txt

c-preload:
	$(MAKE) -C c-preload
