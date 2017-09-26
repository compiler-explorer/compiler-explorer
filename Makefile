ifneq "" "$(NODE_DIR)"
NPM:=$(NODE_DIR)/bin/npm
NODE:=$(NODE_DIR)/bin/node
default: run
else
ifeq "" "$(shell which npm)"
default:
	@echo "Please install node.js"
	@echo "Visit http://nodejs.org/ for more details"
	@echo "On Ubuntu/Debian try: sudo apt-get install nodejs npm"
	exit 1
else
NPM:= $(shell which npm)
NODE:= $(shell which node || which nodejs)
default: run
endif
endif

.PHONY: clean run test run-amazon c-preload optional-haskell-support optional-d-support optional-rust-support
.PHONY: dist lint prereqs node_modules bower_modules
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
optional-rust-support:
	cd rust && cargo build --release
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
	$(NODE) ./node_modules/.bin/jshint app.js $(shell find lib static -name '*.js' -not -path 'static/ext/*' -not -path static/analytics.js)

LANG:=C++

node_modules: $(NODE_MODULES)
bower_modules: $(BOWER_MODULES)

test: $(NODE_MODULES) lint
	$(MAKE) -C c-preload test
	@echo Tests pass

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

c-preload:
	$(MAKE) -C c-preload
