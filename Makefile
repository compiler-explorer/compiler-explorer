ifneq "" "$(NODE_DIR)"
NPM:=$(NODE_DIR)/bin/npm
NODE:=$(NODE_DIR)/bin/npm
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

.PHONY: clean run test run-amazon c-preload optional-d-support prereqs node_modules bower_modules
.PHONY: dist lint
prereqs: optional-d-support node_modules c-preload bower_modules

ifneq "" "$(shell which gdc)"
optional-d-support:
	$(MAKE) -C d
else ifneq "" "$(shell which ${DMD})"
optional-d-support:
	$(MAKE) -C d
else
optional-d-support:
	@echo "D language support disabled"
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

lint: $(NODE_MODULES)
	$(NODE) ./node_modules/.bin/jshint app.js $(shell find lib static -name '*.js' -not -path 'static/ext/*')

LANG:=C++

node_modules: $(NODE_MODULES)
bower_modules: $(BOWER_MODULES)

test: $(NODE_MODULES) lint
	$(NPM) test
	$(MAKE) -C c-preload test
	@echo Tests pass

clean:
	rm -rf bower_modules node_modules .npm-updated .bower-updated out static/ext
	$(MAKE) -C d clean
	$(MAKE) -C c-preload clean

run: prereqs
	$(NODE) ./node_modules/.bin/supervisor -w app.js,lib,etc/config -e 'js|node|properties' --exec $(NODE) -- ./app.js --language $(LANG) $(EXTRA_ARGS)

HASH := $(shell git rev-parse HEAD)
dist: prereqs
	rm -rf out/dist
	$(NODE) ./node_modules/requirejs/bin/r.js -o app.build.js
	# Move all assets to a versioned directory
	mkdir -p out/dist/v/$(HASH)
	# main.js
	mv out/dist/main.js* out/dist/v/$(HASH)/
	sed -i -e 's!data-main="main"!data-main="v/'"$(HASH)"'/main"'! out/dist/*.html
	# explorer.css
	mv out/dist/explorer.css out/dist/v/$(HASH)/
	sed -i -e 's!href="explorer.css"!href="v/'"$(HASH)"'/explorer.css"'! out/dist/*.html
	# any actual assets
	mv out/dist/assets/ out/dist/v/$(HASH)/
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
	# rewrite any src refs
	sed -i -e 's!src="!src="v/'"$(HASH)"'/'! out/dist/*.html

c-preload:
	$(MAKE) -C c-preload
