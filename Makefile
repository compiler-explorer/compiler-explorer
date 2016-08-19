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
.PHONY: dist
prereqs: optional-d-support node_modules c-preload bower_modules

ifeq "" "$(shell which gdc)"
optional-d-support:
	@echo "D language support disabled"
else
optional-d-support:
	$(MAKE) -C d
endif

NODE_MODULES=.npm-updated
$(NODE_MODULES): package.json
	$(NPM) install
	@touch $@

BOWER_MODULES=.bower-updated
$(BOWER_MODULES): bower.json $(NODE_MODULES)
	$(NODE) ./node_modules/bower/bin/bower install
	@touch $@

LANG:=C++

node_modules: $(NODE_MODULES)
bower_modules: $(BOWER_MODULES)

test:
	(cd test; $(NODE) test.js)
	$(MAKE) -C c-preload test
	@echo Tests pass

clean:
	rm -rf bower_modules node_modules .npm-updated .bower-updated out
	$(MAKE) -C d clean
	$(MAKE) -C c-preload clean

run: prereqs
	$(NODE) ./node_modules/.bin/supervisor -e 'js|node|properties' --exec $(NODE) -- ./app.js --language $(LANG)

dist: prereqs
	$(NODE) ./node_modules/requirejs/bin/r.js -o app.build.js

c-preload:
	$(MAKE) -C c-preload

