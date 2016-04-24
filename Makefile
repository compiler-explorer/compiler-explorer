ifeq "" "$(shell which npm)"
default:
	@echo "Please install node.js"
	@echo "Visit http://nodejs.org/ for more details"
	@echo "On Ubuntu/Debian try: sudo apt-get install nodejs npm"
	exit 1
else
NODE:= $(shell which node || which nodejs)
default: run
endif

.PHONY: clean run test run-amazon c-preload optional-d-support prereqs
prereqs: optional-d-support node_modules c-preload

ifeq "" "$(shell which gdc)"
optional-d-support:
	@echo "D language support disabled"
else
optional-d-support:
	$(MAKE) -C d
endif

NODE_MODULES=.npm-updated
$(NODE_MODULES): package.json
	npm install
	@touch $@

LANG:=C++

node_modules: $(NODE_MODULES)

test:
	(cd test; $(NODE) test.js)
	$(MAKE) -C c-preload test
	@echo Tests pass

clean:
	rm -rf node_modules .npm-updated
	$(MAKE) -C d clean
	$(MAKE) -C c-preload clean

run: node_modules optional-d-support c-preload
	$(NODE) ./node_modules/.bin/supervisor -e 'js|node|properties' --exec $(NODE) -- ./app.js --language $(LANG)

c-preload:
	$(MAKE) -C c-preload
