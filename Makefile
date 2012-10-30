ifeq "" "$(shell which npm)"
default:
	@echo "Please install node.js"
	@echo "Visit http://nodejs.org/ for more details"
	@echo "On Ubuntu/Debian try: sudo apt-get install nodejs npm"
	exit 1
else
default: run
endif

ifeq "" "$(shell which gdc)"
optional_d_support:
	@echo "D language support disabled"
else
optional_d_support:
	$(MAKE) -C d
endif

NODE_MODULES=.npm-updated
$(NODE_MODULES): package.json
	npm install
	@touch $@

node_modules: $(NODE_MODULES)

test:
	(cd test; node test.js)
	@echo Tests pass

clean:
	rm -rf node_modules

.PHONY: clean run test run-amazon

run: node_modules optional_d_support
	./node_modules/.bin/supervisor ./app.js

c-preload:
	$(MAKE) -C c-preload

run-amazon: node_modules optional_d_support c-preload
	./node_modules/.bin/supervisor -- ./app.js --env amazon

run-amazon-d: node_modules optional_d_support c-preload
	./node_modules/.bin/supervisor -- ./app.js --env amazon-d

