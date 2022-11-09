default: run

help: # with thanks to Ben Rady
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# If you see "node-not-found" then you need to depend on node-installed.
NODE:=node-not-found
NPM:=npm-not-found
NODE_MODULES:=./node_modules/.npm-updated

# These 'find' scripts cache their results in a dotfile.
# Doing it this way instead of NODE:=$(shell etc/script/find-node) means
# if they fail, they stop the make process. As best I can tell there's no
# way to get make to fail if a sub-shell command fails.
.node-bin: etc/scripts/find-node
	@etc/scripts/find-node .node-bin

# All targets that need node must depend on this to ensure the NODE variable
# is appropriately set, and that PATH is updated.
.PHONY: node-installed
node-installed: .node-bin
	@$(eval NODE:=$(shell cat .node-bin))
	@$(eval NPM:=$(shell dirname $(shell cat .node-bin))/npm)
	@$(eval PATH=$(shell dirname $(realpath $(NODE))):$(PATH))

.PHONY: info
info: node-installed ## print out some useful variables
	@echo Using node from $(NODE)
	@echo Using npm from $(NPM)
	@echo PATH is $(PATH)

.PHONY: prereqs
prereqs: $(NODE_MODULES)

$(NODE_MODULES): package.json | node-installed
	$(NPM) install $(NPM_FLAGS)
	@rm -rf node_modules/.cache/esm/*
	@touch $@

.PHONY: lint
lint: $(NODE_MODULES)  ## Checks if the source currently matches code conventions
	$(NPM) run ts-check
	$(NPM) run lint-check

.PHONY: lint-fix
lint-fix: $(NODE_MODULES)  ## Checks if everything matches code conventions & fixes those which are trivial to do so
	$(NPM) run lint

.PHONY: ci-lint
ci-lint: $(NODE_MODULES)
	$(NPM) run ci-lint

.PHONY: test
test: $(NODE_MODULES)  ## Runs the tests
	$(NPM) run test
	@echo Tests pass

.PHONY: test-min
test-min: $(NODE_MODULES)  ## Runs the minimal tests
	$(NPM) run test-min
	@echo Tests pass

.PHONY: check
check: $(NODE_MODULES) lint test  ## Runs all checks required before committing (fixing trivial things automatically)

.PHONY: pre-commit
pre-commit: $(NODE_MODULES) test-min lint

.PHONY: clean
clean:  ## Cleans up everything
	rm -rf node_modules .*-updated .*-bin out

.PHONY: run
run: prereqs  ## Runs the site like it runs in production
	$(NPM) run webpack
	$(NPM) run ts-compile
	env NODE_ENV=production $(NODE) $(NODE_ARGS) -r esm ./out/dist/app.js --webpackContent ./out/webpack/static $(EXTRA_ARGS)

.PHONY: dev
dev: prereqs ## Runs the site as a developer; including live reload support and installation of git hooks
	./node_modules/.bin/supervisor -w app.js,lib,etc/config,static/tsconfig.json -e 'js|ts|node|properties|yaml' -n exit --exec $(NODE) $(NODE_ARGS) -- -r esm -r ts-node/register ./app.js $(EXTRA_ARGS)

.PHONY: gpu-dev
gpu-dev: prereqs ## Runs the site as a developer; including live reload support and installation of git hooks
	./node_modules/.bin/supervisor -w app.js,lib,etc/config,static/tsconfig.json -e 'js|ts|node|properties|yaml' -n exit --exec $(NODE) $(NODE_ARGS) -- -r esm -r ts-node/register ./app.js --env gpu $(EXTRA_ARGS)

.PHONY: debug
debug: prereqs ## Runs the site as a developer with full debugging; including live reload support and installation of git hooks
	./node_modules/.bin/supervisor -w app.js,lib,etc/config,static/tsconfig.json -e 'js|ts|node|properties|yaml' -n exit --inspect 9229 --exec $(NODE) $(NODE_ARGS) -- -r esm -r ts-node/register ./app.js --debug $(EXTRA_ARGS)
