default: run

help: # with thanks to Ben Rady
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

export XZ_OPT := -1 -T 0
export SENTRY_ORG := compiler-explorer

# If you see "node-not-found" then you need to depend on node-installed.
NODE:=node-not-found
NPM:=npm-not-found

# These 'find' scripts cache their results in a dotfile.
# Doing it this way instead of NODE:=$(shell etc/script/find-node) means
# if they fail, they stop the make process. As best I can tell there's no
# way to get make to fail if a sub-shell command fails.
.node-bin: etc/scripts/find-node
	@etc/scripts/find-node .node-bin

# All targets that need node must depend on this to ensure the NODE variable
# is appropriately set, and that PATH is updated.
node-installed: .node-bin
	@$(eval NODE:=$(shell cat .node-bin))
	@$(eval NPM:=$(shell dirname $(shell cat .node-bin))/npm)
	@$(eval PATH=$(shell dirname $(realpath $(NODE))):$(PATH))

info: node-installed ## print out some useful variables
	@echo Using node from $(NODE)
	@echo Using npm from $(NPM)
	@echo PATH is $(PATH)

.PHONY: clean run test run-amazon
.PHONY: dist lint lint-fix ci-lint prereqs node_modules gh-dist check pre-commit
prereqs: node_modules

NODE_MODULES=.npm-updated
$(NODE_MODULES): package.json | node-installed
	$(NPM) install --only=prod $(NPM_FLAGS)
	$(NPM) install --only=dev $(NPM_FLAGS)
	@touch $@

WEBPACK:=./node_modules/webpack-cli/bin/cli.js
$(WEBPACK): $(NODE_MODULES)

lint: $(NODE_MODULES)  ## Checks if the source currently matches code conventions
	$(NPM) run lint

lint-fix: $(NODE_MODULES)  ## Checks if everything matches code conventions & fixes those which are trivial to do so
	$(NPM) run lint-fix

ci-lint: $(NODE_MODULES)
	$(NPM) run ci-lint

node_modules: $(NODE_MODULES)
webpack: $(WEBPACK)  ## Runs webpack (useful only for debugging webpack)
	rm -rf out/dist/static out/dist/manifest.json
	$(WEBPACK) $(WEBPACK_ARGS)

test: $(NODE_MODULES)  ## Runs the tests
	$(NPM) run test
	@echo Tests pass

check: $(NODE_MODULES) test lint  ## Runs all checks required before committing (fixing trivial things automatically)
pre-commit: $(NODE_MODULES) test ci-lint

clean:  ## Cleans up everything
	rm -rf node_modules .*-updated .*-bin out

# Don't use $(NODE) ./node_modules/<path to node_module> as sometimes that's not actually a node script. Instead, rely
# on PATH ensuring "node" is found in our distribution first.
run: export NODE_ENV=production
run: export WEBPACK_ARGS="-p"
run: prereqs webpack  ## Runs the site normally
	./node_modules/.bin/supervisor -w app.js,lib,etc/config -e 'js|node|properties|yaml' --exec $(NODE) $(NODE_ARGS) -- -r esm ./app.js $(EXTRA_ARGS)

dev: export NODE_ENV=development
dev: prereqs install-git-hooks ## Runs the site as a developer; including live reload support and installation of git hooks
	./node_modules/.bin/supervisor -w app.js,lib,etc/config -e 'js|node|properties|yaml' -n exit --exec $(NODE) $(NODE_ARGS) -- -r esm ./app.js $(EXTRA_ARGS)

debug: export NODE_ENV=development
debug: prereqs install-git-hooks ## Runs the site as a developer with full debugging; including live reload support and installation of git hooks
	./node_modules/.bin/supervisor -w app.js,lib,etc/config -e 'js|node|properties|yaml' -n exit --exec $(NODE) $(NODE_ARGS) -- -r esm ./app.js --debug $(EXTRA_ARGS)

HASH := $(shell git rev-parse HEAD)
dist: export NODE_ENV=production
dist: export WEBPACK_ARGS=-p
dist: prereqs webpack  ## Creates a distribution
	echo $(HASH) > out/dist/git_hash

RELEASE_FILE_NAME=$(GITHUB_RUN_NUMBER)
RELEASE_NAME=gh-$(RELEASE_FILE_NAME)
gh-dist: dist  ## Creates a distribution as if we were running on github
	# Output some magic for GH to set the branch name
	echo "::set-output name=branch::$${GITHUB_REF#refs/heads/}"
	echo $(RELEASE_NAME) > out/dist/release_build
	rm -rf out/dist-bin
	mkdir -p out/dist-bin
	tar -Jcf out/dist-bin/$(RELEASE_FILE_NAME).tar.xz -T gh-dist-files.txt
	tar -Jcf out/dist-bin/$(RELEASE_FILE_NAME).static.tar.xz --transform="s,^out/dist/static/,," out/dist/static/*
	echo $(HASH) > out/dist-bin/$(RELEASE_FILE_NAME).txt
	du -ch out/**/*
	# Create and set commits for a sentry release if and only if we have the secure token set
	# External GitHub PRs etc won't have the variable set.
	@[ -z "$(SENTRY_AUTH_TOKEN)" ] || $(NPM) run sentry -- releases new -p compiler-explorer $(RELEASE_NAME)
	@[ -z "$(SENTRY_AUTH_TOKEN)" ] || $(NPM) run sentry -- releases set-commits --auto $(RELEASE_NAME)
	@[ -z "$(SENTRY_AUTH_TOKEN)" ] || $(NPM) run sentry -- releases files $(RELEASE_NAME) upload-sourcemaps out/dist/static

install-git-hooks:  ## Install git hooks that will ensure code is linted and tests are run before allowing a check in
	mkdir -p "$(shell git rev-parse --git-dir)/hooks"
	ln -sf "$(shell pwd)/etc/scripts/pre-commit" "$(shell git rev-parse --git-dir)/hooks/pre-commit"
.PHONY: install-git-hooks

changelog:  ## Create the changelog
	python3 ./etc/scripts/changelog.py

policies:
	python3 ./etc/scripts/politic.py

.PHONY: changelog
