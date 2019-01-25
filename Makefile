default: run

export XZ_OPT=-1 -T 0

# If you see "node-not-found" or "yarn-not-found" then you need to depend
# on either node-installed or yarn-installed.
NODE:=node-not-found
YARN:=yarn-not-found

# These 'find' scripts cache their results in a dotfile.
# Doing it this way instead of NODE:=$(shell etc/script/find-node) means
# if they fail, they stop the make process. As best I can tell there's no
# way to get make to fail if a sub-shell command fails.
.node-bin: etc/scripts/find-node
	@etc/scripts/find-node .node-bin
.yarn-bin: etc/scripts/find-yarn node-installed
	@etc/scripts/find-yarn .yarn-bin

# All targets that need node must depend on this to ensure the NODE variable
# is appropriately set, and that PATH is updated so that yarn etc will use this
# node and not any other random node on the PATH.
node-installed: .node-bin
	@$(eval NODE:=$(shell cat .node-bin))
	@$(eval PATH=$(shell dirname $(realpath $(NODE))):${PATH})
# All targets that need yarn must depend on this to ensure YARN is set.
yarn-installed: .yarn-bin
	@$(eval YARN:=$(shell cat .yarn-bin))

debug: node-installed yarn-installed
	@echo Using node from $(NODE)
	@echo Using yarn from $(YARN)
	@echo PATH is $(PATH)

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


NODE_MODULES=.yarn-updated
$(NODE_MODULES): package.json yarn-installed
	$(YARN) install $(YARN_FLAGS)
	@touch $@

webpack: $(NODE_MODULES)
	$(NODE) node_modules/webpack/bin/webpack.js ${WEBPACK_ARGS}

lint: $(NODE_MODULES)
	$(YARN) run lint

node_modules: $(NODE_MODULES)
webpack: $(WEBPACK)

test: $(NODE_MODULES)
	$(YARN) run test
	-$(MAKE) -C c-preload test
	@echo Tests pass

check: $(NODE_MODULES) test lint

clean:
	rm -rf node_modules .*-updated .*-bin out static/dist static/vs
	$(MAKE) -C d clean
	$(MAKE) -C c-preload clean

run: export NODE_ENV=LOCAL WEBPACK_ARGS="-p"
run: prereqs
	$(NODE) ./node_modules/.bin/supervisor -w app.js,lib,etc/config -e 'js|node|properties' --exec $(NODE) $(NODE_ARGS) -- ./app.js $(EXTRA_ARGS)

dev: export NODE_ENV=DEV
dev: prereqs install-git-hooks
	 $(NODE) ./node_modules/.bin/supervisor -w app.js,lib,etc/config -e 'js|node|properties' --exec $(NODE) $(NODE_ARGS) -- ./app.js $(EXTRA_ARGS)


HASH := $(shell git rev-parse HEAD)
dist: export WEBPACK_ARGS=-p
dist: prereqs
	rm -rf out/dist/
	mkdir -p out/dist
	mkdir -p out/dist/vs
	cp -r static/dist/ out/dist/
	cp -r static/vs/ out/dist/
	cp -r static/policies/ out/dist/
	echo ${HASH} > out/dist/git_hash

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

changelog:
	python ./etc/scripts/changelog.py
.PHONY: changelog
