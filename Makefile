ifeq "" "$(shell which npm)"
default:
	@echo "Please install node.js"
	@echo "Visit http://nodejs.org/ for more details"
	exit 1
else
default: run
endif

node_modules: package.json
	npm install
	@touch $@

clean:
	rm -rf node_modules
.PHONY: clean run

run: node_modules
	./node_modules/.bin/supervisor ./app.js

run-amazon: node_modules
	./node_modules/.bin/supervisor ./app.js --env amazon
