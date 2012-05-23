default: run

node_modules: package.json
	npm install
	@touch $@

clean:
	rm -rf node_modules
.PHONY: clean run

run: node_modules
	./node_modules/.bin/supervisor ./app.js
