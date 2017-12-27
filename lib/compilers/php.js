var Compile = require('../base-compiler'),
	fs = require('fs-extra'),
	php = require('../php'),
	utils = require('../utils');

function compilePHP(info, env) {
    var compiler = new Compile(info, env);
    compiler.supportsIntel = false;
    compiler.couldSupportASTDump = function() { return false; };
    compiler.asm = new php.PHPByteCodeParser(env.compilerProps);
    compiler.optionsForFilter = function (filters, outputFilename, userOptions) {
    	// Add options for vld extension.
        return ['-dvld.active=1','-dvld.execute=0'];
    };

	compiler.prepareArguments = function (userOptions, filters, backendOptions, inputFilename, outputFilename) {
		let options = this.optionsForFilter(filters, outputFilename, userOptions);
		backendOptions = backendOptions || {};

		if (this.compiler.options) {
			options = options.concat(this.compiler.options.split(" "));
		}

		if (this.compiler.supportsOptOutput && backendOptions.produceOptInfo) {
			options = options.concat(this.compiler.optArg);
		}

		// Only allow "-e" user option, everthing else looks dangerous
		userOptions = (userOptions || []).filter(option => option == '-e');

		// Hack: Pass output filename as the last option
		return options.concat(userOptions || []).concat([this.filename(inputFilename), outputFilename]);
	};

    compiler.runCompiler = function (compiler, options, inputFilename, execOptions) {
    	// Hack: Get output filename from last option
    	let outputFilename = options.pop();

        return this.exec(compiler, options, execOptions).then(function (result) {
            result.inputFilename = inputFilename;
            result.stdout = utils.parseOutput(result.stdout, inputFilename);
            if (result.code === 0) {
            	// Actual bytecode is emitted to stderror, write it to the output file
            	fs.writeFileSync(outputFilename, result.stderr);
            	result.stderr = "";
            } else {
            	result.stderr = utils.parseOutput(result.stderr, inputFilename);
            }
            return result;
        });
    };

    return compiler.initialise();
}

module.exports = compilePHP;
