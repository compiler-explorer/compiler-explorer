var Compile = require('../base-compiler');

function compileFPC(info, env) {
    var compiler = new Compile(info, env);
    compiler.supportsOptOutput = false;
    compiler.supportsBinary = false;

    compiler.getOutputFilename = function (dirPath) {
       var logger = require('../logger').logger;
       var path = require("path");
       var outputFilebase = "output";
       var outputFilename = path.join(dirPath, outputFilebase + ".o");
       logger.info("outputFilename is " + outputFilename);
       return outputFilename;
    };

    compiler.optionsForFilter = function (filters, outputFilename, userOptions) {
        filters.execute = false;

        return [];
    };

    return compiler.initialise();
}

module.exports = compileFPC;
