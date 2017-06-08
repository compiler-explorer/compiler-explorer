var Compile = require('../base-compiler');

function compileHaskell(info, env) {
    var compiler = new Compile(info, env);
    compiler.optionsForFilter = function (filters, outputFilename) {
        return ['-S', '-g', '-o', this.filename(outputFilename)];
    };
    return compiler.initialise();
}

module.exports = compileHaskell;
