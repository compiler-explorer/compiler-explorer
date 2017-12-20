var Compile = require('../base-compiler');

function compileHaskell(info, env, langId) {
    var compiler = new Compile(info, env, langId);
    compiler.optionsForFilter = function (filters, outputFilename, userOptions) {
        return ['-S', '-g', '-o', this.filename(outputFilename)];
    };
    return compiler.initialise();
}

module.exports = compileHaskell;
