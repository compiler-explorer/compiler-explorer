var Compile = require('../base-compiler');

function compileISPC(info, env) {
    var compiler = new Compile(info, env);
    compiler.optionsForFilter = function (filters, outputFilename) {
        return ['--target=sse2-i32x4', '--emit-asm', '-g', '-o', this.filename(outputFilename)];
    };
    return compiler.initialise();
}

module.exports = compileISPC;
