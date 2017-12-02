var Compile = require('../base-compiler');

function compileISPC(info, env, langId) {
    var compiler = new Compile(info, env, langId);
    compiler.optionsForFilter = function (filters, outputFilename, userOptions) {
        return ['--target=sse2-i32x4', '--emit-asm', '-g', '-o', this.filename(outputFilename)];
    };
    return compiler.initialise();
}

module.exports = compileISPC;
