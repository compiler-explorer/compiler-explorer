
const Compile = require('../base-compiler'),
    exec = require('../exec');

function compilePpci(info, env) {
    const compiler = new Compile(info, env);

    compiler.exec = function (compiler, args, options) {
        const python = 'python3';
        options = options || {};
        options.env = {'PYTHONPATH': compiler};
        const python_args = ['-m', 'ppci.cli.cc'].concat(args);
        return exec.execute(python, python_args, options);
    };

    return compiler.initialise();
}

module.exports = compilePpci;
