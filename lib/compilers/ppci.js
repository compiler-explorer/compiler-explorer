
const Compile = require('../base-compiler'),
    utils = require('../utils');

function compilePpci(info, env) {
    const compiler = new Compile(info, env);

    compiler.runCompiler = function (compiler, options, inputFilename, execOptions) {
        if (!execOptions) {
            execOptions = compiler.getDefaultExecOptions();
        }

        compiler = 'python3';
        python_path = '/opt/compiler-explorer/ppci-0.5.5/';
        execOptions.env['PYTHONPATH'] = python_path;
        var python_options = ['-m', 'ppci.cli.cc'];
        options = python_options.concat(options);
        return this.exec(compiler, options, execOptions).then(function (result) {
            result.inputFilename = inputFilename;
            result.stdout = utils.parseOutput(result.stdout, inputFilename);
            result.stderr = utils.parseOutput(result.stderr, inputFilename);
            return result;
        });
    };

    return compiler.initialise();
}

module.exports = compilePpci;
