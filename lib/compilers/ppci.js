
const Compile = require('../base-compiler'),
    exec = require('../exec'),
    logger = require('../logger').logger;

function compilePpci(info, env) {
    const compiler = new Compile(info, env);
    const originalExec = compiler.exec;

    compiler.exec = function (compiler, args, options) {
        if (compiler.endsWith('.py')) {
            const python = env.ceProps("python3");
            options = options || {};

            const matches = compiler.match(/^(.*)(\/ppci\/)(.*).py/);
            if (matches) {
                const pythonpath = matches[1];
                const ppciname = "ppci." + matches[3].replace('/', '.');
                options.env = {'PYTHONPATH': pythonpath};
                let python_args = ['-m', ppciname].concat(args);
                return exec.execute(python, python_args, options);
            } else {
                logger.error('invalid ppci path');
            }
        } else {
            return originalExec(compiler, args, options);
        }
    };

    return compiler.initialise();
}

module.exports = compilePpci;
