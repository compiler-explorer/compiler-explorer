var Compile = require('../base-compiler');
var logger = require('../../lib/logger').logger;
var child_process = require('child_process');
var path = require('path');
var _ = require('underscore-node');

function compileHaskell(info, env) {
    var compiler = new Compile(info, env);
    compiler.optionsForFilter = function (filters, outputFilename, userOptions) {

        usedStg = _.any(userOptions, function(opt) {
            return opt.indexOf("-ddump-stg") > -1;
        });

        logger.warn("usedStg: ", usedStg);

        if (usedStg) {
            return ["-ddump-stg", "-ddump-to-file", "-o", this.filename(outputFilename)];
        } else {
            return ['-S', '-g', '-o', this.filename(outputFilename)];
        }
    };

    compiler.handlePostProcessResult = function (result, postResult, outputFilename) {
        if (usedStg) {
            logger.info("handlePostProcessResult::usedStg");
            const dirExploded = path.parse(outputFilename);
            logger.info(dirExploded);
            const catStgFileCommand = 'cat ' + dirExploded.dir + '/' + 'example' + '.dump-stg';
            logger.info(catStgFileCommand);
            const stgData = child_process.execSync(catStgFileCommand).toString();
            result.asm = stgData;
            return result;
        }
        else {
            result.asm = postResult.stdout;
            return result;
        }
    };

    return compiler.initialise();
}

module.exports = compileHaskell;
