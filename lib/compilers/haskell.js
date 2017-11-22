var Compile = require('../base-compiler');
var child_process = require('child_process');
var path = require('path');
var _ = require('underscore-node');

function compileHaskell(info, env) {
    var compiler = new Compile(info, env);
    // this is "global" to share state between optionsForFilter and
    // handlePostProcessResult.
    var usedStg = false;

    compiler.optionsForFilter = function (filters, outputFilename, userOptions) {

        usedStg = _.any(userOptions, function(opt) {
            return opt.indexOf("-ddump-stg") > -1;
        });

        if (usedStg) {
            return ["-ddump-stg", "-ddump-to-file", "-o", this.filename(outputFilename)];
        } else {
            return ['-S', '-g', '-o', this.filename(outputFilename)];
        }
    };

    compiler.handlePostProcessResult = function (result, postResult, outputFilename) {
        if (usedStg) {
            const dirExploded = path.parse(outputFilename);
            const catStgFileCommand = 'cat ' + dirExploded.dir + '/' + 'example' + '.dump-stg';
            result.asm = child_process.execSync(catStgFileCommand).toString();
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
