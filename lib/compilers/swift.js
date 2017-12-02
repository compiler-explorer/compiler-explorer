const Compile = require('../base-compiler'),
    logger = require('../logger').logger;

function compileSwift(info, env, langId) {
    const compiler = new Compile(info, env, langId);

    compiler.handlePostProcessResult = function (result, postResult) {
        result.asm = postResult.stdout;
        // Seems swift-demangle like to exit with error 1
        if (postResult.code !== 0 && !result.asm) {
            result.asm = "<Error during post processing: " + postResult.code + ">";
            logger.error("Error during post-processing", result);
        }
        return result;
    };
    return compiler.initialise();
}

module.exports = compileSwift;
