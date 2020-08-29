const
    GccCompiler = require('./gcc');

class TenDRACompiler extends GccCompiler {
    optionsForFilter(filters, outputFilename) {
        let options = ['-o', this.filename(outputFilename)];
        if (!filters.binary) options = options.concat('-S');
        return options;
    }
}

module.exports = TenDRACompiler;
