const
    GccCompiler = require('./gcc');

class TenDRACompiler extends GccCompiler {
    static get key() { return 'tendra'; }

    optionsForFilter(filters, outputFilename) {
        let options = ['-o', this.filename(outputFilename)];
        if (!filters.binary) options = options.concat('-S');
        return options;
    }
}

module.exports = TenDRACompiler;
