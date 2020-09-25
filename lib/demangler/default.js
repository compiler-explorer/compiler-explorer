const BaseDemangler = require('./base').Demangler;

class DefaultDemangler extends BaseDemangler {
    static get key() { return 'default'; }
}

exports.Demangler = DefaultDemangler;
