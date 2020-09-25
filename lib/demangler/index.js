const makeKeyedTypeGetter = require('../keyed-type').makeKeyedTypeGetter;
const all = require('./_all');

module.exports = {
    getDemanglerTypeByKey: makeKeyedTypeGetter('demangler', all),
};
