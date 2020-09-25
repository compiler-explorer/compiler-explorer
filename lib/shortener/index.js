const makeKeyedTypeGetter = require('../keyed-type').makeKeyedTypeGetter;
const all = require('./_all');

module.exports = {
    getShortenerTypeByKey: makeKeyedTypeGetter('shortener', all),
};
