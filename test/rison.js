import * as rison from '../static/rison';

// Copied from https://github.com/Nanonid/rison/blob/master/python/rison/tests.py
const py_testcases = {
    '(a:0,b:1)': {a: 0, b: 1},
    "(a:0,b:foo,c:'23skidoo')": {a: 0, c: '23skidoo', b: 'foo'},
    '!t': true,
    '!f': false,
    '!n': null,
    "''": '',
    0: 0,
    1.5: 1.5,
    '-3': -3,
    '1e30': 1e30,
    '1e-30': 1.0000000000000001e-30, // eslint-disable-line quote-props
    'G.': 'G.',
    a: 'a',
    "'0a'": '0a',
    "'abc def'": 'abc def',
    '()': {},
    '(a:0)': {a: 0},
    '(id:!n,type:/common/document)': {type: '/common/document', id: null},
    '!()': [],
    "!(!t,!f,!n,'')": [true, false, null, ''],
    "'-h'": '-h',
    'a-z': 'a-z',
    "'wow!!'": 'wow!',
    'domain.com': 'domain.com',
    "'user@domain.com'": 'user@domain.com',
    "'US $10'": 'US $10',
    "'can!'t'": "can't",
};

const encode_testcases = {
    "can't": "'can!'t'",
    '"can\'t"': "'\"can!'t\"'",
    "'can't'": "'!'can!'t!''",
};

describe('Rison test cases', () => {
    for (const [r, obj] of Object.entries(py_testcases)) {
        it(`Should decode "${r}"`, () => {
            // hack to get around "TypeError: Cannot read properties of null (reading 'should')"
            ({x: rison.decode(r)}.should.deep.equal({x: obj}));
        });
        it(`Should encode ${JSON.stringify(obj)}`, () => {
            rison.encode(obj).should.deep.equal(r);
        });
    }
    for (const [obj, r] of Object.entries(encode_testcases)) {
        it(`Should encode ${JSON.stringify(obj)}`, () => {
            rison.encode(obj).should.deep.equal(r);
        });
    }
});
