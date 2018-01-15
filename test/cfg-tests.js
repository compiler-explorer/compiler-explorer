const chai = require('chai'),
    cfg = require('../lib/cfg'),
    fs = require('fs');

const cases = fs.readdirSync(__dirname + '/cases/')
    .filter(x => x.match(/cfg-\w*/))
    .map(x => __dirname + '/cases/' + x);

const assert = chai.assert;

function common(cases, filterArg, cfgArg) {
    cases.filter(x => x.includes(filterArg))
        .forEach(filename => {
            const file = fs.readFileSync(filename, 'utf-8');
            if (file) {
                const contents = JSON.parse(file);
                assert.deepEqual(cfg.generateStructure(cfgArg, contents.asm), contents.cfg, `${filename}`);
            }
        });
}

describe('Cfg test cases', () => {
    it('works for gcc', () => {
        common(cases, "gcc", "g++");
    });

    it('works for clang', () => {
        common(cases, "clang", "clang");
    });
});
