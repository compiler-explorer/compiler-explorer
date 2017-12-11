

var should = require('chai').should();
var cfg = require('../lib/cfg');
var fs = require('fs');

var cases = fs.readdirSync(__dirname + '/cases')
    .filter(function (x) {
        return x.match(/cfg-\w*/);
    })
    .map(function (x) {
        return __dirname + '/cases/' + x;
    });
    
function common(cases, filterArg, cfgArg){
    cases.filter(function (x) {return x.includes(filterArg);})
        .forEach(function (filename) {
            var file = fs.readFileSync(filename, 'utf-8');
            var content = JSON.parse(file);

            if (file) {
                it(filename, function () {
                    var cfg_ = new cfg.ControlFlowGraph(cfgArg);
                    var result = cfg_.generateCfgStructure(content.asm);
                    result.should.deep.equal(content.cfg);
                });
            }
        });
}    
    
    
describe('Cfg test cases', function () {
    describe('gcc tests', function () {
        common(cases, "gcc", "g++");
    });
    
    describe('clang tests', function () {
        common(cases, "clang", "clang");
    });
});
