// Copyright (c) 2016, Matt Godbolt
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
//     * Redistributions of source code must retain the above copyright notice, 
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright 
//       notice, this list of conditions and the following disclaimer in the 
//       documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ,
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.

const fs = require('fs');
const AsmParser = require('../lib/asm-parser');
const AsmParserVC = require('../lib/asm-parser-vc');
const utils = require('../lib/utils');
require('chai').should();

function processAsm(filename, filters) {
    const file = fs.readFileSync(filename, 'utf-8');
    let parser;
    if (file.indexOf('Microsoft') >= 0)
        parser = new AsmParserVC();
    else
        parser = new AsmParser();
    return parser.process(file, filters);
}

const cases = fs.readdirSync(__dirname + '/cases')
    .filter(function (x) {
        return x.match(/\.asm$/);
    })
    .map(function (x) {
        return __dirname + '/cases/' + x;
    });

function bless(filename, output, filters) {
    const result = processAsm(__dirname + '/' + filename, filters);
    fs.writeFileSync(__dirname + '/' + output, JSON.stringify(result, null, 2));
}

function dump(file) {
    for (let i = 0; i < file.length; ++i) {
        console.log((i + 1) + " : " + JSON.stringify(file[i]));
    }
}

function testFilter(filename, suffix, filters) {
    const result = processAsm(filename, filters);

    const expected = filename + suffix;
    let json = false;
    let file;
    try {
        file = fs.readFileSync(expected + '.json', 'utf-8');
        json = true;
    } catch (e) {
    }
    if (!file) {
        try {
            file = fs.readFileSync(expected, 'utf-8');
        } catch (e) {
            return;
        }
    }
    it(filename, function () {
        if (json) {
            file = JSON.parse(file);
        } else {
            file = utils.splitLines(file);
        }
        if (json) {
            result.should.deep.equal(file);
        } else {
            result.asm.map(function (x) {
                return x.text;
            }).should.deep.equal(file);
        }
    });
}

// bless("cases/mips5-square.asm", "cases/mips5-square.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/gcc-sum.asm", "cases/gcc-sum.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/gcc-arm-sum.asm", "cases/gcc-arm-sum.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/gcc-avr-sum.asm", "cases/gcc-avr-sum.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/cl-regex.asm", "cases/cl-regex.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/cl-main-opt-out.asm", "cases/cl-main-opt-out.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/avr-loop.asm", "cases/avr-loop.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/bug-192.asm", "cases/bug-192.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/arm-moose.asm", "cases/arm-moose.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/gcc-x86-vector.asm", "cases/gcc-x86-vector.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/clang-on-mac.asm", "cases/clang-on-mac.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/bug-349.asm", "cases/bug-349.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/bug-348.asm", "cases/bug-348.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/bug-660.asm", "cases/bug-660.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/nvcc-example.asm", "cases/nvcc-example.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/clang-cuda-example.asm", "cases/clang-cuda.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/bug-995.asm", "cases/bug-995.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/arm-jump-table.asm", "cases/arm-jump-table.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/bug-1179.asm", "cases/bug-1179.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// bless("cases/6502-square.asm", "cases/6502-square.asm.directives.labels.comments.json", {directives: true, labels: true, commentOnly: true});
// describe('A test', function() {
//     it('should work', function(){
//         console.log(processAsm(__dirname + '/cases/6502-square.asm', {directives: true, labels: true, commentOnly: true}));
//     });
// });

describe('Filter test cases', function () {
    describe('No filters', function () {
        cases.forEach(function (x) {
            testFilter(x, ".none", {});
        });
    });
    describe('Directive filters', function () {
        cases.forEach(function (x) {
            testFilter(x, ".directives", {directives: true});
        });
    });
    describe('Directives and labels together', function () {
        cases.forEach(function (x) {
            testFilter(x, ".directives.labels",
                {directives: true, labels: true});
        });
    });
    describe('Directives, labels and comments', function () {
        cases.forEach(function (x) {
            testFilter(x, ".directives.labels.comments",
                {directives: true, labels: true, commentOnly: true});
        });
    });
    describe('Directives and comments', function () {
        cases.forEach(function (x) {
            testFilter(x, ".directives.comments",
                {directives: true, commentOnly: true});
        });
    });
    describe('Directives and library code', function () {
        cases.forEach(function (x) {
            testFilter(x, ".directives.library",
                {directives: true, libraryCode: true});
        });
    });
    describe('Directives, labels, comments and library code', function () {
        cases.forEach(function (x) {
            testFilter(x, ".directives.labels.comments.library",
                {directives: true, labels: true, commentOnly: true, libraryCode: true});
        });
    });
});
