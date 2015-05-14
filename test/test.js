#!/usr/bin/env node

// Copyright (c) 2012-2015, Matt Godbold
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

var fs = require('fs');
var asm = require('../static/asm.js');

function processAsm(filename, filters) {
    var file = fs.readFileSync(filename, 'utf-8');
    return asm.processAsm(file, filters);
}

var cases = fs.readdirSync('./cases')
    .filter(function (x) {
        return x.match(/\.asm$/)
    })
    .map(function (x) {
        return './cases/' + x;
    });

var failures = 0;

function assertEq(a, b, context) {
    if (a != b) {
        console.log("Fail: ", a, " != ", b, context);
        failures++;
    }
}

function testFilter(filename, suffix, filters) {
    var result = processAsm(filename, filters);
    var expected = filename + suffix;
    try {
        var file = fs.readFileSync(expected, 'utf-8').split('\n');
    } catch (e) {
        return;
    }
    assertEq(file.length, result.length, expected);
    if (file.length != result.length) return;
    for (var i = 0; i < file.length; ++i) {
        assertEq(file[i], result[i].text, expected + ":" + (i + 1));
    }
}


cases.forEach(function (x) {
    testFilter(x, "", {})
});
cases.forEach(function (x) {
    testFilter(x, ".directives", {directives: true})
});
cases.forEach(function (x) {
    testFilter(x, ".directives.labels",
        {directives: true, labels: true})
});
cases.forEach(function (x) {
    testFilter(x, ".directives.labels.comments",
        {directives: true, labels: true, commentOnly: true})
});

if (failures) {
    console.log(failures + " failures");
}
