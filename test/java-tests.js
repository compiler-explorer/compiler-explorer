// Copyright (c) 2017, Compiler Explorer Authors
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
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

const chai = require('chai');
const chaiAsPromised = require('chai-as-promised');
const JavaCompiler = require('../lib/compilers/java');
const fs = require('fs-extra');
const utils = require('../lib/utils');
const {makeCompilationEnvironment} = require('./utils');

chai.use(chaiAsPromised);
chai.should();

const languages = {
    java: {id: 'java'},
};

const info = {
    exe: null,
    remote: true,
    lang: languages.java.id,
};




// Temporarily disabled: see #1438
describe.skip('Basic compiler setup', function () {
    let env;

    before(() => {
        env = makeCompilationEnvironment({languages});
    });

    it('Should not crash on instantiation', function () {
        new JavaCompiler(info, env);
    });


    it('should ignore second param for getOutputFilename', function () {
        // Because javac produces a class files based on user provided class names,
        // it's not possible to determine the main class file before compilation/parsing
        const compiler = new JavaCompiler(info, env);
        if (process.platform === 'win32') {
            compiler.getOutputFilename('/tmp/', 'Ignored.java').should.equal('\\tmp\\example.class');
        } else {
            compiler.getOutputFilename('/tmp/', 'Ignored.java').should.equal('/tmp/example.class');
        }
    });

    describe.skip('Forbidden compiler arguments', function () {
        it('JavaCompiler should not allow -d parameter', () => {
            const compiler = new JavaCompiler(info, env);
            compiler.filterUserOptions(['hello', '-d', '--something', '--something-else']).should.deep.equal(
                ['hello', '--something-else'],
            );
            compiler.filterUserOptions(['hello', '-d']).should.deep.equal(
                ['hello'],
            );
            compiler.filterUserOptions(['-d', 'something', 'something-else']).should.deep.equal(
                ['something-else'],
            );
        });

        it('JavaCompiler should not allow -s parameter', () => {
            const compiler = new JavaCompiler(info, env);
            compiler.filterUserOptions(['hello', '-s', '--something', '--something-else']).should.deep.equal(
                ['hello', '--something-else'],
            );
            compiler.filterUserOptions(['hello', '-s']).should.deep.equal(
                ['hello'],
            );
            compiler.filterUserOptions(['-s', 'something', 'something-else']).should.deep.equal(
                ['something-else'],
            );
        });

        it('JavaCompiler should not allow --source-path parameter', () => {
            const compiler = new JavaCompiler(info, env);
            compiler.filterUserOptions(['hello', '--source-path', '--something', '--something-else']).should.deep.equal(
                ['hello', '--something-else'],
            );
            compiler.filterUserOptions(['hello', '--source-path']).should.deep.equal(
                ['hello'],
            );
            compiler.filterUserOptions(['--source-path', 'something', 'something-else']).should.deep.equal(
                ['something-else'],
            );
        });

        it('JavaCompiler should not allow -sourcepath parameter', () => {
            const compiler = new JavaCompiler(info, env);
            compiler.filterUserOptions(['hello', '-sourcepath', '--something', '--something-else']).should.deep.equal(
                ['hello', '--something-else'],
            );
            compiler.filterUserOptions(['hello', '-sourcepath']).should.deep.equal(
                ['hello'],
            );
            compiler.filterUserOptions(['-sourcepath', 'something', 'something-else']).should.deep.equal(
                ['something-else'],
            );
        });
    });
});

describe.skip('javap parsing', () => {
    let compiler;
    let env;
    before(() => {
        const env = makeCompilationEnvironment({languages});
        compiler = new JavaCompiler(info, env);
    });

    function testJava(baseFolder, ...classNames) {
        const compiler = new JavaCompiler(info, env);

        const asm = classNames.map(className => fs.readFileSync(`${baseFolder}/${className}.asm`).toString());

        const output = utils.splitLines(fs.readFileSync(`${baseFolder}/output.asm`).toString());
        const expectedSegments = output.map(line => {
            const match = line.match(/^line (\d+):(.*)$/);
            if (match) {
                return {
                    text: match[2],
                    source: {
                        line: parseInt(match[1]),
                        file: null,
                    },
                };
            }
            return {
                text: line,
                source: null,
            };
        });

        const result = {
            asm,
        };

        const processed = compiler.processAsm(result).asm;
        processed.should.deep.equal(expectedSegments);
    }

    it('should handle errors', () => {
        const result = {
            asm: '<Compilation failed>',
        };

        compiler.processAsm(result).should.deep.equal([
            {text: '<Compilation failed>', source: null},
        ]);
    });

    it('Parses simple class with one method', () => {
        return Promise.all([
            testJava('test/java/square', 'javap-square'),
        ]);
    });

    it('Preserves ordering of multiple classes', () => {
        return Promise.all([
            testJava('test/java/two-classes', 'ZFirstClass', 'ASecondClass'),
            testJava('test/java/two-classes', 'ASecondClass', 'ZFirstClass'),
        ]);
    });
});


