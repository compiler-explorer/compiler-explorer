// Copyright (c) 2017, Patrick Quist
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
const chaiAsPromised = require("chai-as-promised");
const PascalDemangler = require('../lib/demangler-pascal').Demangler;
const PascalCompiler = require('../lib/compilers/pascal');
const CompilationEnvironment = require('../lib/compilation-env');
const fs = require('fs-extra');
const utils = require('../lib/utils');
const properties = require('../lib/properties');

chai.use(chaiAsPromised);
chai.should();

const languages = {
    pascal: {id: 'pascal'}
};

describe('Pascal', () => {
    let compiler;

    before(() => {
        const compilerProps = new properties.CompilerProps(languages, properties.fakeProps({}));
        const ce = new CompilationEnvironment(compilerProps);
        const info = {
            exe: null,
            remote: true,
            lang: languages.pascal.id
        };

        compiler = new PascalCompiler(info, ce);
    });

    it('Basic compiler setup', () => {
        if (process.platform === "win32") {
            compiler.getOutputFilename("/tmp/", "output.pas").should.equal("\\tmp\\output.s");
        } else {
            compiler.getOutputFilename("/tmp/", "output.pas").should.equal("/tmp/output.s");
        }
    });

    describe('Pascal signature composer function', function () {
        const demangler = new PascalDemangler();

        it('Handle 0 parameter methods', function () {
            demangler.composeReadableMethodSignature("", "", "myfunc", "").should.equal("myfunc()");
            demangler.composeReadableMethodSignature("output", "", "myfunc", "").should.equal("myfunc()");
            demangler.composeReadableMethodSignature("output", "tmyclass", "myfunc", "").should.equal("tmyclass.myfunc()");
        });

        it('Handle 1 parameter methods', function () {
            demangler.composeReadableMethodSignature("output", "", "myfunc", "integer").should.equal("myfunc(integer)");
            demangler.composeReadableMethodSignature("output", "tmyclass", "myfunc", "integer").should.equal("tmyclass.myfunc(integer)");
        });

        it('Handle 2 parameter methods', function () {
            demangler.composeReadableMethodSignature("output", "", "myfunc", "integer,string").should.equal("myfunc(integer,string)");
            demangler.composeReadableMethodSignature("output", "tmyclass", "myfunc", "integer,string").should.equal("tmyclass.myfunc(integer,string)");
        });
    });

    describe('Pascal Demangling FPC 2.6', function () {
        const demangler = new PascalDemangler();

        it('Should demangle OUTPUT_MAXARRAY$array_of_DOUBLE$array_of_DOUBLE', function () {
            demangler.demangle("OUTPUT_MAXARRAY$array_of_DOUBLE$array_of_DOUBLE:").should.equal("maxarray(array_of_double,array_of_double)");
        });

        it('Should demangle OUTPUT_TMYCLASS_$__MYPROC$ANSISTRING', function () {
            demangler.demangle("OUTPUT_TMYCLASS_$__MYPROC$ANSISTRING:").should.equal("tmyclass.myproc(ansistring)");
        });

        it('Should demangle OUTPUT_TMYCLASS_$__MYFUNC$$ANSISTRING', function () {
            demangler.demangle("OUTPUT_TMYCLASS_$__MYFUNC$$ANSISTRING:").should.equal("tmyclass.myfunc()");
        });

        it('Should demangle OUTPUT_NOPARAMFUNC$$ANSISTRING', function () {
            demangler.demangle("OUTPUT_NOPARAMFUNC$$ANSISTRING:").should.equal("noparamfunc()");
        });

        it('Should demangle OUTPUT_NOPARAMPROC', function () {
            demangler.demangle("OUTPUT_NOPARAMPROC:").should.equal("noparamproc()");
        });

        it('Should demangle U_OUTPUT_MYGLOBALVAR', function () {
            demangler.demangle("U_OUTPUT_MYGLOBALVAR:").should.equal("myglobalvar");
        });

        it('Should demangle OUTPUT_INIT (custom method)', function () {
            demangler.demangle("OUTPUT_INIT:").should.equal("init()");
        });

        it('Should demangle OUTPUT_init (builtin symbol)', function () {
            demangler.demangle("OUTPUT_init:").should.equal("unit_initialization");
        });
    });

    describe('Pascal Demangling FPC 3.2', function () {
        const demangler = new PascalDemangler();

        it('Should demangle OUTPUT_$$_SQUARE$LONGINT$$LONGINT', function () {
            demangler.demangle("OUTPUT_$$_SQUARE$LONGINT$$LONGINT:").should.equal("square(longint)");
        });

        it('Should demangle OUTPUT_$$_MAXARRAY$array_of_DOUBLE$array_of_DOUBLE', function () {
            demangler.demangle("OUTPUT_$$_MAXARRAY$array_of_DOUBLE$array_of_DOUBLE:").should.equal("maxarray(array_of_double,array_of_double)");
        });

        it('Should demangle OUTPUT$_$TMYCLASS_$__$$_MYPROC$ANSISTRING', function () {
            demangler.demangle("OUTPUT$_$TMYCLASS_$__$$_MYPROC$ANSISTRING:").should.equal("tmyclass.myproc(ansistring)");
        });

        it('Should demangle OUTPUT$_$TMYCLASS_$__$$_MYFUNC$$ANSISTRING', function () {
            demangler.demangle("OUTPUT$_$TMYCLASS_$__$$_MYFUNC$$ANSISTRING:").should.equal("tmyclass.myfunc()");
        });

        it('Should demangle OUTPUT$_$TMYCLASS_$__$$_MYFUNC$ANSISTRING$$INTEGER', function () {
            demangler.demangle("OUTPUT$_$TMYCLASS_$__$$_MYFUNC$ANSISTRING$$INTEGER:").should.equal("tmyclass.myfunc(ansistring)");
        });

        it('Should demangle OUTPUT$_$TMYCLASS_$__$$_MYFUNC$ANSISTRING$INTEGER$INTEGER$$INTEGER', function () {
            demangler.demangle("OUTPUT$_$TMYCLASS_$__$$_MYFUNC$ANSISTRING$INTEGER$INTEGER$$INTEGER:").should.equal("tmyclass.myfunc(ansistring,integer,integer)");
        });

        it('Should demangle OUTPUT_$$_NOPARAMFUNC$$ANSISTRING', function () {
            demangler.demangle("OUTPUT_$$_NOPARAMFUNC$$ANSISTRING:").should.equal("noparamfunc()");
        });

        it('Should demangle OUTPUT_$$_NOPARAMPROC', function () {
            demangler.demangle("OUTPUT_$$_NOPARAMPROC:").should.equal("noparamproc()");
        });

        it('Should demangle OUTPUT_$$_INIT', function () {
            demangler.demangle("OUTPUT_$$_INIT:").should.equal("init()");
        });

        it('Should demangle U_$OUTPUT_$$_MYGLOBALVAR', function () {
            demangler.demangle("U_$OUTPUT_$$_MYGLOBALVAR:").should.equal("myglobalvar");
        });
    });

    describe('Pascal Demangling Fixed Symbols FPC 2.6', function () {
        const demangler = new PascalDemangler();

        it('Should demangle OUTPUT_finalize_implicit', function () {
            demangler.demangle("OUTPUT_finalize_implicit:").should.equal("unit_finalization_implicit");
        });
    });

    describe('Pascal Demangling Fixed Symbols FPC 3.2', function () {
        const demangler = new PascalDemangler();

        it('Should demangle OUTPUT_$$_init', function () {
            demangler.demangle("OUTPUT_$$_init:").should.equal("unit_initialization");
        });

        it('Should demangle OUTPUT_$$_finalize', function () {
            demangler.demangle("OUTPUT_$$_finalize:").should.equal("unit_finalization");
        });

        it('Should demangle OUTPUT_$$_init_implicit', function () {
            demangler.demangle("OUTPUT_$$_init_implicit:").should.equal("unit_initialization_implicit");
        });

        it('Should demangle OUTPUT_$$_finalize_implicit', function () {
            demangler.demangle("OUTPUT_$$_finalize_implicit:").should.equal("unit_finalization_implicit");
        });

        it('Should demangle OUTPUT_$$_finalize_implicit', function () {
            demangler.demangle("OUTPUT_$$_finalize_implicit:").should.equal("unit_finalization_implicit");
        });
    });

    describe('Pascal NOT Demangling certain symbols FPC 2.6', function () {
        const demangler = new PascalDemangler();

        it('Should NOT demangle VMT_OUTPUT_TMYCLASS', function () {
            demangler.demangle("VMT_OUTPUT_TMYCLASS:").should.equal(false);
        });

        it('Should NOT demangle RTTI_OUTPUT_TMYCLASS', function () {
            demangler.demangle("RTTI_OUTPUT_TMYCLASS:").should.equal(false);
        });

        it('Should NOT demangle INIT$_OUTPUT', function () {
            demangler.demangle("INIT$_OUTPUT:").should.equal(false);
        });

        it('Should NOT demangle FINALIZE$_OUTPUT', function () {
            demangler.demangle("FINALIZE$_OUTPUT:").should.equal(false);
        });

        it('Should NOT demangle DEBUGSTART_OUTPUT', function () {
            demangler.demangle("DEBUGSTART_OUTPUT:").should.equal(false);
        });

        it('Should NOT demangle DBGREF_OUTPUT_THELLO', function () {
            demangler.demangle("DBGREF_OUTPUT_THELLO:").should.equal(false);
        });

        it('Should NOT demangle non-label', function () {
            demangler.demangle("  call OUTPUT$_$TMYCLASS_$__$$_MYTEST2").should.equal(false);
        });
    });

    describe('Pascal NOT Demangling certain symbols FPC 3.2', function () {
        const demangler = new PascalDemangler();

        it('Should NOT demangle RTTI_$OUTPUT_$$_TMYCLASS', function () {
            demangler.demangle("RTTI_$OUTPUT_$$_TMYCLASS:").should.equal(false);
        });

        it('Should NOT demangle .Ld1', function () {
            demangler.demangle(".Ld1:").should.equal(false);
        });

        it('Should NOT demangle _$OUTPUT$_Ld3 (Same in FPC 2.6 and 3.2)', function () {
            demangler.demangle("_$OUTPUT$_Ld3:").should.equal(false);
        });

        it('Should NOT demangle INIT$_$OUTPUT', function () {
            demangler.demangle("INIT$_$OUTPUT:").should.equal(false);
        });

        it('Should NOT demangle DEBUGSTART_$OUTPUT', function () {
            demangler.demangle("DEBUGSTART_$OUTPUT:").should.equal(false);
        });

        it('Should NOT demangle DBGREF_$OUTPUT_$$_THELLO', function () {
            demangler.demangle("DBGREF_$OUTPUT_$$_THELLO:").should.equal(false);
        });
    });

    describe('Add, order and demangle inline', function () {
        const demangler = new PascalDemangler();

        demangler.demangle("OUTPUT$_$TMYCLASS_$__$$_MYTEST:");
        demangler.demangle("U_$OUTPUT_$$_MYGLOBALVAR:");
        demangler.demangle("OUTPUT$_$TMYCLASS_$__$$_MYTEST2:");
        demangler.demangle("OUTPUT$_$TMYCLASS_$__$$_MYOVERLOAD$ANSISTRING:");
        demangler.demangle("OUTPUT$_$TMYCLASS_$__$$_MYOVERLOAD$INTEGER:");

        demangler.demangleIfNeeded("  call OUTPUT$_$TMYCLASS_$__$$_MYTEST2").should.equal("  call tmyclass.mytest2()");
        demangler.demangleIfNeeded("  movl U_$OUTPUT_$$_MYGLOBALVAR,%eax").should.equal("  movl myglobalvar,%eax");
        demangler.demangleIfNeeded("  call OUTPUT$_$TMYCLASS_$__$$_MYTEST2").should.equal("  call tmyclass.mytest2()");
        demangler.demangleIfNeeded("  call OUTPUT$_$TMYCLASS_$__$$_MYTEST").should.equal("  call tmyclass.mytest()");
        demangler.demangleIfNeeded("  call OUTPUT$_$TMYCLASS_$__$$_MYOVERLOAD$ANSISTRING").should.equal("  call tmyclass.myoverload(ansistring)");
        demangler.demangleIfNeeded("  call OUTPUT$_$TMYCLASS_$__$$_MYOVERLOAD$INTEGER").should.equal("  call tmyclass.myoverload(integer)");

        demangler.demangleIfNeeded(".Le1").should.equal(".Le1");
        demangler.demangleIfNeeded("_$SomeThing").should.equal("_$SomeThing");
    });

    describe('Add, order and demangle inline - using addDemangleToCache()', function () {
        const demangler = new PascalDemangler();

        demangler.addDemangleToCache("OUTPUT$_$TMYCLASS_$__$$_MYTEST:");
        demangler.addDemangleToCache("U_$OUTPUT_$$_MYGLOBALVAR:");
        demangler.addDemangleToCache("OUTPUT$_$TMYCLASS_$__$$_MYTEST2:");
        demangler.addDemangleToCache("OUTPUT$_$TMYCLASS_$__$$_MYOVERLOAD$ANSISTRING:");
        demangler.addDemangleToCache("OUTPUT$_$TMYCLASS_$__$$_MYOVERLOAD$INTEGER:");

        demangler.demangleIfNeeded("  call OUTPUT$_$TMYCLASS_$__$$_MYTEST2").should.equal("  call tmyclass.mytest2()");
        demangler.demangleIfNeeded("  movl U_$OUTPUT_$$_MYGLOBALVAR,%eax").should.equal("  movl myglobalvar,%eax");
        demangler.demangleIfNeeded("  call OUTPUT$_$TMYCLASS_$__$$_MYTEST2").should.equal("  call tmyclass.mytest2()");
        demangler.demangleIfNeeded("  call OUTPUT$_$TMYCLASS_$__$$_MYTEST").should.equal("  call tmyclass.mytest()");
        demangler.demangleIfNeeded("  call OUTPUT$_$TMYCLASS_$__$$_MYOVERLOAD$ANSISTRING").should.equal("  call tmyclass.myoverload(ansistring)");
        demangler.demangleIfNeeded("  call OUTPUT$_$TMYCLASS_$__$$_MYOVERLOAD$INTEGER").should.equal("  call tmyclass.myoverload(integer)");

        demangler.demangleIfNeeded(".Le1").should.equal(".Le1");
    });

    describe('Pascal Ignored Symbols', function () {
        const demangler = new PascalDemangler();

        it('Should ignore certain labels', function () {
            demangler.shouldIgnoreSymbol(".Le1").should.equal(true);
            demangler.shouldIgnoreSymbol("_$SomeThing").should.equal(true);
        });

        it('Should be able to differentiate between System and User functions', function () {
            demangler.shouldIgnoreSymbol("RTTI_OUTPUT_MyProperty").should.equal(true);
            demangler.shouldIgnoreSymbol("Rtti_Output_UserFunction").should.equal(false);
        });
    });

    describe('Pascal ASM line number injection', function () {
        before(() => {
            compiler.demanglerClass = require("../lib/demangler-pascal").Demangler;
            compiler.demangler = new compiler.demanglerClass(null, compiler);
        });

        it('Should have line numbering', function () {
            return new Promise(function (resolve, reject) {
                fs.readFile("test/pascal/asm-example.s", function (err, buffer) {
                    const asmLines = utils.splitLines(buffer.toString());
                    compiler.preProcessLines(asmLines);

                    resolve(Promise.all([
                        asmLines.should.include("# [output.pas]"),
                        asmLines.should.include("  .file 1 \"<stdin>\""),
                        asmLines.should.include("# [13] Square := num * num + 14;"),
                        asmLines.should.include("  .loc 1 13 0"),
                        asmLines.should.include(".Le0:"),
                        asmLines.should.include("  .cfi_endproc")
                    ]));
                });
            });
        });
    });

    describe('Pascal objdump filtering', function () {
        it('Should filter out most of the runtime', function () {
            return new Promise(function (resolve, reject) {
                fs.readFile("test/pascal/objdump-example.s", function (err, buffer) {
                    const output = PascalCompiler.preProcessBinaryAsm(buffer.toString());
                    resolve(Promise.all([
                        utils.splitLines(output).length.should.be.below(500),
                        output.should.not.include("fpc_zeromem():"),
                        output.should.include("SQUARE():")
                    ]));
                });
            });
        });
    });

    describe('Pascal parseOutput', () => {
        it('should return parsed output', () => {
            const result = {
                stdout: "Hello, world!",
                stderr: ""
            };

            compiler.parseOutput(result, "/tmp/path/output.pas", "/tmp/path").should.deep.equal({
                inputFilename: "output.pas",
                stdout: [{
                    "text": "Hello, world!"
                }],
                stderr: []
            });
        });
    });
});
