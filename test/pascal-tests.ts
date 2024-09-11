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

import path from 'path';

import {beforeAll, describe, expect, it} from 'vitest';

import {PascalUtils} from '../lib/compilers/pascal-utils.js';
import {PascalWinCompiler} from '../lib/compilers/pascal-win.js';
import {FPCCompiler} from '../lib/compilers/pascal.js';
import {PascalDemangler} from '../lib/demangler/index.js';
import * as utils from '../lib/utils.js';

import {fs, makeCompilationEnvironment} from './utils.js';

const languages = {
    pascal: {id: 'pascal'},
};

describe('Pascal', () => {
    let compiler;

    beforeAll(() => {
        const ce = makeCompilationEnvironment({languages});
        const info = {
            exe: null,
            remote: true,
            lang: languages.pascal.id,
        };

        compiler = new FPCCompiler(info as any, ce);
    });

    it('Basic compiler setup', () => {
        if (process.platform === 'win32') {
            expect(compiler.getOutputFilename('/tmp/', 'prog', {source: 'unit prog;\n//etc'})).toEqual('\\tmp\\prog.s');
        } else {
            expect(compiler.getOutputFilename('/tmp/', 'prog', {source: 'unit prog;\n//etc'})).toEqual('/tmp/prog.s');
        }
    });

    describe('Pascal signature composer function', () => {
        const demangler = new PascalDemangler('demangler-exe', compiler);

        it('Handle 0 parameter methods', () => {
            expect(demangler.composeReadableMethodSignature('', '', 'myfunc', '')).toEqual('myfunc()');
            expect(demangler.composeReadableMethodSignature('output', '', 'myfunc', '')).toEqual('myfunc()');
            expect(demangler.composeReadableMethodSignature('output', 'tmyclass', 'myfunc', '')).toEqual(
                'tmyclass.myfunc()',
            );
        });

        it('Handle 1 parameter methods', () => {
            expect(demangler.composeReadableMethodSignature('output', '', 'myfunc', 'integer')).toEqual(
                'myfunc(integer)',
            );
            expect(demangler.composeReadableMethodSignature('output', 'tmyclass', 'myfunc', 'integer')).toEqual(
                'tmyclass.myfunc(integer)',
            );
        });

        it('Handle 2 parameter methods', () => {
            expect(demangler.composeReadableMethodSignature('output', '', 'myfunc', 'integer,string')).toEqual(
                'myfunc(integer,string)',
            );
            expect(demangler.composeReadableMethodSignature('output', 'tmyclass', 'myfunc', 'integer,string')).toEqual(
                'tmyclass.myfunc(integer,string)',
            );
        });
    });

    describe('Pascal Demangling FPC 2.6', () => {
        const demangler = new PascalDemangler('demangler-exe', compiler);

        it('Should demangle OUTPUT_MAXARRAY$array_of_DOUBLE$array_of_DOUBLE', () => {
            expect(demangler.demangle('OUTPUT_MAXARRAY$array_of_DOUBLE$array_of_DOUBLE:')).toEqual(
                'maxarray(array_of_double,array_of_double)',
            );
        });

        it('Should demangle OUTPUT_TMYCLASS_$__MYPROC$ANSISTRING', () => {
            expect(demangler.demangle('OUTPUT_TMYCLASS_$__MYPROC$ANSISTRING:')).toEqual('tmyclass.myproc(ansistring)');
        });

        it('Should demangle OUTPUT_TMYCLASS_$__MYFUNC$$ANSISTRING', () => {
            expect(demangler.demangle('OUTPUT_TMYCLASS_$__MYFUNC$$ANSISTRING:')).toEqual('tmyclass.myfunc()');
        });

        it('Should demangle OUTPUT_NOPARAMFUNC$$ANSISTRING', () => {
            expect(demangler.demangle('OUTPUT_NOPARAMFUNC$$ANSISTRING:')).toEqual('noparamfunc()');
        });

        it('Should demangle OUTPUT_NOPARAMPROC', () => {
            expect(demangler.demangle('OUTPUT_NOPARAMPROC:')).toEqual('noparamproc()');
        });

        it('Should demangle U_OUTPUT_MYGLOBALVAR', () => {
            expect(demangler.demangle('U_OUTPUT_MYGLOBALVAR:')).toEqual('myglobalvar');
        });

        it('Should demangle OUTPUT_INIT (custom method)', () => {
            expect(demangler.demangle('OUTPUT_INIT:')).toEqual('init()');
        });

        it('Should demangle OUTPUT_init (builtin symbol)', () => {
            expect(demangler.demangle('OUTPUT_init:')).toEqual('unit_initialization');
        });
    });

    describe('Pascal Demangling FPC 3.2', () => {
        const demangler = new PascalDemangler('demangler-exe', compiler);

        it('Should demangle OUTPUT_$$_SQUARE$LONGINT$$LONGINT', () => {
            expect(demangler.demangle('OUTPUT_$$_SQUARE$LONGINT$$LONGINT:')).toEqual('square(longint)');
        });

        it('Should demangle OUTPUT_$$_MAXARRAY$array_of_DOUBLE$array_of_DOUBLE', () => {
            expect(demangler.demangle('OUTPUT_$$_MAXARRAY$array_of_DOUBLE$array_of_DOUBLE:')).toEqual(
                'maxarray(array_of_double,array_of_double)',
            );
        });

        it('Should demangle OUTPUT$_$TMYCLASS_$__$$_MYPROC$ANSISTRING', () => {
            expect(demangler.demangle('OUTPUT$_$TMYCLASS_$__$$_MYPROC$ANSISTRING:')).toEqual(
                'tmyclass.myproc(ansistring)',
            );
        });

        it('Should demangle OUTPUT$_$TMYCLASS_$__$$_MYFUNC$$ANSISTRING', () => {
            expect(demangler.demangle('OUTPUT$_$TMYCLASS_$__$$_MYFUNC$$ANSISTRING:')).toEqual('tmyclass.myfunc()');
        });

        it('Should demangle OUTPUT$_$TMYCLASS_$__$$_MYFUNC$ANSISTRING$$INTEGER', () => {
            expect(demangler.demangle('OUTPUT$_$TMYCLASS_$__$$_MYFUNC$ANSISTRING$$INTEGER:')).toEqual(
                'tmyclass.myfunc(ansistring)',
            );
        });

        it('Should demangle OUTPUT$_$TMYCLASS_$__$$_MYFUNC$ANSISTRING$INTEGER$INTEGER$$INTEGER', () => {
            expect(demangler.demangle('OUTPUT$_$TMYCLASS_$__$$_MYFUNC$ANSISTRING$INTEGER$INTEGER$$INTEGER:')).toEqual(
                'tmyclass.myfunc(ansistring,integer,integer)',
            );
        });

        it('Should demangle OUTPUT_$$_NOPARAMFUNC$$ANSISTRING', () => {
            expect(demangler.demangle('OUTPUT_$$_NOPARAMFUNC$$ANSISTRING:')).toEqual('noparamfunc()');
        });

        it('Should demangle OUTPUT_$$_NOPARAMPROC', () => {
            expect(demangler.demangle('OUTPUT_$$_NOPARAMPROC:')).toEqual('noparamproc()');
        });

        it('Should demangle OUTPUT_$$_INIT', () => {
            expect(demangler.demangle('OUTPUT_$$_INIT:')).toEqual('init()');
        });

        it('Should demangle U_$OUTPUT_$$_MYGLOBALVAR', () => {
            expect(demangler.demangle('U_$OUTPUT_$$_MYGLOBALVAR:')).toEqual('myglobalvar');
        });
    });

    describe('Pascal Demangling Fixed Symbols FPC 2.6', () => {
        const demangler = new PascalDemangler('demangler-exe', compiler);

        it('Should demangle OUTPUT_finalize_implicit', () => {
            expect(demangler.demangle('OUTPUT_finalize_implicit:')).toEqual('unit_finalization_implicit');
        });
    });

    describe('Pascal Demangling Fixed Symbols FPC 3.2', () => {
        const demangler = new PascalDemangler('demangler-exe', compiler);

        it('Should demangle OUTPUT_$$_init', () => {
            expect(demangler.demangle('OUTPUT_$$_init:')).toEqual('unit_initialization');
        });

        it('Should demangle OUTPUT_$$_finalize', () => {
            expect(demangler.demangle('OUTPUT_$$_finalize:')).toEqual('unit_finalization');
        });

        it('Should demangle OUTPUT_$$_init_implicit', () => {
            expect(demangler.demangle('OUTPUT_$$_init_implicit:')).toEqual('unit_initialization_implicit');
        });

        it('Should demangle OUTPUT_$$_finalize_implicit', () => {
            expect(demangler.demangle('OUTPUT_$$_finalize_implicit:')).toEqual('unit_finalization_implicit');
        });

        it('Should demangle OUTPUT_$$_finalize_implicit', () => {
            expect(demangler.demangle('OUTPUT_$$_finalize_implicit:')).toEqual('unit_finalization_implicit');
        });
    });

    describe('Pascal NOT Demangling certain symbols FPC 2.6', () => {
        const demangler = new PascalDemangler('demangler-exe', compiler);

        it('Should NOT demangle VMT_OUTPUT_TMYCLASS', () => {
            expect(demangler.demangle('VMT_OUTPUT_TMYCLASS:')).toEqual(false);
        });

        it('Should NOT demangle RTTI_OUTPUT_TMYCLASS', () => {
            expect(demangler.demangle('RTTI_OUTPUT_TMYCLASS:')).toEqual(false);
        });

        it('Should NOT demangle INIT$_OUTPUT', () => {
            expect(demangler.demangle('INIT$_OUTPUT:')).toEqual(false);
        });

        it('Should NOT demangle FINALIZE$_OUTPUT', () => {
            expect(demangler.demangle('FINALIZE$_OUTPUT:')).toEqual(false);
        });

        it('Should NOT demangle DEBUGSTART_OUTPUT', () => {
            expect(demangler.demangle('DEBUGSTART_OUTPUT:')).toEqual(false);
        });

        it('Should NOT demangle DBGREF_OUTPUT_THELLO', () => {
            expect(demangler.demangle('DBGREF_OUTPUT_THELLO:')).toEqual(false);
        });

        it('Should NOT demangle non-label', () => {
            expect(demangler.demangle('  call OUTPUT$_$TMYCLASS_$__$$_MYTEST2')).toEqual(false);
        });
    });

    describe('Pascal NOT Demangling certain symbols FPC 3.2', () => {
        const demangler = new PascalDemangler('demangler-exe', compiler);

        it('Should NOT demangle RTTI_$OUTPUT_$$_TMYCLASS', () => {
            expect(demangler.demangle('RTTI_$OUTPUT_$$_TMYCLASS:')).toEqual(false);
        });

        it('Should NOT demangle .Ld1', () => {
            expect(demangler.demangle('.Ld1:')).toEqual(false);
        });

        it('Should NOT demangle _$OUTPUT$_Ld3 (Same in FPC 2.6 and 3.2)', () => {
            expect(demangler.demangle('_$OUTPUT$_Ld3:')).toEqual(false);
        });

        it('Should NOT demangle INIT$_$OUTPUT', () => {
            expect(demangler.demangle('INIT$_$OUTPUT:')).toEqual(false);
        });

        it('Should NOT demangle DEBUGSTART_$OUTPUT', () => {
            expect(demangler.demangle('DEBUGSTART_$OUTPUT:')).toEqual(false);
        });

        it('Should NOT demangle DBGREF_$OUTPUT_$$_THELLO', () => {
            expect(demangler.demangle('DBGREF_$OUTPUT_$$_THELLO:')).toEqual(false);
        });
    });

    describe('Add, order and demangle inline', () => {
        const demangler = new PascalDemangler('demangler-exe', compiler);

        it('should work', () => {
            demangler.demangle('OUTPUT$_$TMYCLASS_$__$$_MYTEST:');
            demangler.demangle('U_$OUTPUT_$$_MYGLOBALVAR:');
            demangler.demangle('OUTPUT$_$TMYCLASS_$__$$_MYTEST2:');
            demangler.demangle('OUTPUT$_$TMYCLASS_$__$$_MYOVERLOAD$ANSISTRING:');
            demangler.demangle('OUTPUT$_$TMYCLASS_$__$$_MYOVERLOAD$INTEGER:');

            expect(demangler.demangleIfNeeded('  call OUTPUT$_$TMYCLASS_$__$$_MYTEST2')).toEqual(
                '  call tmyclass.mytest2()',
            );
            expect(demangler.demangleIfNeeded('  movl U_$OUTPUT_$$_MYGLOBALVAR,%eax')).toEqual(
                '  movl myglobalvar,%eax',
            );
            expect(demangler.demangleIfNeeded('  call OUTPUT$_$TMYCLASS_$__$$_MYTEST2')).toEqual(
                '  call tmyclass.mytest2()',
            );
            expect(demangler.demangleIfNeeded('  call OUTPUT$_$TMYCLASS_$__$$_MYTEST')).toEqual(
                '  call tmyclass.mytest()',
            );
            expect(demangler.demangleIfNeeded('  call OUTPUT$_$TMYCLASS_$__$$_MYOVERLOAD$ANSISTRING')).toEqual(
                '  call tmyclass.myoverload(ansistring)',
            );
            expect(demangler.demangleIfNeeded('  call OUTPUT$_$TMYCLASS_$__$$_MYOVERLOAD$INTEGER')).toEqual(
                '  call tmyclass.myoverload(integer)',
            );

            expect(demangler.demangleIfNeeded('.Le1')).toEqual('.Le1');
            expect(demangler.demangleIfNeeded('_$SomeThing')).toEqual('_$SomeThing');
        });
    });

    describe('Add, order and demangle inline - using addDemangleToCache()', () => {
        const demangler = new PascalDemangler('demangler-exe', compiler);

        it('should work', () => {
            demangler.addDemangleToCache('OUTPUT$_$TMYCLASS_$__$$_MYTEST:');
            demangler.addDemangleToCache('U_$OUTPUT_$$_MYGLOBALVAR:');
            demangler.addDemangleToCache('OUTPUT$_$TMYCLASS_$__$$_MYTEST2:');
            demangler.addDemangleToCache('OUTPUT$_$TMYCLASS_$__$$_MYOVERLOAD$ANSISTRING:');
            demangler.addDemangleToCache('OUTPUT$_$TMYCLASS_$__$$_MYOVERLOAD$INTEGER:');

            expect(demangler.demangleIfNeeded('  call OUTPUT$_$TMYCLASS_$__$$_MYTEST2')).toEqual(
                '  call tmyclass.mytest2()',
            );
            expect(demangler.demangleIfNeeded('  movl U_$OUTPUT_$$_MYGLOBALVAR,%eax')).toEqual(
                '  movl myglobalvar,%eax',
            );
            expect(demangler.demangleIfNeeded('  call OUTPUT$_$TMYCLASS_$__$$_MYTEST2')).toEqual(
                '  call tmyclass.mytest2()',
            );
            expect(demangler.demangleIfNeeded('  call OUTPUT$_$TMYCLASS_$__$$_MYTEST')).toEqual(
                '  call tmyclass.mytest()',
            );
            expect(demangler.demangleIfNeeded('  call OUTPUT$_$TMYCLASS_$__$$_MYOVERLOAD$ANSISTRING')).toEqual(
                '  call tmyclass.myoverload(ansistring)',
            );
            expect(demangler.demangleIfNeeded('  call OUTPUT$_$TMYCLASS_$__$$_MYOVERLOAD$INTEGER')).toEqual(
                '  call tmyclass.myoverload(integer)',
            );

            expect(demangler.demangleIfNeeded('.Le1')).toEqual('.Le1');
        });
    });

    describe('Pascal Ignored Symbols', () => {
        const demangler = new PascalDemangler('demangler-exe', compiler);

        it('Should ignore certain labels', () => {
            expect(demangler.shouldIgnoreSymbol('.Le1')).toEqual(true);
            expect(demangler.shouldIgnoreSymbol('_$SomeThing')).toEqual(true);
        });

        it('Should be able to differentiate between System and User functions', () => {
            expect(demangler.shouldIgnoreSymbol('RTTI_OUTPUT_MyProperty')).toEqual(true);
            expect(demangler.shouldIgnoreSymbol('Rtti_Output_UserFunction')).toEqual(false);
        });
    });

    describe('Pascal ASM line number injection', () => {
        beforeAll(() => {
            compiler.demanglerClass = PascalDemangler;
            compiler.demangler = new PascalDemangler('demangler-exe', compiler);
        });

        it('Should have line numbering', async () => {
            const asmLines = utils.splitLines((await fs.readFile('test/pascal/asm-example.s')).toString());
            compiler.preProcessLines(asmLines);
            expect(asmLines).toContain('# [output.pas]');
            expect(asmLines).toContain('  .file 1 "output.pas"');
            expect(asmLines).toContain('# [13] Square := num * num + 14;');
            expect(asmLines).toContain('  .loc 1 13 0');
            expect(asmLines).toContain('.Le0:');
            expect(asmLines).toContain('  .cfi_endproc');
        });
    });

    // describe('Pascal objdump filtering', function () {
    //     it('Should filter out most of the runtime', function () {
    //         return new Promise(function (resolve) {
    //             fs.readFile('test/pascal/objdump-example.s', function (err, buffer) {
    //                 const output = FPCCompiler.preProcessBinaryAsm(buffer.toString());
    //                 resolve(Promise.all([
    //                     utils.splitLines(output).length.should.be.below(500),
    //                     output.should.not.include('fpc_zeromem():'),
    //                     output.should.include('SQUARE():'),
    //                 ]));
    //             });
    //         });
    //     });
    // });

    describe('Pascal parseOutput', () => {
        it('should return parsed output', () => {
            const result = {
                stdout: 'Hello, world!',
                stderr: '',
            };

            expect(compiler.parseOutput(result, '/tmp/path/output.pas', '/tmp/path')).toEqual({
                inputFilename: 'output.pas',
                stdout: [
                    {
                        text: 'Hello, world!',
                    },
                ],
                stderr: [],
            });
        });
    });

    describe('Pascal filetype detection', () => {
        const pasUtils = new PascalUtils();
        const progSource = fs.readFileSync('test/pascal/prog.dpr').toString('utf8');
        const unitSource = fs.readFileSync('test/pascal/example.pas').toString('utf8');

        it('Should detect simple program', () => {
            expect(pasUtils.isProgram(progSource)).toEqual(true);
            expect(pasUtils.isProgram(unitSource)).toEqual(false);
        });

        it('Should detect simple unit', () => {
            expect(pasUtils.isUnit(progSource)).toEqual(false);
            expect(pasUtils.isUnit(unitSource)).toEqual(true);
        });
    });

    describe('Multifile writing behaviour', () => {
        let compiler;

        beforeAll(() => {
            const ce = makeCompilationEnvironment({languages});
            const info = {
                exe: null,
                remote: true,
                lang: languages.pascal.id,
            };

            compiler = new FPCCompiler(info as unknown as any, ce);
        });

        it('Original behaviour (old unitname)', async () => {
            const dirPath = await compiler.newTempDir();
            const filters = {};
            const files = [];
            const source = fs.readFileSync('examples/pascal/default.pas').toString('utf8');

            const writeSummary = await compiler.writeAllFiles(dirPath, source, files, filters);

            expect(writeSummary.inputFilename).toEqual(path.join(dirPath, 'output.pas'));
            await expect(utils.fileExists(path.join(dirPath, 'output.pas'))).resolves.toBe(true);
            await expect(utils.fileExists(path.join(dirPath, 'prog.dpr'))).resolves.toBe(false); // note: will be written somewhere else
        });

        it('Original behaviour (just a unit file)', async () => {
            const dirPath = await compiler.newTempDir();
            const filters = {};
            const files = [];
            const source = fs.readFileSync('test/pascal/example.pas').toString('utf8');

            const writeSummary = await compiler.writeAllFiles(dirPath, source, files, filters);

            expect(writeSummary.inputFilename).toEqual(path.join(dirPath, 'example.pas'));
            await expect(utils.fileExists(path.join(dirPath, 'example.pas'))).resolves.toBe(true);
            await expect(utils.fileExists(path.join(dirPath, 'prog.dpr'))).resolves.toBe(false); // note: will be written somewhere else
        });

        it('Writing program instead of a unit', async () => {
            const dirPath = await compiler.newTempDir();
            const filters = {};
            const files = [];
            const source = fs.readFileSync('test/pascal/prog.dpr').toString('utf8');

            const writeSummary = await compiler.writeAllFiles(dirPath, source, files, filters);

            expect(writeSummary.inputFilename).toEqual(path.join(dirPath, 'prog.dpr'));
            await expect(utils.fileExists(path.join(dirPath, 'example.pas'))).resolves.toBe(false);
            await expect(utils.fileExists(path.join(dirPath, 'prog.dpr'))).resolves.toBe(true);
        });

        it('Writing program with a unit', async () => {
            const dirPath = await compiler.newTempDir();
            const filters = {};
            const files = [
                {
                    filename: 'example.pas',
                    contents: '{ hello\n   world }',
                },
            ];
            const source = fs.readFileSync('test/pascal/prog.dpr').toString('utf8');

            const writeSummary = await compiler.writeAllFiles(dirPath, source, files, filters);

            expect(writeSummary.inputFilename).toEqual(path.join(dirPath, 'prog.dpr'));
            await expect(utils.fileExists(path.join(dirPath, 'example.pas'))).resolves.toBe(true);
            await expect(utils.fileExists(path.join(dirPath, 'prog.dpr'))).resolves.toBe(true);
        });
    });

    describe('Multifile writing behaviour Pascal-WIN', () => {
        let compiler;

        beforeAll(() => {
            const ce = makeCompilationEnvironment({languages});
            const info = {
                exe: null,
                remote: true,
                lang: languages.pascal.id,
            };

            compiler = new PascalWinCompiler(info as any, ce);
        });

        it('Original behaviour (old unitname)', async () => {
            const dirPath = await compiler.newTempDir();
            const filters = {};
            const files = [];
            const source = fs.readFileSync('examples/pascal/default.pas').toString('utf8');

            const writeSummary = await compiler.writeAllFiles(dirPath, source, files, filters);

            expect(writeSummary.inputFilename).toEqual(path.join(dirPath, 'output.pas'));
            await expect(utils.fileExists(path.join(dirPath, 'output.pas'))).resolves.toBe(true);
            await expect(utils.fileExists(path.join(dirPath, 'prog.dpr'))).resolves.toBe(false); // note: will be written somewhere else
        });

        it('Original behaviour (just a unit file)', async () => {
            const dirPath = await compiler.newTempDir();
            const filters = {};
            const files = [];
            const source = fs.readFileSync('test/pascal/example.pas').toString('utf8');

            const writeSummary = await compiler.writeAllFiles(dirPath, source, files, filters);

            expect(writeSummary.inputFilename).toEqual(path.join(dirPath, 'example.pas'));
            await expect(utils.fileExists(path.join(dirPath, 'example.pas'))).resolves.toBe(true);
            await expect(utils.fileExists(path.join(dirPath, 'prog.dpr'))).resolves.toBe(false); // note: will be written somewhere else
        });

        it('Writing program instead of a unit', async () => {
            const dirPath = await compiler.newTempDir();
            const filters = {};
            const files = [];
            const source = fs.readFileSync('test/pascal/prog.dpr').toString('utf8');

            const writeSummary = await compiler.writeAllFiles(dirPath, source, files, filters);

            expect(writeSummary.inputFilename).toEqual(path.join(dirPath, 'prog.dpr'));
            await expect(utils.fileExists(path.join(dirPath, 'example.pas'))).resolves.toBe(false);
            await expect(utils.fileExists(path.join(dirPath, 'prog.dpr'))).resolves.toBe(true);
        });

        it('Writing program with a unit', async () => {
            const dirPath = await compiler.newTempDir();
            const filters = {};
            const files = [
                {
                    filename: 'example.pas',
                    contents: '{ hello\n   world }',
                },
            ];
            const source = fs.readFileSync('test/pascal/prog.dpr').toString('utf8');

            const writeSummary = await compiler.writeAllFiles(dirPath, source, files, filters);

            expect(writeSummary.inputFilename).toEqual(path.join(dirPath, 'prog.dpr'));
            await expect(utils.fileExists(path.join(dirPath, 'example.pas'))).resolves.toBe(true);
            await expect(utils.fileExists(path.join(dirPath, 'prog.dpr'))).resolves.toBe(true);
        });
    });
});
