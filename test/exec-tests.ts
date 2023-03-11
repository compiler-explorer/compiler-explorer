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

import * as exec from '../lib/exec.js';
import * as props from '../lib/properties.js';

import {chai} from './utils.js';

const expect = chai.expect;

function testExecOutput(x) {
    // Work around chai not being able to deepEquals with a function
    x.filenameTransform.should.be.a('function');
    delete x.filenameTransform;
    delete x.execTime;
    return x;
}

describe('Execution tests', () => {
    if (process.platform === 'win32') {
        // win32
        describe('Executes external commands', () => {
            // note: we use powershell, since echo is a builtin, and false doesn't exist
            it('supports output', () => {
                return exec
                    .execute('powershell', ['-Command', 'echo "hello world"'], {})
                    .then(testExecOutput)
                    .should.eventually.deep.equals({
                        code: 0,
                        okToCache: true,
                        stderr: '',
                        stdout: 'hello world\r\n',
                        timedOut: false,
                    });
            });
            it('limits output', () => {
                return exec
                    .execute('powershell', ['-Command', 'echo "A very very very very very long string"'], {
                        maxOutput: 10,
                    })
                    .then(testExecOutput)
                    .should.eventually.deep.equals({
                        code: 0,
                        okToCache: true,
                        stderr: '',
                        stdout: 'A very ver\n[Truncated]',
                        timedOut: false,
                    });
            });
            it('handles failing commands', () => {
                return exec
                    .execute('powershell', ['-Command', 'function Fail { exit 1 }; Fail'], {})
                    .then(testExecOutput)
                    .should.eventually.deep.equals({
                        code: 1,
                        okToCache: true,
                        stderr: '',
                        stdout: '',
                        timedOut: false,
                    });
            });
            it('handles timouts', () => {
                return exec
                    .execute('powershell', ['-Command', '"sleep 5"'], {timeoutMs: 10})
                    .then(testExecOutput)
                    .should.eventually.deep.equals({
                        code: 1,
                        okToCache: false,
                        stderr: '\nKilled - processing time exceeded',
                        stdout: '',
                        timedOut: false,
                    });
            });
            it('handles missing executables', () => {
                return exec.execute('__not_a_command__', [], {}).should.be.rejectedWith('ENOENT');
            });
        });
    } else {
        // POSIX
        describe('Executes external commands', () => {
            it('supports output', () => {
                return exec.execute('echo', ['hello', 'world'], {}).then(testExecOutput).should.eventually.deep.equals({
                    code: 0,
                    okToCache: true,
                    stderr: '',
                    stdout: 'hello world\n',
                    timedOut: false,
                });
            });
            it('limits output', () => {
                return exec
                    .execute('echo', ['A very very very very very long string'], {maxOutput: 10})
                    .then(testExecOutput)
                    .should.eventually.deep.equals({
                        code: 0,
                        okToCache: true,
                        stderr: '',
                        stdout: 'A very ver\n[Truncated]',
                        timedOut: false,
                    });
            });
            it('handles failing commands', () => {
                return exec.execute('false', [], {}).then(testExecOutput).should.eventually.deep.equals({
                    code: 1,
                    okToCache: true,
                    stderr: '',
                    stdout: '',
                    timedOut: false,
                });
            });
            it('handles timouts', () => {
                return exec
                    .execute('sleep', ['5'], {timeoutMs: 10})
                    .then(testExecOutput)
                    .should.eventually.deep.equals({
                        code: -1,
                        okToCache: false,
                        stderr: '\nKilled - processing time exceeded',
                        stdout: '',
                        timedOut: true,
                    });
            });
            it('handles missing executables', () => {
                return exec.execute('__not_a_command__', [], {}).should.be.rejectedWith('ENOENT');
            });
            it('handles input', () => {
                return exec
                    .execute('cat', [], {input: 'this is stdin'})
                    .then(testExecOutput)
                    .should.eventually.deep.equals({
                        code: 0,
                        okToCache: true,
                        stderr: '',
                        stdout: 'this is stdin',
                        timedOut: false,
                    });
            });
        });
    }

    describe('nsjail unit tests', () => {
        before(() => {
            props.initialize(path.resolve('./test/test-properties/execution'), ['test']);
        });
        after(() => {
            props.reset();
        });
        it('should handle simple cases', () => {
            const {args, options, filenameTransform} = exec.getNsJailOptions(
                'sandbox',
                '/path/to/compiler',
                ['1', '2', '3'],
                {},
            );
            args.should.deep.equals([
                '--config',
                exec.getNsJailCfgFilePath('sandbox'),
                '--env=HOME=/app',
                '--',
                '/path/to/compiler',
                '1',
                '2',
                '3',
            ]);
            options.should.deep.equals({});
            expect(filenameTransform).to.be.undefined;
        });
        it('should pass through options', () => {
            const options = exec.getNsJailOptions('sandbox', '/path/to/compiler', [], {
                timeoutMs: 42,
                maxOutput: -1,
            }).options;
            options.should.deep.equals({timeoutMs: 42, maxOutput: -1});
        });
        it('should not pass through unknown configs', () => {
            expect(() => exec.getNsJailOptions('custom-config', '/path/to/compiler', ['1', '2', '3'], {})).to.throw();
        });
        it('should remap paths when using customCwd', () => {
            const {args, options, filenameTransform} = exec.getNsJailOptions(
                'sandbox',
                './exec',
                ['/some/custom/cwd/file', '/not/custom/file'],
                {customCwd: '/some/custom/cwd'},
            );
            args.should.deep.equals([
                '--config',
                exec.getNsJailCfgFilePath('sandbox'),
                '--cwd',
                '/app',
                '--bindmount',
                '/some/custom/cwd:/app',
                '--env=HOME=/app',
                '--',
                './exec',
                '/app/file',
                '/not/custom/file',
            ]);
            options.should.deep.equals({});
            expect(filenameTransform).to.not.be.undefined;
            filenameTransform('moo').should.equal('moo');
            filenameTransform('/some/custom/cwd/file').should.equal('/app/file');
        });
        it('should handle timeouts', () => {
            const args = exec.getNsJailOptions('sandbox', '/path/to/compiler', [], {timeoutMs: 1234}).args;
            args.should.include('--time_limit=2');
        });
        it('should handle linker paths', () => {
            const {args, options} = exec.getNsJailOptions('sandbox', '/path/to/compiler', [], {
                ldPath: ['/a/lib/path', '/b/lib2'],
            });
            options.should.deep.equals({});
            args.should.include('--env=LD_LIBRARY_PATH=/a/lib/path:/b/lib2');
        });
        it('should handle envs', () => {
            const {args, options} = exec.getNsJailOptions('sandbox', '/path/to/compiler', [], {
                env: {ENV1: '1', ENV2: '2'},
            });
            options.should.deep.equals({});
            args.should.include('--env=ENV1=1');
            args.should.include('--env=ENV2=2');
        });
    });

    describe('Subdirectory execution', () => {
        before(() => {
            props.initialize(path.resolve('./test/test-properties/execution'), ['test']);
        });
        after(() => {
            props.reset();
        });

        it('Normal situation without customCwd', () => {
            const {args, options} = exec.getSandboxNsjailOptions('/tmp/hellow/output.s', [], {});

            options.should.deep.equals({});
            args.should.deep.equals([
                '--config',
                'etc/nsjail/sandbox.cfg',
                '--cwd',
                '/app',
                '--bindmount',
                '/tmp/hellow:/app',
                '--env=HOME=/app',
                '--',
                './output.s',
            ]);
        });

        it('Normal situation', () => {
            const {args, options} = exec.getSandboxNsjailOptions('/tmp/hellow/output.s', [], {
                customCwd: '/tmp/hellow',
            });

            options.should.deep.equals({});
            args.should.deep.equals([
                '--config',
                'etc/nsjail/sandbox.cfg',
                '--cwd',
                '/app',
                '--bindmount',
                '/tmp/hellow:/app',
                '--env=HOME=/app',
                '--',
                './output.s',
            ]);
        });

        it('Subdirectory', () => {
            const {args, options} = exec.getSandboxNsjailOptions('/tmp/hellow/subdir/output.s', [], {
                customCwd: '/tmp/hellow',
            });

            options.should.deep.equals({});
            if (process.platform !== 'win32') {
                args.should.deep.equals([
                    '--config',
                    'etc/nsjail/sandbox.cfg',
                    '--cwd',
                    '/app',
                    '--bindmount',
                    '/tmp/hellow:/app',
                    '--env=HOME=/app',
                    '--',
                    'subdir/output.s',
                ]);
            }
        });

        it('CMake outside tree building', () => {
            const {args, options} = exec.getNsJailOptions('execute', '/opt/compiler-explorer/cmake/bin/cmake', ['..'], {
                customCwd: '/tmp/hellow/build',
                appHome: '/tmp/hellow',
            });

            options.should.deep.equals({
                appHome: '/tmp/hellow',
            });
            if (process.platform !== 'win32') {
                args.should.deep.equals([
                    '--config',
                    'etc/nsjail/execute.cfg',
                    '--cwd',
                    '/app/build',
                    '--bindmount',
                    '/tmp/hellow:/app',
                    '--env=HOME=/app',
                    '--',
                    '/opt/compiler-explorer/cmake/bin/cmake',
                    '..',
                ]);
            }
        });

        it('Use cwd inside appHome', () => {
            const {args, options} = exec.getNsJailOptions(
                'execute',
                '/opt/compiler-explorer/compiler/bin/g++',
                [
                    '-c',
                    '-S',
                    '-I/usr/include',
                    '-I/tmp/hellow/myincludes',
                    '/tmp/hellow/example.cpp',
                    '-o',
                    '/tmp/hellow/build/example.o',
                ],
                {
                    customCwd: '/tmp/hellow/build',
                    appHome: '/tmp/hellow',
                },
            );

            options.should.deep.equals({
                appHome: '/tmp/hellow',
            });
            if (process.platform !== 'win32') {
                args.should.deep.equals([
                    '--config',
                    'etc/nsjail/execute.cfg',
                    '--cwd',
                    '/app/build',
                    '--bindmount',
                    '/tmp/hellow:/app',
                    '--env=HOME=/app',
                    '--',
                    '/opt/compiler-explorer/compiler/bin/g++',
                    '-c',
                    '-S',
                    '-I/usr/include',
                    '-I/app/myincludes',
                    '/app/example.cpp',
                    '-o',
                    '/app/build/example.o',
                ]);
            }
        });
    });
});
