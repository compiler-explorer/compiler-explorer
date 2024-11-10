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

import path from 'node:path';

import {afterAll, beforeAll, describe, expect, it} from 'vitest';

import * as exec from '../lib/exec.js';
import * as props from '../lib/properties.js';
import {UnprocessedExecResult} from '../types/execution/execution.interfaces.js';

async function testExecOutput(execPromise: Promise<Partial<UnprocessedExecResult>>) {
    const x = await execPromise;
    expect(x.filenameTransform).toBeInstanceOf(Function);
    delete x.filenameTransform;
    delete x.execTime;
    return x;
}

describe('Execution tests', async () => {
    if (process.platform === 'win32') {
        // win32
        describe('Executes external commands', () => {
            // note: we use powershell, since echo is a builtin, and false doesn't exist
            it('supports output', async () => {
                await expect(
                    testExecOutput(exec.execute('powershell', ['-Command', 'echo "hello world"'], {})),
                ).resolves.toEqual({
                    code: 0,
                    okToCache: true,
                    stderr: '',
                    stdout: 'hello world\r\n',
                    truncated: false,
                    timedOut: false,
                });
            });
            it('limits output', async () => {
                await expect(
                    testExecOutput(
                        exec.execute('powershell', ['-Command', 'echo "A very very very very very long string"'], {
                            maxOutput: 22,
                        }),
                    ),
                ).resolves.toEqual({
                    code: 0,
                    okToCache: true,
                    stderr: '',
                    stdout: 'A very ver\n[Truncated]',
                    truncated: true,
                    timedOut: false,
                });
            });
            it('handles failing commands', async () => {
                await expect(
                    testExecOutput(exec.execute('powershell', ['-Command', 'function Fail { exit 1 }; Fail'], {})),
                ).resolves.toEqual({
                    code: 1,
                    okToCache: true,
                    stderr: '',
                    stdout: '',
                    truncated: false,
                    timedOut: false,
                });
            });
            it('handles timouts', async () => {
                await expect(
                    testExecOutput(exec.execute('powershell', ['-Command', '"sleep 5"'], {timeoutMs: 10})),
                ).resolves.toEqual({
                    code: 1,
                    okToCache: false,
                    stderr: '\nKilled - processing time exceeded\n',
                    stdout: '',
                    truncated: false,
                    timedOut: true,
                });
            });
            it('handles missing executables', async () => {
                await expect(exec.execute('__not_a_command__', [], {})).rejects.toThrow('ENOENT');
            });
        });
    } else {
        // POSIX
        describe('Executes external commands', () => {
            it('supports output', async () => {
                await expect(testExecOutput(exec.execute('echo', ['hello', 'world'], {}))).resolves.toEqual({
                    code: 0,
                    okToCache: true,
                    stderr: '',
                    stdout: 'hello world\n',
                    truncated: false,
                    timedOut: false,
                });
            });
            it('limits output', async () => {
                return expect(
                    testExecOutput(exec.execute('echo', ['A very very very very very long string'], {maxOutput: 22})),
                ).resolves.toEqual({
                    code: 0,
                    okToCache: true,
                    stderr: '',
                    stdout: 'A very ver\n[Truncated]',
                    truncated: true,
                    timedOut: false,
                });
            });
            it('handles failing commands', async () => {
                await expect(testExecOutput(exec.execute('false', [], {}))).resolves.toEqual({
                    code: 1,
                    okToCache: true,
                    stderr: '',
                    stdout: '',
                    truncated: false,
                    timedOut: false,
                });
            });
            it('handles timouts', async () => {
                await expect(testExecOutput(exec.execute('sleep', ['5'], {timeoutMs: 10}))).resolves.toEqual({
                    code: -1,
                    okToCache: false,
                    stderr: '\nKilled - processing time exceeded\n',
                    stdout: '',
                    truncated: false,
                    timedOut: true,
                });
            });
            it('handles missing executables', async () => {
                await expect(exec.execute('__not_a_command__', [], {})).rejects.toThrow('ENOENT');
            });
            it('handles input', async () => {
                await expect(testExecOutput(exec.execute('cat', [], {input: 'this is stdin'}))).resolves.toEqual({
                    code: 0,
                    okToCache: true,
                    stderr: '',
                    stdout: 'this is stdin',
                    truncated: false,
                    timedOut: false,
                });
            });
        });
    }

    describe('nsjail unit tests', () => {
        beforeAll(() => {
            props.initialize(path.resolve('./test/test-properties/execution'), ['test']);
        });
        afterAll(() => {
            props.reset();
        });
        it('should handle simple cases', () => {
            const {args, options, filenameTransform} = exec.getNsJailOptions(
                'sandbox',
                '/path/to/compiler',
                ['1', '2', '3'],
                {},
            );
            expect(args).toEqual([
                '--config',
                exec.getNsJailCfgFilePath('sandbox'),
                '--env=HOME=/app',
                '--',
                '/path/to/compiler',
                '1',
                '2',
                '3',
            ]);
            expect(options).toEqual({});
            expect(filenameTransform).to.be.undefined;
        });
        it('should pass through options', () => {
            const options = exec.getNsJailOptions('sandbox', '/path/to/compiler', [], {
                timeoutMs: 42,
                maxOutput: -1,
            }).options;
            expect(options).toEqual({timeoutMs: 42, maxOutput: -1});
        });
        it('should not pass through unknown configs', () => {
            expect(() => exec.getNsJailOptions('custom-config', '/path/to/compiler', ['1', '2', '3'], {})).toThrow();
        });
        it('should remap paths when using customCwd', () => {
            const {args, options, filenameTransform} = exec.getNsJailOptions(
                'sandbox',
                './exec',
                ['/some/custom/cwd/file', '/not/custom/file'],
                {customCwd: '/some/custom/cwd'},
            );
            expect(args).toEqual([
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
            expect(options).toEqual({});
            expect(filenameTransform).toBeTruthy();
            if (filenameTransform) {
                expect(filenameTransform('moo')).toEqual('moo');
                expect(filenameTransform('/some/custom/cwd/file')).toEqual('/app/file');
            }
        });
        it('should handle timeouts', () => {
            const args = exec.getNsJailOptions('sandbox', '/path/to/compiler', [], {timeoutMs: 1234}).args;
            expect(args).toContain('--time_limit=2');
        });
        it('should handle linker paths', () => {
            const {args, options} = exec.getNsJailOptions('sandbox', '/path/to/compiler', [], {
                ldPath: ['/a/lib/path', '/b/lib2'],
            });
            expect(options).toEqual({});
            if (process.platform === 'win32') {
                expect(args).toContain('--env=LD_LIBRARY_PATH=/a/lib/path;/b/lib2');
            } else {
                expect(args).toContain('--env=LD_LIBRARY_PATH=/a/lib/path:/b/lib2');
            }
        });
        it('should handle envs', () => {
            const {args, options} = exec.getNsJailOptions('sandbox', '/path/to/compiler', [], {
                env: {ENV1: '1', ENV2: '2'},
            });
            expect(options).toEqual({});
            expect(args).toContain('--env=ENV1=1');
            expect(args).toContain('--env=ENV2=2');
        });
    });

    describe('cewrapper unit tests', () => {
        beforeAll(() => {
            props.initialize(path.resolve('./test/test-properties/execution'), ['test']);
        });
        afterAll(() => {
            props.reset();
        });
        it('passed as arguments', () => {
            const options = exec.getCeWrapperOptions('sandbox', '/path/to/something', ['--help'], {
                timeoutMs: 42,
                maxOutput: -1,
                env: {
                    TEST: 'Hello, World!',
                },
            });

            expect(options.args).toEqual([
                '--config=' + path.resolve('etc/cewrapper/user-execution.json'),
                '--time_limit=1',
                '/path/to/something',
                '--help',
            ]);
            expect(options.options).toEqual({timeoutMs: 42, maxOutput: -1, env: {TEST: 'Hello, World!'}});
        });
    });

    describe('Subdirectory execution', () => {
        beforeAll(() => {
            props.initialize(path.resolve('./test/test-properties/execution'), ['test']);
        });
        afterAll(() => {
            props.reset();
        });

        it('Normal situation without customCwd', () => {
            const {args, options} = exec.getSandboxNsjailOptions('/tmp/hellow/output.s', [], {});

            expect(options).toEqual({});
            expect(args).toEqual([
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

            expect(options).toEqual({});
            expect(args).toEqual([
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
        it('Should remap env vars', () => {
            const {args, options} = exec.getSandboxNsjailOptions('/tmp/hellow/output.s', [], {
                customCwd: '/tmp/hellow',
                env: {SOME_DOTNET_THING: '/tmp/hellow/dotnet'},
            });

            expect(options).toEqual({});
            expect(args).toEqual([
                '--config',
                'etc/nsjail/sandbox.cfg',
                '--cwd',
                '/app',
                '--bindmount',
                '/tmp/hellow:/app',
                '--env=SOME_DOTNET_THING=/app/dotnet',
                '--env=HOME=/app',
                '--',
                './output.s',
            ]);
        });

        it('Should remap longer env vars with multiple paths', () => {
            const {args, options} = exec.getSandboxNsjailOptions('/tmp/hellow/output.s', [], {
                customCwd: '/tmp/hellow',
                env: {CXX_FLAGS: '-L/usr/lib -L/tmp/hellow/curl/lib -L/tmp/hellow/fmt/lib'},
            });

            expect(options).toEqual({});
            expect(args).toEqual([
                '--config',
                'etc/nsjail/sandbox.cfg',
                '--cwd',
                '/app',
                '--bindmount',
                '/tmp/hellow:/app',
                '--env=CXX_FLAGS=-L/usr/lib -L/app/curl/lib -L/app/fmt/lib',
                '--env=HOME=/app',
                '--',
                './output.s',
            ]);
        });

        it('Should remap ldPath env vars', () => {
            const {args, options} = exec.getSandboxNsjailOptions('/tmp/hellow/output.s', [], {
                customCwd: '/tmp/hellow',
                ldPath: ['/usr/lib', '', '/tmp/hellow/lib'],
            });

            expect(options).toEqual({});
            expect(args).toEqual([
                '--config',
                'etc/nsjail/sandbox.cfg',
                '--cwd',
                '/app',
                '--bindmount',
                '/tmp/hellow:/app',
                '--env=LD_LIBRARY_PATH=' + ['/usr/lib', '/app/lib'].join(path.delimiter),
                '--env=HOME=/app',
                '--',
                './output.s',
            ]);
        });

        it('Subdirectory', () => {
            const {args, options} = exec.getSandboxNsjailOptions('/tmp/hellow/subdir/output.s', [], {
                customCwd: '/tmp/hellow',
            });

            expect(options).toEqual({});
            if (process.platform !== 'win32') {
                expect(args).toEqual([
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

            expect(options).toEqual({
                appHome: '/tmp/hellow',
            });
            if (process.platform !== 'win32') {
                expect(args).toEqual([
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

            expect(options).toEqual({
                appHome: '/tmp/hellow',
            });
            if (process.platform !== 'win32') {
                expect(args).toEqual([
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
