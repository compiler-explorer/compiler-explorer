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

import './utils';
import * as exec from '../lib/exec';

function testExecOutput(x) {
    // Work around chai not being able to deepEquals with a function
    x.filenameTransform.should.be.a('function');
    delete x.filenameTransform;
    delete x.execTime;
    return x;
}

if (process.platform !== 'win32') { // POSIX
    describe('Executes external commands', () => {
        it('supports output', () => {
            return exec.execute('echo', ['hello', 'world'], {})
                .then(testExecOutput)
                .should.eventually.deep.equals(
                    {
                        code: 0,
                        okToCache: true,
                        stderr: '',
                        stdout: 'hello world\n',
                    });
        });
        it('limits output', () => {
            return exec.execute('echo', ['A very very very very very long string'], {maxOutput: 10})
                .then(testExecOutput)
                .should.eventually.deep.equals(
                    {
                        code: 0,
                        okToCache: true,
                        stderr: '',
                        stdout: 'A very ver\n[Truncated]',
                    });
        });
        it('handles failing commands', () => {
            return exec.execute('false', [], {})
                .then(testExecOutput)
                .should.eventually.deep.equals(
                    {
                        code: 1,
                        okToCache: true,
                        stderr: '',
                        stdout: '',
                    });
        });
        it('handles timouts', () => {
            return exec.execute('sleep', ['5'], {timeoutMs: 10})
                .then(testExecOutput)
                .should.eventually.deep.equals(
                    {
                        code: -1,
                        okToCache: false,
                        stderr: '\nKilled - processing time exceeded',
                        stdout: '',
                    });
        });
        it('handles missing executables', () => {
            return exec.execute('__not_a_command__', [], {})
                .should.be.rejectedWith('ENOENT');
        });
        it('handles input', () => {
            return exec.execute('cat', [], {input: 'this is stdin'})
                .then(testExecOutput)
                .should.eventually.deep.equals(
                    {
                        code: 0,
                        okToCache: true,
                        stderr: '',
                        stdout: 'this is stdin',
                    });
        });
    });
} else { // win32
    describe('Executes external commands', () => {
        // note: we use powershell, since echo is a builtin, and false doesn't exist
        it('supports output', () => {
            return exec.execute('powershell', ['-Command', 'echo "hello world"'], {})
                .then(testExecOutput)
                .should.eventually.deep.equals(
                    {
                        code: 0,
                        okToCache: true,
                        stderr: '',
                        stdout: 'hello world\r\n',
                    });
        });
        it('limits output', () => {
            return exec.execute('powershell', ['-Command', 'echo "A very very very very very long string"'], {maxOutput: 10})
                .then(testExecOutput)
                .should.eventually.deep.equals(
                    {
                        code: 0,
                        okToCache: true,
                        stderr: '',
                        stdout: 'A very ver\n[Truncated]',
                    });
        });
        it('handles failing commands', () => {
            return exec.execute('powershell', ['-Command', 'function Fail { exit 1 }; Fail'], {})
                .then(testExecOutput)
                .should.eventually.deep.equals(
                    {
                        code: 1,
                        okToCache: true,
                        stderr: '',
                        stdout: '',
                    });
        });
        it('handles timouts', () => {
            return exec.execute('powershell', ['-Command', '"sleep 5"'], {timeoutMs: 10})
                .then(testExecOutput)
                .should.eventually.deep.equals(
                    {
                        code: 1,
                        okToCache: false,
                        stderr: '\nKilled - processing time exceeded',
                        stdout: '',
                    });
        });
        it('handles missing executables', () => {
            return exec.execute('__not_a_command__', [], {})
                .should.be.rejectedWith('ENOENT');
        });
    });
}
