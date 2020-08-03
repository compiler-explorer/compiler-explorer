// Copyright (c) 2019, Compiler Explorer Authors
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

const
    path = require('path'),
    chai = require('chai'),
    CompilerDropinTool = require('../lib/tooling/compiler-dropin-tool');

chai.should();

describe('CompilerDropInTool', () => {
    it('Should support llvm based compilers', () => {
        const tool = new CompilerDropinTool({}, {});

        const compilationInfo = {
            compiler: {
                exe: '/opt/compiler-explorer/clang-8.0.0/bin/clang++',
                options: '--gcc-toolchain=/opt/compiler-explorer/gcc-7.2.0',
            },
            options: [],
        };
        const includeflags = [];
        const args = [];
        const sourcefile = 'example.cpp';

        const orderedArgs = tool.getOrderedArguments(compilationInfo, includeflags, args, sourcefile);
        orderedArgs.should.deep.equal(
            [
                '--gcc-toolchain=/opt/compiler-explorer/gcc-7.2.0',
                '--gcc-toolchain=/opt/compiler-explorer/gcc-7.2.0',
            ],
        );
    });

    it('Should support gcc based compilers', () => {
        const tool = new CompilerDropinTool({}, {});

        const compilationInfo = {
            compiler: {
                exe: '/opt/compiler-explorer/gcc-8.0/bin/g++',
                options: '',
            },
            options: [],
        };
        const includeflags = [];
        const args = [];
        const sourcefile = 'example.cpp';

        const orderedArgs = tool.getOrderedArguments(compilationInfo, includeflags, args, sourcefile);
        orderedArgs.should.deep.equal(
            [
                '--gcc-toolchain=' + path.resolve('/opt/compiler-explorer/gcc-8.0'),
                '--gcc-toolchain=' + path.resolve('/opt/compiler-explorer/gcc-8.0'),
            ],
        );
    });

    it('Should not support riscv gcc compilers', () => {
        const tool = new CompilerDropinTool({}, {});

        const compilationInfo = {
            compiler: {
                exe: '/opt/compiler-explorer/riscv64/gcc-8.2.0/riscv64-unknown-linux-gnu/bin/riscv64-unknown-linux-gnu-g++',
                options: '',
            },
            options: [],
        };
        const includeflags = [];
        const args = [];
        const sourcefile = 'example.cpp';

        const orderedArgs = tool.getOrderedArguments(compilationInfo, includeflags, args, sourcefile);
        orderedArgs.should.deep.equal(false);
    });

    it('Should support ICC compilers', () => {
        const tool = new CompilerDropinTool({}, {});

        const compilationInfo = {
            compiler: {
                exe: '/opt/compiler-explorer/intel-2019.1/bin/icc',
                options: '--gxx-name=/opt/compiler-explorer/gcc-8.2.0/bin/g++',
            },
            options: [],
        };
        const includeflags = [];
        const args = [];
        const sourcefile = 'example.cpp';

        const orderedArgs = tool.getOrderedArguments(compilationInfo, includeflags, args, sourcefile);
        orderedArgs.should.deep.equal(
            [
                '--gcc-toolchain=' + path.resolve('/opt/compiler-explorer/gcc-8.2.0'),
                '--gcc-toolchain=' + path.resolve('/opt/compiler-explorer/gcc-8.2.0'),
            ],
        );
    });

    it('Should not support WINE MSVC compilers', () => {
        const tool = new CompilerDropinTool({}, {});

        const compilationInfo = {
            compiler: {
                exe: '/opt/compiler-explorer/windows/19.14.26423/bin/cl.exe',
                options: 
                    '/I/opt/compiler-explorer/windows/10.0.10240.0/ucrt/ ' +
                    '/I/opt/compiler-explorer/windows/19.14.26423/include/',
                internalIncludePaths: [
                    '/opt/compiler-explorer/windows/19.14.26423/include',
                ],
            },
            options: [],
        };
        const includeflags = [];
        const args = ['/MD', '/STD:c++latest', '/Ox'];
        const sourcefile = 'example.cpp';

        const orderedArgs = tool.getOrderedArguments(compilationInfo, includeflags, args, sourcefile);
        orderedArgs.should.deep.equal(false);
    });

    it('Should not support using libc++', () => {
        const tool = new CompilerDropinTool({}, {});

        const compilationInfo = {
            compiler: {
                exe: '/opt/compiler-explorer/clang-concepts-trunk/bin/clang++',
                options: '-stdlib=libc++',
                internalIncludePaths: [
                    '/opt/compiler-explorer/clang-concepts-trunk/something/etc/include',
                ],
            },
            options: [],
        };
        const includeflags = [];
        const args = [];
        const sourcefile = 'example.cpp';

        const orderedArgs = tool.getOrderedArguments(compilationInfo, includeflags, args, sourcefile);
        orderedArgs.should.deep.equal(false);
    });
});
