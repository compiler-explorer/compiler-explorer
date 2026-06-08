// Copyright (c) 2026, Compiler Explorer Authors
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

import {GCCCompiler} from './gcc.js';

export class SfpiCompiler extends GCCCompiler {
    static override get key() {
        return 'sfpi';
    }

    private getSelectedCpu(userOptions?: string[]): string {
        const options = userOptions ?? [];
        for (let i = 0; i < options.length; ++i) {
            const option = options[i];
            if (option.startsWith('-mcpu=')) {
                return option.slice('-mcpu='.length);
            }
            if (option === '-mcpu' && i + 1 < options.length) {
                return options[i + 1];
            }
        }
        return 'tt-wh-tensix';
    }

    private getBase() {
        const exeDir = path.dirname(this.compiler.exe);
        return path.resolve(exeDir, '../../');
    }

    override optionsForFilter(filters, outputFilename: string, userOptions?: string[]) {
        const options = super.optionsForFilter(filters, outputFilename, userOptions);
        const cpu = this.getSelectedCpu(userOptions);
        const isBlackhole = cpu.includes('bh') || cpu.includes('blackhole');
        const base = this.getBase();
        const llkArchDir = isBlackhole ? 'tt_llk_blackhole' : 'tt_llk_wormhole_b0';
        const ckernelArchDir = isBlackhole ? 'blackhole' : 'wormhole_b0';

        options.push(
            '-std=c++17',
            '-O2',
            '-ffast-math',
            '-fno-exceptions',
            '-fno-use-cxa-atexit',
            '-fno-tree-loop-distribute-patterns',
            '-DCOMPILE_FOR_TRISC',
            '-DTENSIX_FIRMWARE',
            '-Wall',
            '-Wextra',
        );

        if (!userOptions?.some(option => option === '-mcpu' || option.startsWith('-mcpu='))) {
            options.push(`-mcpu=${cpu}`);
        }

        options.push(
            `-I${path.join(base, 'include')}`,
            `-I${path.join(base, 'build/sfpi/include')}`,
            `-I${path.join(base, 'tt_metal/hw/inc')}`,
            `-I${path.join(base, 'tt_metal/tt-llk/common')}`,
            `-I${path.join(base, 'tt_metal/tt-llk/tests/helpers/include')}`,
            `-I${path.join(base, `tt_metal/tt-llk/${llkArchDir}/common/inc`)}`,
            `-I${path.join(base, `tt_metal/tt-llk/${llkArchDir}/llk_lib`)}`,
            `-I${path.join(base, `tt_metal/hw/ckernels/${ckernelArchDir}/metal/llk_api`)}`,
            `-I${path.join(base, `tt_metal/hw/ckernels/${ckernelArchDir}/metal/llk_api/llk_sfpu`)}`,
        );

        return options;
    }
}
