// Copyright (c) 2021, Compiler Explorer Authors
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

import Semver from 'semver';

import {CompilerFilters, ParseFilters} from '../../types/features/filters.interfaces';
import {asSafeVer} from '../utils';

import {ClangCompiler} from './clang';

export class ZigCC extends ClangCompiler {
    private readonly needsForcedBinary: boolean;

    static override get key() {
        return 'zigcc';
    }

    constructor(info: any, env: any) {
        super(info, env);
        this.needsForcedBinary =
            Semver.gte(asSafeVer(this.compiler.semver), '0.6.0', true) &&
            Semver.lt(asSafeVer(this.compiler.semver), '0.9.0', true);
    }

    override preProcess(source: string, filters: CompilerFilters): string {
        if (this.needsForcedBinary) {
            // note: zig versions > 0.6 don't emit asm, only binary works - https://github.com/ziglang/zig/issues/8153
            filters.binary = true;
        }

        return super.preProcess(source, filters);
    }

    override optionsForFilter(filters: ParseFilters, outputFilename: string, userOptions?: string[]): string[] {
        if (this.needsForcedBinary) {
            // note: zig versions > 0.6 don't emit asm, only binary works - https://github.com/ziglang/zig/issues/8153
            filters.binary = true;
        }

        let options = ['cc', '-g', '-o', this.filename(outputFilename)];
        if (this.compiler.intelAsm && filters.intel && !filters.binary) {
            options = options.concat(this.compiler.intelAsm.split(' '));
        }
        if (!filters.binary) options = options.concat('-S');
        return options;
    }
}
