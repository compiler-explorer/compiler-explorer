// Copyright (c) 2022, Compiler Explorer Authors
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

import _ from 'underscore';

import {CacheKey} from '../../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {VersionInfo} from '../options-handler.js';

import {ExecCompilerCachedFunc} from './base.js';
import {BuildEnvSetupCeConanDirect} from './ceconan.js';

export class BuildEnvSetupCeConanRustDirect extends BuildEnvSetupCeConanDirect {
    static override get key() {
        return 'ceconan-rust';
    }

    constructor(compilerInfo: CompilerInfo, env: CompilationEnvironment) {
        super(compilerInfo, env);

        this.onlyonstaticliblink = false;
        this.extractAllToRoot = false;
    }

    override async initialise(execCompilerCachedFunc: ExecCompilerCachedFunc) {
        if (this.compilerArch) return;
        this.compilerSupportsX86 = true;
    }

    override getLibcxx(key: CacheKey) {
        return '';
    }

    override getDestinationFilepath(downloadPath: string, zippedPath: string, libId: string): string {
        // libId is already included in rust packages
        return path.join(downloadPath, zippedPath);
    }

    getArchFromTriple(triple: string) {
        if (triple && triple.split) {
            const arr = triple.split('-');
            if (arr && arr[0]) {
                return arr[0];
            } else {
                return triple;
            }
        } else {
            return '';
        }
    }

    override getTarget(key: CacheKey) {
        if (!this.compilerSupportsX86) return '';
        if (this.compilerArch) return this.compilerArch;

        const target = _.find(key.options, option => {
            return option.startsWith('-target=') || option.startsWith('--target=');
        });

        if (target) {
            const triple = target.substring(target.indexOf('=') + 1);
            return this.getArchFromTriple(triple);
        } else {
            const idx = key.options.indexOf('--target');
            if (idx !== -1) {
                const triple = key.options[idx + 1];
                return this.getArchFromTriple(triple);
            }
        }

        return 'x86_64';
    }

    override hasBinariesToLink(details: VersionInfo) {
        return true;
    }

    override shouldDownloadPackage(details: VersionInfo) {
        return true;
    }
}
