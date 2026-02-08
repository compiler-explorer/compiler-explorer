// Copyright (c) 2025, Compiler Explorer Authors
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

import _ from 'underscore';

import {CacheKey} from '../../types/compilation/compilation.interfaces.js';
import {CompilerInfo} from '../../types/compiler.interfaces.js';
import {CompilationEnvironment} from '../compilation-env.js';
import {VersionInfo} from '../options-handler.js';
import {ExecCompilerCachedFunc} from './base.js';
import type {BuildEnvDownloadInfo} from './buildenv.interfaces.js';
import {BuildEnvSetupCeConanDirect} from './ceconan.js';

export class BuildEnvSetupCeConanGoDirect extends BuildEnvSetupCeConanDirect {
    static override get key() {
        return 'ceconan-go';
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

    override getTarget(key: CacheKey) {
        const goarch = (this.compiler.goarch || 'amd64').toString();
        return BuildEnvSetupCeConanGoDirect.goArchToConanArch(goarch);
    }

    static goArchToConanArch(goarch: string): string {
        switch (goarch) {
            case 'amd64':
                return 'x86_64';
            case '386':
                return 'x86';
            case 'arm64':
                return 'aarch64';
            default:
                return goarch;
        }
    }

    override hasBinariesToLink(details: VersionInfo) {
        return true;
    }

    override shouldDownloadPackage(details: VersionInfo) {
        return true;
    }

    override async download(
        key: CacheKey,
        dirPath: string,
        libraryDetails: Record<string, VersionInfo>,
    ): Promise<BuildEnvDownloadInfo[]> {
        const modifiedLibraryDetails: Record<string, VersionInfo> = {};

        _.each(libraryDetails, (details: VersionInfo, libId: string) => {
            const modifiedDetails = {...details};
            if (!modifiedDetails.lookupname) {
                modifiedDetails.lookupname = `go_${libId}`;
            }
            modifiedLibraryDetails[libId] = modifiedDetails;
        });

        return super.download(key, dirPath, modifiedLibraryDetails);
    }
}
