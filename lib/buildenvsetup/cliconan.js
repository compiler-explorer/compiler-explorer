// Copyright (c) 2020, Compiler Explorer Authors
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

import fs from 'fs-extra';
import _ from 'underscore';

import * as exec from '../exec';
import { logger } from '../logger';

import { BuildEnvSetupBase } from './base';

export class BuildEnvSetupCliConan extends BuildEnvSetupBase {
    static get key() {
        return 'cliconan';
    }

    constructor(compilerInfo, env, execCompilerCachedFunc) {
        super(compilerInfo, env, execCompilerCachedFunc);

        this.exe = compilerInfo.buildenvsetup.props('exe', 'conan');
        this.remote = compilerInfo.buildenvsetup.props('remote', false);
        this.onlyonstaticliblink = compilerInfo.buildenvsetup.props('onlyonstaticliblink', true);
    }

    async setup(key, dirPath, libraryDetails) {
        if (!this.onlyonstaticliblink || this.hasAtLeastOneBinaryToLink(libraryDetails)) {
            await this.prepareConanRequest(libraryDetails, dirPath);
            return this.installLibrariesViaConan(key, dirPath);
        } else {
            return Promise.resolve();
        }
    }

    hasBinariesToLink(details) {
        return (details.libpath.length === 0) && (details.staticliblink.length > 0);
    }

    hasAtLeastOneBinaryToLink(libraryDetails) {
        return _.some(libraryDetails, (details) => this.hasBinariesToLink(details));
    }

    async prepareConanRequest(libraryDetails, dirPath) {
        let data = '[requires]\n';

        _.each(libraryDetails, (details, libId) => {
            if (this.hasBinariesToLink(details)) {
                data += `${libId}/${details.version}@${libId}/${details.version}\n`;
            }
        });

        data +=
            '[imports]\n' +
            'lib, *.a -> .\n' +
            'lib, *.so* -> .\n';

        return fs.writeFile(path.join(dirPath, 'conanfile.txt'), data);
    }

    async installLibrariesViaConan(key, dirPath) {
        const arch = this.getTarget(key);
        const libcxx = this.getLibcxx(key);
        const stdver = '';
        const flagcollection = '';

        const args = ['install', '.'];
        if (this.remote) args.push('-r', this.remote);
        args.push(
            '-s', 'os=Linux',
            '-s', 'build_type=Debug',
            '-s', `compiler=${this.compilerTypeOrGCC}`,
            '-s', `compiler.version=${this.compiler.id}`,
            '-s', `compiler.libcxx=${libcxx}`,
            '-s', `arch=${arch}`,
            '-s', `stdver=${stdver}`,
            '-s', `flagcollection=${flagcollection}`);

        logger.info('Conan install: ', args);
        return exec.execute(this.exe, args, {customCwd: dirPath});
    }
}
