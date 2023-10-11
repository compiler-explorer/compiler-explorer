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

import * as exec from '../exec.js';
import {logger} from '../logger.js';

import {BuildEnvSetupBase} from './base.js';
import type {BuildEnvDownloadInfo} from './buildenv.interfaces.js';

export class BuildEnvSetupCliConan extends BuildEnvSetupBase {
    private exe: any;
    private remote: any;
    private onlyonstaticliblink: any;

    static get key() {
        return 'cliconan';
    }

    constructor(compilerInfo, env) {
        super(compilerInfo, env);

        this.exe = compilerInfo.buildenvsetup.props('exe', 'conan');
        this.remote = compilerInfo.buildenvsetup.props('remote', false);
        this.onlyonstaticliblink = compilerInfo.buildenvsetup.props('onlyonstaticliblink', true);
    }

    override async setup(key, dirPath, libraryDetails, binary): Promise<BuildEnvDownloadInfo[]> {
        if (this.onlyonstaticliblink && !binary) return [];

        const librariesToDownload = _.filter(libraryDetails, details => this.shouldDownloadPackage(details));

        await this.prepareConanRequest(librariesToDownload, dirPath);
        return this.installLibrariesViaConan(key, dirPath);
    }

    async prepareConanRequest(libraryDetails, dirPath) {
        let data = '[requires]\n';

        _.each(libraryDetails, (details, libId) => {
            if (this.hasBinariesToLink(details)) {
                data += `${libId}/${details.version}@${libId}/${details.version}\n`;
            }
        });

        // prettier-ignore
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
        // prettier-ignore
        args.push(
            '-s', 'os=Linux',
            '-s', 'build_type=Debug',
            '-s', `compiler=${this.compilerTypeOrGCC}`,
            '-s', `compiler.version=${this.compiler.id}`,
            '-s', `compiler.libcxx=${libcxx}`,
            '-s', `arch=${arch}`,
            '-s', `stdver=${stdver}`,
            '-s', `flagcollection=${flagcollection}`,
        );

        logger.info('Conan install: ', args);

        const result = await exec.execute(this.exe, args, {customCwd: dirPath});
        const info: BuildEnvDownloadInfo = {
            step: 'Conan install',
            packageUrl: args.join(' '),
            time: result.execTime,
        };

        return [info];
    }
}
