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

import urljoin from 'url-join';

import {getRemoteId} from '../../shared/remote-utils.js';
import {LanguageLibs} from '../options.interfaces.js';
import {optionsHash} from '../options.js';
import {SentryCapture} from '../sentry.js';

export class LibsService {
    private readonly loadPromises = new Map<string, Promise<LanguageLibs>>();

    async getLibsForLang(langId: string): Promise<LanguageLibs> {
        let promise = this.loadPromises.get(langId);
        if (!promise) {
            promise = this.fetchLibsForLang(langId);
            this.loadPromises.set(langId, promise);
            promise.catch(e => {
                SentryCapture(e, `fetchLibsForLang(${langId})`);
                this.loadPromises.delete(langId);
            });
        }
        return promise;
    }

    async getRemoteLibs(langId: string, remoteUrl: string): Promise<LanguageLibs> {
        const remoteId = getRemoteId(remoteUrl, langId);
        let promise = this.loadPromises.get(remoteId);
        if (!promise) {
            promise = this.fetchRemoteLibs(langId, remoteUrl);
            this.loadPromises.set(remoteId, promise);
        }
        return promise;
    }

    private async fetchLibsForLang(langId: string): Promise<LanguageLibs> {
        const response = await fetch(
            `${window.httpRoot}api/libraries/${encodeURIComponent(langId)}?hash=${optionsHash}`,
            {headers: {Accept: 'application/json'}},
        );
        const libsArr = await response.json();
        return this.libArrayToRecord(libsArr);
    }

    private async fetchRemoteLibs(langId: string, remoteUrl: string): Promise<LanguageLibs> {
        const url = urljoin(remoteUrl, 'api/libraries', encodeURIComponent(langId));
        try {
            const response = await fetch(url, {headers: {Accept: 'application/json'}});
            const libsArr = await response.json();
            return this.libArrayToRecord(libsArr);
        } catch (e) {
            SentryCapture(e, `fetchRemoteLibs(${langId}, ${remoteUrl})`);
            return {};
        }
    }

    private libArrayToRecord(libsArr: any[]): LanguageLibs {
        const libs: LanguageLibs = {};
        for (const lib of libsArr) {
            const versions = Object.fromEntries(lib.versions.map((v: any) => [v.id, v]));
            libs[lib.id] = {...lib, versions};
        }
        return libs;
    }
}

export const libsService = new LibsService();
