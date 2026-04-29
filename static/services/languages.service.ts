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

import {Language, LanguageKey} from '../../types/languages.interfaces.js';
import {optionsHash} from '../options.js';
import {SentryCapture} from '../sentry.js';

export type LanguageMap = Partial<Record<LanguageKey, Language>>;

export class LanguagesService {
    private loadPromise: Promise<LanguageMap> | null = null;
    private cache: LanguageMap | null = null;

    async getLanguages(): Promise<LanguageMap> {
        if (!this.loadPromise) {
            const promise = this.fetchLanguages();
            this.loadPromise = promise;
            try {
                this.cache = await promise;
            } catch (e) {
                SentryCapture(e, 'fetchLanguages');
                if (this.loadPromise === promise) this.loadPromise = null;
                throw e;
            }
        }
        return this.loadPromise;
    }

    getLanguagesOrFail(): LanguageMap {
        if (!this.cache) {
            throw new Error(
                'Languages have not been loaded yet. Ensure languagesService.getLanguages() has completed.',
            );
        }
        return this.cache;
    }

    private static readonly languageFields = [
        'id',
        'name',
        'extensions',
        'monaco',
        'alias',
        'formatter',
        'supportsExecute',
        'logoFilename',
        'logoFilenameDark',
        'example',
        'monacoDisassembly',
        'noAsmHint',
        'tooltip',
        'digitSeparator',
        'defaultCompiler',
        'defaultLibs',
    ].join(',');

    private async fetchLanguages(): Promise<LanguageMap> {
        const response = await fetch(
            `${window.httpRoot}api/languages?fields=${LanguagesService.languageFields}&hash=${optionsHash}`,
            {headers: {Accept: 'application/json'}},
        );
        const languages: Language[] = await response.json();
        const result: LanguageMap = {};
        for (const lang of languages) {
            result[lang.id] = lang;
        }
        return result;
    }
}

export const languagesService = new LanguagesService();
