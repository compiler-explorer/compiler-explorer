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

import * as fs from 'fs';

import {SiteTemplatesType} from '../../types/features/site-templates.interfaces';

const siteTemplates: SiteTemplatesType = {
    meta: {},
    templates: {},
};

function splitProperty(line: string) {
    return [line.substring(0, line.indexOf('=')), line.substring(line.indexOf('=') + 1)];
}

function partition<T>(array: T[], filter: (value: T) => boolean): [T[], T[]] {
    const pass: T[] = [],
        fail: T[] = [];
    for (const item of array) {
        if (filter(item)) {
            pass.push(item);
        } else {
            fail.push(item);
        }
    }
    return [pass, fail];
}

export function loadSiteTemplates(configDir: string) {
    const [meta, templates] = partition(
        fs
            .readFileSync(configDir + '/site-templates.conf', 'utf8')
            .split('\n')
            .filter(l => l !== '')
            .map(splitProperty)
            .map(pair => [pair[0], pair[1].replace(/^https:\/\/godbolt.org\/#/, '')]),
        ([name, _]) => name.startsWith('meta.')
    );
    siteTemplates.meta = Object.fromEntries(meta);
    siteTemplates.templates = Object.fromEntries(templates);
}

export function getSiteTemplates() {
    return siteTemplates;
}
