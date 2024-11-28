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

import * as fsp from 'node:fs/promises';
import path from 'node:path';

import _ from 'underscore';

import {SiteTemplatesType} from '../types/features/site-templates.interfaces.js';

let siteTemplates: SiteTemplatesType;

export async function getSiteTemplates(): Promise<SiteTemplatesType> {
    siteTemplates ??= await loadSiteTemplates('etc/config');
    return siteTemplates;
}

/**
 * Load all the site templates from the given config directory
 *
 * The configuration keys that start with "meta" are returned as metadata keys in the 0th element of the returned tuple
 */
async function loadSiteTemplates(configDir: string): Promise<SiteTemplatesType> {
    const config = await fsp.readFile(path.join(configDir, 'site-templates.conf'), 'utf8');
    const properties = config
        .split('\n')
        .filter(l => l.length > 0)
        .map(property => {
            // Rison does not have equal signs in its syntax, so we do not need to account for any trailing equal signs
            // after the first one.
            const [name, value] = property.split('=');
            return [name, value] as const;
        });
    const [meta, templates] = _.partition(properties, ([name]) => name.startsWith('meta.'));
    return {meta: Object.fromEntries(meta), templates: Object.fromEntries(templates)};
}
