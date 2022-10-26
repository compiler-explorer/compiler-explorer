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

import yaml from 'yaml';

import {Sponsor, Sponsors} from './sponsors.interfaces';

export function parse(mapOrString: Record<string, any> | string): Sponsor {
    if (typeof mapOrString == 'string') mapOrString = {name: mapOrString};
    return {
        name: mapOrString.name,
        description: typeof mapOrString.description === 'string' ? [mapOrString.description] : mapOrString.description,
        url: mapOrString.url,
        onclick: mapOrString.url ? `window.onSponsorClick(${JSON.stringify(mapOrString.url)});` : '',
        img: mapOrString.img,
        icon: mapOrString.icon || mapOrString.img,
        icon_dark: mapOrString.icon_dark,
        topIcon: !!mapOrString.topIcon,
        sideBySide: !!mapOrString.sideBySide,
        priority: mapOrString.priority || 0,
        statsId: mapOrString.statsId,
    };
}

function compareSponsors(lhs: Sponsor, rhs: Sponsor): number {
    const lhsPrio = lhs.priority;
    const rhsPrio = rhs.priority;
    if (lhsPrio !== rhsPrio) return rhsPrio - lhsPrio;
    return lhs.name.localeCompare(rhs.name);
}

export function loadSponsorsFromString(stringConfig: string): Sponsors {
    const sponsorConfig = yaml.parse(stringConfig);
    sponsorConfig.icons = [];
    for (const level of sponsorConfig.levels) {
        for (const required of ['name', 'description', 'sponsors'])
            if (!level[required]) throw new Error(`Level is missing '${required}'`);
        level.sponsors = level.sponsors.map(parse).sort(compareSponsors);
        sponsorConfig.icons.push(...level.sponsors.filter(sponsor => sponsor.topIcon && sponsor.icon));
    }
    return sponsorConfig;
}
