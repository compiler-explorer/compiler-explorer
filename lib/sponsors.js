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
"use strict";

const yaml = require('yaml');

function namify(mapOrString) {
    if (typeof (mapOrString) == "string") return {name: mapOrString};
    return mapOrString;
}

function clickify(sponsor) {
    if (sponsor.url) {
        sponsor.onclick=`window.onSponsorClick(${JSON.stringify(sponsor)});`;
    }
    return sponsor;
}

function compareSponsors(lhs, rhs) {
    const lhsPrio = lhs.priority || 0;
    const rhsPrio = rhs.priority || 0;
    if (lhsPrio !== rhsPrio) return rhsPrio - lhsPrio;
    return lhs.name.localeCompare(rhs.name);
}


function loadFromString(stringConfig) {
    const sponsorConfig = yaml.parse(stringConfig);
    // TODO: on final merge, update Patreon information.
    // TODO: on final merge, ensure thanks.html matches sponsors.yaml
    sponsorConfig.icons = [];
    for (const level of sponsorConfig.levels) {
        for (const required of ['name', 'description'])
            if (!level[required])
                throw new Error(`Level is missing '${required}'`);
        level.sponsors = level.sponsors
            .map(namify)
            .map(clickify)
            .sort(compareSponsors);
        sponsorConfig.icons.push(...level.sponsors.filter(sponsor => sponsor.topIcon && sponsor.img));
    }
    return sponsorConfig;
}

exports.loadFromString = loadFromString;
