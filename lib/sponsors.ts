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

import {Level, Sponsor, Sponsors} from './sponsors.interfaces';

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
        topIconShowEvery: mapOrString.topIconShowEvery || 0,
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

function calcMean(values: number[]): number {
    return values.reduce((x, y) => x + y, 0) / values.length;
}

function squareSumFromMean(values: number[]): number {
    const mean = calcMean(values);
    return values.reduce((x, y) => x + (y - mean) * (y - mean), 0);
}

function standardDeviation(values: number[]): number {
    return values.length < 2 ? 0 : Math.sqrt(squareSumFromMean(values) / (values.length - 1));
}

// A sponsor icon set is ok if:
// - each sponsor is shown at least every "topIconShowEvery"
// - the standard deviation for the number of showings between sponsors at the same "show every' is not too high: that
//   is we fairly distribute showings of sponsors at the same level
function sponsorIconSetsOk(
    sponsorAppearanceCount: Map<Sponsor, number>,
    totalAppearances: number,
    maxStandardDeviation: number
): boolean {
    const countsByShowEvery: Map<number, number[]> = new Map();
    for (const [icon, count] of sponsorAppearanceCount.entries()) {
        const seenEvery = count > 0 ? totalAppearances / count : Infinity;
        if (seenEvery > icon.topIconShowEvery) {
            return false;
        }
        const others = countsByShowEvery.get(icon.topIconShowEvery) || [];
        others.push(seenEvery);
        countsByShowEvery.set(icon.topIconShowEvery, others);
    }
    return Math.max(...[...countsByShowEvery.values()].map(standardDeviation)) <= maxStandardDeviation;
}

export function makeIconSets(
    icons: Sponsor[],
    maxIcons: number,
    maxIters = 100,
    maxStandardDeviation = 0.5
): Sponsor[][] {
    const result: Sponsor[][] = [];
    const sponsorAppearanceCount: Map<Sponsor, number> = new Map();
    for (const icon of icons) sponsorAppearanceCount.set(icon, 0);
    while (!sponsorIconSetsOk(sponsorAppearanceCount, result.length, maxStandardDeviation)) {
        if (result.length > maxIters) {
            throw new Error(`Unable to find a solution in ${maxIters}`);
        }
        const toPick = icons.map(icon => {
            return {
                icon: icon,
                // Number of times we'd expect to see this, divided by number of times we saw it
                error: result.length / icon.topIconShowEvery / (sponsorAppearanceCount.get(icon) || 0.00001),
            };
        });
        toPick.sort((lhs, rhs) => rhs.error - lhs.error);
        const chosen = toPick
            .slice(0, maxIcons)
            .map(x => x.icon)
            .sort(compareSponsors);
        for (const c of chosen) sponsorAppearanceCount.set(c, (sponsorAppearanceCount.get(c) || 0) + 1);
        result.push(chosen);
    }
    return result;
}

class SponsorsImpl implements Sponsors {
    private readonly _levels: Level[];
    private readonly _icons: Sponsor[];
    private readonly _iconSets: Sponsor[][];
    private _nextSet: number;

    constructor(levels: Level[], maxTopIcons) {
        this._levels = levels;
        this._icons = [];
        for (const level of levels) {
            this._icons.push(...level.sponsors.filter(sponsor => sponsor.topIconShowEvery && sponsor.icon));
        }
        this._iconSets = makeIconSets(this._icons, maxTopIcons);
        this._nextSet = 0;
    }

    getLevels(): Level[] {
        return this._levels;
    }

    getAllTopIcons(): Sponsor[] {
        return this._icons;
    }

    pickTopIcons(): Sponsor[] {
        const result = this._iconSets[this._nextSet];
        this._nextSet = (this._nextSet + 1) % this._iconSets.length;
        return result;
    }
}

export function loadSponsorsFromLevels(levels: Level[], maxTopIcons: number): Sponsors {
    return new SponsorsImpl(levels, maxTopIcons);
}

export function loadSponsorsFromString(stringConfig: string): Sponsors {
    const sponsorConfig = yaml.parse(stringConfig);
    for (const level of sponsorConfig.levels) {
        for (const required of ['name', 'description', 'sponsors'])
            if (!level[required]) throw new Error(`Level is missing '${required}'`);
        level.sponsors = level.sponsors.map(parse).sort(compareSponsors);
    }
    return loadSponsorsFromLevels(sponsorConfig.levels, sponsorConfig.maxTopIcons || 3);
}
