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

function gcd(a: number, b: number): number {
    return a === 0 ? b : gcd(b % a, a);
}

function gcd_array(arr: number[]): number {
    if (arr.length === 0) return 1;
    let result = arr[0];
    for (let i = 1; i < arr.length; ++i) {
        result = gcd(arr[i], result);

        if (result === 1) return 1;
    }
    return result;
}

function lcm(a: number, b: number): number {
    return Math.floor((a * b) / gcd(a, b));
}

export function makeIconSets(icons: Sponsor[], maxIcons: number): Sponsor[][] {
    const gcd = gcd_array(icons.map(s => s.topIconShowEvery));
    const every: Map<number, Sponsor[]> = new Map();
    let setSize = 1;
    for (const icon of icons) {
        const index = Math.floor(icon.topIconShowEvery / gcd);
        if (!every.has(index)) {
            every.set(index, [icon]);
            setSize = lcm(setSize, index);
        }
        else every.set(index, icons);
    }
    const result: Sponsor[][] = [];
    for (let index = 0; index < setSize; ++index) {
        const thisSlot: Sponsor[] = [];
        const notChosen: Sponsor[] = [];
        for (const [showEvery, icons] of every) {
            if (index % showEvery === 0) thisSlot.push(...icons);
            else notChosen.push(...icons);
        }
        if (thisSlot.length > maxIcons) {
            throw new Error(`Unable to evenly distribute icons, slot ${index} has ${thisSlot.length} vs ${maxIcons}`);
        }
        while (thisSlot.length < maxIcons && notChosen.length > 0) {
            // const totalProb = notChosen.reduce((x, y) => x + 1 / y.topIconShowEvery, 0);
            // const randomChoice = Math.random() * totalProb;
            const randomChoice = Math.floor(Math.random() * notChosen.length);
            thisSlot.push(notChosen[randomChoice]);
            notChosen.splice(randomChoice, 1);
        }
        result.push(thisSlot.sort(compareSponsors));
    }
    return result;
}

class SponsorsImpl implements Sponsors {
    private readonly _levels: Level[];
    private readonly _icons: Sponsor[];

    constructor(levels: Level[]) {
        this._levels = levels;
        this._icons = [];
        for (const level of levels) {
            this._icons.push(...level.sponsors.filter(sponsor => sponsor.topIconShowEvery && sponsor.icon));
        }
    }

    getLevels(): Level[] {
        return this._levels;
    }

    getAllTopIcons(): Sponsor[] {
        return this._icons;
    }

    pickTopIcons(maxIcons: number, randFunc: () => number = Math.random): Sponsor[] {
        // // Grab the icons we always show.
        // const result = this._icons.filter(sponsor => sponsor.topIconShowEvery);
        // // get number of slots left...
        // // generate 100 pick patterns until  the "at least 1 in N" fits? then cycle them.
        // // can do this in the constructor se we know ahead of time
        // // X: one 1 in two
        // // Y, Z: one in 3
        // // need a pattern length of 2*3*3 = 18?
        // // XY XZ YZ XY XZ
        // if (result.length > maxIcons) {
        //     throw new Error('Unable to do the thing with the stuff');
        // }
        // const possibleChoices = this._icons.filter(sponsor => typeof sponsor.topIcon === 'number');
        // while (result.length < maxIcons && possibleChoices.length > 0) {
        //     const totalVisibility = possibleChoices
        //         .map(sponsor => sponsor.topIcon as number)
        //         .reduce((x, y) => x + y, 0);
        //     const randomPick = Math.floor(randFunc() * totalVisibility);
        //     result.push(possibleChoices[randomPick]);
        //     possibleChoices.splice(randomPick, 1);
        // }
        // // some icons on always
        return this._icons.slice(0, maxIcons);
    }
}

export function loadSponsorsFromLevels(levels: Level[]): Sponsors {
    return new SponsorsImpl(levels);
}

export function loadSponsorsFromString(stringConfig: string): Sponsors {
    const sponsorConfig = yaml.parse(stringConfig);
    for (const level of sponsorConfig.levels) {
        for (const required of ['name', 'description', 'sponsors'])
            if (!level[required]) throw new Error(`Level is missing '${required}'`);
        level.sponsors = level.sponsors.map(parse).sort(compareSponsors);
    }
    return loadSponsorsFromLevels(sponsorConfig.levels);
}
