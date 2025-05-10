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

import fs from 'fs';

import {describe, expect, it} from 'vitest';

import {Sponsor} from '../lib/sponsors.interfaces.js';
import {loadSponsorsFromString, makeIconSets, parse} from '../lib/sponsors.js';

import {resolvePathFromTestRoot} from './utils.js';

describe('Sponsors', () => {
    it('should expand names to objects', () => {
        expect(parse('moo').name).toEqual('moo');
    });
    it('should handle just names', () => {
        expect(parse({name: 'moo'}).name).toEqual('moo');
    });
    it('should default empty params', () => {
        const obj = parse('moo');
        expect(obj.description).toBeUndefined();
        expect(obj.url).toBeUndefined();
        expect(obj.onclick).toEqual('');
        expect(obj.img).toBeUndefined();
        expect(obj.icon).toBeUndefined();
        expect(obj.icon_dark).toBeUndefined();
        expect(obj.topIconShowEvery).toEqual(0);
        expect(obj.displayType).toEqual('Above');
        expect(obj.statsId).toBeUndefined();
        expect(obj.style).toEqual({});
    });
    it('should make descriptions always one-sized arrays', () => {
        expect(parse({name: 'moo', description: 'desc'}).description).toEqual(['desc']);
    });
    it('should pass through descriptions', () => {
        expect(parse({name: 'moo', description: ['desc1', 'desc2']}).description).toEqual(['desc1', 'desc2']);
    });
    it('should pass through icons', () => {
        expect(parse({name: 'bob', icon: 'icon'}).icon).toEqual('icon');
    });
    it('should pick icons over images', () => {
        expect(parse({name: 'bob', img: 'img', icon: 'icon'}).icon).toEqual('icon');
    });
    it('should pick icons if not img', () => {
        expect(parse({name: 'bob', img: 'img'}).icon).toEqual('img');
    });
    it('should pick dark icons if specified', () => {
        expect(parse({name: 'bob', icon: 'icon', icon_dark: 'icon_dark'}).icon_dark).toEqual('icon_dark');
    });
    it('should handle styles', () => {
        expect(parse({name: 'bob', bgColour: 'red'}).style).toEqual({'background-color': 'red'});
    });
    it('should handle clicks', () => {
        expect(
            parse({
                name: 'bob',
                url: 'https://some.host/click',
            }).onclick,
        ).toEqual('window.onSponsorClick("https://some.host/click");');
    });

    it('should load a simple example', () => {
        const sample = loadSponsorsFromString(`
---
levels:
  - name: Patreon Legends
    description: 'These amazing people have pledged at the highest level of support. Huge thanks to these fine folks:'
    sponsors:
      - name: Kitty Kat
        url: https://meow.meow
      - name: Aardvark
        url: https://ard.vark
  - name: Patreons
    description: 'Thanks to all my patrons:'
    sponsors:
      - Lovely
      - People
      - Yay
`);
        expect(sample).not.toBeNull();
        const levels = sample.getLevels();
        expect(levels.length).toEqual(2);
        expect(levels[0].name).toEqual('Patreon Legends');
        expect(levels[1].name).toEqual('Patreons');
    });

    it('should sort sponsors by name', () => {
        const peeps = loadSponsorsFromString(`
---
levels:
  - name: a
    description: d
    sponsors:
    - D
    - C
    - A
    - B
        `).getLevels()[0].sponsors;
        expect(peeps.map(sponsor => sponsor.name)).toEqual(['A', 'B', 'C', 'D']);
    });
    it('should sort sponsors by priority then name', () => {
        const peeps = loadSponsorsFromString(`
---
levels:
  - name: a
    description: d
    sponsors:
    - name: D
      priority: 100
    - name: C
      priority: 50
    - name: B
      priority: 50
        `).getLevels()[0].sponsors;
        expect(
            peeps.map(sponsor => {
                return {name: sponsor.name, priority: sponsor.priority};
            }),
        ).toEqual([
            {name: 'D', priority: 100},
            {name: 'B', priority: 50},
            {name: 'C', priority: 50},
        ]);
    });

    it('should pick out all the top level icons', () => {
        const icons = loadSponsorsFromString(`
---
levels:
  - name: a
    description: d
    sponsors:
    - name: one
      img: pick_me
      topIconShowEvery: 1
    - name: two
      img: not_me
  - name: another level
    description: more
    sponsors:
    - name: three
      img: not_me_either
      topIconShowEvery: 0
    - name: four
      img: pick_me_also
      topIconShowEvery: 2
    - name: five
      topIconShowEvery: 3
        `).getAllTopIcons();
        expect(icons.map(s => s.name)).toEqual(['one', 'four']);
    });

    it('should pick icons appropriately when all required every 3', () => {
        const sponsor1 = parse({name: 'Sponsor1', topIconShowEvery: 3, icon: '1'});
        const sponsor2 = parse({name: 'Sponsor2', topIconShowEvery: 3, icon: '2'});
        const sponsor3 = parse({name: 'Sponsor3', topIconShowEvery: 3, icon: '3'});
        const icons = [sponsor1, sponsor2, sponsor3];
        expect(makeIconSets(icons, 10)).toEqual([icons]);
        expect(makeIconSets(icons, 3)).toEqual([icons]);
        expect(makeIconSets(icons, 2)).toEqual([
            [sponsor1, sponsor2],
            [sponsor1, sponsor3],
            [sponsor2, sponsor3],
        ]);
        expect(makeIconSets(icons, 1)).toEqual([[sponsor1], [sponsor2], [sponsor3]]);
    });
    it('should pick icons appropriately when not required on different schedules', () => {
        const sponsor1 = parse({name: 'Sponsor1', topIconShowEvery: 1, icon: '1'});
        const sponsor2 = parse({name: 'Sponsor2', topIconShowEvery: 2, icon: '2'});
        const sponsor3 = parse({name: 'Sponsor3', topIconShowEvery: 3, icon: '3'});
        const icons = [sponsor1, sponsor2, sponsor3];
        expect(makeIconSets(icons, 10)).toEqual([icons]);
        expect(makeIconSets(icons, 3)).toEqual([icons]);
        expect(makeIconSets(icons, 2)).toEqual([
            [sponsor1, sponsor2],
            [sponsor1, sponsor3],
        ]);
        expect(() => makeIconSets(icons, 1)).toThrow();
    });
    it('should pick icons appropriately with a lot of sponsors on representative schedules', () => {
        const sponsor1 = parse({name: 'Sponsor1', topIconShowEvery: 1, icon: '1'});
        const sponsor2 = parse({name: 'Sponsor2', topIconShowEvery: 3, icon: '2'});
        const sponsor3 = parse({name: 'Sponsor3', topIconShowEvery: 3, icon: '3'});
        const sponsor4 = parse({name: 'Sponsor4', topIconShowEvery: 3, icon: '3'});
        const sponsor5 = parse({name: 'Sponsor5', topIconShowEvery: 3, icon: '3'});
        const icons = [sponsor1, sponsor2, sponsor3, sponsor4, sponsor5];
        expect(makeIconSets(icons, 10)).toEqual([icons]);
        expect(makeIconSets(icons, 3)).toEqual([
            [sponsor1, sponsor2, sponsor3],
            [sponsor1, sponsor4, sponsor5],
        ]);
        expect(() => makeIconSets(icons, 1)).toThrow();
    });
    it('should handle alternating', () => {
        const sponsor1 = parse({name: 'Sponsor1', topIconShowEvery: 1, icon: '1'});
        const sponsor2 = parse({name: 'Sponsor2', topIconShowEvery: 1, icon: '2'});
        const sponsor3 = parse({name: 'Sponsor3', topIconShowEvery: 2, icon: '3'});
        const sponsor4 = parse({name: 'Sponsor4', topIconShowEvery: 2, icon: '4'});
        const icons = [sponsor1, sponsor2, sponsor3, sponsor4];
        expect(makeIconSets(icons, 4)).toEqual([icons]);
        expect(makeIconSets(icons, 3)).toEqual([
            [sponsor1, sponsor2, sponsor3],
            [sponsor1, sponsor2, sponsor4],
        ]);
        expect(() => makeIconSets(icons, 2)).toThrow();
    });
    it('should pick icons appropriately when one is required every time and the others fit in ok every 3', () => {
        const sponsor1 = parse({name: 'Sponsor1', topIconShowEvery: 1, icon: '1'});
        const sponsor2 = parse({name: 'Sponsor2', topIconShowEvery: 3, icon: '2'});
        const sponsor3 = parse({name: 'Sponsor3', topIconShowEvery: 3, icon: '3'});
        const sponsor4 = parse({name: 'Sponsor4', topIconShowEvery: 3, icon: '4'});
        const sponsor5 = parse({name: 'Sponsor5', topIconShowEvery: 3, icon: '5'});
        const sponsor6 = parse({name: 'Sponsor6', topIconShowEvery: 3, icon: '6'});
        const icons = [sponsor1, sponsor2, sponsor3, sponsor4, sponsor5, sponsor6];
        expect(makeIconSets(icons, 10)).toEqual([icons]);
        expect(makeIconSets(icons, 3)).toEqual([
            [sponsor1, sponsor2, sponsor3],
            [sponsor1, sponsor4, sponsor5],
            [sponsor1, sponsor2, sponsor6],
            [sponsor1, sponsor3, sponsor4],
            [sponsor1, sponsor5, sponsor6],
        ]);
    });
});

describe('Our specific sponsor file', () => {
    const stringConfig = fs.readFileSync(resolvePathFromTestRoot('../etc/config/sponsors.yaml')).toString();
    it('should parse the current config', () => {
        loadSponsorsFromString(stringConfig);
    });
    it('should pick appropriate sponsor icons', () => {
        const numLoads = 100;
        const expectedNumIcons = 3;

        const sponsors = loadSponsorsFromString(stringConfig);
        const picks: Sponsor[][] = [];
        for (let load = 0; load < numLoads; ++load) {
            picks.push(sponsors.pickTopIcons());
        }
        const countBySponsor = new Map();
        for (const pick of picks) {
            for (const sponsor of pick) {
                countBySponsor.set(sponsor, (countBySponsor.get(sponsor) || 0) + 1);
            }
            expect(pick.length).toEqual(expectedNumIcons);
        }
        for (const topIcon of sponsors.getAllTopIcons()) {
            const appearsEvery = countBySponsor.get(topIcon) / numLoads;
            expect(appearsEvery).toBeLessThanOrEqual(topIcon.topIconShowEvery);
        }
    });
});
