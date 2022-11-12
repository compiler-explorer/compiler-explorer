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

import {loadSponsorsFromString, makeIconSets, parse} from '../lib/sponsors';

import {resolvePathFromTestRoot, should} from './utils';

describe('Sponsors', () => {
    it('should expand names to objects', () => {
        parse('moo').name.should.eq('moo');
    });
    it('should handle just names', () => {
        parse({name: 'moo'}).name.should.eq('moo');
    });
    it('should default empty params', () => {
        const obj = parse('moo');
        should.equal(obj.description, undefined);
        should.equal(obj.url, undefined);
        obj.onclick.should.eq('');
        should.equal(obj.img, undefined);
        should.equal(obj.icon, undefined);
        should.equal(obj.icon_dark, undefined);
        obj.topIconShowEvery.should.eq(0);
        obj.sideBySide.should.be.false;
        should.equal(obj.statsId, undefined);
    });
    it('should make descriptions always one-sized arrays', () => {
        parse({name: 'moo', description: 'desc'}).description.should.deep.eq(['desc']);
    });
    it('should pass through descriptions', () => {
        parse({name: 'moo', description: ['desc1', 'desc2']}).description.should.deep.eq(['desc1', 'desc2']);
    });
    it('should pass through icons', () => {
        parse({name: 'bob', icon: 'icon'}).icon.should.eq('icon');
    });
    it('should pick icons over images', () => {
        parse({name: 'bob', img: 'img', icon: 'icon'}).icon.should.eq('icon');
    });
    it('should pick icons if not img', () => {
        parse({name: 'bob', img: 'img'}).icon.should.eq('img');
    });
    it('should pick dark icons if specified', () => {
        parse({name: 'bob', icon: 'icon', icon_dark: 'icon_dark'}).icon_dark.should.eq('icon_dark');
    });
    it('should handle topIcons', () => {
        parse({name: 'bob', topIconShowEvery: 2}).topIconShowEvery.should.eq(2);
    });
    it('should handle clicks', () => {
        parse({
            name: 'bob',
            url: 'https://some.host/click',
        }).onclick.should.eq('window.onSponsorClick("https://some.host/click");');
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
        sample.should.not.be.null;
        const levels = sample.getLevels();
        levels.length.should.eq(2);
        levels[0].name.should.eq('Patreon Legends');
        levels[1].name.should.eq('Patreons');
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
        peeps.map(sponsor => sponsor.name).should.deep.equals(['A', 'B', 'C', 'D']);
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
        peeps
            .map(sponsor => {
                return {name: sponsor.name, priority: sponsor.priority};
            })
            .should.deep.equals([
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
        icons.map(s => s.name).should.deep.equals(['one', 'four']);
    });

    it('should pick icons appropriately when all required every 3', () => {
        const sponsor1 = parse({name: 'Sponsor1', topIconShowEvery: 3, icon: '1'});
        const sponsor2 = parse({name: 'Sponsor2', topIconShowEvery: 3, icon: '2'});
        const sponsor3 = parse({name: 'Sponsor3', topIconShowEvery: 3, icon: '3'});
        const icons = [sponsor1, sponsor2, sponsor3];
        makeIconSets(icons, 10).should.deep.eq([icons]);
        makeIconSets(icons, 3).should.deep.eq([icons]);
        makeIconSets(icons, 2).should.deep.eq([
            [sponsor1, sponsor2],
            [sponsor1, sponsor3],
            [sponsor2, sponsor3],
        ]);
        makeIconSets(icons, 1).should.deep.eq([[sponsor1], [sponsor2], [sponsor3]]);
    });
    it('should pick icons appropriately when not required on different schedules', () => {
        const sponsor1 = parse({name: 'Sponsor1', topIconShowEvery: 1, icon: '1'});
        const sponsor2 = parse({name: 'Sponsor2', topIconShowEvery: 2, icon: '2'});
        const sponsor3 = parse({name: 'Sponsor3', topIconShowEvery: 3, icon: '3'});
        const icons = [sponsor1, sponsor2, sponsor3];
        makeIconSets(icons, 10).should.deep.eq([icons]);
        makeIconSets(icons, 3).should.deep.eq([icons]);
        makeIconSets(icons, 2).should.deep.eq([
            [sponsor1, sponsor2],
            [sponsor1, sponsor3],
        ]);
        (() => makeIconSets(icons, 1)).should.throw();
    });
    it('should pick icons appropriately with a lot of sponsors on representative schedules', () => {
        const sponsor1 = parse({name: 'Sponsor1', topIconShowEvery: 1, icon: '1'});
        const sponsor2 = parse({name: 'Sponsor2', topIconShowEvery: 3, icon: '2'});
        const sponsor3 = parse({name: 'Sponsor3', topIconShowEvery: 3, icon: '3'});
        const sponsor4 = parse({name: 'Sponsor4', topIconShowEvery: 3, icon: '3'});
        const sponsor5 = parse({name: 'Sponsor5', topIconShowEvery: 3, icon: '3'});
        const icons = [sponsor1, sponsor2, sponsor3, sponsor4, sponsor5];
        makeIconSets(icons, 10).should.deep.eq([icons]);
        makeIconSets(icons, 3).should.deep.eq([
            [sponsor1, sponsor2, sponsor3],
            [sponsor1, sponsor4, sponsor5],
        ]);
        (() => makeIconSets(icons, 1)).should.throw();
    });
    it('should handle alternating', () => {
        const sponsor1 = parse({name: 'Sponsor1', topIconShowEvery: 1, icon: '1'});
        const sponsor2 = parse({name: 'Sponsor2', topIconShowEvery: 1, icon: '2'});
        const sponsor3 = parse({name: 'Sponsor3', topIconShowEvery: 2, icon: '3'});
        const sponsor4 = parse({name: 'Sponsor4', topIconShowEvery: 2, icon: '4'});
        const icons = [sponsor1, sponsor2, sponsor3, sponsor4];
        makeIconSets(icons, 4).should.deep.eq([icons]);
        makeIconSets(icons, 3).should.deep.eq([
            [sponsor1, sponsor2, sponsor3],
            [sponsor1, sponsor2, sponsor4],
        ]);
        (() => makeIconSets(icons, 2)).should.throw();
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
        const picks = [];
        for (let load = 0; load < numLoads; ++load) {
            picks.push(sponsors.pickTopIcons());
        }
        const countBySponsor = new Map();
        for (const pick of picks) {
            for (const sponsor of pick) {
                countBySponsor.set(sponsor, (countBySponsor.get(sponsor) || 0) + 1);
            }
            pick.length.should.eq(expectedNumIcons);
        }
        for (const topIcon of sponsors.getAllTopIcons()) {
            const appearsEvery = countBySponsor.get(topIcon) / numLoads;
            appearsEvery.should.lte(topIcon.topIconShowEvery);
        }
    });
});
