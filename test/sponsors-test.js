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
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ,
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

const chai = require('chai');
chai.should();
chai.use(require('deep-equal-in-any-order'));
const sponsors = require('../lib/sponsors');

describe('Sponsors', () => {
    it('should load a simple example', () => {
        const sample = sponsors.loadFromString(`
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
        sample.levels.length.should.eq(2);
        sample.levels[0].name.should.eq('Patreon Legends');
        sample.levels[1].name.should.eq('Patreons');
    });

    it('should expand names to objects', () => {
        const folks = sponsors.loadFromString(`
---
levels:
  - name: a
    description: d
    sponsors:
    - Just a string
    - name: An object
        `).levels[0].sponsors;
        folks.should.deep.equalInAnyOrder([
          {name: "An object"},
          {name: "Just a string"}
      ]);
    });

    it('should sort sponsors by name', () => {
        const peeps = sponsors.loadFromString(`
---
levels:
  - name: a
    description: d
    sponsors:
    - D
    - C
    - A
    - B
        `).levels[0].sponsors;
        peeps.should.deep.equals([
            {name: 'A'},
            {name: 'B'},
            {name: 'C'},
            {name: 'D'}
        ]);
    });
    it('should sort sponsors by priority then name', () => {
        const peeps = sponsors.loadFromString(`
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
        `).levels[0].sponsors;
        peeps.should.deep.equals([
            {name: 'D', priority: 100},
            {name: 'B', priority: 50},
            {name: 'C', priority: 50}
        ]);
    });
    it('should pick icon over img', () => {
        const things = sponsors.loadFromString(`
---
levels:
  - name: a
    description: d
    sponsors:
    - name: one
      img: image
    - name: two
      img: not_an_icon
      icon: icon
        `).levels[0].sponsors;
        things.should.deep.equalInAnyOrder([
            {name: "one", icon: "image", img: "image"},
            {name: "two", icon: "icon", img: "not_an_icon"}
        ]);
    });

    it('should pick out the top level icons', () => {
        const icons = sponsors.loadFromString(`
---
levels:
  - name: a
    description: d
    sponsors:
    - name: one
      img: pick_me
      topIcon: true
    - name: two
      img: not_me
  - name: another level
    description: more
    sponsors:
    - name: three
      img: not_me_either
      topIcon: false
    - name: four
      img: pick_me_also
      topIcon: true
    - name: five
      topIcon: true
        `).icons;
        icons.should.deep.equalInAnyOrder([
            {name: "one", icon: "pick_me", img: "pick_me", topIcon: true},
            {name: "four", icon: "pick_me_also", img: "pick_me_also", topIcon: true}
        ]);
    });
});