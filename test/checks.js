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

import _ from 'underscore';

import { languages } from '../lib/languages';
import * as properties from '../lib/properties';

import { fs } from './utils';

describe('Live site checks', () => {
    let ceProps;
    let compilerProps;

    before(() => {
        properties.initialize('etc/config/', ['amazon']);
        ceProps = properties.propsFor('compiler-explorer');
        compilerProps = new properties.CompilerProps(languages, ceProps);
    });
    after(() => {
        properties.reset();
    });

    it('there should not be any orphan libraries', () => {
        const langsLibs = compilerProps.get(compilerProps.languages, 'libs', '');

        const libs = _.pick(langsLibs, libs => !!libs);

        const differences = {};
        _.each(libs, (libs, lang) => {
            const langLibs = libs.split(':');

            const filePath = `etc/config/${lang}.amazon.properties`;
            const fileContents = fs.readFileSync(filePath, 'utf-8');

            const matches = fileContents.match(/^libs\..*?\.name/gm);
            let found = [];
            if (matches) {
                found = _.map(matches, line => line.match(/libs\.(.*?)\.name/)[1]);
            }
            const difference = _.difference(found, langLibs);
            if (difference.length > 0) {
                differences[lang] = difference;
            }
        });
        differences.should.be.eql({}, 'One or more defined libraries are not listed on their corresponding language libs property array');
    });
});
