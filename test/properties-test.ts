// Copyright (c) 2017, Compiler Explorer Authors
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

import {unwrap} from '../lib/assert';
import * as properties from '../lib/properties';

import {should} from './utils';

const languages = {
    a: {id: 'a'},
};

describe('Properties', () => {
    let casesProps, overridingProps, compilerProps;

    before(() => {
        properties.initialize('test/example-config/', ['test', 'overridden-base', 'overridden-tip']);
        casesProps = properties.propsFor('cases');
        overridingProps = properties.propsFor('overwrite');
        compilerProps = new properties.CompilerProps(
            languages,
            properties.fakeProps({
                foo: '1',
            }),
        );
    });

    after(() => {
        properties.reset();
    });

    it('Has working propsFor', () => {
        should.equal(properties.get('cases', 'exampleProperty'), casesProps('exampleProperty'));
    });
    it('Does not find non existent properties when no default is set', () => {
        should.equal(casesProps('nonexistentProp'), undefined);
    });
    it('Falls back to default if value not found and default is set', () => {
        // Randomly generated number...
        casesProps('nonexistentProp', 4).should.be.equal(4);
        should.equal(casesProps('nonexistentProp', 4), 4);
    });
    it('Handles empty properties as empty strings', () => {
        should.equal(casesProps('emptyProperty'), '');
    });
    it('Handles bad numbers properties as strings', () => {
        should.equal(casesProps('001string'), '001');
    });
    it('Handles bad numbers properties as strings', () => {
        should.equal(casesProps('0985string'), '0985');
    });
    it('Ignores commented out properties', () => {
        should.equal(casesProps('commentedProperty'), undefined);
    });
    it('Ignores bad lines', () => {
        should.equal(casesProps('badLineIfYouSeeThisWithAnErrorItsOk'), undefined);
    });
    it('Understands positive integers', () => {
        should.equal(casesProps('numericPropertyPositive'), 42);
    });
    it('Understands zero as integer', () => {
        should.equal(casesProps('numericPropertyZero'), 0);
    });
    it('Understands negative integers', () => {
        should.equal(casesProps('numericPropertyNegative'), -11);
    });
    it('Understands positive floats', () => {
        should.equal(casesProps('floatPropertyPositive'), 3.14);
    });
    it('Understands negative floats', () => {
        should.equal(casesProps('floatPropertyNegative'), -9000);
    });
    it('Does not understand comma decimal as float', () => {
        should.equal(casesProps('commaAsDecimalProperty'), '3,14');
    });
    it('Does not understand DASH-SPACE-NUMBER as a negative number', () => {
        should.equal(casesProps('stringPropertyNumberLike'), '- 97');
    });
    it('Understands yes as true boolean', () => {
        should.equal(casesProps('truePropertyYes'), true);
    });
    it('Understands true as true boolean', () => {
        should.equal(casesProps('truePropertyTrue'), true);
    });
    it('Does not understand Yes as boolean', () => {
        should.equal(casesProps('stringPropertyYes'), 'Yes');
    });
    it('Does not understand True as boolean', () => {
        should.equal(casesProps('stringPropertyTrue'), 'True');
    });
    it('Understands no as false boolean', () => {
        should.equal(casesProps('falsePropertyNo'), false);
    });
    it('Understands false as false boolean', () => {
        should.equal(casesProps('falsePropertyFalse'), false);
    });
    it('Does not understand No as boolean', () => {
        should.equal(casesProps('stringPropertyNo'), 'No');
    });
    it('Does not understand False as boolean', () => {
        should.equal(casesProps('stringPropertyFalse'), 'False');
    });
    it('Should find non overridden properties', () => {
        should.equal(overridingProps('nonOverriddenProperty'), '.... . .-.. .-.. ---');
    });
    it('Should handle overridden properties', () => {
        should.equal(overridingProps('overrodeProperty'), 'ACTUALLY USED');
    });
    it('Should fall back from overridden', () => {
        should.equal(overridingProps('localProperty'), 11235813);
    });
    it('should have an identity function if none provided', () => {
        should.equal(compilerProps.get('a', 'foo', '0'), '1');
        compilerProps.get(languages, 'foo', '0').should.deep.equal({a: '1'});
    });
    it('should return an object of languages if the languages arg is an object itself', () => {
        compilerProps.get(languages, 'foo', '0').should.deep.equal({a: '1'});
    });
    it('should return a direct result if the language is an ID', () => {
        compilerProps.propsByLangId[languages.a.id] = properties.fakeProps({foo: 'b'});
        should.equal(compilerProps.get('a', 'foo', '0'), 'b');
        compilerProps.propsByLangId[languages.a.id] = undefined;
    });
    it('should have backwards compatibility compilerProps behaviour', () => {
        should.equal(compilerProps.get('', 'foo', '0'), '1');
    });
    it('should report the default value if an unknown language is used', () => {
        should.equal(compilerProps.get('b', 'foo', '0'), '0');
    });
    it('should not check ceProps for falsey values', () => {
        // Set bar to be falsey in the language specific setting.
        compilerProps.propsByLangId[languages.a.id] = properties.fakeProps({bar: false});
        // Now query it with a default of true. We should see false...
        should.equal(compilerProps.get('a', 'bar', true), false);
        compilerProps.get(languages, 'bar', true).should.deep.equal({a: false});
        compilerProps.propsByLangId[languages.a.id] = undefined;
    });
    it('should not parse version properies as numbers', () => {
        should.equal(casesProps('libs.example.versions.010.version'), '0.10');
    });
    it('should not parse semver properies as numbers', () => {
        should.equal(casesProps('compiler.example110.semver'), '1.10');
    });
});

describe('Properties blob parsing', () => {
    it('Normal properties', () => {
        const props = properties.parseProperties(
            // prettier-ignore
            'hello = test \n' +
            'etc=123\n' +
            'mybool=false\n',
            '<test props>',
        );
        unwrap(props.hello).should.equal('test');
        unwrap(props.etc).should.equal(123);
        unwrap(props.mybool).should.equal(false);
    });
});
