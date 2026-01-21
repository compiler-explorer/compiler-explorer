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

import {afterAll, beforeAll, describe, expect, it} from 'vitest';

import {PropertyGetter} from '../lib/properties.interfaces.js';
import * as properties from '../lib/properties.js';

const languages = {
    a: {id: 'a'},
};

describe('Properties', () => {
    let casesProps: PropertyGetter;
    let overridingProps: PropertyGetter;
    let compilerProps;

    beforeAll(() => {
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

    afterAll(() => {
        properties.reset();
    });

    it('Has working propsFor', () => {
        expect(properties.get('cases', 'exampleProperty')).toEqual(casesProps('exampleProperty'));
    });
    it('Does not find non existent properties when no default is set', () => {
        expect(casesProps('nonexistentProp')).toEqual(undefined);
    });
    it('Falls back to default if value not found and default is set', () => {
        // Randomly generated number...
        expect(casesProps('nonexistentProp', 4)).toEqual(4);
        expect(casesProps('nonexistentProp', 4)).toEqual(4);
    });
    it('Handles empty properties as empty strings', () => {
        expect(casesProps('emptyProperty')).toEqual('');
    });
    it('Handles bad numbers properties as strings', () => {
        expect(casesProps('001string')).toEqual('001');
    });
    it('Handles bad numbers properties as strings', () => {
        expect(casesProps('0985string')).toEqual('0985');
    });
    it('Ignores commented out properties', () => {
        expect(casesProps('commentedProperty')).toBeUndefined();
    });
    it('Ignores bad lines', () => {
        expect(casesProps('badLineIfYouSeeThisWithAnErrorItsOk')).toBeUndefined();
    });
    it('Understands positive integers', () => {
        expect(casesProps('numericPropertyPositive')).toEqual(42);
    });
    it('Understands zero as integer', () => {
        expect(casesProps('numericPropertyZero')).toEqual(0);
    });
    it('Understands negative integers', () => {
        expect(casesProps('numericPropertyNegative')).toEqual(-11);
    });
    it('Understands positive floats', () => {
        expect(casesProps('floatPropertyPositive')).toEqual(3.14);
    });
    it('Understands negative floats', () => {
        expect(casesProps('floatPropertyNegative')).toEqual(-9000);
    });
    it('Does not understand comma decimal as float', () => {
        expect(casesProps('commaAsDecimalProperty')).toEqual('3,14');
    });
    it('Does not understand DASH-SPACE-NUMBER as a negative number', () => {
        expect(casesProps('stringPropertyNumberLike')).toEqual('- 97');
    });
    it('Understands yes as true boolean', () => {
        expect(casesProps('truePropertyYes')).toBe(true);
    });
    it('Understands true as true boolean', () => {
        expect(casesProps('truePropertyTrue')).toBe(true);
    });
    it('Does not understand Yes as boolean', () => {
        expect(casesProps('stringPropertyYes')).toEqual('Yes');
    });
    it('Does not understand True as boolean', () => {
        expect(casesProps('stringPropertyTrue')).toEqual('True');
    });
    it('Understands no as false boolean', () => {
        expect(casesProps('falsePropertyNo')).toBe(false);
    });
    it('Understands false as false boolean', () => {
        expect(casesProps('falsePropertyFalse')).toBe(false);
    });
    it('Does not understand No as boolean', () => {
        expect(casesProps('stringPropertyNo')).toEqual('No');
    });
    it('Does not understand False as boolean', () => {
        expect(casesProps('stringPropertyFalse')).toEqual('False');
    });
    it('Should find non overridden properties', () => {
        expect(overridingProps('nonOverriddenProperty')).toEqual('.... . .-.. .-.. ---');
    });
    it('Should handle overridden properties', () => {
        expect(overridingProps('overrodeProperty')).toEqual('ACTUALLY USED');
    });
    it('Should fall back from overridden', () => {
        expect(overridingProps('localProperty')).toEqual(11235813);
    });
    it('should have an identity function if none provided', () => {
        expect(compilerProps.get('a', 'foo', '0')).toEqual('1');
        expect(compilerProps.get(languages, 'foo', '0')).toEqual({a: '1'});
    });
    it('should return an object of languages if the languages arg is an object itself', () => {
        expect(compilerProps.get(languages, 'foo', '0')).toEqual({a: '1'});
    });
    it('should return a direct result if the language is an ID', () => {
        compilerProps.propsByLangId[languages.a.id] = properties.fakeProps({foo: 'b'});
        expect(compilerProps.get('a', 'foo', '0')).toEqual('b');
        compilerProps.propsByLangId[languages.a.id] = undefined;
    });
    it('should have backwards compatibility compilerProps behaviour', () => {
        expect(compilerProps.get('', 'foo', '0')).toEqual('1');
    });
    it('should report the default value if an unknown language is used', () => {
        expect(compilerProps.get('b', 'foo', '0')).toEqual('0');
    });
    it('should not check ceProps for falsey values', () => {
        // Set bar to be falsey in the language specific setting.
        compilerProps.propsByLangId[languages.a.id] = properties.fakeProps({bar: false});
        // Now query it with a default of true. We should see false...
        expect(compilerProps.get('a', 'bar', true)).toBe(false);
        expect(compilerProps.get(languages, 'bar', true)).toEqual({a: false});
        compilerProps.propsByLangId[languages.a.id] = undefined;
    });
    it('should not parse version properties as numbers', () => {
        expect(casesProps('libs.example.versions.010.version')).toEqual('0.10');
    });
    it('should not parse semver properties as numbers', () => {
        expect(casesProps('compiler.example110.semver')).toEqual('1.10');
    });
});

describe('Properties blob parsing', () => {
    it('Normal properties', () => {
        // biome-ignore format: keep as-is for readability
        const props = properties.parseProperties(
            'hello = test \n' +
            'etc=123\n' +
            'mybool=false\n',
            '<test props>',
        );
        expect(props.hello).toEqual('test');
        expect(props.etc).toEqual(123);
        expect(props.mybool).toBe(false);
    });

    it('should trim both leading and trailing whitespace for = assignment', () => {
        const props = properties.parseProperties('spaced=   value   \n', '<test props>');
        expect(props.spaced).toEqual('value');
    });
});

describe('Properties append syntax', () => {
    it('should append to existing string properties with +=', () => {
        const props = properties.parseProperties('list=a:b\n' + 'list+=:c:d\n', '<test props>');
        expect(props.list).toEqual('a:b:c:d');
    });

    it('should skip += on undefined property and log error', () => {
        const props = properties.parseProperties('newprop+=value\n', '<test props>');
        expect(props.newprop).toBeUndefined();
    });

    it('should handle multiple += operations', () => {
        const props = properties.parseProperties(
            'items=first\n' + 'items+=second\n' + 'items+=third\n',
            '<test props>',
        );
        expect(props.items).toEqual('firstsecondthird');
    });

    it('should skip += on boolean properties and log error', () => {
        const props = properties.parseProperties('flag=true\n' + 'flag+=more\n', '<test props>');
        // Append skipped, original value preserved
        expect(props.flag).toBe(true);
    });

    it('should preserve leading whitespace for += but trim trailing', () => {
        const props = properties.parseProperties('opts=-Wall\n' + 'opts+= -Wextra   \n', '<test props>');
        // Leading space preserved, trailing trimmed
        expect(props.opts).toEqual('-Wall -Wextra');
    });
});

describe('Cross-file += interaction', () => {
    let crossFileProps: PropertyGetter;

    beforeAll(() => {
        properties.reset();
        properties.initialize('test/example-config/', ['crossfile-base', 'crossfile-tip']);
        crossFileProps = properties.propsFor('crossfile');
    });

    afterAll(() => {
        properties.reset();
    });

    it('should not allow += in tip file to append to property defined only in base file', () => {
        // += in crossfile.crossfile-tip.properties tries to append to appendTarget
        // But appendTarget is not defined in that file, only in crossfile-base
        // So the += fails (logs error), and the base file's original value is returned
        expect(crossFileProps('appendTarget')).toEqual('base-value');
    });

    it('should allow = to override base value then += to append in same file', () => {
        // In crossfile.crossfile-tip.properties:
        //   overrideTarget=tip-value      (overrides base-value from base file)
        //   overrideTarget+= appended     (appends to the value defined above)
        // The tip file first redefines the property with =, then appends with +=
        // This is the correct way to "extend" a base property
        expect(crossFileProps('overrideTarget')).toEqual('tip-value appended');
    });
});
