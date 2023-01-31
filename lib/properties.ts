// Copyright (c) 2012, Compiler Explorer Authors
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
import path from 'path';

import _ from 'underscore';

import {LanguageKey} from '../types/languages.interfaces';

import {logger} from './logger';
import {PropertyGetter, PropertyValue, Widen} from './properties.interfaces';
import {toProperty} from './utils';

let properties: Record<string, Record<string, PropertyValue>> = {};

let hierarchy: string[] = [];

let propDebug = false;

function findProps(base: string, elem: string): Record<string, PropertyValue> {
    return properties[`${base}.${elem}`];
}

function debug(string) {
    if (propDebug) logger.info(`prop: ${string}`);
}

export function get(base: string, property: string, defaultValue: undefined): PropertyValue;
export function get<T extends PropertyValue>(
    base: string,
    property: string,
    defaultValue: Widen<T>,
): typeof defaultValue;
export function get<T extends PropertyValue>(base: string, property: string, defaultValue?: unknown): T;
export function get(base: string, property: string, defaultValue?: unknown): unknown {
    let result = defaultValue;
    let source = 'default';
    for (const elem of hierarchy) {
        const propertyMap = findProps(base, elem);
        if (propertyMap && property in propertyMap) {
            debug(`${base}.${property}: overriding ${source} value (${result}) with ${propertyMap[property]}`);
            result = propertyMap[property];
            source = elem;
        }
    }
    debug(`${base}.${property}: returning ${result} (from ${source})`);
    return result;
}

export type RawPropertiesGetter = typeof get;

export function parseProperties(blob, name) {
    const props = {};
    for (const [index, lineOrig] of blob.split('\n').entries()) {
        const line = lineOrig.replace(/#.*/, '').trim();
        if (!line) continue;
        const split = line.match(/([^=]+)=(.*)/);
        if (!split) {
            logger.error(`Bad line: ${line} in ${name}: ${index + 1}`);
            continue;
        }
        const prop = split[1].trim();
        let val = split[2].trim();
        // hack to avoid applying toProperty to version properties
        // so that they're not parsed as numbers
        if (!prop.endsWith('.version') && !prop.endsWith('.semver')) {
            val = toProperty(val);
        }
        props[prop] = val;
        debug(`${prop} = ${val}`);
    }
    return props;
}

export function initialize(directory, hier) {
    if (hier === null) throw new Error('Must supply a hierarchy array');
    hierarchy = _.map(hier, x => x.toLowerCase());
    logger.info(`Reading properties from ${directory} with hierarchy ${hierarchy}`);
    const endsWith = /\.properties$/;
    const propertyFiles = fs.readdirSync(directory).filter(filename => filename.match(endsWith));
    properties = {};
    for (let file of propertyFiles) {
        const baseName = file.replace(endsWith, '');
        file = path.join(directory, file);
        debug('Reading config from ' + file);
        properties[baseName] = parseProperties(fs.readFileSync(file, 'utf8'), file);
    }
    logger.debug('props.properties = ', properties);
}

export function reset() {
    hierarchy = [];
    properties = {};
    logger.debug('Properties reset');
}

export function propsFor(base): PropertyGetter {
    return function (property, defaultValue) {
        return get(base, property, defaultValue);
    };
}

// function mappedOf(fn, funcA, funcB) {
//     const resultA = funcA();
//     if (resultA !== undefined) return resultA;
//     return funcB();
// }

type LanguageDef = {
    id: string;
};

/***
 * Compiler property fetcher
 */
export class CompilerProps {
    public readonly languages: Record<string, any>;
    public readonly propsByLangId: Record<string, PropertyGetter>;
    public readonly ceProps: PropertyGetter;

    /***
     * Creates a CompilerProps lookup function
     */
    constructor(languages: Record<string, LanguageDef>, ceProps: PropertyGetter) {
        this.languages = languages;
        this.propsByLangId = {};

        this.ceProps = ceProps;

        // Instantiate a function to access records concerning the chosen language in hidden object props.properties
        _.each(this.languages, lang => (this.propsByLangId[lang.id] = propsFor(lang.id)));
    }

    $getInternal(base: string, property: string, defaultValue: undefined): PropertyValue;
    $getInternal<T extends PropertyValue>(base: string, property: string, defaultValue: Widen<T>): typeof defaultValue;
    $getInternal<T extends PropertyValue>(base: string, property: string, defaultValue?: PropertyValue): T;
    $getInternal(langId: string, key: string, defaultValue: PropertyValue): PropertyValue {
        const languagePropertyValue = this.propsByLangId[langId](key);
        if (languagePropertyValue !== undefined) {
            return languagePropertyValue;
        }
        return this.ceProps(key, defaultValue);
    }

    /***
     * Gets a value for a given key associated to the given languages from the properties
     *
     * @param langs - Which langs to search in
     *  For compatibility, {null} means looking into the Compiler Explorer properties (Not on any language)
     *  If langs is a {string}, it refers to the language id we want to search into
     *  If langs is a {CELanguages}, it refers to which languages we want to search into
     *  TODO: Add a {Language} version?
     * @param {string} key - Key to look for
     * @param {*} defaultValue - What to return if the key is not found
     * @param {?function} fn - Transformation to give to each value found
     * @returns {*} Transformed value(s) found or fn(defaultValue)
     */

    // A lot of overloads for a lot of different variants:
    // const a = this.compilerProps(lang, property); // PropertyValue
    // const b = this.compilerProps<number>(lang, property); // number
    // const c = this.compilerProps(lang, property, 42); // number
    // const d = this.compilerProps(lang, property, undefined, (x) => ["foobar"]); // string[]
    // const e = this.compilerProps(lang, property, 42, (x) => ["foobar"]); // number | string[]
    // if more than one language:
    // const f = this.compilerProps(languages, property); // Record<LanguageKey, PropertyValue>
    // const g = this.compilerProps<number>(languages, property); // Record<LanguageKey, number>
    // const h = this.compilerProps(languages, property, 42); // Record<LanguageKey, number>
    // const i = this.compilerProps(languages, property, undefined, (x) => ["foobar"]); // Record<LanguageKey, string[]>
    // const j = this.compilerProps(languages, property, 42, (x) => ["foobar"]);//Record<LanguageKey, number | string[]>

    // general overloads
    get(base: string, property: string, defaultValue?: undefined, fn?: undefined): PropertyValue;
    get<T extends PropertyValue>(
        base: string,
        property: string,
        defaultValue: Widen<T>,
        fn?: undefined,
    ): typeof defaultValue;
    get<T extends PropertyValue>(base: string, property: string, defaultValue?: PropertyValue, fn?: undefined): T;
    // fn overloads
    get<R>(
        base: string,
        property: string,
        defaultValue?: undefined,
        fn?: (item: PropertyValue, language?: any) => R,
    ): R;
    get<T extends PropertyValue, R>(
        base: string,
        property: string,
        defaultValue: Widen<T>,
        fn?: (item: typeof defaultValue, language?: any) => R,
    ): R;
    // container base general overloads
    get(
        base: LanguageDef[] | Record<string, any>,
        property: string,
        defaultValue?: undefined,
        fn?: undefined,
    ): Record<LanguageKey, PropertyValue>;
    get<T extends PropertyValue>(
        base: LanguageDef[] | Record<string, any>,
        property: string,
        defaultValue: Widen<T>,
        fn?: undefined,
    ): Record<LanguageKey, typeof defaultValue>;
    get<T extends PropertyValue>(
        base: LanguageDef[] | Record<string, any>,
        property: string,
        defaultValue?: PropertyValue,
        fn?: undefined,
    ): Record<LanguageKey, T>;
    // container base fn overloads
    get<R>(
        base: LanguageDef[] | Record<string, any>,
        property: string,
        defaultValue?: undefined,
        fn?: (item: PropertyValue, language?: any) => R,
    ): Record<LanguageKey, R>;
    get<T extends PropertyValue, R>(
        base: LanguageDef[] | Record<string, any>,
        property: string,
        defaultValue: Widen<T>,
        fn?: (item: typeof defaultValue, language?: any) => R,
    ): Record<LanguageKey, R>;

    get(
        langs: string | LanguageDef[] | Record<string, any>,
        key: string,
        defaultValue?: PropertyValue,
        fn?: (item: PropertyValue, language?: any) => unknown,
    ) {
        const map_fn = fn || _.identity;
        if (_.isEmpty(langs)) {
            return map_fn(this.ceProps(key, defaultValue));
        }
        if (_.isString(langs)) {
            if (this.propsByLangId[langs]) {
                return map_fn(this.$getInternal(langs, key, defaultValue), this.languages[langs]);
            } else {
                logger.error(`Tried to pass ${langs} as a language ID`);
                return map_fn(defaultValue);
            }
        } else {
            return _.chain(langs)
                .map(lang => [lang.id, map_fn(this.$getInternal(lang.id, key, defaultValue), lang)])
                .object()
                .value();
        }
    }
}

export function setDebug(debug: boolean) {
    propDebug = debug;
}

export function fakeProps(fake: Record<string, PropertyValue>): PropertyGetter {
    return (prop, def) => (fake[prop] === undefined ? def : fake[prop]);
}
