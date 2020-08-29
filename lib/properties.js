// Copyright (c) 2012, Matt Godbolt
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

const fs = require('fs'),
    logger = require('./logger').logger,
    _ = require('underscore'),
    toProperty = require('./utils').toProperty,
    path = require('path');

let properties = {};

let hierarchy = [];

let propDebug = false;

function findProps(base, elem) {
    const name = base + '.' + elem;
    return properties[name];
}

function debug(string) {
    if (propDebug) logger.info(`prop: ${string}`);
}

function get(base, property, defaultValue) {
    let result = defaultValue;
    let source = 'default';
    hierarchy.forEach(elem => {
        const propertyMap = findProps(base, elem);
        if (propertyMap && property in propertyMap) {
            debug(`${base}.${property}: overriding ${source} value (${result}) with ${propertyMap[property]}`);
            result = propertyMap[property];
            source = elem;
        }
    });
    debug(`${base}.${property}: returning ${result} (from ${source})`);
    return result;
}

function parseProperties(blob, name) {
    const props = {};
    blob.split('\n').forEach((line, index) => {
        line = line.replace(/#.*/, '').trim();
        if (!line) return;
        let split = line.match(/([^=]+)=(.*)/);
        if (!split) {
            logger.error(`Bad line: ${line} in ${name}: ${index + 1}`);
            return;
        }
        let prop = split[1].trim();
        let val = split[2].trim();
        // hack to avoid applying toProperty to version properties
        // so that they're not parsed as numbers
        if (!prop.endsWith('.version') && !prop.endsWith('.semver')) {
            val = toProperty(val);
        }
        props[prop] = val;
        debug(`${prop} = ${val}`);
    });
    return props;
}

function initialize(directory, hier) {
    if (hier === null) throw new Error('Must supply a hierarchy array');
    hierarchy = _.map(hier, x => x.toLowerCase());
    logger.info(`Reading properties from ${directory} with hierarchy ${hierarchy}`);
    const endsWith = /\.properties$/;
    const propertyFiles = fs.readdirSync(directory).filter(filename => filename.match(endsWith));
    properties = {};
    propertyFiles.forEach(file => {
        const baseName = file.replace(endsWith, '');
        file = path.join(directory, file);
        debug('Reading config from ' + file);
        properties[baseName] = parseProperties(fs.readFileSync(file, 'utf-8'), file);
    });
    logger.debug('props.properties = ', properties);
}

function reset() {
    hierarchy = [];
    properties = {};
    logger.debug('Properties reset');
}

function propsFor(base) {
    return function (property, defaultValue) {
        return get(base, property, defaultValue);
    };
}

// function mappedOf(fn, funcA, funcB) {
//     const resultA = funcA();
//     if (resultA !== undefined) return resultA;
//     return funcB();
// }

/***
 * Compiler property fetcher
 */
class CompilerProps {
    /***
     * Creates a CompilerProps lookup function
     *
     * @param {CELanguages} languages - Supported languages
     * @param {function} ceProps - propsFor function to get Compiler Explorer values from
     */
    constructor(languages, ceProps) {
        this.languages = languages;
        this.propsByLangId = {};

        this.ceProps = ceProps;

        // Instantiate a function to access records concerning the chosen language in hidden object props.properties
        _.each(this.languages, lang => this.propsByLangId[lang.id] = propsFor(lang.id));
    }

    $getInternal(langId, key, defaultValue) {
        const languagePropertyValue = this.propsByLangId[langId](key);
        if (languagePropertyValue !== undefined) {
            return languagePropertyValue;
        }
        return this.ceProps(key, defaultValue);
    }

    /***
     * Gets a value for a given key associated to the given languages from the properties
     *
     * @param {?(string|CELanguages)} langs - Which langs to search in
     *  For compatibility, {null} means looking into the Compiler Explorer properties (Not on any language)
     *  If langs is a {string}, it refers to the language id we want to search into
     *  If langs is a {CELanguages}, it refers to which languages we want to search into
     *  TODO: Add a {Language} version?
     * @param {string} key - Key to look for
     * @param {*} defaultValue - What to return if the key is not found
     * @param {?function} fn - Transformation to give to each value found
     * @returns {*} Transformed value(s) found or fn(defaultValue)
     */
    get(langs, key, defaultValue, fn = _.identity) {
        fn = fn || _.identity;
        if (_.isEmpty(langs)) {
            return fn(this.ceProps(key, defaultValue));
        }
        if (!_.isString(langs)) {
            return _.chain(langs)
                .map(lang => [lang.id, fn(this.$getInternal(lang.id, key, defaultValue), lang)])
                .object()
                .value();
        } else {
            if (this.propsByLangId[langs]) {
                return fn(this.$getInternal(langs, key, defaultValue), this.languages[langs]);
            } else {
                logger.error(`Tried to pass ${langs} as a language ID`);
                return fn(defaultValue);
            }
        }
    }
}

module.exports = {
    get: get,
    propsFor: propsFor,
    initialize: initialize,
    reset: reset,
    CompilerProps: CompilerProps,
    setDebug: debug => {
        propDebug = debug;
    },
    fakeProps: fake => (prop, def) => fake[prop] === undefined ? def : fake[prop],
    parseProperties: parseProperties,
};
