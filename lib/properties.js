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

function toProperty(prop) {
    if (prop === 'true' || prop === 'yes') return true;
    if (prop === 'false' || prop === 'no') return false;
    if (prop.match(/^-?(0|[1-9][0-9]*)$/)) return parseInt(prop);
    if (prop.match(/^-?[0-9]*\.[0-9]+$/)) return parseFloat(prop);
    return prop;
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
        props[split[1].trim()] = toProperty(split[2].trim());
        debug(`${split[1].trim()} = ${split[2].trim()}`);
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
    logger.debug("props.properties = ", properties);
}

function propsFor(base) {
    return function (property, defaultValue) {
        return get(base, property, defaultValue);
    };
}

class CompilerProps {
    constructor(languages, ceProps) {
        this.languages = languages;
        this.propsByLangId = {};

        this.ceProps = ceProps;

        // Instantiate a function to access records concerning the chosen language in hidden object props.properties
        _.each(this.languages, lang => this.propsByLangId[lang.id] = propsFor(lang.id));
    }

    get(langs, key, defaultValue, fn) {
        fn = fn || _.identity;
        if (_.isEmpty(langs)) {
            return fn(this.ceProps(key, defaultValue));
        }
        if (!_.isString(langs)) {
            return _.chain(langs)
                .map(lang => [lang.id, fn(this.propsByLangId[lang.id](key) || this.ceProps(key, defaultValue), lang)])
                .object()
                .value();
        } else {
            if (this.propsByLangId[langs]) {
                return fn(this.propsByLangId[langs](key) || this.ceProps(key, defaultValue), this.languages[langs]);
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
    CompilerProps: CompilerProps,
    setDebug: debug => {
        propDebug = debug;
    },
    fakeProps: fake => (prop, def) => fake[prop] === undefined ? def : fake[prop]
};
