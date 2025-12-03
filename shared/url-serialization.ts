// Copyright (c) 2025, Compiler Explorer Authors
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

import lzstring from 'lz-string';
import * as rison from './rison.js';

/**
 * ConfigMinifier - Reimplementation of GoldenLayout's ConfigMinifier
 *
 * Minifies and unminifies configs by replacing frequent keys and values
 * with one letter substitutes using base36 encoding.
 *
 * This is a Node-compatible reimplementation that doesn't depend on
 * the browser-only GoldenLayout library.
 */
class ConfigMinifier {
    private readonly _keys: string[];
    private readonly _values: (string | boolean)[];

    constructor() {
        // Array position matters - these map to base36 characters (0-9, a-z)
        this._keys = [
            'settings',
            'hasHeaders',
            'constrainDragToContainer',
            'selectionEnabled',
            'dimensions',
            'borderWidth',
            'minItemHeight',
            'minItemWidth',
            'headerHeight',
            'dragProxyWidth',
            'dragProxyHeight',
            'labels',
            'close',
            'maximise',
            'minimise',
            'popout',
            'content',
            'componentName',
            'componentState',
            'id',
            'width',
            'type',
            'height',
            'isClosable',
            'title',
            'popoutWholeStack',
            'openPopouts',
            'parentId',
            'activeItemIndex',
            'reorderEnabled',
            'borderGrabWidth',
        ];

        this._values = [
            true,
            false,
            'row',
            'column',
            'stack',
            'component',
            'close',
            'maximise',
            'minimise',
            'open in new window',
        ];

        if (this._keys.length > 36) {
            throw new Error('Too many keys in config minifier map');
        }
    }

    /**
     * Takes a GoldenLayout configuration object and replaces its keys
     * and values recursively with one letter counterparts
     */
    minifyConfig(config: any): any {
        const min: any = {};
        this._nextLevel(config, min, '_min');
        return min;
    }

    /**
     * Takes a configuration Object that was previously minified
     * using minifyConfig and returns its original version
     */
    unminifyConfig(minifiedConfig: any): any {
        const orig: any = {};
        this._nextLevel(minifiedConfig, orig, '_max');
        return orig;
    }

    /**
     * Recursive function, called for every level of the config structure
     */
    private _nextLevel(from: any, to: any, translationFn: '_min' | '_max'): void {
        for (const key in from) {
            // Skip prototype properties
            if (!from.hasOwnProperty(key)) continue;

            // For arrays, cast keys to numbers (not strings!)
            // This is important because the single-char check in _min/_max
            // should not trigger for numeric indices like "0", "1", etc.
            let processedKey: string | number = key;
            if (Array.isArray(from)) {
                processedKey = parseInt(key, 10);
            }

            // Translate the key to a one letter substitute
            const minKey = this[translationFn](processedKey, this._keys);

            // For Arrays and Objects, create a new Array/Object and recurse
            if (typeof from[key] === 'object' && from[key] !== null) {
                to[minKey] = Array.isArray(from[key]) ? [] : {};
                this._nextLevel(from[key], to[minKey], translationFn);
            } else {
                // For primitive values, minify the value
                to[minKey] = this[translationFn](from[key], this._values);
            }
        }
    }

    /**
     * Minifies value based on a dictionary
     */
    private _min(value: any, dictionary: readonly (string | boolean)[]): any {
        // If a value actually is a single character, prefix it with ___
        // to avoid mistaking it for a minification code
        if (typeof value === 'string' && value.length === 1) {
            return '___' + value;
        }

        const index = dictionary.indexOf(value);

        // Value not found in the dictionary, return it unmodified
        if (index === -1) {
            return value;
        }

        // Value found in dictionary, return its base36 counterpart
        return index.toString(36);
    }

    /**
     * Unminifies value based on a dictionary
     */
    private _max(value: any, dictionary: readonly (string | boolean)[]): any {
        // Value is a single character - assume it's a translation
        if (typeof value === 'string' && value.length === 1) {
            return dictionary[parseInt(value, 36)];
        }

        // Value originally was a single character and was prefixed with ___
        if (typeof value === 'string' && value.substr(0, 3) === '___') {
            return value[3];
        }

        // Value was not minified
        return value;
    }
}

// Create a singleton instance
const configMinifier = new ConfigMinifier();

/**
 * Minify a GoldenLayout config object
 */
export function minifyConfig(config: any): any {
    return configMinifier.minifyConfig(config);
}

/**
 * Unminify a GoldenLayout config object
 */
export function unminifyConfig(config: any): any {
    return configMinifier.unminifyConfig(config);
}

/**
 * Convert object to rison-encoded string
 */
export function risonify(obj: rison.JSONValue): string {
    return rison.quote(rison.encode_object(obj));
}

/**
 * Convert rison-encoded string to object
 */
export function unrisonify(text: string): any {
    return rison.decode_object(decodeURIComponent(text.replace(/\+/g, '%20')));
}

/**
 * Serialise state object to URL hash string
 *
 * Process:
 * 1. Minify the config (replace common keys/values with single chars)
 * 2. Rison encode the minified config
 * 3. If compression saves >20%, compress with lzstring and wrap in {z: ...}
 * 4. Return the final rison-encoded string
 */
export function serialiseState(stateText: any): string {
    const ctx = minifyConfig({content: stateText.content});
    ctx.version = 4;
    const uncompressed = risonify(ctx);
    const compressed = risonify({z: lzstring.compressToBase64(uncompressed)});
    const MinimalSavings = 0.2;
    if (compressed.length < uncompressed.length * (1.0 - MinimalSavings)) {
        return compressed;
    }
    return uncompressed;
}

/**
 * Deserialise URL hash string to state object
 *
 * Process:
 * 1. Rison decode the hash
 * 2. If it contains {z: ...}, decompress the lzstring data
 * 3. Rison decode again if decompressed
 * 4. Unminify the config (expand single chars back to full keys/values)
 * 5. Handle version migrations for old state formats
 */
export function deserialiseState(stateText: string): any {
    let state;
    try {
        state = unrisonify(stateText);
        if (state?.z) {
            const data = lzstring.decompressFromBase64(state.z);
            // lzstring returns empty string on failure rather than throwing
            if (data === '') {
                throw new Error('lzstring decompress error, url is corrupted');
            }
            state = unrisonify(data);
        }
    } catch (ex) {
        // If we can't parse it, return false so caller can handle
        console.warn('Failed to deserialise state:', ex);
        return false;
    }

    // Handle version migrations
    if (!state || state.version === undefined) return false;

    switch (state.version) {
        case 4:
            state = unminifyConfig(state);
            break;
        default:
            // Versions 1-3 require GoldenLayout and Components, which are browser-only
            // These should be handled by the frontend-specific code in static/url.ts
            return state;
    }

    return state;
}
