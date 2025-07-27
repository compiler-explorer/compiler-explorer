// Copyright (c) 2021, Compiler Explorer Authors
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

import GoldenLayout from 'golden-layout';
import lzstring from 'lz-string';
import _ from 'underscore';
import {CURRENT_LAYOUT_VERSION, GoldenLayoutConfig} from './components.interfaces.js';
import * as Components from './components.js';

import * as rison from './rison.js';

// Legacy state format for version migration
interface LegacyState {
    compilers: Array<{
        sourcez?: string;
        source?: string;
        options: unknown;
        compiler: string;
    }>;
    filterAsm: {
        colouriseAsm?: boolean;
        [key: string]: unknown;
    };
}

export function convertOldState(state: LegacyState): GoldenLayoutConfig {
    const sc = state.compilers[0];
    if (!sc) throw new Error('Unable to determine compiler from old state');
    const content: unknown[] = [];
    let source;
    if (sc.sourcez) {
        source = lzstring.decompressFromBase64(sc.sourcez);
    } else {
        source = sc.source;
    }
    const options = {
        compileOnChange: true,
        colouriseAsm: state.filterAsm.colouriseAsm,
    };
    const filters = _.clone(state.filterAsm);
    if ('colouriseAsm' in filters) {
        delete filters.colouriseAsm;
    }
    // TODO(junlarsen): find the missing language field here
    // @ts-expect-error: this is missing the language field, which was never noticed because the import was untyped
    content.push(Components.getEditorWith(1, source, options));
    // @ts-expect-error: legacy state conversion - filters may not match exact type
    content.push(Components.getCompilerWith(1, filters, sc.options, sc.compiler));
    return {version: CURRENT_LAYOUT_VERSION, content: [{type: 'row', content: content}]} as GoldenLayoutConfig;
}

/**
 * Validation function that checks item structure and basic type safety.
 * Returns an error message string if invalid, or null if valid.
 */
function validateItemConfig(item: any, depth = 0): string | null {
    // Prevent infinite recursion with very deep layouts
    if (depth > 50) {
        return 'layout nesting too deep (max 50 levels)';
    }
    if (!item || typeof item !== 'object') {
        return 'must be an object';
    }
    if (!item.type) {
        return "missing 'type' property";
    }

    if (item.type === 'component') {
        if (!item.componentName) {
            return "missing 'componentName' property";
        }
        if (typeof item.componentName !== 'string') {
            return "'componentName' must be a string";
        }
        if (!item.componentState) {
            return "missing 'componentState' property";
        }
        if (typeof item.componentState !== 'object') {
            return "'componentState' must be an object";
        }
        // TODO(#7808): Add component-specific state validation
        // - Validate component names against known components
        // - Validate component state structure matches expected types
        // - Validate required properties are present
        // - Validate property types (e.g. numbers vs strings)
        return null;
    }
    if (item.type === 'row' || item.type === 'column' || item.type === 'stack') {
        if (!Array.isArray(item.content)) {
            return "layout items must have a 'content' array";
        }
        // Validate nested items
        for (let i = 0; i < item.content.length; i++) {
            const nestedError = validateItemConfig(item.content[i], depth + 1);
            if (nestedError) {
                return `nested item ${i}: ${nestedError}`;
            }
        }
        return null;
    }

    return `unknown type '${item.type}'`;
}

/**
 * Validates and loads a layout state configuration.
 * Handles version migration and structural validation for layout configurations.
 * @param state - The state object to validate (can be any format including legacy)
 * @param shouldUnminify - Whether to unminify the config (true for URL sources, false for localStorage)
 * @returns Validated GoldenLayoutConfig ready for use by GoldenLayout
 * @throws Error if validation fails with detailed error message
 */
export function loadState(state: any, shouldUnminify: boolean): GoldenLayoutConfig {
    if (!state || typeof state !== 'object') {
        throw new Error('Invalid state: must be an object');
    }
    if (state.version === undefined) {
        throw new Error('Invalid state: missing version information');
    }

    // Handle version migration
    switch (state.version) {
        case 1:
            state.filterAsm = {};
            state.version = 2;
        /* falls through */
        case 2:
            state.compilers = [state];
            state.version = 3;
        /* falls through */
        case 3:
            state = convertOldState(state);
            break; // no fall through
        case CURRENT_LAYOUT_VERSION:
            if (shouldUnminify) {
                state = GoldenLayout.unminifyConfig(state);
            }
            break;
        default:
            throw new Error("Invalid version '" + state.version + "'");
    }

    // Validate structure after version migration
    if (!Array.isArray(state.content)) {
        throw new Error('Configuration content must be an array');
    }

    // Validate each item in content using the detailed validator
    for (let i = 0; i < state.content.length; i++) {
        const item = state.content[i];
        const error = validateItemConfig(item);
        if (error) {
            const itemType = item?.type || 'unknown';
            const componentName = item?.componentName || '';
            const context = componentName
                ? ` (type: '${itemType}', componentName: '${componentName}')`
                : ` (type: '${itemType}')`;
            throw new Error(`Invalid item ${i}${context}: ${error}`);
        }
    }

    // Return validated config (no unsafe cast needed)
    return state;
}

export function risonify(obj: rison.JSONValue): string {
    return rison.quote(rison.encode_object(obj));
}

export function unrisonify(text: string): unknown {
    return rison.decode_object(decodeURIComponent(text.replace(/\+/g, '%20')));
}

export function deserialiseState(stateText: string): GoldenLayoutConfig {
    let state;
    let exception;
    try {
        state = unrisonify(stateText);
        if (state?.z) {
            const data = lzstring.decompressFromBase64(state.z);
            // If lzstring fails to decompress this it'll return an empty string rather than throwing an error
            if (data === '') {
                throw new Error('lzstring decompress error, url is corrupted');
            }
            state = unrisonify(data);
        }
    } catch (ex) {
        exception = ex;
    }

    // This handles prehistoric urls, assumes rison fails with an error
    if (!state) {
        try {
            state = JSON.parse(decodeURIComponent(stateText));
            exception = null;
        } catch (ex) {
            if (!exception) exception = ex;
        }
    }
    if (exception) throw exception;
    return loadState(state, true);
}

export function serialiseState(config: GoldenLayoutConfig): string {
    const ctx = GoldenLayout.minifyConfig({content: config.content});
    // Always assign current version when serializing - we only serialize current states
    ctx.version = CURRENT_LAYOUT_VERSION;
    const uncompressed = risonify(ctx);
    const compressed = risonify({z: lzstring.compressToBase64(uncompressed)});
    const MinimalSavings = 0.2; // at least this ratio smaller
    if (compressed.length < uncompressed.length * (1.0 - MinimalSavings)) {
        return compressed;
    }
    return uncompressed;
}
