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

export function convertOldState(state: any): GoldenLayoutConfig {
    const sc = state.compilers[0];
    if (!sc) throw new Error('Unable to determine compiler from old state');
    const content: any[] = [];
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
    delete filters.colouriseAsm;
    // TODO(junlarsen): find the missing language field here
    // @ts-expect-error: this is missing the language field, which was never noticed because the import was untyped
    content.push(Components.getEditorWith(1, source, options));
    content.push(Components.getCompilerWith(1, filters, sc.options, sc.compiler));
    return {version: CURRENT_LAYOUT_VERSION, content: [{type: 'row', content: content}]} as GoldenLayoutConfig;
}

export function loadState(state: any): GoldenLayoutConfig {
    if (!state || state.version === undefined) {
        throw new Error('Invalid state: missing version information');
    }
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
            state = GoldenLayout.unminifyConfig(state);
            break;
        default:
            throw new Error("Invalid version '" + state.version + "'");
    }
    return state as GoldenLayoutConfig;
}

export function risonify(obj: rison.JSONValue): string {
    return rison.quote(rison.encode_object(obj));
}

export function unrisonify(text: string): any {
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
    return loadState(state);
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
