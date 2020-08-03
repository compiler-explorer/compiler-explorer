// Copyright (c) 2016, Matt Godbolt
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

'use strict';
var GoldenLayout = require('golden-layout');
var rison = require('rison');
var $ = require('jquery');
var Components = require('components');
var lzstring = require('lz-string');
var _ = require('underscore');

function convertOldState(state) {
    var sc = state.compilers[0];
    if (!sc) throw new Error('Unable to determine compiler from old state');
    var content = [];
    var source;
    if (sc.sourcez) {
        source = lzstring.decompressFromBase64(sc.sourcez);
    } else {
        source = sc.source;
    }
    var options = {compileOnChange: true, colouriseAsm: state.filterAsm.colouriseAsm};
    var filters = _.clone(state.filterAsm);
    delete filters.colouriseAsm;
    content.push(Components.getEditorWith(1, source, options));
    content.push(Components.getCompilerWith(1, filters, sc.options, sc.compiler));
    return {version: 4, content: [{type: 'row', content: content}]};
}

function loadState(state) {
    if (!state || state.version === undefined) return false;
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
            break;  // no fall through
        case 4:
            state = GoldenLayout.unminifyConfig(state);
            break;
        default:
            throw new Error("Invalid version '" + state.version + "'");
    }
    return state;
}

function risonify(obj) {
    return rison.quote(rison.encode_object(obj));
}

function unrisonify(text) {
    return rison.decode_object(decodeURIComponent(text.replace(/\+/g, '%20')));
}

function deserialiseState(stateText) {
    var state;
    var exception;
    try {
        state = unrisonify(stateText);
        if (state && state.z) {
            state = unrisonify(lzstring.decompressFromBase64(state.z));
        }
    } catch (ex) {
        exception = ex;
    }

    if (!state) {
        try {
            state = $.parseJSON(decodeURIComponent(stateText));
        } catch (ex) {
            if (!exception) exception = ex;
        }
    }
    if (!state && exception) throw exception;
    return loadState(state);
}

function serialiseState(stateText) {
    var ctx = GoldenLayout.minifyConfig({content: stateText.content});
    ctx.version = 4;
    var uncompressed = risonify(ctx);
    var compressed = risonify({z: lzstring.compressToBase64(uncompressed)});
    var MinimalSavings = 0.20;  // at least this ratio smaller
    if (compressed.length < uncompressed.length * (1.0 - MinimalSavings)) {
        return compressed;
    } else {
        return uncompressed;
    }
}

module.exports = {
    deserialiseState: deserialiseState,
    serialiseState: serialiseState,
    unrisonify: unrisonify,
    risonify: risonify,
};
