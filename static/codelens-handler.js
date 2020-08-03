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
'use strict';

var _ = require('underscore'),
    monaco = require('monaco-editor');

var registeredCodelenses = [];
var providersPerLanguage = {};

function registerLensesForCompiler(compilerId, editorModel, lenses) {
    var item = _.find(registeredCodelenses, function (item) {
        return item.compilerId === compilerId;
    });

    if (item) {
        item.lenses = lenses;
    } else {
        registeredCodelenses.push({
            compilerId: compilerId,
            editorModel: editorModel,
            lenses: lenses,
        });
    }
}

function provide(model) {
    var item = _.find(registeredCodelenses, function (item) {
        return item.editorModel === model;
    });

    if (item) {
        return {
            lenses: item.lenses,
            dispose: function () {},
        };
    } else {
        return {
            lenses: [],
            dispose: function () {},
        };
    }
}

function unregister(compilerId) {
    var item = _.find(registeredCodelenses, function (item) {
        return item.compilerId === compilerId;
    });

    if (item) {
        registeredCodelenses = _.without(registeredCodelenses, item);
    }
}

function registerProviderForLanguage(language) {
    if (!providersPerLanguage[language]) {
        providersPerLanguage[language] = monaco.languages.registerCodeLensProvider(language, {
            provideCodeLenses: provide,
        });
    }
}

module.exports = {
    registerLensesForCompiler: registerLensesForCompiler,
    unregister: unregister,
    registerProviderForLanguage: registerProviderForLanguage,
};
