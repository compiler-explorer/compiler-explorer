// Copyright (c) 2019, Compiler Explorer Authors
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
var
    local = require('./local'),
    _ = require('underscore');

var maxHistoryEntries = 30;

function extractEditorSources(content) {
    var sources = [];

    for (var i = 0; i < content.length; i++) {
        var component = content[i];
        if (component.content) {
            var subsources = extractEditorSources(component.content);
            if (subsources.length > 0) {
                sources = sources.concat(subsources);
            }
        } else if (component.componentName === 'codeEditor') {
            sources.push({
                lang: component.componentState.lang,
                source: component.componentState.source,
            });
        }
    }

    return sources;
}

function list() {
    var stringifiedHistory = local.get('history');
    return JSON.parse(stringifiedHistory ? stringifiedHistory : '[]');
}

function getArrayWithJustTheCode(editorSources) {
    return _.pluck(editorSources, 'source');
}

function getSimilarSourcesIndex(completeHistory, sourcesToCompareTo) {
    var duplicateIdx = -1;

    for (var i = 0; i < completeHistory.length; i++) {
        var diff = _.difference(sourcesToCompareTo, getArrayWithJustTheCode(completeHistory[i].sources));
        if (diff.length === 0) {
            duplicateIdx = i;
            break;
        }
    }

    return duplicateIdx;
}

function push(stringifiedConfig) {
    var config = JSON.parse(stringifiedConfig);
    var sources = extractEditorSources(config.content);
    if (sources.length > 0) {
        var completeHistory = list();
        var duplicateIdx = getSimilarSourcesIndex(completeHistory, getArrayWithJustTheCode(sources));

        if (duplicateIdx === -1) {
            while (completeHistory.length >= maxHistoryEntries) {
                completeHistory.shift();
            }
    
            completeHistory.push({
                dt: Date.now(),
                sources: sources,
                config: config,
            });
        } else {
            var entry = completeHistory[duplicateIdx];
            entry.dt = Date.now();
        }

        local.set('history', JSON.stringify(completeHistory));
    }
}

function sortedList() {
    var sorted = list();

    sorted.sort(function (a, b) {
        return b.dt - a.dt;
    });

    return sorted;
}

function sources(language) {
    var sourcelist = [];
    _.each(sortedList(), function (entry) {
        _.each(entry.sources, function (source) {
            if (source.lang === language) {
                sourcelist.push({
                    dt: entry.dt,
                    source: source.source,
                });
            }
        });
    });

    return sourcelist;
}

module.exports = {
    push: _.debounce(push, 500),
    list: list,
    sortedList: sortedList,
    sources: sources,
};
