// Copyright (c) 2022, Compiler Explorer Authors
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

import * as local from './local';
import _ from 'underscore';
import { Sharing } from './sharing';


const maxHistoryEntries = 30;
type Source = {dt: number, source: string};
export type HistoryEntry = {dt: number, sources: EditorSource[], config: any};
export type EditorSource = {lang: string, source: string};

function extractEditorSources(content: any[]): EditorSource[] {
    const sources: EditorSource[] = [];
    for (const component of content) {
        if (component.content) {
            const subsources = extractEditorSources(component.content);
            if (subsources.length > 0) {
                sources.push(...subsources);
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

function list(): HistoryEntry[] {
    return JSON.parse(local.get('history', '[]'));
}

function getArrayWithJustTheCode(editorSources: Record<string, any>[]): string[] {
    return editorSources.map(s => s.source);
}

function getSimilarSourcesIndex(completeHistory: HistoryEntry[], sourcesToCompareTo: any[]): number {
    let duplicateIdx = -1;

    for (let i = 0; i < completeHistory.length; i++) {
        const diff = _.difference(sourcesToCompareTo, getArrayWithJustTheCode(completeHistory[i].sources));
        if (diff.length === 0) {
            duplicateIdx = i;
            break;
        }
    }

    return duplicateIdx;
}

function push(stringifiedConfig: string) {
    const config = JSON.parse(stringifiedConfig);
    const sources = extractEditorSources(config.content);
    if (sources.length > 0) {
        const completeHistory = list();
        const duplicateIdx = getSimilarSourcesIndex(completeHistory, getArrayWithJustTheCode(sources));

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
            completeHistory[duplicateIdx].dt = Date.now();
        }

        local.set('history', JSON.stringify(completeHistory));
    }
}

export function trackHistory(layout: any) {
    let lastState: string | null = null;
    const debouncedPush = _.debounce(push, 500);
    layout.on('stateChanged', () => {
        const stringifiedConfig = JSON.stringify(Sharing.filterComponentState(layout.toConfig()));
        if (stringifiedConfig !== lastState) {
            lastState = stringifiedConfig;
            debouncedPush(stringifiedConfig);
        }
    });
}

export function sortedList(): HistoryEntry[] {
    return list().sort((a, b) => b.dt - a.dt);
}

export function sources(language: string): Source[] {
    const sourcelist: Source[] = [];
    for (const entry of sortedList()) {
        for (const source of entry.sources) {
            if (source.lang === language) {
                sourcelist.push({
                    dt: entry.dt,
                    source: source.source,
                });
            }
        }
    }
    return sourcelist;
}
