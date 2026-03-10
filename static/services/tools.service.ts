// Copyright (c) 2026, Compiler Explorer Authors
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

import {optionsHash} from '../options.js';

export type ToolEntry = {
    id: string;
    name: string;
    type: string;
    languageId: string;
    allowStdin: boolean;
    args?: string;
    monacoStdin?: string;
    icon?: string;
    darkIcon?: string;
    stdinHint?: string;
};

export class ToolsService {
    private readonly loadPromises = new Map<string, Promise<Record<string, ToolEntry>>>();
    private readonly cache = new Map<string, Record<string, ToolEntry>>();

    getCachedToolsForLang(langId: string): Record<string, ToolEntry> | undefined {
        return this.cache.get(langId);
    }

    async getToolsForLang(langId: string): Promise<Record<string, ToolEntry>> {
        let promise = this.loadPromises.get(langId);
        if (!promise) {
            promise = this.fetchToolsForLang(langId);
            this.loadPromises.set(langId, promise);
            promise.then(result => this.cache.set(langId, result));
        }
        return promise;
    }

    private async fetchToolsForLang(langId: string): Promise<Record<string, ToolEntry>> {
        const response = await fetch(`${window.httpRoot}api/tools/${encodeURIComponent(langId)}?hash=${optionsHash}`, {
            headers: {Accept: 'application/json'},
        });
        const toolsArr: ToolEntry[] = await response.json();
        const result: Record<string, ToolEntry> = {};
        for (const tool of toolsArr) {
            result[tool.id] = tool;
        }
        return result;
    }
}

export const toolsService = new ToolsService();
