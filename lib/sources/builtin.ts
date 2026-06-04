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

import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';

import type {Source, SourceApiEntry, SourceEntry} from '../../types/source.interfaces.js';
import * as props from '../properties.js';

const NAME_SUBSTUTION_PATTERN = /_/g;
let examplesPath: string | undefined;
let allExamples: SourceEntry[] | undefined;

function readExamples(configuredExamplesPath: string): SourceEntry[] {
    // Recurse through the language folders
    return fs.readdirSync(configuredExamplesPath).flatMap(folder => {
        const folderPath = path.join(configuredExamplesPath, folder);
        return fs.readdirSync(folderPath).map(file => {
            // Recurse through the source files
            const filePath = path.join(folderPath, file);
            const fileName = path.parse(filePath).name;
            return {
                lang: folder,
                name: fileName.replaceAll(NAME_SUBSTUTION_PATTERN, ' '),
                path: filePath,
                file: fileName,
            };
        });
    });
}

function getExamples(): SourceEntry[] {
    const configuredExamplesPath = props.get<string>('builtin', 'sourcePath', './examples/');
    if (allExamples === undefined || examplesPath !== configuredExamplesPath) {
        examplesPath = configuredExamplesPath;
        allExamples = readExamples(configuredExamplesPath);
    }
    return allExamples;
}

export const builtin: Source = {
    name: 'Examples',
    urlpart: 'builtin',
    async load(language: string, filename: string): Promise<{file: string}> {
        const example = getExamples().find(e => e.lang === language && e.file === filename);
        if (example === undefined) {
            return {file: 'No path found'};
        }
        try {
            return {file: await fsp.readFile(example.path, 'utf8')};
        } catch {
            return {file: 'Could not read file'};
        }
    },
    async list(): Promise<SourceApiEntry[]> {
        return getExamples().map(e => ({
            lang: e.lang,
            name: e.name,
            file: e.file,
        }));
    },
};
