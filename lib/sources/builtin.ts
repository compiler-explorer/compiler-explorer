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

import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';

import * as props from '../properties';

import type {Source, SourceEntry} from './index';

const EXAMPLES_PATH = props.get('builtin', 'sourcePath', './examples/');
const NAME_SUBSTUTION_PATTERN = new RegExp('_', 'g');
const ALL_EXAMPLES: SourceEntry[] = fs.readdirSync(EXAMPLES_PATH).flatMap(folder => {
    // Recurse through the language folders
    const folderPath = path.join(EXAMPLES_PATH, folder);
    return fs.readdirSync(folderPath).map(file => {
        // Recurse through the source files
        const filePath = path.join(folderPath, file);
        const fileName = path.parse(filePath).name;
        return {
            lang: folder,
            name: fileName.replace(NAME_SUBSTUTION_PATTERN, ' '),
            path: filePath,
            file: fileName,
        };
    });
});

export const builtin: Source = {
    name: 'Examples',
    urlpart: 'builtin',
    async load(language: string, filename: string): Promise<{file: string}> {
        const example = ALL_EXAMPLES.find(e => e.lang === language && e.file === filename);
        if (example === undefined) {
            return {file: 'No path found'};
        }
        try {
            return {file: await fsp.readFile(example.path, 'utf8')};
        } catch (err: unknown) {
            return {file: 'Could not read file'};
        }
    },
    async list(): Promise<Omit<SourceEntry, 'path'>[]> {
        return ALL_EXAMPLES.map(e => ({
            lang: e.lang,
            name: e.name,
            file: e.file,
        }));
    },
    save: null,
};
