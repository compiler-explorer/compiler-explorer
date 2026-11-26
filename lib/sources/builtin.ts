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

function readExamples(examplesPath: string): SourceEntry[] {
    return fs.readdirSync(examplesPath).flatMap(folder => {
        const folderPath = path.join(examplesPath, folder);
        return fs.readdirSync(folderPath).map(file => {
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

export class BuiltinSource implements Source {
    readonly name = 'Examples';
    readonly urlpart = 'builtin';
    private readonly examples: SourceEntry[];

    constructor(examplesPath: string) {
        this.examples = readExamples(examplesPath);
    }

    async load(language: string, filename: string): Promise<{file: string}> {
        const example = this.examples.find(e => e.lang === language && e.file === filename);
        if (example === undefined) {
            return {file: 'No path found'};
        }
        try {
            return {file: await fsp.readFile(example.path, 'utf8')};
        } catch {
            return {file: 'Could not read file'};
        }
    }

    async list(): Promise<SourceApiEntry[]> {
        return this.examples.map(e => ({
            lang: e.lang,
            name: e.name,
            file: e.file,
        }));
    }
}

export function getExamplesRoot(): string {
    return props.get<string>('builtin', 'sourcePath', './examples/');
}

export function createBuiltinSource(): BuiltinSource {
    const examplesRoot = getExamplesRoot();
    try {
        return new BuiltinSource(examplesRoot);
    } catch (e) {
        throw new Error(`Unable to read builtin examples from "${examplesRoot}" (set via builtin.sourcePath)`, {
            cause: e,
        });
    }
}
