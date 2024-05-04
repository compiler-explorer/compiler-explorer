// Copyright (c) 2024, Compiler Explorer Authors
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

import path from 'path';

import fs from 'fs-extra';

import {BufferOkFunc, BuildResult, CompilationResult} from '../types/compilation/compilation.interfaces.js';
import {Artifact, ArtifactType} from '../types/tool.interfaces.js';

import {HeaptrackWrapper} from './runtime-tools/heaptrack-wrapper.js';
import * as utils from './utils.js';

export async function addArtifactToResult(
    result: CompilationResult,
    filepath: string,
    customType?: string,
    customTitle?: string,
    checkFunc?: BufferOkFunc,
): Promise<void> {
    const file_buffer = await fs.readFile(filepath);

    if (checkFunc && !checkFunc(file_buffer)) return;

    const artifact: Artifact = {
        content: file_buffer.toString('base64'),
        type: customType || 'application/octet-stream',
        name: path.basename(filepath),
        title: customTitle || path.basename(filepath),
    };

    if (!result.artifacts) result.artifacts = [];

    result.artifacts.push(artifact);
}

export async function addHeaptrackResults(result: CompilationResult, dirPath?: string): Promise<void> {
    let dirPathToUse: string = '';
    if (dirPath) {
        dirPathToUse = dirPath;
    } else if (result.buildResult && result.buildResult.dirPath) {
        dirPathToUse = result.buildResult.dirPath;
    }

    if (dirPathToUse === '') return;

    const flamegraphFilepath = path.join(dirPathToUse, HeaptrackWrapper.FlamegraphFilename);
    if (await utils.fileExists(flamegraphFilepath)) {
        await addArtifactToResult(result, flamegraphFilepath, ArtifactType.heaptracktxt, 'Heaptrack results');
    }
}

export function moveArtifactsIntoResult(movefrom: BuildResult, moveto: CompilationResult): CompilationResult {
    if (movefrom.artifacts && movefrom.artifacts.length > 0) {
        if (!moveto.artifacts) {
            moveto.artifacts = [];
        }
        moveto.artifacts.push(...movefrom.artifacts);
        delete movefrom.artifacts;
    }

    return moveto;
}
