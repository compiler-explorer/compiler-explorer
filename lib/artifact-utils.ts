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
