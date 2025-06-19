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

import fs from 'node:fs/promises';
import path from 'node:path';

import type {SelectedLibraryVersion} from '../../types/libraries/libraries.interfaces.js';
import {logger} from '../logger.js';

function hasCopperSpiceGui(libraries: SelectedLibraryVersion[]): boolean {
    return libraries.some(lib => lib.id === 'copperspice');
}

async function findCopperSpicePlugins(libraryPaths: string[]): Promise<string[]> {
    const pluginFiles: string[] = [];

    for (const libPath of libraryPaths) {
        try {
            const files = await fs.readdir(libPath);
            for (const file of files) {
                if (file.startsWith('CsGuiXcb') && file.endsWith('.so')) {
                    const fullPath = path.join(libPath, file);
                    pluginFiles.push(fullPath);
                    logger.debug(`Found CopperSpice plugin: ${fullPath}`);
                }
                if (file.startsWith('CsXcbSupport') && file.endsWith('.so')) {
                    const fullPath = path.join(libPath, file);
                    pluginFiles.push(fullPath);
                    logger.debug(`Found CopperSpice support plugin: ${fullPath}`);
                }
            }
        } catch (e) {
            logger.debug(`Could not read directory ${libPath}: ${e}`);
        }
    }

    return pluginFiles;
}

export async function copyCopperSpicePlugins(
    dirPath: string,
    executableFilename: string,
    libraries: SelectedLibraryVersion[],
    libraryPaths: string[] = [],
): Promise<void> {
    if (!hasCopperSpiceGui(libraries)) {
        return;
    }

    logger.debug(`CopperSpice GUI detected, setting up Xcb plugins for ${executableFilename}`);

    try {
        if (libraryPaths.length === 0) {
            libraryPaths = ['/opt/compiler-explorer/libs/copperspice/1.8.0/lib'];
        }

        const pluginFiles = await findCopperSpicePlugins(libraryPaths);

        if (pluginFiles.length === 0) {
            logger.debug('No CopperSpice Xcb plugins found in library paths');
            return;
        }

        const platformsDir = path.join(dirPath, 'platforms');
        await fs.mkdir(platformsDir, {recursive: true});

        for (const pluginFile of pluginFiles) {
            const targetPath = path.join(platformsDir, path.basename(pluginFile));
            await fs.copyFile(pluginFile, targetPath);
            logger.debug(`Copied ${path.basename(pluginFile)} to platforms directory`);
        }

        logger.debug(`Successfully set up CopperSpice plugins in ${platformsDir}`);
    } catch (e) {
        logger.error(`Error while setting up CopperSpice plugins for ${executableFilename}: ${e}`);
    }
}
