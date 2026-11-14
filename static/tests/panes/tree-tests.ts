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

import $ from 'jquery';
import {describe, expect, it, vi} from 'vitest';

import {MultifileFile} from '../../multifile-service.js';
import {Tree} from '../../panes/tree.js';

const rowTemplate = `<div>
    <li class="list-group-item tree-editor-file">
        <span class="filename"></span>
        <button class="btn delete-file"></button>
        <button class="btn rename-file"></button>
        <button class="btn stage-file"></button>
        <button class="btn unstage-file"></button>
    </li>
</div>`;

/** A tree listing `example.cpp`, included in the build, with its editor closed. */
function treeWithClosedFile() {
    const file: MultifileFile = {
        fileId: 1,
        isIncluded: true,
        isOpen: false,
        isMainSource: true,
        filename: 'example.cpp',
        content: 'int main() {}',
        editorId: -1,
        langId: 'c++',
    };
    const namedItems = $('<ul>');
    const multifileService = {
        getFileByFileId: () => file,
        renameFile: vi.fn(async () => true),
        includeByFileId: vi.fn(async () => {}),
        excludeByFileId: vi.fn(async () => {}),
    };
    const alertSystem = {ask: vi.fn()};
    const tree = Object.create(Tree.prototype) as Tree;
    Object.assign(tree, {
        rowTemplate: $(rowTemplate),
        namedItems,
        unnamedItems: $('<ul>'),
        multifileService,
        alertSystem,
        editFile: vi.fn(),
        refresh: vi.fn(),
        sendChangesToAllEditors: vi.fn(),
    });

    (tree as any).addRowToTreelist(file);

    return {tree: tree as any, row: namedItems.children().first(), multifileService, alertSystem};
}

describe('Tree file rows', () => {
    it('opens the file when its row is clicked', () => {
        const {tree, row} = treeWithClosedFile();

        row.find('.filename')[0].click();

        expect(tree.editFile).toHaveBeenCalledWith(1);
    });

    it.each<[string, (made: ReturnType<typeof treeWithClosedFile>) => unknown]>([
        ['delete-file', made => made.alertSystem.ask],
        ['rename-file', made => made.multifileService.renameFile],
        ['stage-file', made => made.multifileService.includeByFileId],
        ['unstage-file', made => made.multifileService.excludeByFileId],
    ])('only acts on the %s button, without opening the file', (button, handler) => {
        // opening it from the delete button left an editor whose pending change outlived it once the delete closed it
        const made = treeWithClosedFile();

        made.row.find(`.${button}`)[0].click();

        expect(handler(made)).toHaveBeenCalled();
        expect(made.tree.editFile).not.toHaveBeenCalled();
    });
});
