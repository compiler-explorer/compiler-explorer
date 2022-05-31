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

import * as monaco from 'monaco-editor';
import {ga} from '../analytics';
import LRUCache from 'lru-cache';
import {getAssemblyDocumentation} from '../api/api';
import {AssemblyInstructionInfo} from '../../lib/asm-docs/base';
import {InstructionSet} from '../../types/features/assembly-documentation.interfaces';
import {Alert} from '../alert';
import {throttle} from 'underscore';
import {SiteSettings} from '../settings';

type OpcodeCacheEntry =
    | {
          found: true;
          body: AssemblyInstructionInfo;
      }
    | {
          found: false;
          error: string;
      };

const VIEW_ASSEMBLY_DOCUMENTATION_ID = 'viewasmdoc';
const IS_ASM_KEYWORD_CTX = 'isAsmKeyword';
const ASSEMBLY_OPCODE_CACHE = new LRUCache<string, OpcodeCacheEntry>({
    max: 64 * 1024,
});

/**
 * Add an extension to the monaco editor which allows to view the assembly documentation for the instruction the
 * cursor is currently on.
 */
export const createViewAssemblyDocumentationAction = (
    editor: monaco.editor.IStandaloneCodeEditor,
    getInstructionSet: () => InstructionSet
) => {
    const contextKey = editor.createContextKey(IS_ASM_KEYWORD_CTX, true);

    editor.onContextMenu(event => {
        if (event.target.position === null) return;
        const position = event.target.position;

        const word = editor.getModel()?.getWordAtPosition(position) ?? null;
        if (word === null) return;

        if (word.word) {
            contextKey.set(
                isAssemblyKeyword(
                    editor,
                    word,
                    new monaco.Range(
                        position.lineNumber,
                        Math.max(1, word.startColumn),
                        position.lineNumber,
                        word.endColumn
                    )
                )
            );
        }
    });

    editor.addAction({
        id: VIEW_ASSEMBLY_DOCUMENTATION_ID,
        label: 'View assembly documentation',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F8],
        precondition: IS_ASM_KEYWORD_CTX,
        contextMenuGroupId: 'help',
        contextMenuOrder: 1.5,
        run: onAssemblyAction(editor, getInstructionSet),
    });
};

// Very hacky solution to get access to the decorations stored on the compiler.js instance
// TODO(supergrecko): refactor the decorations storage to be more accessible
type DecorationStorageValue = {
    range: monaco.IRange;
    options: monaco.editor.IModelDecorationOptions;
};
type WithDecorationContext = {
    prevDecorations: string[];
    decorations: Record<string, DecorationStorageValue>;
    updateDecorations(): void;
};

export const createShowAssemblyDocumentationEvent = (
    editor: monaco.editor.IStandaloneCodeEditor,
    getInstructionSet: () => InstructionSet,
    getSettings: () => SiteSettings,
    self: WithDecorationContext
) => {
    const handler = throttle((event: monaco.editor.IEditorMouseEvent) => {
        onEditorMouseMove(event, editor, getInstructionSet, getSettings, self);
    }, 50);
    editor.onMouseMove(handler);
};

const onEditorMouseMove = async (
    event: monaco.editor.IEditorMouseEvent,
    editor: monaco.editor.IStandaloneCodeEditor,
    getInstructionSet: () => InstructionSet,
    getSettings: () => SiteSettings,
    self: WithDecorationContext
) => {
    if (event.target.position === null) return;
    const settings = getSettings();
    if (!settings.hoverShowAsmDoc) return;

    const model = editor.getModel();
    if (model === null) return;

    const word = model.getWordAtPosition(event.target.position);
    if (word === null || !word.word) return;

    const range = new monaco.Range(
        event.target.position.lineNumber,
        Math.max(1, word.startColumn),
        event.target.position.lineNumber,
        word.endColumn
    );
    const isAsmKeyword = isAssemblyKeyword(editor, word, range);

    if (isAsmKeyword) {
        const response = await getAssemblyInfo(word.word.toUpperCase(), getInstructionSet());
        if (response.found) {
            self.decorations.asmToolTip = {
                range,
                options: {
                    isWholeLine: false,
                    hoverMessage: [
                        {
                            value: response.body.tooltip + '\n\nMore information available in the context menu.',
                            isTrusted: true,
                        },
                    ],
                },
            };
            self.updateDecorations();
        }
    }
};

const isAssemblyKeyword = (
    editor: monaco.editor.IStandaloneCodeEditor,
    word: monaco.editor.IWordAtPosition,
    range: monaco.IRange
): boolean => {
    const model = editor.getModel();
    if (model === null || range.startLineNumber > model.getLineCount()) return false;
    const language = model.getLanguageId();
    const tokens = monaco.editor.tokenize(model.getLineContent(range.startLineNumber), language);

    const line = tokens.length > 0 ? tokens[0] : [];

    return line.some(t => {
        return t.offset + 1 === word.startColumn && t.type === 'keyword.asm';
    });
};

const onAssemblyAction =
    (editor: monaco.editor.IStandaloneCodeEditor, getInstructionSet: () => InstructionSet) => async () => {
        ga.proxy('send', {
            hitType: 'event',
            eventCategory: 'OpenModalPane',
            eventAction: 'AsmDocs',
        });

        const position = editor.getPosition();
        const model = editor.getModel();
        if (position === null || model === null) return;

        const word = model.getWordAtPosition(position);
        if (word === null || word.word === '') return;

        const opcode = word.word.toUpperCase();
        const alertSystem = new Alert();

        try {
            const response = await getAssemblyInfo(opcode, getInstructionSet());
            if (response.found) {
                alertSystem.alert(
                    opcode + ' help',
                    response.body.html + createDisplayableHtml(response.body.url, opcode),
                    () => {
                        editor.focus();
                        editor.setPosition(position);
                    }
                );
            } else {
                alertSystem.notify('This token was not found in the documentation. Sorry!', {
                    group: 'notokenindocs',
                    alertClass: 'notification-error',
                    dismissTime: 5000,
                });
            }
        } catch (error) {
            alertSystem.notify(
                'There was a network error fetching the documentation for this opcode (' + error + ').',
                {
                    group: 'notokenindocs',
                    alertClass: 'notification-error',
                    dismissTime: 5000,
                }
            );
        }
    };

export const getAssemblyInfo = async (opcode: string, instructionSet: InstructionSet): Promise<OpcodeCacheEntry> => {
    const entryName = `asm/${instructionSet}/${opcode}`;

    if (ASSEMBLY_OPCODE_CACHE.has(entryName)) {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- logically sound
        return ASSEMBLY_OPCODE_CACHE.get(entryName)!;
    }

    try {
        const response = await getAssemblyDocumentation({opcode, instructionSet});
        const json = await response.json();
        if (response.status === 200) {
            ASSEMBLY_OPCODE_CACHE.set(entryName, {found: true, body: json});
        } else {
            // TODO(supergrecko): make prettier with XOR type
            const jsonWithError = json as unknown as {error: string};
            ASSEMBLY_OPCODE_CACHE.set(entryName, {found: false, error: jsonWithError.error});
        }

        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion -- logically sound
        return ASSEMBLY_OPCODE_CACHE.get(entryName)!;
    } catch (error) {
        throw new Error('Fetch Assembly Documentation failed: ' + error);
    }
};

const createDisplayableHtml = (url: string, opcode: string) => {
    const title = encodeURI(`[BUG] Problem with ${opcode} opcode documentation`);
    const github = `https://github.com/compiler-explorer/compiler-explorer/issues/new?title=${title}`;
    return `
<br><br>
For more information, visit 
<a href='${url}' target='_blank' rel='noopener noreferrer'>the ${opcode} documentation
  <sup>
    <small class='fas fa-external-link-alt opens-new-window' title='Opens in a new window'>
    </small>
  </sup>
<a/>.
If the documentation for this opcode is wrong or broken in some way, please feel free to 
<a href='${github}'>
  open an issue on GitHub 
  <sup>
    <small class='fas fa-external-link-alt opens-new-window' title='Opens in a new window'>
    </small>
  </sup>
</a>`;
};
