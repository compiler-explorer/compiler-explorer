import * as monaco from 'monaco-editor';
import {ga} from '../analytics';
import LRUCache from 'lru-cache';
import {getAssemblyDocumentation} from '../api/api';
import {AssemblyInstructionInfo} from '../../lib/asm-docs/base';
import {InstructionSet} from '../../types/features/assembly-documentation.interfaces';
import {Alert} from '../alert';

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
const ASSEMBLY_OPCODE_CACHE = new LRUCache<string, OpcodeCacheEntry>({
    max: 64 * 1024,
});

/**
 * Add an extension to the monaco editor which allows to view the assembly documentation for the instruction the
 * cursor is currently on.
 */
export const createViewAssemblyDocumentationAction = (
    editor: monaco.editor.IStandaloneCodeEditor,
    instructionSet: InstructionSet
) => {
    editor.addAction({
        id: VIEW_ASSEMBLY_DOCUMENTATION_ID,
        label: 'View assembly documentation',
        keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.F8],
        precondition: 'isAsmKeyword',
        contextMenuGroupId: 'help',
        contextMenuOrder: 1.5,
        run: onAssemblyAction(editor, instructionSet),
    });
};

const onAssemblyAction = (editor: monaco.editor.IStandaloneCodeEditor, instructionSet: InstructionSet) => async () => {
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
        const response = await getAssemblyInfo(opcode, instructionSet);
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
        alertSystem.notify('There was a network error fetching the documentation for this opcode (' + error + ').', {
            group: 'notokenindocs',
            alertClass: 'notification-error',
            dismissTime: 5000,
        });
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
<a href="${github}">
  open an issue on GitHub 
  <sup>
    <small class='fas fa-external-link-alt opens-new-window' title='Opens in a new window'>
    </small>
  </sup>
</a>`;
};
