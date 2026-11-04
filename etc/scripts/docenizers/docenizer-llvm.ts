import {Document} from 'domhandler';
import {load, Cheerio, CheerioAPI} from 'cheerio';
import {readFile} from 'fs/promises';

const LANGREF_PATH = './vendor/LangRef.html';

type InstructionInfo = {
    name: string;
    url: string;
    html: string;
    tooltip: string;
}

/** Retrieves the list of all LLVM instruction names */
const getInstructionList = (root: Cheerio<Document>, $: CheerioAPI) => {
    const ids = root.find('section[id$="-instruction"]')
        .map((_, el) => $(el).attr('id')!.replace(/-instruction$/, ''))
        .get();
    return [...new Set(ids)];
};

/** Crawls information about one LLVM instruction */
const getInstructionInfo = (instruction: string, root: Cheerio<Document>, $: CheerioAPI): InstructionInfo => {
    const anchor$ = root.find(`#${instruction.replace(' ', '-')}-instruction`).first();
    const url = `https://llvm.org/docs/LangRef.html#${instruction}-instruction`;

    const overviewHeaders$ = anchor$.find('h5').filter((_, el) => $(el).text().startsWith('Overview'));
    const overviewPars$ = overviewHeaders$.parent().find('> p');
    
    // Load the HTML content of the anchor element into a new Cheerio instance
    const myhtml$ = load(anchor$.html()!);
    // <a class="headerlink" href="#id210" title="Permalink to this heading">¶</a>

    // Find and remove all <a> elements with text equal to '¶'
    myhtml$('a').filter((_, el) => myhtml$(el).text().trim() === '¶').remove();

    // Remove all <a class="headerlink"> elements
    myhtml$('a.headerlink').remove();

    // Make relative links absolute so they point at llvm.org rather than the host page
    myhtml$('a[href]').each((_, el) => {
        const a$ = myhtml$(el);
        const href = a$.attr('href')!;
        if (/^(https?:|mailto:)/.test(href)) return;
        a$.attr(
            'href',
            href.startsWith('#')
                ? `https://llvm.org/docs/LangRef.html${href}`
                : new URL(href, 'https://llvm.org/docs/').toString(),
        );
        a$.attr('target', '_blank');
        a$.attr('rel', 'noopener noreferrer');
    });

    // Extract the modified HTML content
    const modifiedHtml = myhtml$.html();

    // Instructions that have '-to' in their anchors are likely conversion
    // instructions. The trailing '-to' has to be trimmed, as otherwise
    // operations like 'fptrunc' won't find a match in the documentation.
    const name = instruction.replace(new RegExp("-to$"),'')

    return {
        url,
        name,
        html: modifiedHtml,
        tooltip: $(overviewPars$).text()
    };
}

const contents = await readFile(LANGREF_PATH, 'utf8');
const $ = load(contents);

const names = getInstructionList($.root(), $);
const info = names.map((x) => getInstructionInfo(x, $.root(), $));

console.log('import type {AssemblyInstructionInfo} from \'../../../types/assembly-docs.interfaces.js\';');
console.log('');
console.log('export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {');
console.log('    if (!opcode) return;');
console.log('    switch (opcode.toUpperCase()) {');

for (const instruction of info) {
    console.log(`        case '${instruction.name.toUpperCase()}':`);
    console.log('            return {');
    console.log(`                url: \`${instruction.url}\`,`);
    console.log(`                html: \`${instruction.html.replaceAll('`', '\\`')}\`,`);
    console.log(`                tooltip: \`${instruction.tooltip.replaceAll('\n', '').replaceAll('`', '\\`')}\`,`);
    console.log('            };');
}

console.log('    }');
console.log('}');
