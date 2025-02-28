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
    const anchor$ = root.find('[href="\\#instruction-reference"]').first();
    const instructions$ = anchor$.parent().find('+ ul > li > ul > li > p > a > code');
    const instructions = instructions$.map((_, el) => {
        const span$ = $(el).find('span');
        return span$.map((_, el) => $(el).text())
            .get()
            .filter((s) => s !== '..')
            .map((x) => {
                // Do any name-to-anchor rewrites here
                if (x == "va_arg") return "va-arg";
                return x;
            })
            .join('-');
    }).get();
    return [...new Set(instructions)].filter(x => x !== '..' && x !== 'to');
}

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

    // Extract the modified HTML content
    const modifiedHtml = myhtml$.html();
    
    return {
        url,
        name: instruction,
        html: modifiedHtml,
        tooltip: $(overviewPars$).text()
    };
}

const contents = await readFile(LANGREF_PATH, 'utf8');
const $ = load(contents);

const names = getInstructionList($.root(), $);
const info = names.map((x) => getInstructionInfo(x, $.root(), $));

console.log('import {AssemblyInstructionInfo} from \'../base.js\';');
console.log('');
console.log('export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {');
console.log('    if (!opcode) return;');
console.log('    switch (opcode.toUpperCase()) {');

for (const instruction of info) {
    console.log(`        case '${instruction.name.toUpperCase()}':`);
    console.log('            return {');
    console.log(`                url: \`${instruction.url}\`,`);
    console.log(`                html: \`${instruction.html.replaceAll('\n', '').replaceAll('`', '\\`')}\`,`);
    console.log(`                tooltip: \`${instruction.tooltip.replaceAll('\n', '').replaceAll('`', '\\`')}\`,`);
    console.log('            };');
}

console.log('    }');
console.log('}');
