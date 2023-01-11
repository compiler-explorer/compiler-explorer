import cheerio, {Cheerio, CheerioAPI, Document} from 'cheerio';
import fs from 'fs/promises';

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
    const instructions$ = anchor$.find('+ ul > li > ul > li > a > code');
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

    const overview$ = anchor$.find('> .section > p')[1];
    return {
        url,
        name: instruction,
        html: $(anchor$).html()!,
        tooltip: $(overview$).text()
    };
}

const contents = await fs.readFile(LANGREF_PATH, 'utf8');
const $ = cheerio.load(contents);

const names = getInstructionList($.root(), $);
const info = names.map((x) => getInstructionInfo(x, $.root(), $));

console.log('import {AssemblyInstructionInfo} from \'../base\';');
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
