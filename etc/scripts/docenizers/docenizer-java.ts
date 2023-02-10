#!/usr/bin/env node

import fs from 'fs/promises';
import * as cheerio from 'cheerio';

const JVMS_SPECIFICATION = './vendor/jvms.html';
const VARIADIC_MAPPINGS = {
    '<n>': ['0', '1', '2', '3'],
    '<d>': ['0', '1'],
    '<f>': ['0, 1', '2'],
    '<i>': ['m1', '0', '1', '2', '3', '4', '5'],
    '<l>': ['0', '1'],
    '<op>': ['g', 'l'],
    '<cond>': ['eq', 'ne'],
};

type InstructionInfo = {
    name: string
    anchor: string
    tooltip: string
    format: string[]
    stack: [string | null, string | null]
    description: string
}

const extract = (node: cheerio.Cheerio<cheerio.Element>, $: cheerio.CheerioAPI) => {
    const anchorElement = node.find('div.titlepage > div > div > h3.title > a[name*="jvms-6.5"]').first();
    const nameElement = anchorElement.parent().find('span.emphasis > em');
    const anchor = anchorElement.attr('name')!;
    const name = nameElement.text();

    const [
        operationSection,
        formatSection,
        _formsSection,
        operandStackSection,
        descriptionSection,
    ] = node.find('div.section').toArray().map(it => $(it));

    const operation = operationSection.find('p.norm').first().text();
    const format = formatSection.find('div.literallayout > p > span.emphasis > em').toArray().map(it => $(it).text());
    const description = descriptionSection.find('p.norm-dynamic').first();
    // rewrite links to oracle.com
    $(description).find('* > a[href*="jvms-"]').toArray().forEach((el) => {
        $(el).attr('href', `https://docs.oracle.com/javase/specs/jvms/se18/html/${$(el).attr('href')}`);
    });

    const [stackBefore, stackAfter] = operandStackSection.find('p.norm')
        .toArray()
        .map(it => $(it));

    const result: InstructionInfo[] = [];
    const hasVariadicMapping = Object.keys(VARIADIC_MAPPINGS).some((pat) => name.endsWith(pat));
    if (hasVariadicMapping) {
        for (const [pattern, mappings] of Object.entries(VARIADIC_MAPPINGS)) {
            if (name.endsWith(pattern)) {
                for (const mapping of mappings) {
                    result.push({
                        name: name.replace(pattern, mapping).replaceAll('<', '[').replaceAll('>', ']'),
                        anchor,
                        description: description.html()!,
                        tooltip: operation,
                        stack: [stackBefore?.html(), stackAfter?.html()],
                        format: format.map(x => x.replaceAll('<', '[').replaceAll('>', ']')),
                    });
                }
            }
        }
    } else {
        result.push({
            name,
            anchor,
            description: description.html()!,
            tooltip: operation,
            stack: [stackBefore?.html(), stackAfter?.html()],
            format,
        });
    }
    return result;
};

const main = async () => {
    const file = await fs.readFile(JVMS_SPECIFICATION, 'utf-8');
    const $ = cheerio.load(file);
    const sections = $('div.section-execution');
    const instructions = sections.toArray()
        .slice(1) // Drop 1 because the first is the "mne monic"
        .map(it => extract($(it), $))
        .flat();
    console.log('import {AssemblyInstructionInfo} from \'../base\';');
    console.log('');
    console.log('export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {');
    console.log('    if (!opcode) return;');
    console.log('    switch (opcode.toUpperCase()) {');
    for (const instruction of instructions) {
        console.log(`        case '${instruction.name.toUpperCase()}':`);
        console.log('            return {');
        console.log(`                url: \`https://docs.oracle.com/javase/specs/jvms/se18/html/jvms-6.html#${instruction.anchor}\`,`);
        const body = `<p>Instruction ${instruction.name}: ${instruction.tooltip}</p><p>Format: ${instruction.format.join(' ')}</p>${instruction.stack[0] && `<p>Operand Stack: ${instruction.stack[0]} ${instruction.stack[1]}</p>`}<p>${instruction.description}</p>`;
        console.log(`                html: \`${body.replace(/\s\s+/g, ' ')}\`,`);
        console.log(`                tooltip: \`${instruction.tooltip.replace(/\s\s+/g, ' ')}\`,`);
        console.log('            };');
    }
    console.log('    }');
    console.log('}');
};

main().then(() =>{}).catch(e => console.error("Caught error", e));
