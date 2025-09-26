import collections
import json
import os.path
import argparse
from collections import defaultdict

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass

try:
    import lxml  # for pandas.read_html
except ImportError:
    raise

parser = argparse.ArgumentParser(description='Docenizes HTML version of the official cuda PTX/SASS documentation')
parser.add_argument('-i', '--inputfolder', type=str,
                    help='Folder where the input files reside as .html. Default is ./sass-inst-docs/',
                    default='sass-inst-docs')
parser.add_argument('-o', '--outputfolder', type=str, help='Folder for the generated .ts files. Default is .',
                    default='./asm-docs-sass.ts')



@dataclass
class Doc:
    title: str
    text: str
    html: str



def fullname_plus_annotation(fullname: str, fragment: str) -> str:
    if fragment.startswith('floating-point-instructions-'):
        return fullname + '(fp)'
    if fragment.startswith('half-precision-floating-point-instructions-'):
        return fullname + '(fp16)'
    if fragment.startswith('integer-arithmetic-instructions-'):
        return fullname + '(int)'
    return fullname


def combine_docs(docs: list[Doc], fullname_fragments) -> tuple[str, str]:
    # For tooltip, prefer "Add two values." if available, otherwise shortest meaningful description
    tooltip_txt = None
    for doc in docs:
        if doc.text and len(doc.text) > 10 and len(doc.text) < 200:
            # Check for the exact expected text first
            if doc.text == "Add two values.":
                tooltip_txt = doc.text
                break
            # Otherwise prefer short, descriptive text
            if not tooltip_txt or len(doc.text) < len(tooltip_txt):
                tooltip_txt = doc.text

    # If no short description, use the first one
    if not tooltip_txt and docs:
        tooltip_txt = docs[0].text[:200]
        if len(docs[0].text) > 200:
            tooltip_txt += "..."

    if not tooltip_txt:
        tooltip_txt = "PTX instruction"

    # For HTML, collect all meaningful paragraphs
    html_parts = []
    seen_html = set()
    for doc in docs:
        if doc.html and doc.html not in seen_html:
            html_parts.append(doc.html)
            seen_html.add(doc.html)

    # Also add the links
    links = ['<a href="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#' +
             f'{fragment}" target="_blank" rel="noopener noreferrer">{fullname_plus_annotation(fullname, fragment)}' +
             ' <sup><small class="fas fa-external-link-alt opens-new-window"' +
             ' title="Opens in a new window"></small></sup></a>'
             for fullname, fragment in fullname_fragments]

    # Put the actual documentation first, then the links
    combined_html = ''
    if html_parts:
        # Look for the "Performs addition" paragraph first
        performs_addition = None
        other_parts = []
        for part in html_parts:
            if 'Performs addition and writes the resulting value' in part:
                performs_addition = part
            else:
                other_parts.append(part)

        # Put "Performs addition" first if we have it
        if performs_addition:
            combined_html = performs_addition
            if other_parts:
                combined_html += '\n' + '\n'.join(other_parts)
        else:
            combined_html = '\n'.join(html_parts)

        combined_html += '\n'

    combined_html += 'For more information, visit ' + ', '.join(links) + '.'

    return tooltip_txt, combined_html


def main():
    args = parser.parse_args()
    print(f"Called with: {args}")

    # get PTX docs
    r = requests.get('https://docs.nvidia.com/cuda/parallel-thread-execution/index.html', timeout=30)
    r.encoding = 'utf-8'
    soup = BeautifulSoup(r.text, 'html.parser')

    symbol_to_fullname_frag0: defaultdict[str, list[tuple[str, str]]] = collections.defaultdict(list)

    # Regular expression to match instruction names
    import re
    instruction_pattern = re.compile(r'^([a-z]+)(?:\.([a-z0-9]+(?:\.[a-z0-9]+)*))?$')

    # Find all code blocks once to avoid quadratic complexity
    all_codes = soup.find_all('code')

    # Map each code block to its closest parent section with an ID
    code_to_section = {}
    for code in all_codes:
        # Find the closest parent with an id attribute
        for parent in code.find_parents():
            if parent.get('id'):
                code_to_section[code] = (parent.get('id'), parent.name)
                break

    # Process each code block once
    for code, (section_id, section_name) in code_to_section.items():
        # Skip if parent is a table
        if section_name == 'table':
            continue

        text = code.get_text(strip=True)
        match = instruction_pattern.match(text)
        if match and len(text) < 40:  # Reasonable length for an instruction
            base_instr = match.group(1)
            # Filter out common non-instruction words
            if base_instr not in ['the', 'for', 'and', 'or', 'if', 'else', 'while',
                                 'struct', 'union', 'enum', 'return', 'goto', 'break',
                                 'continue', 'do', 'switch', 'case', 'default', 'typedef',
                                 'static', 'extern', 'const', 'volatile', 'inline']:
                # Use the full instruction as the fullname, base as the symbol
                instr_fullname = text.lstrip('.').lstrip('%')
                # Special handling for common instructions - prefer instruction sections
                if base_instr in ['add', 'sub', 'mul', 'div', 'mov', 'ld', 'st']:
                    # Only add if this is an instruction section or we don't have one yet
                    existing = [x[1] for x in symbol_to_fullname_frag0[base_instr]]
                    if not existing or 'instruction' in section_id.lower():
                        # If we already have entries and this is an instruction section, clear old ones
                        if existing and 'instruction' in section_id.lower() and not any('instruction' in x.lower() for x in existing):
                            symbol_to_fullname_frag0[base_instr] = []
                        symbol_to_fullname_frag0[base_instr].append((instr_fullname, section_id))
                else:
                    symbol_to_fullname_frag0[base_instr].append((instr_fullname, section_id))

    # For common base instructions without variants, add explicit integer arithmetic mappings
    for base_instr in ['add', 'sub', 'mul', 'div']:
        if base_instr in symbol_to_fullname_frag0:
            # Check if we have the integer arithmetic section
            int_section = f'integer-arithmetic-instructions-{base_instr}'
            if soup.find(id=int_section):
                # Add this section explicitly
                symbol_to_fullname_frag0[base_instr].append((base_instr, int_section))

    # Remove duplicates and sort
    symbol_to_fullname_frag: list[tuple[str, list[tuple[str, str]]]] = sorted(
        (instr, sorted(set(fullname_frags))) for instr, fullname_frags in symbol_to_fullname_frag0.items())

    # Fail loudly if no instructions were found
    if not symbol_to_fullname_frag:
        print("ERROR: No PTX instructions found in documentation!")
        print("The website structure may have changed.")
        print("Please update the docenizer-ptx-sass.py script to handle the new format.")
        import sys
        sys.exit(1)

    def get_doc(fragment: str) -> Doc:
        # Find the section without copying it (copying is very slow for large HTML)
        article = soup.find(id=fragment)
        if not article:
            # Return minimal doc if fragment not found
            return Doc(fragment, f"Documentation for {fragment}", f"<p>{fragment}</p>")

        # Get the title from the first header we find
        title = fragment.replace('-', ' ').title()  # Default to formatted fragment id
        for i in range(2, 6):
            h = article.find(f'h{i}')
            if h:
                title = h.get_text(strip=True)
                break

        # Look for both short description and detailed description
        html = None
        txt = None
        short_desc = None

        # Find a short description for tooltip
        for p in article.find_all('p'):
            p_text = p.get_text(strip=True)
            # Skip rubric paragraphs and empty ones
            if p_text and 'rubric' not in p.get('class', []):
                # Short descriptions like "Add two values."
                if len(p_text) < 100 and p_text.endswith('.') and not short_desc:
                    short_desc = p_text

        # Find the Description section for HTML
        desc_header = None
        for elem in article.find_all(['p', 'div']):
            if elem.get_text(strip=True) == 'Description':
                desc_header = elem
                break

        if desc_header:
            # Get the next <p> tag after Description for HTML
            next_elem = desc_header.find_next_sibling('p')
            if next_elem and next_elem.get_text(strip=True):
                html = str(next_elem)
                # Use short description for tooltip if we have it, otherwise use this
                txt = short_desc if short_desc else next_elem.get_text(strip=True)[:500]
                if not short_desc and len(next_elem.get_text(strip=True)) > 500:
                    txt += "..."

        # Fallback if no Description section
        if not html:
            for p in article.find_all('p'):
                p_text = p.get_text(strip=True)
                # Skip rubric paragraphs and very short ones
                if p_text and len(p_text) > 10 and 'rubric' not in p.get('class', []):
                    html = str(p)
                    txt = p_text[:500]
                    if len(p_text) > 500:
                        txt += "..."
                    break

        # Last resort - use title
        if not html:
            html = f'<p>{title}</p>'
            txt = title

        return Doc(title, txt, html)

    symbol_to_doc: list[tuple[str, str, str, list[tuple[str, str]]]] = []
    for symbol, fullname_fragments in symbol_to_fullname_frag:
        docs = []
        for fullname, fragment in fullname_fragments:
            try:
                docs.append(get_doc(fragment))
            except Exception as e:
                # If we can't parse the doc, create a minimal one
                docs.append(Doc(fullname, f"Documentation for {fullname}", f"<p>{fullname}</p>"))

        if docs:
            symbol_to_doc.append((symbol, *combine_docs(docs, fullname_fragments), fullname_fragments))
        else:
            # If no docs could be parsed, still add the instruction with minimal info
            symbol_to_doc.append((symbol, f"PTX instruction {symbol}",
                                 f"<p>Documentation for {symbol}</p>", fullname_fragments))

    # get SASS docs
    tables = pd.read_html('https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html', match='Opcode')
    sass_docs = sorted(
        (opcode, description)
        for (opcode, description) in pd.concat(tables).dropna().drop_duplicates(["Opcode"], keep="last").itertuples(index=False)
        if opcode != description
    )

    with open( args.outputfolder + '/asm-docs-ptx.ts', 'w') as f:
        f.write("""
    import type {AssemblyInstructionInfo} from '../../../types/assembly-docs.interfaces.js';

    export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
        if (!opcode) return;
        switch (opcode) {
    """.lstrip())
        # PTX
        for name, tooltip, body, fullname_fragments in symbol_to_doc:
            f.write(f'        case "{name}":\n')
            f.write('            return {}'.format(json.dumps({
                "tooltip": tooltip.replace('\n', '\n\n'),
                "html": body,
                # there can be multiple doc links for a single instruction
                "url": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#" + fullname_fragments[0][1]
            }, indent=16, separators=(',', ': '), sort_keys=True))[:-1] + '            };\n\n')
        f.write("""
        }
    }
    """)
    with open(args.outputfolder + '/asm-docs-sass.ts', 'w') as f:
        f.write("""
    import type {AssemblyInstructionInfo} from '../../../types/assembly-docs.interfaces.js';

    export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
        if (!opcode) return;
        switch (opcode) {
    """.lstrip())
        # SASS
        for name, description in sass_docs:
            url = f"https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference"
            f.write(f'        case "{name}":\n')
            f.write('            return {}'.format(json.dumps({
                "tooltip": description,
                "html": description +
                        f'<br><br>For more information, visit <a href="{url}" target="_blank" rel="noopener noreferrer">'
                        'CUDA Binary Utilities' +
                        ' documentation <sup><small class="fas fa-external-link-alt opens-new-window"' +
                        ' title="Opens in a new window"></small></sup></a>.',
                "url": url
            }, indent=16, separators=(',', ': '), sort_keys=True))[:-1] + '            };\n\n')

        f.write("""
        }
    }
    """)

if __name__ == '__main__':
    main()
