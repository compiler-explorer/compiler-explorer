import argparse
import collections
import json
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import requests
from bs4 import BeautifulSoup

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
    # For tooltip, prefer the shortest meaningful description
    tooltip_txt = None
    for doc in docs:
        if doc.text and len(doc.text) > 10 and len(doc.text) < 200:
            # Prefer short, descriptive text
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
        combined_html = '\n'.join(html_parts) + '\n'

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

    # Find all sections with instruction documentation structure
    all_sections = soup.find_all(['section', 'div'], id=lambda x: x and '-instructions-' in x)

    # Build priority map based on section hierarchy
    instruction_priority = {}

    for section in all_sections:
        section_id = section.get('id', '')

        # Check if this section has the standard instruction documentation structure
        rubrics = section.find_all('p', class_='rubric')
        rubric_texts = [r.get_text(strip=True) for r in rubrics]

        # Only process sections with proper instruction documentation structure
        if not any(r in rubric_texts for r in ['Syntax', 'Description', 'Semantics', 'Examples']):
            continue

        # Extract instruction name from section ID (last segment after final dash)
        if '-' not in section_id:
            continue

        instruction_name = section_id.rsplit('-', 1)[-1]

        # Determine priority based on section type
        priority = 0
        if 'integer-arithmetic-instructions' in section_id:
            priority = 10
        elif 'floating-point-instructions' in section_id:
            priority = 9
        elif 'half-precision-floating-point-instructions' in section_id:
            priority = 8
        elif 'extended-precision-arithmetic-instructions' in section_id:
            priority = 7
        elif 'comparison-and-selection-instructions' in section_id:
            priority = 6
        elif 'logical-instructions' in section_id:
            priority = 5
        elif 'data-movement-and-conversion-instructions' in section_id:
            priority = 4
        elif 'texture-instructions' in section_id:
            priority = 3
        elif 'surface-instructions' in section_id:
            priority = 2
        else:
            priority = 1

        # Find the first code element in this section for fullname
        first_code = section.find('code')
        if first_code:
            fullname = first_code.get_text(strip=True).lstrip('.').lstrip('%')
        else:
            fullname = instruction_name

        # Store or update if higher priority
        if instruction_name not in instruction_priority or instruction_priority[instruction_name][1] < priority:
            instruction_priority[instruction_name] = ((fullname, section_id), priority)

    # Build the final mapping from the priority-filtered results
    for instruction_name, ((fullname, section_id), _) in instruction_priority.items():
        symbol_to_fullname_frag0[instruction_name].append((fullname, section_id))

    # Also check for instruction variants with dots (e.g., add.s32)
    for section in all_sections:
        section_id = section.get('id', '')

        # Look for code elements that might be instruction variants
        codes = section.find_all('code')
        for code in codes[:5]:  # Check first few codes only
            text = code.get_text(strip=True)
            if '.' in text and len(text) < 40:
                base = text.split('.')[0].lstrip('%@')
                if base and base.isalpha() and len(base) < 20:
                    # Check if this base is already in our instruction map
                    if base in instruction_priority:
                        fullname = text.lstrip('.').lstrip('%')
                        # Add variant
                        symbol_to_fullname_frag0[base].append((fullname, section_id))

    # Remove duplicates and sort
    symbol_to_fullname_frag: list[tuple[str, list[tuple[str, str]]]] = sorted(
        (instr, sorted(set(fullname_frags))) for instr, fullname_frags in symbol_to_fullname_frag0.items()
        if fullname_frags)  # Only include if we have fragments

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

        # First, try to find the short description that appears between instruction name and Syntax
        short_desc = None
        instruction_rubric = article.find('p', class_='rubric')  # First rubric is usually instruction name
        if instruction_rubric:
            # Look for the next paragraph after instruction name
            next_p = instruction_rubric.find_next_sibling('p')
            if next_p and 'rubric' not in next_p.get('class', []):
                # This is likely the short description
                short_desc_text = next_p.get_text(strip=True)
                if short_desc_text and len(short_desc_text) < 100:  # Short descriptions are typically brief
                    short_desc = short_desc_text

        # Find Description rubric to locate the detailed description structurally
        desc_rubric = article.find('p', class_='rubric', string='Description')

        html = None
        txt = None

        if desc_rubric:
            # The actual description is in the paragraph following the Description rubric
            next_elem = desc_rubric.find_next_sibling('p')
            if next_elem and next_elem.get_text(strip=True):
                # Skip if this is another rubric
                if 'rubric' not in next_elem.get('class', []):
                    html = str(next_elem)
                    # Use short description for tooltip if available, otherwise use detailed description
                    txt = short_desc if short_desc else next_elem.get_text(strip=True)
                    # Truncate for tooltip if too long
                    if len(txt) > 200:
                        txt = txt[:200] + "..."

        # Fallback: look for any non-rubric paragraph with meaningful content
        if not html:
            for p in article.find_all('p'):
                # Skip rubric paragraphs
                if 'rubric' in p.get('class', []):
                    continue
                p_text = p.get_text(strip=True)
                # Look for paragraphs with actual content
                if p_text and len(p_text) > 10:
                    html = str(p)
                    # Still prefer short description for tooltip if we found one
                    txt = short_desc if short_desc else p_text[:200]
                    if not short_desc and len(p_text) > 200:
                        txt += "..."
                    break

        # Last resort - use title
        if not html:
            html = f'<p>{title}</p>'
            txt = short_desc if short_desc else title

        return Doc(title, txt, html)

    symbol_to_doc: list[tuple[str, str, str, list[tuple[str, str]]]] = []
    for symbol, fullname_fragments in symbol_to_fullname_frag:
        docs = []
        for fullname, fragment in fullname_fragments:
            try:
                docs.append(get_doc(fragment))
            except Exception:
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
            url = "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference"
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
