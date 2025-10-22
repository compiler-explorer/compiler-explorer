import argparse
import collections
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Constants for documentation structure
DOC_RUBRIC_TYPES = ['Syntax', 'Description', 'Semantics', 'Examples']
PTX_DOCS_BASE_URL = 'https://docs.nvidia.com/cuda/parallel-thread-execution/index.html'
CUDA_BINARY_UTILS_URL = 'https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html'

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


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to max_length characters, adding ellipsis if truncated."""
    if len(text) > max_length:
        return f"{text[:max_length]}..."
    return text



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
        tooltip_txt = truncate_text(docs[0].text, 200)

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
    links = [f'<a href="{PTX_DOCS_BASE_URL}#{fragment}" target="_blank" rel="noopener noreferrer">'
             f'{fullname_plus_annotation(fullname, fragment)}'
             f' <sup><small class="fas fa-external-link-alt opens-new-window"'
             f' title="Opens in a new window"></small></sup></a>'
             for fullname, fragment in fullname_fragments]

    # Put the actual documentation first, then the links
    combined_html = ''
    if html_parts:
        combined_html = f"{'\n'.join(html_parts)}\n"

    combined_html += f'For more information, visit {", ".join(links)}.'

    return tooltip_txt, combined_html


def main():
    args = parser.parse_args()
    print(f"Called with: {args}")

    # get PTX docs
    r = requests.get(PTX_DOCS_BASE_URL, timeout=30)
    r.encoding = 'utf-8'
    soup = BeautifulSoup(r.text, 'html.parser')

    symbol_to_fullname_frag0: defaultdict[str, list[tuple[str, str]]] = collections.defaultdict(list)

    # Find all sections with instruction documentation structure
    all_sections = soup.find_all(['section', 'div'], id=lambda x: x and '-instructions-' in x)

    # Helper to check if a section is a valid instruction section
    def is_valid_instruction_section(section_id):
        if '-' not in section_id:
            return False
        last_part = section_id.rsplit('-', 1)[-1]
        # Skip non-instruction sections and composite instructions
        if last_part in ['restrictions', 'mechanisms', 'smem', 'notes'] or '-' in last_part:
            return False
        # Also skip sections that end with "mechanisms-mbarrier" etc
        if 'mechanisms' in section_id or 'restrictions' in section_id:
            return False
        return True

    # Define section preference order (most preferred first)
    SECTION_PREFERENCE = [
        'integer-arithmetic-instructions',
        'floating-point-instructions',
        'half-precision-floating-point-instructions',
        'extended-precision-arithmetic-instructions',
        'comparison-and-selection-instructions',
        'logical-instructions',
        'data-movement-and-conversion-instructions',
        'texture-instructions',
        'surface-instructions',
    ]

    # Track which instructions we've already documented
    seen_instructions = {}

    # First, group sections by their preference order
    sections_by_preference = {pref: [] for pref in SECTION_PREFERENCE}
    sections_other = []

    for section in all_sections:
        section_id = section.get('id', '')

        if not is_valid_instruction_section(section_id):
            continue

        # Check if this section has the standard instruction documentation structure
        rubrics = section.find_all('p', class_='rubric')
        rubric_texts = [r.get_text(strip=True) for r in rubrics]

        # Only process sections with proper instruction documentation structure
        if not any(r in rubric_texts for r in DOC_RUBRIC_TYPES):
            continue

        # Categorize by preference
        categorized = False
        for pref in SECTION_PREFERENCE:
            if pref in section_id:
                sections_by_preference[pref].append(section)
                categorized = True
                break
        if not categorized:
            sections_other.append(section)

    # Process sections in preference order
    for pref in SECTION_PREFERENCE:
        for section in sections_by_preference[pref]:
            section_id = section.get('id', '')

            # Get actual instruction name from first code element
            first_code = section.find('code')
            if not first_code:
                continue

            fullname = first_code.get_text(strip=True).lstrip('.').lstrip('%')
            # Extract base instruction name (before first dot or space)
            base_instruction = fullname.split('.')[0].split()[0]

            # Only add if we haven't seen this base instruction before
            if base_instruction not in seen_instructions:
                seen_instructions[base_instruction] = (fullname, section_id)
                symbol_to_fullname_frag0[base_instruction].append((fullname, section_id))

    # Process remaining sections
    for section in sections_other:
        section_id = section.get('id', '')
        first_code = section.find('code')
        if not first_code:
            continue

        fullname = first_code.get_text(strip=True).lstrip('.').lstrip('%')
        base_instruction = fullname.split('.')[0].split()[0]

        if base_instruction not in seen_instructions:
            seen_instructions[base_instruction] = (fullname, section_id)
            symbol_to_fullname_frag0[base_instruction].append((fullname, section_id))

    # Collect instruction variants (e.g., add.s32, add.f32)
    for section in all_sections:
        section_id = section.get('id', '')

        if not is_valid_instruction_section(section_id):
            continue

        # Check structure
        rubrics = section.find_all('p', class_='rubric')
        rubric_texts = [r.get_text(strip=True) for r in rubrics]
        if not any(r in rubric_texts for r in DOC_RUBRIC_TYPES):
            continue

        # Look for instruction variants in syntax section
        syntax_rubric = section.find('p', class_='rubric', string='Syntax')
        if syntax_rubric:
            # Find the syntax block (usually next sibling)
            next_elem = syntax_rubric.find_next_sibling()
            if next_elem:
                # Look for code elements that are instruction variants
                codes = next_elem.find_all('code') if next_elem.name == 'div' else section.find_all('code')[:10]
                for code in codes:
                    text = code.get_text(strip=True).lstrip('.').lstrip('%')
                    # Check if this looks like an instruction (not an operand)
                    if text and not text.startswith(('d,', 'a,', 'b,', 'c,')) and len(text) < 40:
                        parts = text.split()
                        if parts:  # Has content
                            instruction = parts[0]
                            if '.' in instruction:  # It's a variant
                                base = instruction.split('.')[0]
                                # Only add variants for instructions we're documenting
                                if base in seen_instructions and (instruction, section_id) not in symbol_to_fullname_frag0[base]:
                                    symbol_to_fullname_frag0[base].append((instruction, section_id))

    # Remove duplicates and sort
    symbol_to_fullname_frag: list[tuple[str, list[tuple[str, str]]]] = sorted(
        (instr, sorted(set(fullname_frags))) for instr, fullname_frags in symbol_to_fullname_frag0.items()
        if fullname_frags)  # Only include if we have fragments

    # Fail loudly if no instructions were found
    if not symbol_to_fullname_frag:
        print("ERROR: No PTX instructions found in documentation!")
        print("The website structure may have changed.")
        print("Please update the docenizer-ptx-sass.py script to handle the new format.")
        sys.exit(1)

    def get_doc(fragment: str) -> Doc:
        # Find the section
        article = soup.find(id=fragment)
        if not article:
            raise ValueError(f"Documentation section not found for fragment: {fragment}")

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
            next_p = instruction_rubric.find_next_sibling('p')
            if next_p and 'rubric' not in next_p.get('class', []):
                short_desc_text = next_p.get_text(strip=True)
                if short_desc_text and len(short_desc_text) < 100:  # Short descriptions are typically brief
                    short_desc = short_desc_text

        # Get description from Description rubric (all PTX instructions have this)
        desc_rubric = article.find('p', class_='rubric', string='Description')
        if not desc_rubric:
            raise ValueError(f"No Description rubric found for {fragment}")

        # Description can be in either a <p> or <dl> element
        next_elem = desc_rubric.find_next_sibling(['p', 'dl'])
        if not next_elem:
            raise ValueError(f"No description element found after Description rubric for {fragment}")

        # If it's a <p>, check it's not another rubric
        if next_elem.name == 'p' and 'rubric' in next_elem.get('class', []):
            raise ValueError(f"No description content found after Description rubric for {fragment}")

        if not next_elem.get_text(strip=True):
            raise ValueError(f"Empty description after Description rubric for {fragment}")

        html = str(next_elem)

        # Tooltip text: Use short description if available, otherwise truncate detailed description.
        # This fallback is necessary because some PTX instructions don't have short summaries -
        # they jump straight to Syntax or have descriptions > 100 chars (e.g., mad, mov, set).
        if short_desc:
            txt = short_desc
        else:
            detailed_desc = next_elem.get_text(strip=True)
            txt = truncate_text(detailed_desc, 200)

        return Doc(title, txt, html)

    symbol_to_doc: list[tuple[str, str, str, list[tuple[str, str]]]] = []
    for symbol, fullname_fragments in symbol_to_fullname_frag:
        docs = []
        for fullname, fragment in fullname_fragments:
            docs.append(get_doc(fragment))

        if docs:
            symbol_to_doc.append((symbol, *combine_docs(docs, fullname_fragments), fullname_fragments))
        else:
            raise ValueError(f"No documentation could be parsed for instruction: {symbol}")

    # get SASS docs
    tables = pd.read_html(CUDA_BINARY_UTILS_URL, match='Opcode')
    sass_docs = sorted(
        (opcode, description)
        for (opcode, description) in pd.concat(tables).dropna().drop_duplicates(["Opcode"], keep="last").itertuples(index=False)
        if opcode != description
    )

    output_dir = Path(args.outputfolder)
    with (output_dir / 'asm-docs-ptx.ts').open('w') as f:
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
                "url": f"{PTX_DOCS_BASE_URL}#{fullname_fragments[0][1]}"
            }, indent=16, separators=(',', ': '), sort_keys=True))[:-1] + '            };\n\n')
        f.write("""
        }
    }
    """)
    with (output_dir / 'asm-docs-sass.ts').open('w') as f:
        f.write("""
    import type {AssemblyInstructionInfo} from '../../../types/assembly-docs.interfaces.js';

    export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
        if (!opcode) return;
        switch (opcode) {
    """.lstrip())
        # SASS
        for name, description in sass_docs:
            url = f"{CUDA_BINARY_UTILS_URL}#instruction-set-reference"
            f.write(f'        case "{name}":\n')
            f.write('            return {}'.format(json.dumps({
                "tooltip": description,
                "html": f'{description}<br><br>For more information, visit <a href="{url}" target="_blank" rel="noopener noreferrer">'
                        f'CUDA Binary Utilities'
                        f' documentation <sup><small class="fas fa-external-link-alt opens-new-window"'
                        f' title="Opens in a new window"></small></sup></a>.',
                "url": url
            }, indent=16, separators=(',', ': '), sort_keys=True))[:-1] + '            };\n\n')

        f.write("""
        }
    }
    """)

if __name__ == '__main__':
    main()
