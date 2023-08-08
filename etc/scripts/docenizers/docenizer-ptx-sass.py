import collections
import copy
import json
import os.path
from collections import defaultdict

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass

try:
    import lxml  # for pandas.read_html
except ImportError:
    raise
# get PTX docs
r = requests.get('https://docs.nvidia.com/cuda/parallel-thread-execution/index.html')
r.encoding = 'utf-8'
soup = BeautifulSoup(r.text, 'html.parser')
symbol_to_fullname_frag0: defaultdict[str, list[tuple[str, str]]] = collections.defaultdict(list)
for links, anchor_text_sep in [
    (soup.find('a', class_='reference internal', href='#instruction-set').parent.find_all('a'), 'Instructions: '),
    (soup.find('a', class_='reference internal', href='#directives').parent.find_all('a'), 'Directives: '),
    (soup.find('a', class_='reference internal', href='#special-registers').parent.find_all('a'),
     'Special Registers: ')]:
    for link in links:
        if anchor_text_sep in link.string:
            topic, instructions0 = link.string.split(anchor_text_sep)
            instructions: list[str] = instructions0.replace(' / ', ', ').split(', ')
            href: str = link.get('href')
            assert href.startswith('#')
            frag = href[1:]
            for instr in instructions:
                if not instr.startswith('@') and ' ' not in instr and instr != '{}':
                    instr_fullname = instr.lstrip('.').lstrip('%')
                    symbol_to_fullname_frag0[instr_fullname.split('.', 1)[0]].append((instr_fullname, frag))
symbol_to_fullname_frag: list[tuple[str, list[tuple[str, str]]]] = sorted(
    (instr, sorted(set(fullname_frags))) for instr, fullname_frags in symbol_to_fullname_frag0.items())


@dataclass
class Doc:
    title: str
    text: str
    html: str


def get_doc(fragment: str) -> Doc:
    article = copy.copy(soup.find(id=fragment))
    txt0 = article.text.replace('\n\n', '\n').replace('\n\n', '\n')
    title, _, txt = txt0.split(' ', 1)[1].split('\n', 2)
    for i in range(2, 6):
        if h := article.find(f'h{i}'):
            h.decompose()
            break
    else:
        print("=====")
        print(article)
        raise AssertionError
    article.p.decompose()  # remove the instruction name
    return Doc(title.rstrip(''), txt, str(article))


def fullname_plus_annotation(fullname: str, fragment: str) -> str:
    if fragment.startswith('floating-point-instructions-'):
        return fullname + '(fp)'
    if fragment.startswith('half-precision-floating-point-instructions-'):
        return fullname + '(fp16)'
    if fragment.startswith('integer-arithmetic-instructions-'):
        return fullname + '(int)'
    return fullname


def combine_docs(docs: list[Doc], fullname_fragments) -> tuple[str, str]:
    common_txt = os.path.commonprefix([doc.text for doc in docs])
    combined_txt = common_txt \
        if len(common_txt) > 100 else \
        '\n'.join(f'====={doc.title}\n\n' + doc.text[:400] + '...' for doc in docs)
    links = ['<a href="https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#' +
             f'{fragment}" target="_blank" rel="noopener noreferrer">{fullname_plus_annotation(fullname, fragment)}' +
             ' <sup><small class="fas fa-external-link-alt opens-new-window"' +
             ' title="Opens in a new window"></small></sup></a>'
             for fullname, fragment in fullname_fragments]
    return combined_txt[:4000] + ' ...', \
           'For more information, visit ' + ', '.join(links) + '.' + \
           '\n'.join([f'<h1>{doc.title}</h1>' + doc.html for doc in docs])


symbol_to_doc: list[tuple[str, str, str, list[tuple[str, str]]]] = []
for symbol, fullname_fragments in symbol_to_fullname_frag:
    docs = [get_doc(fragment) for _, fragment in fullname_fragments]
    symbol_to_doc.append((symbol, *combine_docs(docs, fullname_fragments), fullname_fragments))
# get SASS docs
tables = pd.read_html('https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html', match='Opcode')
sass_docs = sorted(dict(pd.concat(tables).dropna().itertuples(index=False)).items())

with open('./asm-docs-ptx.ts', 'w') as f:
    f.write("""
import {AssemblyInstructionInfo} from '../base.js';

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
with open('./asm-docs-sass.ts', 'w') as f:
    f.write("""
import {AssemblyInstructionInfo} from '../base.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode) {
""".lstrip())
    # SASS
    for name, description in sass_docs:
        f.write(f'        case "{name}":\n')
        f.write('            return {}'.format(json.dumps({
            "tooltip": description,
            "html": description +
                    f'<br><br>For more information, visit <a href="https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14" target="_blank" rel="noopener noreferrer">'
                    'CUDA Binary Utilities' +
                    ' documentation <sup><small class="fas fa-external-link-alt opens-new-window"' +
                    ' title="Opens in a new window"></small></sup></a>.',
            "url": "https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#id14"
        }, indent=16, separators=(',', ': '), sort_keys=True))[:-1] + '            };\n\n')

    f.write("""
    }
}
""")
