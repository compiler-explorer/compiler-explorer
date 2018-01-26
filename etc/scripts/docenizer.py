#! /usr/bin/env python2
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import shutil
import urllib
import zipfile

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError("Please install BeautifulSoup (apt-get install python-bs4 should do it)")

parser = argparse.ArgumentParser(description='Docenizes HTML version of the official Intel Asm PDFs')
parser.add_argument('-i', '--inputfolder', type=str,
                    help='Folder where the input files reside as .html. Default is current folder')
parser.add_argument('-o', '--outputpath', type=str, help='Final path of the .js file. Default is ./asm-docs.js',
                    default='./asm-docs.js')

# The maximum number of paragraphs from the description to copy.
MAX_DESC_PARAS = 5
STRIP_PREFIX = re.compile(r'^(([0-9a-fA-F]{2}|(REX|VEX\.)[.0-9A-Z]*|/.|[a-z]+)\b\s*)*')
INSTRUCTION_RE = re.compile(r'^([A-Z][A-Z0-9]+)\*?(\s+|$)')
# Some instructions are so broken we just take their names from the filename
UNPARSEABLE_INSTR_NAMES = ['PSRLW:PSRLD:PSRLQ', 'PSLLW:PSLLD:PSLLQ']
# Some instructions are defined in multiple files. We ignore a specific set of the
# duplicates here.
IGNORED_DUPLICATES = [
    'MOV-1',  # move to control reg
    'MOV-2',  # move to debug reg
    'CMPSD',  # compare doubleword (defined in CMPS:CMPSB:CMPSW:CMPSD:CMPSQ)
    'MOVQ',  # defined in MOVD:MOVQ
    'MOVSD'  # defined in MOVS:MOVSB:MOVSW:MOVSD:MOVSQ
]
# Where to extract the asmdoc archive.
ASMDOC_DIR = "asm-docs"


class Instruction(object):
    def __init__(self, name, names, tooltip, body):
        self.name = name
        self.url = "http://www.felixcloutier.com/x86/{}.html".format(urllib.quote(name))
        self.names = names
        self.tooltip = tooltip.rstrip(': ,')
        self.body = body

    def __str__(self):
        return "{} = {}\n{}".format(self.names, self.tooltip, self.body)

def download_asm_doc_archive():
    print("Downloading archive...")
    urllib.urlretrieve("http://www.felixcloutier.com/x86/x86.zip", "x86.zip")
    if os.path.isdir(ASMDOC_DIR):
        shutil.rmtree(ASMDOC_DIR)
    zip_ref = zipfile.ZipFile("x86.zip", 'r')
    zip_ref.extractall(ASMDOC_DIR)
    zip_ref.close()
    shutil.rmtree(os.path.join(ASMDOC_DIR, "__MACOSX"));


def strip_non_instr(i):
    # removes junk from encodings where the opcode is in the middle
    # of prefix stuff. e.g.
    # 66 0f 38 30 /r PMOVZXBW xmm1, xmm2/m64
    return STRIP_PREFIX.sub('', i)


def instr_name(i):
    match = INSTRUCTION_RE.match(strip_non_instr(i))
    if match:
        return match.group(1)


def get_description(section):
    for sub in section:
        descr = sub.get_text().strip()
        if len(descr) > 20:
            return descr
    raise RuntimeError("Couldn't find decent description in {}".format(section))


def parse(filename, f):
    doc = BeautifulSoup(f, 'html.parser')
    table = read_table(doc.table)
    names = set()

    def add_all(instrs):
        for i in instrs:
            instruction_name = instr_name(i)
            if instruction_name:
                names.add(instruction_name)

    for inst in table:
        if 'Opcode/Instruction' in inst:
            add_all(inst['Opcode/Instruction'].split("\n"))
        elif 'Opcode*/Instruction' in inst:
            add_all(inst['Opcode*/Instruction'].split("\n"))
        else:
            instruction_name = instr_name(inst['Instruction'])
            if not instruction_name:
                print "Unable to get instruction from:", inst['Instruction']
            else:
                names.add(instruction_name)
    if not names:
        if filename in UNPARSEABLE_INSTR_NAMES:
            for inst in filename.split(":"):
                names.add(inst)
        else:
            return None
    sections = {}
    for section_header in doc.find_all("h2"):
        children = []
        first = section_header.next_sibling
        while first and first.name != 'h2':
            if str(first).strip():
                children.append(first)
            first = first.next_sibling
        sections[section_header.text] = children
    return Instruction(
        filename,
        names,
        get_description(sections['Description']),
        "".join(str(x) for x in sections['Description'][:MAX_DESC_PARAS]).strip())


def read_table(table):
    headers = [h.get_text() for h in table.find_all('th')]
    result = []
    if headers:
        # common case
        for row in table.find_all('tr'):
            obj = {}
            for column, name in zip(row.find_all('td'), headers):
                obj[name] = column.get_text()
            if obj:
                result.append(obj)
    else:
        # Cases like BEXTR and BZHI
        rows = table.find_all('tr')
        if len(rows) != 1:
            return []
        obj = {}
        for td in rows[0].find_all('td'):
            header = td.p.strong.get_text()
            td.p.strong.decompose()
            obj[header] = td.get_text()
        result.append(obj)

    return result


def parse_html(directory):
    instructions = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".html") and file != 'index.html':
                with open(os.path.join(root, file)) as f2:
                    name = os.path.splitext(file)[0]
                    if name in IGNORED_DUPLICATES:
                        continue
                    instruction = parse(name, f2)
                    if not instruction:
                        print "Unable to get instructions for " + file
                        continue
                    instructions.append(instruction)
    return instructions


def self_test(instructions, directory):
    directory = os.path.join(directory, "html")
    for inst in instructions:
        if not os.path.isfile(os.path.join(directory, inst.name + ".html")):
            print("Warning: {} has not file associated".format(inst.name))

if __name__ == '__main__':
    args = parser.parse_args()
    if args.inputfolder is None:
        download_asm_doc_archive()
        args.inputfolder = ASMDOC_DIR
    instructions = parse_html(args.inputfolder)
    instructions.sort(lambda x, y: cmp(x.name, y.name))
    self_test(instructions, args.inputfolder)
    all_inst = set()
    for inst in instructions:
        if not all_inst.isdisjoint(inst.names):
            print "Overlap in instruction names: {} for {}".format(
                inst.names.intersection(all_inst), inst.name)
        all_inst = all_inst.union(inst.names)

    with open(args.outputpath, 'w') as f:
        f.write("""
function getAsmOpcode(opcode) {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
""")
        for inst in instructions:
            for name in inst.names:
                f.write('        case "{}":\n'.format(name))
            f.write('            return {}'.format(json.dumps({
                "tooltip": inst.tooltip,
                "html": inst.body,
                "url": inst.url
                }, indent=16, separators=(',', ': ')))[:-1] + '            };\n\n')
        f.write("""
    }
}

module.exports = {
    getAsmOpcode: getAsmOpcode
};
""")
