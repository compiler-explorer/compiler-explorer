#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import sys
import tarfile
import urllib
from urllib import request
from urllib import parse

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError("Please install BeautifulSoup (apt-get install python3-bs4 or pip install beautifulsoup4 should do it)")

parser = argparse.ArgumentParser(description='Docenizes HTML version of the official Intel Asm PDFs')
parser.add_argument('-i', '--inputfolder', type=str,
                    help='Folder where the input files reside as .html. Default is ./asm-docs/',
                    default='asm-docs')
parser.add_argument('-o', '--outputpath', type=str, help='Final path of the .ts file. Default is ./asm-docs-amd64.ts',
                    default='./asm-docs-amd64.ts')
parser.add_argument('-d', '--downloadfolder', type=str,
                    help='Folder where the archive will be downloaded and extracted', default='asm-docs')

# The maximum number of paragraphs from the description to copy.
MAX_DESC_PARAS = 5
STRIP_PREFIX = re.compile(r'^(([0-9a-fA-F]{2}|m64|NP|(REX|E?VEX\.)[.0-9A-Z]*|/[0-9a-z]+|[a-z]+)\b\s*)*')
INSTRUCTION_RE = re.compile(r'^([A-Z][A-Z0-9]+)\*?(\s+|$)')
# Some instructions are so broken we just take their names from the filename
UNPARSEABLE_INSTR_NAMES = ['PSRLW:PSRLD:PSRLQ', 'PSLLW:PSLLD:PSLLQ', 'MOVBE']
# Some files contain instructions which cannot be parsed and which compilers are unlikely to emit
IGNORED_FILE_NAMES = [
    # SGX pseudo-instructions
    "EADD",
    "EACCEPT",
    "EAUG",
    "EACCEPTCOPY",
    "EDECVIRTCHILD",
    "EINCVIRTCHILD",
    "EINIT",
    "ELDB:ELDU:ELDBC:ELBUC",
    "EMODPE",
    "EMODPR",
    "EMODT",
    "ERDINFO",
    "ESETCONTEXT",
    "ETRACKC",
    "EBLOCK",
    "ECREATE",
    "EDBGRD",
    "EDBGWR",
    "EENTER",
    "EEXIT",
    "EEXTEND",
    "EGETKEY",
    "ELDB",
    "ELDU",
    "ENCLS",
    "ENCLU",
    "EPA",
    "EREMOVE",
    "EREPORT",
    "ERESUME",
    "ETRACK",
    "EWB",
    # VMX instructions
    "INVEPT",
    "INVVPID",
    "VMCALL",
    "VMCLEAR",
    "VMFUNC",
    "VMLAUNCH",
    "VMLAUNCH:VMRESUME",
    "VMPTRLD",
    "VMPTRST",
    "VMREAD",
    "VMRESUME",
    "VMWRITE",
    "VMXOFF",
    "VMXON",
    # Other instructions
    "INVLPG",
    "LAHF",
    "RDMSR",
    "SGDT",
    # Unparsable instructions
    # These instructions should be supported in the future
    "MONITOR",
    "MOVDQ2Q",
    "MFENCE",
]
# Some instructions are defined in multiple files. We ignore a specific set of the
# duplicates here.
IGNORED_DUPLICATES = [
    'MOV-1',  # move to control reg
    'MOV-2',  # move to debug reg
    'CMPSD',  # compare doubleword (defined in CMPS:CMPSB:CMPSW:CMPSD:CMPSQ)
    'MOVQ',  # defined in MOVD:MOVQ
    'MOVSD',  # defined in MOVS:MOVSB:MOVSW:MOVSD:MOVSQ
    'VPBROADCASTB:VPBROADCASTW:VPBROADCASTD:VPBROADCASTQ',  # defined in VPBROADCAST
    "VGATHERDPS:VGATHERDPD",
    "VGATHERQPS:VGATHERQPD",
    "VPGATHERDD:VPGATHERQD",
    "VPGATHERDQ:VPGATHERQQ",
]
# Where to extract the asmdoc archive.
ASMDOC_DIR = "asm-docs"
ARCHIVE_URL = "https://www.felixcloutier.com/x86/x86.tbz2"
ARCHIVE_NAME = "x86.tbz2"


class Instruction(object):
    def __init__(self, name, names, tooltip, body):
        self.name = name
        self.names = names
        self.tooltip = tooltip.rstrip(': ,')
        self.body = body

    def __str__(self):
        return f"{self.name} = {self.tooltip}\n{self.body}"


def get_url_for_instruction(instr):
    return f"https://www.felixcloutier.com/x86/{urllib.parse.quote(instr.name)}.html"


def download_asm_doc_archive(downloadfolder):
    if not os.path.exists(downloadfolder):
        print(f"Creating {downloadfolder} as download folder")
        os.makedirs(downloadfolder)
    elif not os.path.isdir(downloadfolder):
        print(f"Error: download folder {downloadfolder} is not a directory")
        sys.exit(1)
    archive_name = os.path.join(downloadfolder, ARCHIVE_NAME)
    print("Downloading archive...")
    urllib.request.urlretrieve(ARCHIVE_URL, archive_name)


def extract_asm_doc_archive(downloadfolder, inputfolder):
    print("Extracting file...")
    if os.path.isdir(os.path.join(inputfolder, "html")):
        for root, dirs, files in os.walk(os.path.join(inputfolder, "html")):
            for file in files:
                if os.path.splitext(file)[1] == ".html":
                    os.remove(os.path.join(root, file))
    tar = tarfile.open(os.path.join(downloadfolder, ARCHIVE_NAME))
    tar.extractall(path=inputfolder)


def strip_non_instr(i):
    # removes junk from encodings where the opcode is in the middle
    # of prefix stuff. e.g.
    # 66 0f 38 30 /r PMOVZXBW xmm1, xmm2/m64
    return STRIP_PREFIX.sub('', i)


def instr_name(i):
    match = INSTRUCTION_RE.match(strip_non_instr(i))
    if match:
        return match.group(1)


def get_description_paragraphs(document_soup):
    description_header_node = document_soup.find(id="description")
    i = 0
    description_paragraph_node = description_header_node.next_sibling.next_sibling
    description_paragraphs = []
    while i < MAX_DESC_PARAS and len(description_paragraph_node.text) > 20:
        if description_paragraph_node.name == "p":
            description_paragraphs.append(description_paragraph_node)
            i = i + 1
            # Move two siblings forward. Next sibling is the line feed.
        description_paragraph_node = description_paragraph_node.next_sibling.next_sibling
    return description_paragraphs


def parse(filename, f):
    doc = BeautifulSoup(f, 'html.parser')
    if doc.table is None:
        print(f"{filename}: Failed to find table")
        return None
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
        elif 'OpcodeInstruction' in inst:
            add_all(inst['OpcodeInstruction'].split("\n"))
        elif 'Opcode Instruction' in inst:
            add_all(inst['Opcode Instruction'].split("\n"))
        elif 'Opcode*/Instruction' in inst:
            add_all(inst['Opcode*/Instruction'].split("\n"))
        elif 'Opcode / Instruction' in inst:
            add_all(inst['Opcode / Instruction'].split("\n"))
        elif 'Instruction' in inst:
            instruction_name = instr_name(inst['Instruction'])
            if not instruction_name:
                print(f"Unable to get instruction from: {inst['Instruction']}")
            else:
                names.add(instruction_name)
        # else, skip the line
    if not names:
        if filename in UNPARSEABLE_INSTR_NAMES:
            for inst in filename.split(":"):
                names.add(inst)
        else:
            print(f"{filename}: Failed to read instruction table")
            return None

    description_paragraphs = get_description_paragraphs(doc)

    for para in description_paragraphs:
        for link in para.find_all('a'):
            # this urljoin will only ensure relative urls are prefixed
            # if a url is already absolute it does nothing
            link['href'] = urllib.parse.urljoin('https://www.felixcloutier.com/x86/', link['href'])
            link['target'] = '_blank'
            link['rel'] = 'noreferrer noopener'

    return Instruction(
        filename,
        names,
        description_paragraphs[0].text.strip(),
        ''.join(map(lambda x: str(x), description_paragraphs)).strip())


def read_table(start_table):
    # Tables on felixcloutier may be split in half, e.g. on https://www.felixcloutier.com/x86/sal:sar:shl:shr
    # This traverses the immediate siblings of the input table
    tables = []
    current_node = start_table
    while current_node:
        if current_node.name == 'table':
            tables.append(current_node)
        elif current_node.name is not None: # whitespace between the tables, i.e. the \n, is a none tag
            break
        current_node = current_node.next_sibling
    # Finding all 'th' is not enough, since some headers are 'td'.
    # Instead, walk through all children of the first 'tr', filter out those
    # that are only whitespace, keep `get_text()` on the others.
    headers = list(
        map(lambda th: th.get_text(),
            filter(lambda th: str(th).strip(), tables[0].tr.children)))

    result = []
    if headers:
        # common case
        for table in tables:
            for row in table.find_all('tr'):
                obj = {}
                for column, name in zip(row.find_all('td'), headers):
                    # Remove '\n's in names that contain it.
                    obj[name.replace('\n', '')] = column.get_text()
                if obj:
                    result.append(obj)
    else:
        # Cases like BEXTR and BZHI
        for table in tables:
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
    print("Parsing instructions...")
    instructions = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".html") and file != 'index.html':
                with open(os.path.join(root, file), encoding='utf-8') as f2:
                    name = os.path.splitext(file)[0]
                    if name in IGNORED_DUPLICATES or name in IGNORED_FILE_NAMES:
                        continue
                    try:
                        instruction = parse(name, f2)
                        if not instruction:
                            continue
                        patch_instruction(instruction)
                        instructions.append(instruction)
                    except Exception as e:
                        print(f"Error parsing {name}:\n{e}")
    return instructions


def self_test(instructions, directory):
    # For each generated instruction, check that there is a path to a file in
    # the documentation.
    directory = os.path.join(directory, "html")
    ok = True
    for inst in instructions:
        if not os.path.isfile(os.path.join(directory, inst.name + ".html")):
            print(f"Warning: {inst.name} has not file associated")
            ok = False
    return ok


def patch_instruction(instruction):
    if instruction.name == "ADDSS":
        print("\nPatching ADDSS")
        print("REMINDER: Check if https://github.com/compiler-explorer/compiler-explorer/issues/2380 is still relevant\n")

        old_body = instruction.body
        old_tooltip = instruction.tooltip
        instruction.body = old_body.replace("stores the double-precision", "stores the single-precision")
        instruction.tooltip = old_tooltip.replace("stores the double-precision", "stores the single-precision")


def main():
    args = parser.parse_args()
    print(f"Called with: {args}")
    # If we don't have the html folder already...
    if not os.path.isdir(os.path.join(args.inputfolder, 'html')):
        # We don't, try with the compressed file
        if not os.path.isfile(os.path.join(args.downloadfolder, "x86.tbz2")):
            # We can't find that either. Download it
            try:
                download_asm_doc_archive(args.downloadfolder)
                extract_asm_doc_archive(args.downloadfolder, args.inputfolder)
            except IOError as e:
                print("Error when downloading archive:")
                print(e)
                sys.exit(1)
        else:
            # We have a file already downloaded
            extract_asm_doc_archive(args.downloadfolder, args.inputfolder)
    instructions = parse_html(args.inputfolder)
    instructions.sort(key=lambda b: b.name)
    self_test(instructions, args.inputfolder)
    all_inst = set()
    for inst in instructions:
        if not all_inst.isdisjoint(inst.names):
            print(f"Overlap in instruction names: {inst.names.intersection(all_inst)} for {inst.name}")
        all_inst = all_inst.union(inst.names)
    if not self_test(instructions, args.inputfolder):
        print("Tests do not pass. Not writing output file. Aborting.")
        sys.exit(3)
    print(f"Writing {len(instructions)} instructions")
    with open(args.outputpath, 'w') as f:
        f.write("""
import {AssemblyInstructionInfo} from '../base';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
""".lstrip())
        for inst in instructions:
            for name in sorted(inst.names):
                f.write(f'        case "{name}":\n')
            f.write('            return {}'.format(json.dumps({
                "tooltip": inst.tooltip,
                "html": inst.body,
                "url": get_url_for_instruction(inst)
            }, indent=16, separators=(',', ': '), sort_keys=True))[:-1] + '            };\n\n')
        f.write("""
    }
}
""")


if __name__ == '__main__':
    main()
