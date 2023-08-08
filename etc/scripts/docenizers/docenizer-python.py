#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import urllib
from urllib import request
from urllib import parse

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError(
        "Please install BeautifulSoup (apt-get install python3-bs4 or pip install beautifulsoup4 should do it)")

parser = argparse.ArgumentParser(description='Docenizes HTML version of the official Python documentation')
parser.add_argument('-i', '--inputfolder', type=str,
                    help='Folder where the input files reside as .html. Default is ./python-inst-docs/',
                    default='python-inst-docs')
parser.add_argument('-o', '--outputpath', type=str, help='Final path of the .ts file. Default is ./asm-docs-python.ts',
                    default='./asm-docs-python.ts')
parser.add_argument('-d', '--downloadfolder', type=str,
                    help='Folder where the archive will be downloaded and extracted', default='python-inst-docs')

# The maximum number of paragraphs from the description to copy.
MAX_DESC_PARAS = 5

# Where to extract the asmdoc archive.
ARCHIVE_URL = "https://docs.python.org/3/library/dis.html"
ARCHIVE_NAME = "dis.html"


class Instruction(object):
    def __init__(self, name, names, tooltip, body):
        self.name = name
        self.names = names
        self.tooltip = tooltip.rstrip(': ,')
        self.body = body

    def __str__(self):
        return f"{self.name} = {self.tooltip}\n{self.body}"


def get_url_for_instruction(instr):
    return f"https://docs.python.org/3/library/dis.html#opcode-{urllib.parse.quote(instr.name)}"


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


def get_description_paragraphs(opcode):
    ps = opcode.find('dd').findAll('p')
    return [p.text for p in ps]


def parse(f):
    doc = BeautifulSoup(f, 'html.parser')
    table = doc.find('section', {'id': 'python-bytecode-instructions'})

    opcodes = table.findAll('dl', {'class': 'std opcode'})
    instructions = []
    for opcode in opcodes:
        opcode_name = opcode.find('span', {'class': 'pre'}).text
        opcode_desc = get_description_paragraphs(opcode)
        instructions.append(Instruction(
            opcode_name,
            [opcode_name],
            opcode_desc[0],
            '\n'.join(opcode_desc))
        )
    return instructions


def parse_html(directory):
    print("Parsing instructions...")
    instructions = []
    try:
        with open(os.path.join(directory, ARCHIVE_NAME), encoding='utf-8') as f:
            instructions = parse(f)
    except Exception as e:
        print(f"Error parsing {ARCHIVE_NAME}:\n{e}")

    return instructions


def main():
    args = parser.parse_args()
    print(f"Called with: {args}")
    # If we don't have the html folder already...
    if not os.path.isdir(os.path.join(args.inputfolder, 'html')):
        try:
            download_asm_doc_archive(args.downloadfolder)
        except IOError as e:
            print("Error when downloading archive:")
            print(e)
            sys.exit(1)
    instructions = parse_html(args.inputfolder)
    instructions.sort(key=lambda b: b.name)
    all_inst = set()
    for inst in instructions:
        if not all_inst.isdisjoint(inst.names):
            print(f"Overlap in instruction names: {inst.names.intersection(all_inst)} for {inst.name}")
        all_inst = all_inst.union(inst.names)
    print(f"Writing {len(instructions)} instructions")
    with open(args.outputpath, 'w') as f:
        f.write("""
import {AssemblyInstructionInfo} from '../base.js';

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
