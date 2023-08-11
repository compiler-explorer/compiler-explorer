#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import urllib
import re
from urllib import request
from urllib import parse

parser = argparse.ArgumentParser(description='Docenizes the EVM documentation')
parser.add_argument('-i', '--inputfolder', type=str,
                    help='Folder where the input files reside as .html. Default is ./evm-inst-docs/',
                    default='evm-inst-docs')
parser.add_argument('-o', '--outputpath', type=str, help='Final path of the .ts file. Default is ./asm-docs-evm.ts',
                    default='./asm-docs-evm.ts')
parser.add_argument('-d', '--downloadfolder', type=str,
                    help='Folder where the archive will be downloaded and extracted', default='evm-inst-docs')

# | `0x00` | STOP | Halts execution | - | 0 |
MNEMONIC_RE = re.compile('^\| `0x([A-Za-z0-9]+)` \| (.*) \| .* \| .* \| .* \|$')

# Where to extract the asmdoc archive.
ARCHIVE_DESC_URL = "https://raw.githubusercontent.com/comitylabs/evm.codes/main/opcodes.json"
ARCHIVE_DESC_NAME = "opcodes.json"
ARCHIVE_MNEM_URL = "https://raw.githubusercontent.com/crytic/evm-opcodes/master/README.md"
ARCHIVE_MNEM_NAME = "README.md"


class Instruction(object):
    def __init__(self, opcode, mnemonic, tooltip, body):
        self.opcode = opcode
        self.mnemonic = mnemonic
        self.tooltip = tooltip.rstrip(': ,')
        self.body = body

    def __str__(self):
        return f"{self.opcode} = {self.tooltip}\n{self.body}"


def get_url_for_instruction(instr):
    return f"https://www.evm.codes/#{urllib.parse.quote(instr.opcode)}"


def download_asm_doc_archive(downloadfolder):
    if not os.path.exists(downloadfolder):
        print(f"Creating {downloadfolder} as download folder")
        os.makedirs(downloadfolder)
    elif not os.path.isdir(downloadfolder):
        print(f"Error: download folder {downloadfolder} is not a directory")
        sys.exit(1)
    archive_desc_name = os.path.join(downloadfolder, ARCHIVE_DESC_NAME)
    print("Downloading archive...")
    urllib.request.urlretrieve(ARCHIVE_DESC_URL, archive_desc_name)

    archive_mnem_name = os.path.join(downloadfolder, ARCHIVE_MNEM_NAME)
    print("Downloading archive...")
    urllib.request.urlretrieve(ARCHIVE_MNEM_URL, archive_mnem_name)


def get_description_paragraphs(opcode):
    stack_input = 'Input: ' + (f'<code>{opcode["input"]}</code>' if opcode["input"] != "" else '-')
    stack_output = 'Output: ' + (f'<code>{opcode["output"]}</code>' if opcode["output"] != "" else '-')
    return [opcode["description"], stack_input, stack_output]


def generate_opcode_mnemonic_map(mnemonic_file):
    mnemonic_map = {}
    for line in mnemonic_file:
        match = MNEMONIC_RE.match(line)
        if match:
            mnemonic_map[match.group(1)] = match.group(2)
    return mnemonic_map


def is_valid_opcode(opcode, mnemonic_map):
    return opcode in mnemonic_map

def parse(descriptions_file, mnemonic_file):
    descriptions = json.load(descriptions_file)
    mnemonic_map = generate_opcode_mnemonic_map(mnemonic_file)
    print(mnemonic_map)
    opcodes = descriptions.items()
    instructions = []
    for opcode, body in opcodes:
        if is_valid_opcode(opcode, mnemonic_map):
            mnemonic = mnemonic_map[opcode]
            opcode_desc = get_description_paragraphs(body)
            instructions.append(Instruction(
                opcode,
                mnemonic,
                opcode_desc[0],
                '\n'.join(opcode_desc))
            )
    return instructions


def parse_html(directory):
    print("Parsing instructions...")
    instructions = []
    try:
        with open(os.path.join(directory, ARCHIVE_DESC_NAME), encoding='utf-8') as description_file:
            with open(os.path.join(directory, ARCHIVE_MNEM_NAME), encoding='utf-8') as mnemonic_file:
                instructions = parse(description_file, mnemonic_file)
    except Exception as e:
        print(f"Error parsing files:\n{e}")

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
    instructions.sort(key=lambda b: b.opcode)
    all_inst = set()
    print(f"Writing {len(instructions)} instructions")
    with open(args.outputpath, 'w') as f:
        f.write("""
import {AssemblyInstructionInfo} from '../base.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
""".lstrip())
        for inst in instructions:
            f.write(f'        case "{inst.mnemonic}":\n')
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
