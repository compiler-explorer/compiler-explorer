#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import urllib
import re
import html
from urllib import request
from urllib import parse

parser = argparse.ArgumentParser(description='Docenizes based on the opcodes list from the perl source')
parser.add_argument('-i', '--inputfolder', type=str,
                    help='Folder where the input files reside as .html. Default is ./perl-inst-docs/',
                    default='perl-inst-docs')
parser.add_argument('-o', '--outputpath', type=str, help='Final path of the .ts file. Default is ./asm-docs-perl.ts',
                    default='./asm-docs-perl.ts')
parser.add_argument('-d', '--downloadfolder', type=str,
                    help='Folder where the archive will be downloaded and extracted', default='perl-inst-docs')

# this is just a text file with comments, blank lines and opcode lines
# opcode lines are:
# opname<tab>opdesc<tab>ck-function<tab>flags<tab>operands?
# new versions of perl add new opcodes, so get the latest
ARCHIVE_URL = "https://raw.githubusercontent.com/Perl/perl5/refs/heads/blead/regen/opcodes"
ARCHIVE_NAME = "opcodes.txt"

# matches a line from the opcodes file: opname, op description
OPCODE_MATCH = re.compile(r"^(\w+)\t+([^\t]+)\t+")

class Instruction(object):
    def __init__(self, name, names, tooltip, body):
        self.name = name
        self.names = names
        self.tooltip = tooltip.rstrip(': ,')
        self.body = body

    def __str__(self):
        return f"{self.name} = {self.tooltip}\n{self.body}"

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


def parse(f):
    instructions = []
    for line in f:
        match = OPCODE_MATCH.search(line);
        if match:
            opcode_name = match[1]
            opcode_desc = match[2]
            instructions.append(Instruction(
                opcode_name,
                [opcode_name],
                html.escape(opcode_desc),
                '<p>' + html.escape(opcode_desc) + '</p>',
            ))
    return instructions


def parse_opcodes(directory):
    print("Parsing opcodes...")
    instructions = []
    try:
        with open(os.path.join(directory, ARCHIVE_NAME), encoding='utf-8') as f:
            instructions = parse(f)
    except Exception as e:
        print(f"Error parsing {ARCHIVE_NAME}:\n{e}")
        sys.exit(1)

    return instructions


def main():
    args = parser.parse_args()
    print(f"Called with: {args}")
    # If we don't have the html folder already...
    if not os.path.exists(os.path.join(args.inputfolder, ARCHIVE_NAME)):
        try:
            download_asm_doc_archive(args.downloadfolder)
        except IOError as e:
            print("Error when downloading archive:")
            print(e)
            sys.exit(1)
    instructions = parse_opcodes(args.inputfolder)
    instructions.sort(key=lambda b: b.name)
    all_inst = set()
    for inst in instructions:
        if not all_inst.isdisjoint(inst.names):
            print(f"Overlap in instruction names: {inst.names.intersection(all_inst)} for {inst.name}")
        all_inst = all_inst.union(inst.names)
    print(f"Writing {len(instructions)} instructions")
    with open(args.outputpath, 'w') as f:
        f.write("""
import type {AssemblyInstructionInfo} from '../../../types/assembly-docs.interfaces.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toLowerCase()) {
""".lstrip())
        for inst in instructions:
            for name in sorted(inst.names):
                f.write(f'        case "{name}":\n')
            f.write('            return {}'.format(json.dumps({
                "tooltip": inst.tooltip,
                "html": inst.body,
                "url": "", # we have no further documentation
            }, indent=16, separators=(',', ': '), sort_keys=True))[:-1] + '            };\n\n')
        f.write("""
    }
}
""")


if __name__ == '__main__':
    main()
