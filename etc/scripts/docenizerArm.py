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

parser = argparse.ArgumentParser(description='Docenizes XML version of the official ARM documents')
parser.add_argument('-i', '--inputfolder', type=str,
                    help='Folder where the input files reside as .xml. Default is ./asm-docs-arm/',
                    default='asm-docs-arm')
parser.add_argument('-o', '--outputpath', type=str, help='Final path of the .js file. Default is ./asm-docs-arm.js',
                    default='./asm-docs-arm.js')
parser.add_argument('-d', '--downloadfolder', type=str,
                    help='Folder where the archive will be downloaded and extracted', default='asm-docs-arm')

# The maximum number of paragraphs from the description to copy.
MAX_DESC_PARAS = 5
STRIP_SUFFIX = re.compile(r'\s*(\(.*\))?\s*--.*')
INSTRUCTION_RE = re.compile(r'^([A-Z][A-Z0-9]+)\*?(\s+|$)')
# Some instructions are so broken we just take their names from the filename
UNPARSEABLE_INSTR_NAMES = []
# Some files contain instructions which cannot be parsed and which compilers are unlikely to emit
IGNORED_FILE_NAMES = [ ]
# Some instructions are defined in multiple files. We ignore a specific set of the
# duplicates here.
IGNORED_DUPLICATES = []
# Where to get the asmdoc archive.
ARCHIVE_URL = "https://developer.arm.com/-/media/developer/products/architecture/armv8-a-architecture/2020-12/AArch32_ISA_xml_v87A-2020-12.tar.gz"
ARCHIVE_NAME = "AArch32_ISA_xml_v87A-2020-12.tar.gz"
ARCHIVE_SUBDIR = "ISA_AArch32_xml_v87A-2020-12"

class Instruction(object):
    def __init__(self, name, names, tooltip, body):
        self.name = name
        self.names = names
        self.tooltip = tooltip.rstrip(': ,')
        self.body = body

    def __str__(self):
        return "{} = {}\n{}".format(self.names, self.tooltip, self.body)


def get_url_for_instruction(instr):
    return "https://developer.arm.com/documentation/ddi0597/2020-12/Base-Instructions/"


def download_asm_doc_archive(downloadfolder):
    if not os.path.exists(downloadfolder):
        print("Creating {} as download folder".format(downloadfolder))
        os.makedirs(downloadfolder)
    elif not os.path.isdir(downloadfolder):
        print("Error: download folder {} is not a directory".format(downloadfolder))
        sys.exit(1)
    archive_name = os.path.join(downloadfolder, ARCHIVE_NAME)
    print("Downloading archive...")
    urllib.request.urlretrieve(ARCHIVE_URL, archive_name)


def extract_asm_doc_archive(downloadfolder, inputfolder):
    print("Extracting file...")
    if os.path.isdir(os.path.join(inputfolder, ARCHIVE_SUBDIR)):
        for root, dirs, files in os.walk(os.path.join(inputfolder, ARCHIVE_SUBDIR)):
            for file in files:
                if os.path.splitext(file)[1] == ".xml":
                    os.remove(os.path.join(root, file))
    tar = tarfile.open(os.path.join(downloadfolder, ARCHIVE_NAME))
    tar.extractall(path=inputfolder)

def instr_name(i):
    match = INSTRUCTION_RE.match(strip_non_instr(i))
    if match:
        return match.group(1)


def get_authored_paragraphs(document_soup):
    authored = document_soup.instructionsection.desc.authored
    if authored is None:
        return None
    paragraphs = authored.find_all('para')[:5]
    authored_paragraphs = []
    for paragraph in paragraphs:
        paragraph = paragraph.wrap(document_soup.new_tag('p'))
        paragraph.para.unwrap()
        authored_paragraphs.append(paragraph)
    return authored_paragraphs

def parse(filename, f):
    doc = BeautifulSoup(f, 'html.parser')
    if doc.instructionsection is None:
        print(filename + ": Failed to find instructionsection")
        return None
    instructionsection = doc.instructionsection
    names = set()

    for name in STRIP_SUFFIX.sub('',instructionsection['title']).split(','):
        name = name.strip()
        names.add(name)

    authored_paragraphs = get_authored_paragraphs(doc)
    if authored_paragraphs is None:
        return None

    return Instruction(
        filename,
        names,
        authored_paragraphs[0].text.strip(),
        ''.join(map(lambda x: str(x), authored_paragraphs)).strip())

def parse_xml(directory):
    print("Parsing instructions...")
    instructions = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml") and file != "onebigfile.xml":
                with open(os.path.join(root, file), encoding='utf-8') as f2:
                    name = os.path.splitext(file)[0]
                    if name in IGNORED_DUPLICATES or name in IGNORED_FILE_NAMES:
                        continue
                    instruction = parse(name, f2)
                    if not instruction:
                        continue
                    instructions.append(instruction)
    return instructions


def self_test(instructions, directory):
    # For each generated instruction, check that there is a path to a file in
    # the documentation.
    directory = os.path.join(directory, ARCHIVE_SUBDIR)
    ok = True
    for inst in instructions:
        if not os.path.isfile(os.path.join(directory, inst.name + ".xml")):
            print("Warning: {} has not file associated".format(inst.name))
            ok = False
    return ok


def docenizer():
    args = parser.parse_args()
    print("Called with: {}".format(args))
    # If we don't have the html folder already...
    if not os.path.isdir(os.path.join(args.inputfolder, ARCHIVE_SUBDIR)):
        # We don't, try with the compressed file
        if not os.path.isfile(os.path.join(args.downloadfolder, ARCHIVE_NAME)):
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
    instructions = parse_xml(os.path.join(args.inputfolder, ARCHIVE_SUBDIR))
    instructions.sort(key=lambda b: b.name)
    self_test(instructions, args.inputfolder)
    all_inst = set()
    for inst in instructions:
        if not all_inst.isdisjoint(inst.names):
            print("Overlap in instruction names: {} for {}".format(
                inst.names.intersection(all_inst), inst.name))
        all_inst = all_inst.union(inst.names)
    if not self_test(instructions, args.inputfolder):
        print("Tests do not pass. Not writing output file. Aborting.")
        sys.exit(3)
    print("Writing {} instructions".format(len(instructions)))
    with open(args.outputpath, 'w') as f:
        f.write("""export function getAsmOpcode(opcode) {
    if (!opcode) return;
    switch (opcode) {
""")
        for inst in instructions:
            for name in sorted(inst.names):
                f.write('        case "{}":\n'.format(name))
            f.write('            return {}'.format(json.dumps({
                "tooltip": inst.tooltip,
                "html": inst.body,
                "url": get_url_for_instruction(inst)
                }, indent=16, separators=(',', ': ')))[:-1] + '            };\n\n')
        f.write("""
    }
}
""")
if __name__ == '__main__':
    docenizer()
