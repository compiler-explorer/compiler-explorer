#!/usr/bin/env python3
import argparse
import io
import os.path
import pdfminer.high_level
import pdfminer.layout
import re
import sys
import urllib.request


FILE = ("https://ww1.microchip.com/downloads/en/DeviceDoc/"
        "AVR-InstructionSet-Manual-DS40002198.pdf")

section_regex = re.compile(r"^(6\.\d{1,3}?)\s+?(?P<mnemonic>\w+?)\s+?(?:\((?P<mnemonic_2>\w+?)\)\s+?)?[-\u2013]\s+?(?P<name>.+?)\s*?$\s+?\1\.1\s+?Description\s+(?P<description>(?s:.+?))\s+?Operation:", re.MULTILINE)
header_footer_regex = re.compile(r"\s+?\w+?-page \d{1,3}?\s+?Manual\s+?\u00a9 2021 Microchip Technology Inc.\s+?AVR\u00ae Instruction Set Manual\s+?Instruction Description\s*", re.MULTILINE)
page_num_regex = re.compile(r"\b\w+?-page (\d{1,3})")


class Instruction:
    def __init__(self, mnemonic):
        self.mnemonic = mnemonic
        self.name = mnemonic
        self.description = ""
        self.page = 2
        self.mnemonic_2 = ""


def main():
    args = get_arguments()
    docs = get_docs_as_string(FILE)
    instructions = parse_docs(docs)
    write_script(args.output, instructions)


def get_arguments():
    parser = argparse.ArgumentParser()
    help_text = "the location to which the script will be written"
    relative_path = "../../../../lib/asm-docs/generated/asm-docs-avr.ts"
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    default_path = os.path.normpath(script_dir + relative_path)
    parser.add_argument("-o", "--output", help=help_text, default=default_path)
    return parser.parse_args()


def get_docs_as_string(url):
    with urllib.request.urlopen(url) as u:
        log_message(f"reading PDF from {url}...")
        pdf_bytes = u.read()
    with io.BytesIO(pdf_bytes) as pdf_io:
        pdf_params = pdfminer.layout.LAParams(boxes_flow=None)
        log_message("extracting text from PDF...")
        return pdfminer.high_level.extract_text(pdf_io, laparams=pdf_params)


def parse_docs(docs):
    instructions = {}
    log_message("searching for pattern matches...")
    for match in section_regex.finditer(docs):
        if match.group("mnemonic") not in instructions:
            instr = Instruction(match.group("mnemonic"))
            instr.name = match.group("name")
            instr.description = process_description(match.group("description"))
            instr.page = page_num_regex.search(docs, match.start()).group(1)
            #print(40 * "-")
            #print(f"Mnemonic: {instr.mnemonic}\nName: {instr.name}")
            #print(f"Description: {instr.description}")
            #print(instr.description)
            instructions[instr.mnemonic] = instr
        else:
            instr = instructions[match.group("mnemonic")]
        if match.group("mnemonic_2"):
            instr.mnemonic_2 = match.group("mnemonic_2")
    return instructions


def process_description(desc):
    # First, remove page header/footer
    desc = header_footer_regex.sub("", desc)
    # Next, combine lines that are separated by a singular newline
    desc = re.sub(r"(?<!\n)\n(?!\n)", " ", desc, flags=re.MULTILINE)
    # Remove leftovers from diagrams
    p = r"^(?:(?:\b\w+?\b\s*?){1,2}|.)$\n{2}"
    desc = re.sub(p, "", desc, flags=re.MULTILINE)
    return desc


def write_script(filename, instructions):
    log_message(f"writing to {filename}...")
    with open(filename, "w") as script:
        script.write("import {AssemblyInstructionInfo} from '../base.js';\n")
        script.write("\n")
        script.write("export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {\n")
        script.write("    if (!opcode) return;\n")
        script.write("    switch (opcode.toUpperCase()) {\n")
        for inst in instructions.values():
            script.write(f"        case \"{inst.mnemonic}\":\n")
            if inst.mnemonic_2:
                script.write(f"        case \"{inst.mnemonic_2}\":\n")
            script.write("            return {\n")
            html = f"{16 * ' '}\"html\": \"<p>"
            html += inst.description.replace("\n\n", "</p><p>")
            html += "</p>\",\n"
            script.write(html)
            script.write(f"{16 * ' '}\"tooltip\": \"{inst.name}\",\n")
            script.write(f"{16 * ' '}\"url\": \"{FILE}#page={inst.page}\",\n")
            script.write(12 * " " + "};\n\n")
        script.write("    }\n}")


def log_message(msg):
    print(f"{sys.argv[0]}: {msg}", file=sys.stderr)


if __name__ == "__main__":
    main()
