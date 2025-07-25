#!/usr/bin/env python3
import argparse
import io
import os.path
import pdfminer.high_level
import pdfminer.layout
import re
import sys
import urllib.request


FILE = ("https://ww1.microchip.com/downloads/aemDocuments/documents/MCU08/ProductDocuments/ReferenceManuals/"
        "AVR-InstructionSet-Manual-DS40002198.pdf")

section_regex = re.compile(r"^(6\.\d{1,3})\s+(?P<mnemonic>\w+)\s*(?:\((?P<mnemonic_2>\w+)\))?\s*[-\u2013]\s*(?P<name>.+?)\s*$\s*\1\.1\s+Description\s+(?P<description>(?s:.+?))\s+Operation:", re.MULTILINE)
header_footer_regex = re.compile(
    r"\s*DS40002198[A-Z] - \d{1,3}\s*Migration Guide\s*© \d{4} Microchip Technology Inc\.?(?: and its subsidiaries)?\s*AVR® Instruction Set Manual\s*Instruction Description\s*",
    re.MULTILINE
)
page_num_regex = re.compile(r"\bDS40002198[A-Z] - (\d{1,3})")


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
            search_start = max(0, match.start() - 2000)
            search_text = docs[search_start:match.start()]
            page_matches = list(page_num_regex.finditer(search_text))
            if page_matches:
                instr.page = int(page_matches[-1].group(1)) + 1
            else:
                # If page not found backwards, try searching forward a bit
                search_end = min(len(docs), match.end() + 1000)
                forward_text = docs[match.end():search_end]
                page_match = page_num_regex.search(forward_text)
                if page_match:
                    # Add 1 to convert from document page number to PDF page number
                    instr.page = int(page_match.group(1)) + 1
                else:
                    print(f"Warning: Could not find page number for {instr.mnemonic}, using default", file=sys.stderr)
                    instr.page = 3
            
            #print(40 * "-")
            #print(f"Mnemonic: {instr.mnemonic}\nName: {instr.name}")
            #print(f"Description: {instr.description}")
            #print(instr.description)
            instructions[instr.mnemonic] = instr
        else:
            # If we already have this instruction, we might want to merge information
            instr = instructions[match.group("mnemonic")]
            # Update with potentially better description if current one is longer/better
            new_desc = process_description(match.group("description"))
            if len(new_desc) > len(instr.description):
                instr.description = new_desc
                # Also update the page number to the more recent one
                search_start = max(0, match.start() - 2000)
                search_text = docs[search_start:match.start()]
                page_matches = list(page_num_regex.finditer(search_text))
                if page_matches:
                    instr.page = int(page_matches[-1].group(1)) + 1
        
        # Handle secondary mnemonic (like STD for ST)
        if (
            match.group("mnemonic_2")
            # The manual lists some instruction set names in the place where we
            # expected to find second mnemonics.
            and match.group("mnemonic_2") not in ("AVRe", "AVRrc")
        ):
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
    # Clean up problematic Unicode characters
    desc = desc.replace('\ufffd', '')  # Remove replacement characters
    desc = desc.replace('\u00a0', ' ')  # Replace non-breaking spaces with regular spaces
    desc = desc.replace('\u2013', '-')  # Replace en-dash with regular dash
    desc = desc.replace('\u2014', '-')  # Replace em-dash with regular dash
    desc = desc.replace('\u201c', '"')  # Replace left double quote
    desc = desc.replace('\u201d', '"')  # Replace right double quote
    desc = desc.replace('\u2018', "'")  # Replace left single quote
    desc = desc.replace('\u2019', "'")  # Replace right single quote
    # Remove any remaining high Unicode characters that might cause display issues
    desc = re.sub(r'[\u0080-\uffff]', '', desc)
    # Escape double quotes for JavaScript/TypeScript string literals
    desc = desc.replace('"', '\\"')
    return desc


def write_script(filename, instructions):
    log_message(f"writing to {filename}...")
    with open(filename, "w") as script:
        script.write("import type {AssemblyInstructionInfo} from '../../../types/assembly-docs.interfaces.js';\n")
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
    log_message(f"wrote {len(instructions)} opcodes to asm-docs-avr.ts")


def log_message(msg):
    print(f"{sys.argv[0]}: {msg}", file=sys.stderr)


if __name__ == "__main__":
    main()
