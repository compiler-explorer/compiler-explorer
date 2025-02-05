#!/usr/bin/env python3
import argparse
import enum
import json
import os.path
import re
import urllib.request

parser = argparse.ArgumentParser(description='Docenizes 6502 family CPU documentation')
parser.add_argument('-o', '--outputpath', type=str, help='Final path of the .ts file. Default is ./asm-docs-6502.ts',
                    default='./asm-docs-6502.ts')
parser.add_argument('-c', '--cpu', type=str, help='CPU to generate documentation for. Default is 6502',
                    default="6502")
parser.add_argument('-m', '--maxcpu', type=str, help='Maximum CPU to include in documentation. Default is 65c02',
                    default="65c02")

DOC_URL_BASE = "https://raw.githubusercontent.com/mist64/c64ref/f5792f46cfa9e05d006e5dd4f24199fed9091100/src/6502/"
doc_files = {f"{DOC_URL_BASE}{filename}":cpu_type for filename, cpu_type in {
    "cpu_6502.txt" : "6502",
    "cpu_65c02.txt" : "65c02",
    "cpu_65c816.txt" : "65c816",
    }.items()
}
mode_change_regex = re.compile(r"\[(?P<mode_name>.*)\]")
comment_regex = re.compile(r"##")
mnemonic_regex = re.compile(r"(?P<mnemonic>\S+)\s+(?P<name>.*)")
description_start_regex = re.compile(r"(?P<mnemonic>\S+)\s+(?P<long_name>.*)")
description_continue_regex = re.compile(r"\s+(?P<description>.*)")


class ParseMode(enum.Enum):
    IGNORE = enum.auto()
    MNEMONICS = enum.auto()
    DESCRIPTIONS = enum.auto()


class Instruction:
    def __init__(self, mnemonic, cpu_type):
        self.mnemonic = mnemonic
        self.cpu_type = cpu_type
        self.name = ""
        self.long_name = ""
        self.description = []
        self.undocumented = False

    def html_description(self):
        if self.description and self.cpu_type != "65c816":
            return "".join(
                f"<p>{escape_quotes(desc_line)}</p>"
                for desc_line in self.description
            )
        elif self.long_name:
            return f"<p>{escape_quotes(self.long_name)}</p>"
        elif self.name:
            return f"<p>{escape_quotes(self.name)}</p>"
        else:
            return f"<p>{self.mnemonic}</p>"


def get_instructions(cpu, maxcpu):
    """Gathers all instruction data and returns it in a dictionary."""
    instructions = {}
    extra_cpu = False
    for f, t in doc_files.items():
        instructions_from_file(f, cpu if not extra_cpu else t, instructions)
        if t == maxcpu:
            break
        if t == cpu:
            extra_cpu = True
    return instructions


def instructions_from_file(filename, cpu_type, instructions):
    """Gathers instruction data from a file and adds it to the dictionary."""
    with open_file(filename) as response:
        print(f"Reading from {filename}...")
        parse_mode = ParseMode.IGNORE
        parse_funcs = {ParseMode.MNEMONICS: parse_mnemonics,
                       ParseMode.DESCRIPTIONS: parse_descriptions}
        for line_num, line in enumerate(response_to_lines(response), start=1):
            #print(str(line_num) + "\t" + str(line))
            line = remove_comments(line)
            if not line or line.isspace():
                continue
            regex_match = mode_change_regex.match(line)
            if regex_match:
                parse_mode = mode_change(regex_match.group("mode_name"), cpu_type)
                continue
            if parse_mode == ParseMode.IGNORE:
                continue
            parse_funcs[parse_mode](line, line_num, cpu_type, instructions)


def open_file(filename):
    """Opens a documentation file from the internet."""
    return urllib.request.urlopen(filename)


def response_to_lines(response):
    """Converts an HTTP response to a list containing each line of text."""
    return response.read().decode("utf-8").replace("\xad", "").split("\n")


def remove_comments(line):
    """Removes comments from a line of a documentation file."""
    regex_match = comment_regex.search(line)
    if regex_match:
        return line[:regex_match.start()]
    else:
        return line


def mode_change(mode_name, cpu_type):
    if mode_name == "mnemos":
        return ParseMode.MNEMONICS
    elif mode_name == "documentation-mnemos":
        return ParseMode.DESCRIPTIONS
    else:
        return ParseMode.IGNORE


def parse_mnemonics(line, line_num, cpu_type, instructions):
    regex_match = mnemonic_regex.match(line)
    if regex_match:
        mnemonic = regex_match.group("mnemonic")
        name = regex_match.group("name")
        if mnemonic not in instructions:
            instructions[mnemonic] = Instruction(mnemonic, cpu_type)
        instructions[mnemonic].name = name
    else:
        print(f"Mnemonic parsing: Match failure on line {str(line_num)}")
        print("    " + line)


def parse_descriptions(line, line_num, cpu_type, instructions):
    start_match = description_start_regex.match(line)
    continue_match = description_continue_regex.match(line)
    if start_match:
        mnemonic = start_match.group("mnemonic")
        parse_descriptions.last_mnemonic = mnemonic
        long_name = start_match.group("long_name")
        if mnemonic not in instructions:
            instructions[mnemonic] = Instruction(mnemonic, cpu_type)
        instructions[mnemonic].long_name = long_name
    elif continue_match:
        mnemonic = parse_descriptions.last_mnemonic
        description = continue_match.group("description")
        instructions[mnemonic].description.append(description)
        if re.search("undocumented", description):
            instructions[mnemonic].undocumented = True


def write_script(filename, instructions):
    script = ["import {AssemblyInstructionInfo} from '../base.js';",
              "",
              "export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {",
              "    if (!opcode) return;",
              "    switch (opcode.toUpperCase()) {"]
    for inst in instructions.values():
        if inst.undocumented and inst.cpu_type != "6502":
            continue
        script.append(f"        case \"{inst.mnemonic}\":")
        script.append("            return {")
        html = f"{16 * ' '}\"html\": \""
        html += inst.html_description()
        html += "\","
        script.append(html)
        if inst.long_name:
            safe_ln = escape_quotes(inst.long_name)
            script.append(f"{16 * ' '}\"tooltip\": \"{safe_ln}\",")
        elif inst.name:
            safe_n = escape_quotes(inst.name)
            script.append(f"{16 * ' '}\"tooltip\": \"{safe_n}\",")
        else:
            script.append(f"{16 * ' '}\"tooltip\": \"{inst.mnemonic}\",")
        s = "https://www.pagetable.com/c64ref/6502/?cpu="
        e = "&tab=2#"
        t = inst.cpu_type
        m = inst.mnemonic
        script.append(f"{16 * ' '}\"url\": \"{s}{t}{e}{m}\",")
        script.append(12 * " " + "};")
        script.append("")
    script.append("    }")
    script.append("}")
    with open(filename, "w") as f:
        print(f"Writing output to {filename}...")
        f.write("\n".join(script))
    #print("\n".join(script))


def escape_quotes(string):
    return string.replace("\"", "\\\"")


def main():
    args = parser.parse_args()
    instructions = get_instructions(args.cpu, args.maxcpu)
    #for inst in instructions.values():
        #print(inst.__dict__)
    write_script(args.outputpath, instructions)


if __name__ == "__main__":
    main()
