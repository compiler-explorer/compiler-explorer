#!/usr/bin/env python3
import argparse
import enum
import json
import os.path
import re
import urllib.request


DOC_URL_BASE = "https://raw.githubusercontent.com/mist64/c64ref/master/6502/"
doc_files = {f"{DOC_URL_BASE}{filename}":cpu_type for filename, cpu_type in {
    "cpu_6502.txt" : "6502",
    "cpu_65c02.txt" : "65c02",
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

    def html_description(self):
        if self.description:
            html = ""
            for desc_line in self.description:
                html += f"<p>{escape_quotes(desc_line)}</p>"
                return html
        elif self.long_name:
            return f"<p>{escape_quotes(self.long_name)}</p>"
        elif self.name:
            return f"<p>{escape_quotes(self.name)}</p>"
        else:
            return f"<p>{self.mnemonic}</p>"


def get_instructions():
    """Gathers all instruction data and returns it in a dictionary."""
    instructions = {}
    for f, t in doc_files.items():
        instructions_from_file(f, t, instructions)
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
                parse_mode = mode_change(regex_match.group("mode_name"))
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


def mode_change(mode_name):
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


def write_script(filename, instructions):
    script = ["export function getAsmOpcode(opcode) {",
              "    if (!opcode) return;",
              "    switch (opcode.toUpperCase()) {"]
    for inst in instructions.values():
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
        # Will need to be replaced when other 65xx CPUs are added
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


def get_arguments():
    parser = argparse.ArgumentParser()
    help_text = "the location to which the script will be written"
    relative_path = "/../../../lib/handlers/asm-docs-6502.js"
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    default_path = os.path.normpath(script_dir + relative_path)
    parser.add_argument("-o", "--output", help=help_text, default=default_path)
    return parser.parse_args()


def main():
    args = get_arguments()
    instructions = get_instructions()
    #for inst in instructions.values():
        #print(inst.__dict__)
    write_script(args.output, instructions)


if __name__ == "__main__":
    main()
