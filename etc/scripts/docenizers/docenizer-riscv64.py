#!/usr/bin/env python3
import argparse
import json
from urllib import request
from urllib import parse

import requests
import yaml
import sys
import re

#opcodes_yaml = "https://five-embeddev.github.io/riscv-docs-html/opcodes.yaml"
opcodes_yaml = "https://five-embeddev.github.io/riscv-docs-html/opcodes.yaml"
htmlhost = "https://five-embeddev.github.io/riscv-docs-html/"

parser = argparse.ArgumentParser(description='Scrape the five-embeddev quick reference page and generate the riscv documentation')
parser.add_argument('-i', '--opcode-data', type=str, help='Input opcode data in YAML format.',
                    default=opcodes_yaml)
parser.add_argument('-o', '--outputpath', type=str, help='Final path of the .ts file. Default is ./asm-docs-riscv64.ts',
                    default='./asm-docs-riscv64.ts')

def bold_keyword(text, keyword):
    for keywordr in (keyword.upper(), keyword.lower()):
        rer = re.compile(f"\\b{keywordr}\\b")
        text,_ = re.subn(rer,f"<b>{keywordr}</b>",text)
    return text

class operation:
    def __init__(self, yaml_record, yaml_data):
        opcode = yaml_record['opcode'][0].upper()
        # Remove "@" and "c." from opcode
        # These are

        self.opcode = opcode
        self.opcode_alias = None
        if 'opcode_alias' in yaml_record:
            self.opcode_alias = yaml_record['opcode_alias'].upper()

        tool_opcode_args = opcode + " " + ', '.join(yaml_record['opcode_args']) + "\n"
        html_opcode_args = f"<span class=\"opcode\"><b>{opcode}</b> {', '.join(yaml_record['opcode_args'])}</span>"

        # What ISA does this opcode belong to?
        # Is is a psuedo opcode?
        group=""

        if 'main_desc' in yaml_record:
            group += yaml_record['main_desc']
        if 'psuedo' == yaml_record['opcode_group']:
            group += "(pseudo)" 

        html_code = ""
        text_code = ""
        html_group = f"<br><div><b>ISA</b>: {group}</div>"

        if 'psuedo_to_base' in yaml_record:
            html_code += "<br><div><b>Equivalent ASM:</b><pre>"
            html_code += "\n".join(yaml_record['psuedo_to_base'])
            html_code += "</pre></div>"
            text_code += "Equivalent ASM:\n\n" + "\n".join(yaml_record['psuedo_to_base'])
        
        if 'main_url_base' in yaml_record:
            main_url_base = yaml_record['main_url_base']
            main_desc = yaml_record['main_desc']
            main_id = yaml_record['main_id']

            opcode_descs = yaml_record['desc'][main_desc][main_id]['text']
            tool_desc = "\n".join(opcode_descs) + "\n\n" + text_code + "\n\n(ISA: " + group + ")"
            html_desc = "<br><div>" + "<br>".join([bold_keyword(x,opcode) for x in opcode_descs]) + "</div>"

                
            self.url = htmlhost + main_url_base + main_id
            self.tooltip = tool_desc
            # extract all elements under the second column and the third column which are td
            # and put them into a new div tag
            self.html = "<div>" + html_opcode_args + html_desc + html_code + html_group + "</div>"
        elif 'psuedo_to_base' in yaml_record:
            self.url = htmlhost
            self.tooltip = "Psuedo Instruction.\n\n" + text_code  + "\n\n"
            self.html = "<div>" + html_opcode_args +  html_code + html_group + "</div>"
        else:
            self.url = htmlhost
            self.tooltip = "\n\n(ISA: " + group + ")"
            self.html = "<div>" + html_opcode_args  + html_group + "</div>"
        
    def __str__(self):
        dic = {
            # no opcode here
            "tooltip": self.tooltip,
            "url": self.url,
            "html": self.html
        }
        return json.dumps(dic, indent=16, separators=(',', ': '), sort_keys=True)

if __name__ == '__main__':
    args = parser.parse_args()
    yaml_text=None
    try:
        if args.opcode_data[:4] == "http":
            r = requests.get(args.opcode_data)
            r.encoding = 'utf-8'
            yaml_text = r.text
        else:
            with open(args.opcode_data, 'r') as fin :
                yaml_text=fin.read()
    except:
        print("ERROR: Loading YAML file or URL: " + args.opcode_data + ":")
        print(exc)
        sys.exit(-1)
    yaml_data=None    
    try:
        yaml_data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        print("ERROR: Parsing YAML file: " + args.opcode_data + ":")
        print(exc)
        sys.exit(-1)

    with open(args.outputpath, "w") as output:
        
        output.write("""
import {AssemblyInstructionInfo} from '../base.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
""".lstrip())

        for record in [operation(o, yaml_data) for o in yaml_data['opcodes'].values()]:
            # for each opcode
            output.write(f'        case "{record.opcode}":\n')
            if record.opcode_alias:
                output.write(f'        case "{record.opcode_alias}":\n')
            output.write(f'            return {str(record)[:-1]}            }};\n\n')
        
        output.write("""
    }
}
""")

