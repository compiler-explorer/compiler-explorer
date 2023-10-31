import argparse
import json
from urllib import request
from urllib import parse

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError("Please install BeautifulSoup (apt-get install python3-bs4 or pip install beautifulsoup4 should do it)")

htmllink = "https://five-embeddev.com/quickref/instructions.html"
htmlhost = "https://five-embeddev.com"

parser = argparse.ArgumentParser(description='Scrape the five-embeddev quick reference page and generate the riscv documentation')
parser.add_argument('-o', '--outputpath', type=str, help='Final path of the .ts file. Default is ./asm-docs-riscv64.ts',
                    default='./asm-docs-riscv64.ts')

class operation(object):
    def __init__(self, soup_record):
        cols = soup_record.find_all("td")
        self.opcode = cols[0].text.strip().upper()
        self.tooltip = "".join(list(cols[2].strings)).strip()
        self.url = htmlhost+cols[0].a["href"]
        # extract all elements under the second column and the third column which are td
        # and put them into a new div tag
        self.html = "<div>" + "".join(
            [str(child) for child in cols[1].children] + 
            [str(child) for child in cols[2].children]) + "</div>"
        
    def __str__(self):
        dic = {
            # no opcode here
            "tooltip": self.tooltip,
            "url": self.url,
            "html": self.html
        }
        return json.dumps(dic, indent=16, separators=(',', ': '), sort_keys=True)

# iterator of records
def record_iterator(soup):
    tables = soup.find_all("table", class_="sortable-theme-dark")
    for table in tables:
        records = table.tbody.find_all("tr")
        for record in records:
            yield operation(record)
       
if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.outputpath, "w") as output:
        soup = BeautifulSoup(requests.get(htmllink).text, "html.parser")

        output.write("""
import {AssemblyInstructionInfo} from '../base.js';

export function getAsmOpcode(opcode: string | undefined): AssemblyInstructionInfo | undefined {
    if (!opcode) return;
    switch (opcode.toUpperCase()) {
""".lstrip())
        
        for record in record_iterator(soup):
            # for each opcode
            output.write(f'        case "{record.opcode}":\n')
            output.write(f'            return {str(record)[:-1]}            }};\n\n')
        
        output.write("""
    }
}
""")

