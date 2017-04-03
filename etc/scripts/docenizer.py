# -*- coding: utf-8 -*-
import os
import argparse

parser = argparse.ArgumentParser(description='Docenizes HTML version of the official Intel Asm PDFs')
parser.add_argument('-i', '--inputfolder', type=str, help='Folder where the input files reside as .html. Default is current folder', default='./')
parser.add_argument('-o', '--outputpath', type=str, help='Final path of the .js file. Default is ./asm-docs.js', default='./asm-docs.js')

args = parser.parse_args()
with open(args.outputpath, 'w') as f:
    f.write('var tokens = { \n')
    for root, dirs, files in os.walk(args.inputfolder):
        for file in files:
            if file.endswith(".html"):
                getDesc = False
                with open(os.path.join(root, file)) as f2:
                    file = file.split('.')[0]
                    for asm in file.split(':'):
                        tooltip = ''
                        helps = ''
                        isFirst = True
                        for line in f2.readlines():
                            if getDesc and "<h2>" in line:
                                f.write('       "' + asm +'": {"html": "' + helps + '</span>", "url": "' + file + '.html", "tooltip": "' + tooltip + '"},\n')
                                break
                            if getDesc:
                                line = line[:-1]
                                escapedLine = line.replace('“', "'").replace('"', "'")
                                helps += escapedLine.replace('\n', '<br>')
                                if isFirst:
                                    tooltip = escapedLine.replace('<p>', '').replace('</p>', '')
                                    helps += "<a href='#' data-toggle='collapse' data-target='#more' onclick='$(this).remove();'>View more...</a><span id='more' class='collapse'>"
                                    isFirst = False
                            if line == "<h2>Description</h2>\n":
                                getDesc = True
    f.seek(-2, os.SEEK_END)
    f.truncate()
    f.write('\n};\n\
        function getAsmOpcode(opcode) {\n\
            return tokens[opcode.toUpperCase()];\n\
        }\n\
        \n\
        module.exports = {\n\
            getAsmOpcode: getAsmOpcode\n\
        };\n')
