# -*- coding: utf-8 -*-
import os
import argparse

parser = argparse.ArgumentParser(description='Docenizes HTML version of the official Intel Asm PDFs')
parser.add_argument('-i', '--inputfolder', type=str, help='Folder where the input files reside as .html', default='./')
parser.add_argument('-o', '--outputfolder', type=str, help='Final path (+name) of the .js file', default='./asm-docs.js')

args = parser.parse_args()
with open(args.outputfolder, 'w') as f:
    f.write('define(function (require) {\n\
    "use strict";\n\
    var tokens = { \n')
    for file in os.listdir(args.inputfolder):
        if file.endswith(".html"):
            getDesc = False
            with open(file) as f2:
                file = file.split('.')[0]
                for asm in file.split(':'):
                    tooltip = ''
                    helps = ''
                    isFirst = True
                    for line in f2.readlines():
                        if getDesc and "<h2>" in line:
                            f.write('       "' + asm +'": {"html": "' + helps + '</span>", "url": "' + file + '.html", "tooltip": "' + tooltip + '. More help available in the context menu"},\n');
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
    f.write('\n    };\n    return {\n\
        tokens: tokens\n\
    };\n\
});')
