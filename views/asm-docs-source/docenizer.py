# -*- coding: utf-8 -*-
import os

with open('../../static/asm-docs.js', 'w') as f:
    f.write('define(function (require) {\n\
    "use strict";\n\
    var tokens = {\n')
    for file in os.listdir("."):
        if file.endswith(".html"):
            getDesc = False
            with open(file) as f2:
                file = file.split('.')[0]
                for asm in file.split(':'):
                    helps = ''
                    isFirst = True
                    for line in f2.readlines():
                        if getDesc and "<h2>" in line:
                            f.write('       "' + asm +'": {"html": "' + helps + '</span>", "url": "' + file + '.html"},\n');
                            break
                        if getDesc:
                            line = line[:-1]
                            helps += line.replace('“', "'").replace('"', "'").replace('\n', '<br>')
                            if isFirst:
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
