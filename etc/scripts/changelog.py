# -*- coding: utf-8 -*-
# Copyright (c) 2018, Compiler Explorer Authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import subprocess

html_escape_table = {
    "&": "&amp;",
    '"': "&quot;",
    "'": "&apos;",
    ">": "&gt;",
    "<": "&lt;",
}

commit_template = '    <div class="row commit-entry">\n' \
        '                <div class="col-sm-12">{}\n' \
        '                  <a href="{}commit/{}" rel="noreferrer noopener" target="_blank">\n' \
        '                    <sup><small class="glyphicon glyphicon-new-window opens-new-window" ' \
        'title="Opens in a new window"></small></sup>\n' \
        '                  </a>\n' \
        '                </div>\n' \
        '              </div>\n'


def html_escape(text):
    return "".join(html_escape_table.get(c, c) for c in text)


def format_commit(url, commit):
    grouped_commit = commit.split(' * ')
    # --grep matches every line on the commit message, we ensure we don't get caught by it
    try:
        formatted_commit = commit_template.format(html_escape(grouped_commit[1]), url, grouped_commit[0])
        print(grouped_commit)
        return formatted_commit
    except:
        return ''


def get_commits(repo):
    coms = subprocess.check_output(['git', 'log', '--date=local', '--after="3 months ago"', '--grep=^\* ', '--oneline'])
    with open('static/changelog.html', 'w') as f:
        f.write('<div class="commits-list">\n')
        for commit in coms.splitlines():
            f.write(format_commit(repo, commit))
        f.write('</div>\n')


if __name__ == '__main__':
    get_commits('https://github.com/compiler-explorer/compiler-explorer/')
