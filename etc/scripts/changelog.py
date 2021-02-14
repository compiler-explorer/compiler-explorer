# -*- coding: utf-8 -*-
# Copyright (c) 2020, Compiler Explorer Authors
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

commit_template = '''  <div class="row commit-entry">
    <div class="col-sm-12">
      <a href="{}commit/{}" rel="noreferrer noopener" target="_blank">{}</a>
    </div>
  </div>
'''


def html_escape(text):
    return "".join(html_escape_table.get(c, c) for c in text)


def format_commit(url, commit):
    # Input format is "<hash> <description>", so split only on the first space and escape the commit message
    grouped_commit = commit.split(' ', 1)
    print(grouped_commit)
    try:
        return commit_template.format(url, grouped_commit[0], html_escape(grouped_commit[1]))
    except Exception as e:
        print(f'There was an error in changelog.py: {e}')
        return ''


def get_commits(repo):
    coms = subprocess.check_output(['git', 'log', '--date=local', '--after="3 months ago"', '--grep=(#[0-9]*)', '--oneline']).decode('utf-8')
    with open('static/changelog.html', 'w') as f:
        f.write('<div class="commits-list">\n')
        for commit in coms.splitlines():
            f.write(format_commit(repo, commit))
        f.write('</div>\n')


if __name__ == '__main__':
    get_commits('https://github.com/compiler-explorer/compiler-explorer/')
