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


def format_commit(url, commit):
    grouped_commit = commit.split(' * ')
    print(grouped_commit)
    return '<div class="row commits-list">\n' \
           '    <div class="col-sm-3">\n' \
           '        <a href="{}commit/{}" rel="noreferrer noopener" target="_blank">{}\n' \
           '            <sup><small class="glyphicon glyphicon-new-window opens-new-window" ' \
           'title="Opens in a new window"></small></sup>\n' \
           '        </a>\n' \
           '    </div>\n' \
           '    <div class="col-sm-9">\n' \
           '        <p>{}</p>\n' \
           '    </div>\n' \
           '</div>\n'.format(url, grouped_commit[0], grouped_commit[0], grouped_commit[1])


def get_commits(repo):
    coms = subprocess.check_output(['git', 'log', '--date=local', '--after="3 months ago"', '--grep=^* ', '--oneline'])
    with open('static/changelog.html', 'w') as f:
        for commit in coms.splitlines():
            f.write(format_commit(repo, commit))


if __name__ == '__main__':
    get_commits('https://github.com/mattgodbolt/compiler-explorer/')
