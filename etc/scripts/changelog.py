import urllib
import urllib2
from datetime import date
from dateutil.relativedelta import relativedelta
import traceback
import json
import subprocess


def format_commit(url, commit):
    splitted_commit = commit.split(' * ')
    print(splitted_commit)
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
           '</div>\n'.format(url, splitted_commit[0], splitted_commit[0], splitted_commit[1])


def get_commits(repo):
    coms = subprocess.check_output(['git', 'log', '--date=local', '--after="3 months ago"', '--grep=^* ', '--oneline'])
    with open('static/changelog.html', 'w') as f:
        for commit in coms.splitlines():
            f.write(format_commit(repo, commit))


if __name__ == '__main__':
    get_commits('https://github.com/mattgodbolt/compiler-explorer/')
