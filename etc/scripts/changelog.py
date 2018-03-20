import urllib
import urllib2
from datetime import date
from dateutil.relativedelta import relativedelta
import traceback
import json


def format_commit(commit):
    return '<div class="row commits-list">\n' \
           '    <div class="col-sm-10">\n' \
           '        <p>{}{}</p>\n' \
           '    </div>\n' \
           '    <div class="col-sm-2">\n' \
           '        <a href="{}" rel="noreferrer noopener" target="_blank">{}\n' \
           '            <sup><small class="glyphicon glyphicon-new-window opens-new-window" ' \
           'title="Opens in a new window"></small></sup>\n' \
           '        </a>\n' \
           '    </div>\n' \
           '</div>\n'.format(commit['message'], commit['ellipsis'], commit['url'], commit['sha'][0:6])


def get_commits(url):
    three_months_ago = (date.today() + relativedelta(months=-3)).isoformat()
    headers = {'Accept': 'application/vnd.github.v3+json',
               'Content-Type': 'application/json'}
    useful_commits = query_commits('{}?{}'.format(url, urllib.urlencode({'since': three_months_ago})), headers, [])
    print('Got {} noteworthy commits. Writing to static/changelog.html'.format(len(useful_commits)))
    with open('static/changelog.html', 'w') as f:
        for commit in useful_commits:
            f.write(format_commit(commit))


def query_commits(url, headers, commits):
    print('Querying {}'.format(url))
    req = urllib2.Request(url, None, headers)
    try:
        response = urllib2.urlopen(req)
        the_page = json.load(response)
        for obj in the_page:
            if obj['commit']['message'].startswith('* '):
                newline_position = obj['commit']['message'].find('\n')
                commits.append({'message': obj['commit']['message'][2:newline_position],
                                'url': obj['commit']['url'],
                                'sha': obj['sha'],
                                'ellipsis': '...' if len(obj['commit']['message']) > 80 else ''})
        if response.headers['link'] and len(commits) < 20:
            first_link = response.headers['link'].split(',')[0]
            if first_link.find('rel="next"') > -1:
                link = first_link.split(';')[0]
                query_commits(link[1:len(link) - 1], headers, commits)
    except urllib2.HTTPError as err:
        print('Got an error of {}'.format(err))
    except Exception as ex:
        print('Got general exception of {}: {}'.format(ex, traceback.format_exc()))
    finally:
        return commits


if __name__ == '__main__':
    get_commits('https://api.github.com/repos/mattgodbolt/compiler-explorer/commits')
