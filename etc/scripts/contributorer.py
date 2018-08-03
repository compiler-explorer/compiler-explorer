# -*- coding: utf-8 -*-

import requests
import os.path
import json
import PySO8601
import argparse
import re

AUTH_TOKEN = 'MISSING AUTH TOKEN'
IS_DEBUG = False
REPOSITORY = ''


def dprint(msg):
    if IS_DEBUG:
        print(msg)


def get_oauth(url, **kwargs):
    return requests.get(url, headers={'Authorization': 'token {}'.format(AUTH_TOKEN)}, **kwargs)


def contributors():
    contributors = []
    link = 'https://api.github.com/repos/{}/contributors'.format(REPOSITORY)
    while link is not None:
        print(link)
        result = get_oauth(link, params={'per_page': 80})
        links = result.headers.get('link')
        if links is None:
            link = None
        else:
            splits = links.split(',')
            for split in splits:
                bits = split.split(';')
                # If there is a next rel link, follow it
                if len(bits) == 2 and bits[1].strip() == 'rel="next"':
                    link = bits[0].strip()[1:-1]
                else:
                    link = None

        for contributor in result.json():
            contributors.append(contributor)
    return contributors


def collaborators():
    collaborators = []
    link = 'https://api.github.com/repos/{}/collaborators'.format(REPOSITORY)
    while link is not None:
        print(link)
        result = get_oauth(link, params={'per_page': 80})
        links = result.headers.get('link')
        if links is None:
            link = None
        else:
            splits = links.split(',')
            for split in splits:
                bits = split.split(';')
                # If there is a next rel link, follow it
                if len(bits) == 2 and bits[1].strip() == 'rel="next"':
                    link = bits[0].strip()[1:-1]
                else:
                    link = None

        for collaborator in result.json():
            collaborators.append(collaborator)
    return collaborators


parser = argparse.ArgumentParser(description='Creates a CONTRIBUTORS.md file')
parser.add_argument('-t', '--token', type=str, help='GitHub token (Only needs public_repo access)', required=True)
parser.add_argument('-d', '--debug', action='store_true', help='Print debug information')
parser.add_argument('-o', '--output', type=str, help='Path of output file', default='CONTRIBUTORS.md')
parser.add_argument('-r', '--repository', type=str, help='Which repository to query', default='mattgodbolt/compiler-explorer')

if __name__ == '__main__':
    args = parser.parse_args()
    AUTH_TOKEN = args.token
    IS_DEBUG = args.debug
    REPOSITORY = args.repository
    repository_safe = "".join([c for c in REPOSITORY if re.match(r'\w', c)])
    collabs_data = collaborators()
    skippable = set([collab['login'].lower() for collab in collabs_data])
    # Remove people that are in CONTRIBUTORS for some reason or another
    skippable.discard('lefticus')
    skippable.discard('ubsan')
    # Added in the thanks to section of the readme
    skippable.update(['filcab', 'voxelf', 'johanengelen', 'jsheard', 'dkm', 'andrewpardoe'])
    # Duplicated people under different accounts
    skippable.add('jaredadobe')
    data = contributors()
    # People already listed somewhere else. Use set diff?
    contributors = [contributor for contributor in data if contributor['login'].lower() not in skippable]
    print('Found {} contributors. Skipping {} collaborators'.format(len(contributors), len(skippable)))
    # Create cache folder, which can be cleared at any moment
    cache_dir_base = 'contributorer-cache-{}'.format(repository_safe)
    if not os.path.isdir(cache_dir_base):
        os.mkdir(cache_dir_base)
    dprint('Cache base dir: {}'.format(cache_dir_base))
    cache_dir_commits = '{}/commits'.format(cache_dir_base)
    if not os.path.isdir(cache_dir_commits):
        os.mkdir(cache_dir_commits)
    dprint('Cache commits dir: {}'.format(cache_dir_commits))
    first_commits = []
    for contributor in contributors:
        commits = {}
        # Where should the commits for this contributor be?
        # This works even if outdated because we are looking for old commits, not new
        contrib_file = '{}/{}-commits.json'.format(cache_dir_commits, contributor['login'])
        dprint('Checking commits file: {}'.format(contrib_file))
        if os.path.isfile(contrib_file):
            dprint('File found, using as commit source')
            with open(contrib_file, 'r') as c:
                commits = json.load(c)
        else:
            dprint('None found, querying to GitHub')
            # TODO: Buffer them and send only 1 request?
            result = get_oauth('https://api.github.com/repos/{}/commits'.format(REPOSITORY),
                               params={'author': contributor['login']})
            if result.status_code == 200:
                commits = result.json()
                dprint('Writting results to file')
                with open(contrib_file, 'w') as c:
                    c.write(result.text)
        if len(commits) > 0:
            first_commit = commits[-1]
            dprint('First commit for {} was in {}'.format(contributor['login'], first_commit['commit']['author']['date']))
            first_commits.append({'date': first_commit['commit']['author']['date'],
                                  'name': first_commit['commit']['author']['name'] or
                                          '"{}"'.format(first_commit['author']['login']),
                                  'url': first_commit['author']['html_url']})
    dprint('Sorting commits from oldest to newest')
    sorted_commits = sorted(first_commits, key=lambda x: PySO8601.parse(x['date']))
    with open(args.output, 'w') as md:
        dprint('Output file: {}'.format(args.output))
        md.write('From oldest to newest contributor, we would like to thank:\n\n')
        md.writelines(['- [{}]({})\n'.format(commit['name'], commit['url']) for commit in sorted_commits])
