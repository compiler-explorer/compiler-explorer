# -*- coding: utf-8 -*-
from os import listdir, path, chdir, walk
from re import compile, match, sub, error
from datetime import datetime
from itertools import chain


if __name__ == '__main__':
    def_name = 'Matt Godbolt'
    # Has to be called from the local folder. This should be improved
    chdir('../../')

    # ?1: Comment slash; 2: Copyright mark; ?3: Starting year; 4: Current year; ?5: Comma separator; ?6: Names
    license_re = compile(r'(// )?(Copyright \(c\) )(2012-)?(\d*)(,)?( ?.*)')
    year = datetime.utcnow().year

    non_recursive_searched_paths = ['static/', 'test/', 'test/handlers/', 'test/compilers/']
    recursive_searched_paths = ['lib/']

    ignored_files = set(['static/ansi-to-html.js', 'static/gccdump-view.js', 'static/gccdump-rtl-gimple-mode.js'])

    found_paths = ['./app.js', './LICENSE']

    for root in non_recursive_searched_paths:
        found_paths.extend(path.join(root, file_name) for file_name in listdir(root) if file_name.endswith('.js'))

    for root, _, files in chain.from_iterable(walk(file_path) for file_path in recursive_searched_paths):
        found_paths.extend(path.join(root, file_name) for file_name in files if file_name.endswith('.js'))

    found_paths = [file for file in found_paths if file not in ignored_files]

    change_count = 0
    for path in found_paths:
        try:
            file_lines = []
            with open(path, 'r') as f:
                for line in f.readlines():
                    res = match(license_re, line)
                    subbed_line = line
                    try:
                        if res:
                            sub_re = r'{}Copyright (c) 2012-{}, {}'.format(res.group(1) or '', year, res.group(6).strip() if res.group(6) else def_name)
                            subbed_line = sub(license_re, sub_re, line)
                            change_count += 1
                    except error as e:
                        print 'Regex exception "{}" raised in {}\n'.format(e, path)
                    finally:
                        file_lines.append(subbed_line)
            with open(path, 'w+') as f:
                for line in file_lines:
                    f.write(line)
        except OSError as os_error:
            print 'OS error: {}'.format(os_error)

    print 'Validated {} files out of {}'.format(change_count, len(found_paths))
