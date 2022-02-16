# -*- coding: utf-8 -*-
# Copyright (c) 2022, Compiler Explorer Authors
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
import sys
from os import listdir
from os.path import isfile, join
import re

COMPILERS_LIST_RE = re.compile(r'compilers=(.*)')
ALIAS_LIST_RE = re.compile(r'compiler.\.*?\.alias=(.*)')
GROUP_NAME_RE = re.compile(r'group\.(.*?)\.')
COMPILER_EXE_RE = re.compile(r'compiler\.(.*?)\.exe')
DISABLED_RE = re.compile(r'^# Disabled:\s*(.*)')


def process_file(file: str):
    listed_groups = set()
    seen_groups = set()
    listed_compilers = set()
    seen_compilers = set()
    disabled = set()
    with open(file) as f:
        for line in f:
            match_compilers = COMPILERS_LIST_RE.search(line)
            if match_compilers:
                ids = match_compilers.group(1).split(':')
                for elem_id in ids:
                    if elem_id.startswith('&'):
                        listed_groups.add(elem_id[1:])
                    elif '@' not in elem_id:
                        listed_compilers.add(elem_id)
            match_aliases = ALIAS_LIST_RE.match(line)
            if match_aliases:
                seen_compilers.update(match_aliases.group(1).split(':'))
                continue
            match_group = GROUP_NAME_RE.match(line)
            if match_group:
                seen_groups.add(match_group.group(1))
                continue
            match_compiler = COMPILER_EXE_RE.match(line)
            if match_compiler:
                seen_compilers.add(match_compiler.group(1))
                continue
            match_disabled = DISABLED_RE.match(line)
            if match_disabled:
                disabled.update(match_disabled.group(1).split(' '))
    bad_compilers = listed_compilers.symmetric_difference(seen_compilers)
    bad_groups = listed_groups.symmetric_difference(seen_groups)
    return file, bad_compilers - disabled, bad_groups - disabled


def process_folder(folder: str):
    return [process_file(join(folder, f))
            for f in listdir(folder)
            if isfile(join(folder, f))
            and not (f.endswith('.defaults.properties') or f.endswith('.local.properties'))
            and f.endswith('.properties')]

def problems_found(file_result):
    return len(file_result[1]) > 0 or len(file_result[2]) > 0


def find_orphans(folder: str):
    result = sorted([r for r in process_folder(folder) if problems_found(r)], key=lambda x: x[0])
    if result:
        print(f"Found {len(result)} property file(s) with mismatching ids:")
        sep = "\n\t"
        for r in result:
            print(r[0])
            if len(r[1]) > 0:
                print(f"COMPILERS:\n\t{sep.join(sorted(r[1]))}")
            if len(r[2]) > 0:
                print(f"GROUPS:\n\t{sep.join(sorted(r[2]))}")
            print("")
        print("To suppress this warning on IDs that are temporally disabled, "
              "add one or more comments to each listed file:")
        print("# Disabled: id1 id2 ...")
    else:
        print("No configuration mismatches found")
    return result


if __name__ == '__main__':
    if find_orphans('./etc/config/'):
        sys.exit(1)
