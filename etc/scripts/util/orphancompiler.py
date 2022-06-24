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
ALIAS_LIST_RE = re.compile(r'alias=(.*)')
GROUP_NAME_RE = re.compile(r'group\.(.*?)\.')
COMPILER_EXE_RE = re.compile(r'compiler\.(.*?)\.exe')
DEFAULT_COMPILER_RE = re.compile(r'defaultCompiler=(.*)')
FORMATTERS_LIST_RE = re.compile(r'formatters=(.*)')
FORMATTER_EXE_RE = re.compile(r'formatter\.(.*?)\.exe')
LIBS_LIST_RE = re.compile(r'libs=(.+)')
LIB_VERSIONS_LIST_RE = re.compile(r'libs\.(.*?)\.versions=(.*)')
LIB_VERSION_RE = re.compile(r'libs\.(.*?)\.versions\.(.*?)\.version')
TOOLS_LIST_RE = re.compile(r'tools=(.+)')
TOOL_EXE_RE = re.compile(r'tools\.(.*?)\.exe')
EMPTY_LIST_RE = re.compile(r'.*(compilers|formatters|versions|tools|alias|exclude|libPath)=.*::.*')
DISABLED_RE = re.compile(r'^# Disabled:\s*(.*)')


def match_and_add(line, expr, s):
    match = expr.match(line)
    if match:
        s.add(match.group(1))
    return match


def match_and_update(line, expr, s, split=':'):
    match = expr.match(line)
    if match:
        s.update(match.group(1).split(split))
    return match


def process_file(file: str):
    default_compiler = set()
    listed_groups = set()
    seen_groups = set()
    listed_compilers = set()
    seen_compilers = set()
    listed_formatters = set()
    seen_formatters = set()
    listed_tools = set()
    seen_tools = set()
    listed_libs_ids = set()
    seen_libs_ids = set()
    listed_libs_versions = set()
    seen_libs_versions = set()
    empty_separators = set()
    disabled = set()
    with open(file) as f:
        for line in f:
            match_empty = EMPTY_LIST_RE.match(line)
            if match_empty:
                empty_separators.add(f"{line}")

            match_compilers = COMPILERS_LIST_RE.search(line)
            if match_compilers:
                ids = match_compilers.group(1).split(':')
                for elem_id in ids:
                    if elem_id.startswith('&'):
                        listed_groups.add(elem_id[1:])
                    elif '@' not in elem_id:
                        listed_compilers.add(elem_id)

            match_libs_versions = LIB_VERSIONS_LIST_RE.match(line)
            if match_libs_versions:
                lib_id = match_libs_versions.group(1)
                versions = match_libs_versions.group(2).split(':')
                seen_libs_ids.add(lib_id)
                listed_libs_versions.update([f"{lib_id} {v}" for v in versions])

            match_libs_version = LIB_VERSION_RE.match(line)
            if match_libs_version:
                lib_id = match_libs_version.group(1)
                version = match_libs_version.group(2)
                seen_libs_versions.add(f"{lib_id} {version}")

            if(
                    match_and_add(line, DEFAULT_COMPILER_RE, default_compiler) or
                    match_and_add(line, GROUP_NAME_RE, seen_groups) or
                    match_and_add(line, COMPILER_EXE_RE, seen_compilers) or
                    match_and_add(line, FORMATTER_EXE_RE, seen_formatters) or
                    match_and_add(line, TOOL_EXE_RE, seen_tools) or
                    match_and_update(line, ALIAS_LIST_RE, seen_compilers) or
                    match_and_update(line, FORMATTERS_LIST_RE, listed_formatters) or
                    match_and_update(line, TOOLS_LIST_RE, listed_tools) or
                    match_and_update(line, LIBS_LIST_RE, listed_libs_ids) or
                    match_and_update(line, DISABLED_RE, disabled, ' ')
            ):
                continue
    bad_compilers = listed_compilers.symmetric_difference(seen_compilers)
    bad_groups = listed_groups.symmetric_difference(seen_groups)
    bad_formatters = listed_formatters.symmetric_difference(seen_formatters)
    bad_libs_ids = listed_libs_ids.symmetric_difference(seen_libs_ids)
    bad_libs_versions = listed_libs_versions.symmetric_difference(seen_libs_versions)
    bad_tools = listed_tools.symmetric_difference(seen_tools)
    bad_default = default_compiler - listed_compilers
    return (file,
            bad_compilers - disabled,
            bad_groups - disabled,
            bad_formatters - disabled,
            bad_libs_ids - disabled,
            bad_libs_versions - disabled,
            bad_tools - disabled,
            bad_default,
            empty_separators)


def process_folder(folder: str):
    return [process_file(join(folder, f))
            for f in listdir(folder)
            if isfile(join(folder, f))
            and not (f.endswith('.defaults.properties') or f.endswith('.local.properties'))
            and f.endswith('.properties')]


def problems_found(file_result):
    return any([len(r) for r in file_result[1:]])


def print_issue(name, result):
    if len(result) > 0:
        sep = "\n\t"
        print(f"{name}:\n\t{sep.join(sorted(result))}")


def find_orphans(folder: str):
    result = sorted([r for r in process_folder(folder) if problems_found(r)], key=lambda x: x[0])
    if result:
        print(f"Found {len(result)} property file(s) with mismatching ids:")
        for r in result:
            print(r[0])
            print_issue("COMPILERS", r[1])
            print_issue("GROUPS", r[2])
            print_issue("FORMATTERS", r[3])
            print_issue("LIB IDS", r[4])
            print_issue("LIB VERSIONS", r[5])
            print_issue("TOOLS", r[6])
            print_issue("UNKNOWN DEFAULT COMPILER", r[7])
            print_issue("EMPTY LISTINGS", r[8])
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
