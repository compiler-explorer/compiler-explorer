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

PROP_RE = re.compile(r'([^#]*)=(.*)#*')
COMPILERS_LIST_RE = re.compile(r'compilers=(.*)')
ALIAS_LIST_RE = re.compile(r'alias=(.*)')
GROUP_NAME_RE = re.compile(r'group\.(.*?)\.')
COMPILER_EXE_RE = re.compile(r'compiler\.(.*?)\.exe=(.*)')
COMPILER_ID_RE = re.compile(r'compiler\.(.*?)\..*')
TYPO_COMPILERS_RE = re.compile(r'(compilers\..*)')
DEFAULT_COMPILER_RE = re.compile(r'defaultCompiler=(.*)')
FORMATTERS_LIST_RE = re.compile(r'formatters=(.*)')
FORMATTER_EXE_RE = re.compile(r'formatter\.(.*?)\.exe=(.*)')
FORMATTER_ID_RE = re.compile(r'formatter\.(.*?)\..*')
LIBS_LIST_RE = re.compile(r'libs=(.+)')
LIB_VERSIONS_LIST_RE = re.compile(r'libs\.(.*?)\.versions=(.*)')
LIB_VERSION_RE = re.compile(r'libs\.(.*?)\.versions\.(.*?)\.version')
TOOLS_LIST_RE = re.compile(r'tools=(.+)')
TOOL_EXE_RE = re.compile(r'tools\.(.*?)\.exe=(.*)')
TOOL_ID_RE = re.compile(r'tools\.(.*?)\..*')
EMPTY_LIST_RE = re.compile(r'(.*(compilers|formatters|versions|tools|alias|exclude|libPath)=((.*::.*)|(:.*)|(.*:)))$')
DISABLED_RE = re.compile(r'^# Disabled?:?\s*(.*)')


class Line:
    def __init__(self, line_number, text):
        self.number = line_number
        self.text = text.strip()

    def __str__(self):
        return f'Line {self.number}: {self.text}'

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.text == other.text

    def __ne__(self, other):
        return self.text != other.text

    def __hash__(self):
        return hash(self.text)

    def __lt__(self, other):
        return self.text < other.text


def as_line(text):
    return Line(-1, text)


def match_and_add(line: Line, expr, s: set):
    match = expr.match(line.text)
    if match:
        s.add(Line(line.number, match.group(1)))
    return match


def match_and_update(line: Line, expr, s: set, split=':'):
    match = expr.match(line.text)
    if match:
        s.update([Line(line.number, text) for text in match.group(1).split(split)])
    return match


def check_suspicious_path_and_add(line: Line, m, s):
    if m and not m.group(2).startswith('/opt/compiler-explorer'):
        s.add(Line(line.number, m.group(2)))


def process_file(file: str):
    default_compiler = set()

    listed_groups = set()
    seen_groups = set()

    listed_compilers = set()
    seen_compilers_exe = set()
    seen_compilers_id = set()

    listed_formatters = set()
    seen_formatters_exe = set()
    seen_formatters_id = set()

    listed_tools = set()
    seen_tools_exe = set()
    seen_tools_id = set()

    listed_libs_ids = set()
    seen_libs_ids = set()

    listed_libs_versions = set()
    seen_libs_versions = set()

    empty_separators = set()

    seen_lines = set()
    duplicate_lines = set()

    duplicated_compiler_references = set()
    duplicated_group_references = set()

    suspicious_path = set()

    seen_typo_compilers = set()

    # By default, consider this one valid as it's in several configs.
    disabled = {as_line('/usr/bin/ldd')}

    with open(file) as f:
        for line_number, text in enumerate(f, start=1):
            text = text.strip()
            if not text:
                continue
            line = Line(line_number, text)
            match_and_update(line, DISABLED_RE, disabled, ' ')

            match_prop = PROP_RE.match(line.text)
            if not match_prop:
                continue

            prop_key = match_prop.group(1)
            if prop_key in seen_lines:
                duplicate_lines.add(Line(line.number, prop_key))
            else:
                seen_lines.add(prop_key)

            match_compilers_list = COMPILERS_LIST_RE.search(line.text)
            if match_compilers_list:
                ids = match_compilers_list.group(1).split(':')
                for elem_id in ids:
                    if elem_id.startswith('&'):
                        if as_line(elem_id[1:]) in listed_groups:
                            duplicated_group_references.add(Line(line.number, elem_id[1:]))
                        listed_groups.add(Line(line.number, elem_id[1:]))
                    elif '@' not in elem_id:
                        if as_line(elem_id) in listed_compilers:
                            duplicated_compiler_references.add(Line(line.number, elem_id))
                        listed_compilers.add(Line(line.number, elem_id))

            match_libs_versions = LIB_VERSIONS_LIST_RE.match(line.text)
            if match_libs_versions:
                lib_id = match_libs_versions.group(1)
                versions = match_libs_versions.group(2).split(':')
                seen_libs_ids.add(Line(line.number, lib_id))
                listed_libs_versions.update([Line(line.number, f"{lib_id} {v}") for v in versions])

            match_libs_version = LIB_VERSION_RE.match(line.text)
            if match_libs_version:
                lib_id = match_libs_version.group(1)
                version = match_libs_version.group(2)
                seen_libs_versions.add(Line(line.number, f"{lib_id} {version}"))

            match_and_add(line, EMPTY_LIST_RE, empty_separators)
            match_and_add(line, DEFAULT_COMPILER_RE, default_compiler)
            match_and_add(line, GROUP_NAME_RE, seen_groups)

            match_compiler_exe = match_and_add(line, COMPILER_EXE_RE, seen_compilers_exe)
            check_suspicious_path_and_add(line, match_compiler_exe, suspicious_path)

            match_formatter_exe = match_and_add(line, FORMATTER_EXE_RE, seen_formatters_exe)
            check_suspicious_path_and_add(line, match_formatter_exe, suspicious_path)

            match_tool_exe = match_and_add(line, TOOL_EXE_RE, seen_tools_exe)
            check_suspicious_path_and_add(line, match_tool_exe, suspicious_path)

            match_and_add(line, COMPILER_ID_RE, seen_compilers_id)
            match_and_add(line, TOOL_ID_RE, seen_tools_id)
            match_and_add(line, FORMATTER_ID_RE, seen_formatters_id)

            match_and_add(line, TYPO_COMPILERS_RE, seen_typo_compilers)

            match_and_update(line, ALIAS_LIST_RE, seen_compilers_exe)
            match_and_update(line, FORMATTERS_LIST_RE, listed_formatters)
            match_and_update(line, TOOLS_LIST_RE, listed_tools)
            match_and_update(line, LIBS_LIST_RE, listed_libs_ids)

    if len(seen_compilers_exe) > 0:
        bad_compilers_exe = listed_compilers.symmetric_difference(seen_compilers_exe)
    else:
        bad_compilers_exe = set()

    if len(seen_compilers_id) > 0:
        bad_compilers_ids = listed_compilers.symmetric_difference(seen_compilers_id)
    else:
        bad_compilers_ids = set()

    bad_groups = listed_groups.symmetric_difference(seen_groups)
    bad_formatters_exe = listed_formatters.symmetric_difference(seen_formatters_exe)
    bad_formatters_id = listed_formatters.symmetric_difference(seen_formatters_id)
    bad_libs_ids = listed_libs_ids.symmetric_difference(seen_libs_ids)
    bad_libs_versions = listed_libs_versions.symmetric_difference(seen_libs_versions)
    bad_tools_exe = listed_tools.symmetric_difference(seen_tools_exe)
    bad_tools_id = listed_tools.symmetric_difference(seen_tools_id)
    bad_default = default_compiler - listed_compilers
    return {
        "bad_compilers_exe": bad_compilers_exe - disabled,
        "bad_compilers_id": bad_compilers_ids - disabled,
        "bad_groups": bad_groups - disabled,
        "bad_formatters_exe": bad_formatters_exe - disabled,
        "bad_formatters_id": bad_formatters_id - disabled,
        "bad_libs_ids": bad_libs_ids - disabled,
        "bad_libs_versions": bad_libs_versions - disabled,
        "bad_tools_exe": bad_tools_exe - disabled,
        "bad_tools_id": bad_tools_id - disabled,
        "bad_default": bad_default,
        "empty_separators": empty_separators,
        "duplicate_lines": duplicate_lines,
        "duplicated_compiler_references": duplicated_compiler_references,
        "duplicated_group_references": duplicated_group_references,
        "suspicious_path": suspicious_path - disabled,
        "typo_compilers": seen_typo_compilers - disabled
    }


def process_folder(folder: str):
    return [(f, process_file(join(folder, f)))
            for f in listdir(folder)
            if isfile(join(folder, f))
            and not (f.endswith('.defaults.properties') or f.endswith('.local.properties'))
            and f.endswith('.properties')]


def problems_found(file_result):
    return any(len(file_result[r]) > 0 for r in file_result if r != "filename")


def print_issue(name, result):
    if len(result) > 0:
        sep = "\n  "
        print(f"{name}:\n  {sep.join(sorted([str(issue) for issue in result]))}")


def find_orphans(folder: str):
    result = sorted([(f, r) for (f, r) in process_folder(folder) if problems_found(r)], key=lambda x: x[0])
    if result:
        print(f"Found {len(result)} property file(s) with issues:")
        for (filename, issues) in result:
            print('################')
            print(f'## {filename}')
            for issue_key in issues:
                print_issue(issue_key, issues[issue_key])
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
