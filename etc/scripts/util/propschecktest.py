import sys
import os
import unittest

from propscheck import process_file, Line


def sline(number, text):
    return str(Line(number, text))


class PropsCheckTests(unittest.TestCase):
    def run_test(self, filename, expected_key, expected_contents):
        base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        test_case_file = os.path.join(base_path, 'test', 'cases', f"{filename}.properties")
        result = process_file(test_case_file)
        self.assertEqual(result[expected_key], expected_contents)

    def test_bad_compilers(self):
        self.run_test("bad_compilers", "bad_compilers", {"b"})

    def test_bad_compilers_alias(self):
        self.run_test("bad_compilers_aliases", "bad_compilers", {"c"})

    def test_bad_compilers_disabled(self):
        self.run_test("bad_compilers_disabled", "bad_compilers", set())

    def test_bad_groups(self):
        self.run_test("bad_groups", "bad_groups", {"b"})

    def test_bad_formatters(self):
        self.run_test("bad_formatters", "bad_formatters", {"b"})

    def test_bad_libs_ids(self):
        self.run_test("bad_libs_ids", "bad_libs_ids", {"b"})
        self.run_test("bad_libs_ids", "bad_libs_versions", set())

    def test_bad_libs_versions(self):
        self.run_test("bad_libs_versions", "bad_libs_versions", {"a a2"})

    def test_bad_tools(self):
        self.run_test("bad_tools", "bad_tools", {"b"})

    def test_bad_default(self):
        self.run_test("bad_default", "bad_default", {"b"})

    def test_empty_separators(self):
        self.run_test("empty_separators", "empty_separators", {
            sline(1, "compilers=a::b"),
            sline(1, "compilers=a::b"),
            sline(2, "compilers=::a:b"),
            sline(3, "compilers=a:b::"),
            sline(4, "compilers=::"),
            sline(5, "compilers=::a"),
            sline(6, "compilers=a::"),
            sline(7, "compilers=:a"),
            sline(8, "compilers=a:")
        })

    def test_duplicate_lines(self):
        self.run_test("duplicate_lines", "duplicate_lines",
                      {sline(5, "duplicated.prop=true")})

    def test_duplicated_compiler(self):
        self.run_test("bad_duplicated_compiler", "duplicated_compiler_references",
                      {"duplicatedname"})

    def test_duplicated_group(self):
        self.run_test("bad_duplicated_group", "duplicated_group_references",
                      {"dupgroup"})

    def test_suspicious_path(self):
        self.run_test("suspicious_path", "suspicious_path",
                      {"/wrong/path/bin/gcc"})

    def test_good_file(self):
        base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        test_case_file = os.path.join(base_path, '..', '..', 'config', 'c++.amazon.properties')
        result = process_file(test_case_file)
        for k in result:
            if k != "filename":
                self.assertEqual(result[k], set(), f"{k} has output in known good file")

    def test_bad_compiler_ids(self):
        self.run_test("bad_compiler_ids", "bad_compiler_ids", {"bb"})

    def test_typo_compilers(self):
        self.run_test("typo_compilers", "typo_compilers", {sline(3, 'compilers.a.name=A')})


if __name__ == '__main__':
    unittest.main()
