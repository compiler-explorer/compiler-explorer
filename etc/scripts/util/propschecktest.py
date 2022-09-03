import sys
import os
import unittest

from propscheck import process_file, Line


class PropsCheckTests(unittest.TestCase):
    def run_test(self, filename, expected_key, expected_contents):
        base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        test_case_file = os.path.join(base_path, 'test', 'cases', f"{filename}.properties")
        result = process_file(test_case_file)
        self.assertEqual(result[expected_key], {Line(-1, text) for text in expected_contents})

    def test_bad_compilers_exe(self):
        self.run_test("bad_compilers_exe", "bad_compilers_exe", {"b"})

    def test_bad_compilers_exe_alias(self):
        self.run_test("bad_compilers_exe_aliases", "bad_compilers_exe", {"c"})

    def test_bad_compilers_exe_disabled(self):
        self.run_test("bad_compilers_exe_disabled", "bad_compilers_exe", set())

    def test_bad_compilers_id(self):
        self.run_test("bad_compilers_id", "bad_compilers_id", {"bb"})

    def test_bad_groups(self):
        self.run_test("bad_groups", "bad_groups", {"b"})

    def test_bad_formatters_exe(self):
        self.run_test("bad_formatters_exe", "bad_formatters_exe", {"b"})

    def test_bad_formatters_id(self):
        self.run_test("bad_formatters_id", "bad_formatters_id", {"aa"})

    def test_bad_libs_ids(self):
        self.run_test("bad_libs_ids", "bad_libs_ids", {"b"})
        self.run_test("bad_libs_ids", "bad_libs_versions", set())

    def test_bad_libs_versions(self):
        self.run_test("bad_libs_versions", "bad_libs_versions", {"a a2"})

    def test_bad_tools_exe(self):
        self.run_test("bad_tools_exe", "bad_tools_exe", {"b"})

    def test_bad_tools_id(self):
        self.run_test("bad_tools_id", "bad_tools_id", {"aa"})

    def test_bad_default(self):
        self.run_test("bad_default", "bad_default", {"b"})

    def test_empty_separators(self):
        self.run_test("empty_separators", "empty_separators", {
            "compilers=a::b",
            "compilers=a::b",
            "compilers=::a:b",
            "compilers=a:b::",
            "compilers=::",
            "compilers=::a",
            "compilers=a::",
            "compilers=:a",
            "compilers=a:"
        })

    def test_duplicate_lines(self):
        self.run_test("duplicate_lines", "duplicate_lines", {"duplicated.prop=true"})

    def test_duplicated_compiler(self):
        self.run_test("bad_duplicated_compiler", "duplicated_compiler_references", {"duplicatedname"})

    def test_duplicated_group(self):
        self.run_test("bad_duplicated_group", "duplicated_group_references", {"dupgroup"})

    def test_suspicious_path(self):
        self.run_test("suspicious_path", "suspicious_path", {"/wrong/path/bin/gcc"})

    def test_good_file(self):
        base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        test_case_file = os.path.join(base_path, '..', '..', 'config', 'c++.amazon.properties')
        result = process_file(test_case_file)
        for k in result:
            self.assertEqual(result[k], set(), f"{k} has output in known good file")

    def test_typo_compilers(self):
        self.run_test("typo_compilers", "typo_compilers", {'compilers.a.name=A'})


if __name__ == '__main__':
    unittest.main()
