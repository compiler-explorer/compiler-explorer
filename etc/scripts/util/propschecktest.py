import unittest

from propscheck import process_file, Line


class PropsCheckTests(unittest.TestCase):
    def run_test(self, file, expected_key, expected_contents):
        result = process_file(file)
        self.assertEqual(result[expected_key], expected_contents)

    def test_bad_compilers(self):
        self.run_test("./test/cases/bad_compilers.properties", "bad_compilers", {"b"})

    def test_bad_compilers_alias(self):
        self.run_test("./test/cases/bad_compilers_aliases.properties", "bad_compilers", {"c"})

    def test_bad_compilers_disabled(self):
        self.run_test("./test/cases/bad_compilers_disabled.properties", "bad_compilers", set())

    def test_bad_groups(self):
        self.run_test("./test/cases/bad_groups.properties", "bad_groups", {"b"})

    def test_bad_formatters(self):
        self.run_test("./test/cases/bad_formatters.properties", "bad_formatters", {"b"})

    def test_bad_libs_ids(self):
        self.run_test("./test/cases/bad_libs_ids.properties", "bad_libs_ids", {"b"})
        self.run_test("./test/cases/bad_libs_ids.properties", "bad_libs_versions", set())

    def test_bad_libs_versions(self):
        self.run_test("./test/cases/bad_libs_versions.properties", "bad_libs_versions", {"a a2"})

    def test_bad_tools(self):
        self.run_test("./test/cases/bad_tools.properties", "bad_tools", {"b"})

    def test_bad_default(self):
        self.run_test("./test/cases/bad_default.properties", "bad_default", {"b"})

    def test_empty_separators(self):
        self.run_test("./test/cases/empty_separators.properties", "empty_separators", {
            str(Line(1, "compilers=a::b")),
            str(Line(1, "compilers=a::b")),
            str(Line(2, "compilers=::a:b")),
            str(Line(3, "compilers=a:b::")),
            str(Line(4, "compilers=::")),
            str(Line(5, "compilers=::a")),
            str(Line(6, "compilers=a::")),
            str(Line(7, "compilers=:a")),
            str(Line(8, "compilers=a:"))
        })

    def test_duplicate_lines(self):
        self.run_test("./test/cases/duplicate_lines.properties", "duplicate_lines",
                      {str(Line(5, "duplicated.prop=true"))})

    def test_duplicated_compiler(self):
        self.run_test("./test/cases/bad_duplicated_compiler.properties", "duplicated_compiler_references",
                      {"duplicatedname"})

    def test_duplicated_group(self):
        self.run_test("./test/cases/bad_duplicated_group.properties", "duplicated_group_references",
                      {"dupgroup"})

    def test_suspicious_path(self):
        self.run_test("./test/cases/suspicious_path.properties", "suspicious_path",
                      {"/wrong/path/bin/gcc"})

    def test_good_file(self):
        result = process_file('../../config/c++.amazon.properties')
        for k in result:
            if k != "filename":
                self.assertEqual(result[k], set(), f"{k} has output in known good file")


if __name__ == '__main__':
    unittest.main()
