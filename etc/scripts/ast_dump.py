"""Dumps the AST of a Python source file."""

import argparse
import ast
import sys


def main():
    parser = argparse.ArgumentParser(description="Parse and dump the AST of an Python source file")
    parser.add_argument("input", help="Path to input Python source code file",)
    args = parser.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            source = f.read()
    except IOError as err:
        # Log error and output empty array so CE handles it gracefully
        print("Error reading file: %s" % err)
        sys.exit(1)

    try:
        tree = ast.parse(source, optimize=0)
    except SyntaxError as err:
        # Output the syntax error as a single entry so CE can display it
        print("SyntaxError: %s" % err)
        sys.exit(1)

    print(ast.dump(tree, indent=2))


if __name__ == "__main__":
    main()
