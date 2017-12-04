// Copyright (c) 2012-2017, Matt Godbolt
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/*  id is what we internally use to identify the language.
        Same as the key so we can always query for it, everywhere.
            Used in: properties, url redirect, lang dropdown
    name is what we display to the user
            Used in: UI
    monaco is how we tell the monaco editor which language is this
            Used in: Monaco.editor.language
    extension is an array for the usual extensions of the language.
        The first one declared will be used as the default, else txt
        Leading point is needed
            Used in: Save to file extension
*/
function languages() {
    return {
        'c++': {
            id: 'c++',
            name: 'C++',
            monaco: 'cppp',
            extensions: ['.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c'],
            example: "// Type your code here, or load an example.\n" +
            "int square(int num) {\n" +
            "    return num * num;\n" +
            "}\n"
        },
        cppx: {
            id: 'cppx',
            name: 'Cppx',
            monaco: 'cppp',
            extensions: ['.cpp', '.cxx', '.h', '.hpp', '.hxx', '.c'],
            example: "//====================================================================\n" +
            "// Library code: implementing the metaclass (once)\n" +
            "\n" +
            "$class basic_value {\n" +
            "    basic_value() = default;\n" +
            "    basic_value(const basic_value& that) = default;\n" +
            "    basic_value(basic_value&& that) = default;\n" +
            "    basic_value& operator=(const basic_value& that) = default;\n" +
            "    basic_value& operator=(basic_value&& that) = default;\n" +
            "\n" +
            "    constexpr {\n" +
            "        for... (auto f : $basic_value.variables())\n" +
            "            if (!f.has_access()) f.make_private();\n" +
            "        for... (auto f : $basic_value.functions()) {\n" +
            "            if (!f.has_access()) f.make_public();\n" +
            "            compiler.require(!f.is_protected(), \"a value type may not have a protected function\");\n" +
            "            compiler.require(!f.is_virtual(),   \"a value type may not have a virtual function\");\n" +
            "            compiler.require(!f.is_destructor() || f.is_public(), \"a value destructor must be public\");\n" +
            "        }\n" +
            "    }\n" +
            "};\n" +
            "\n" +
            "$class value : basic_value { };\n" +
            "\n" +
            "\n" +
            "//====================================================================\n" +
            "// User code: using the metaclass to write a type (many times)\n" +
            "\n" +
            "value Point {\n" +
            "    int x = 0, y = 0;\n" +
            "    Point(int xx, int yy) : x{xx}, y{yy} { }\n" +
            "};\n" +
            "\n" +
            "Point get_some_point() { return {1,1}; }\n" +
            "\n" +
            "int main() {\n" +
            "\n" +
            "    Point p1(50,100), p2;\n" +
            "    p2 = get_some_point();\n" +
            "    p2.x = 42;\n" +
            "\n" +
            "}\n" +
            "\n" +
            "// Compiler Explorer note: Click the \"triangle ! icon\" to see the output:\n" +
            "constexpr {\n" +
            "    compiler.debug($Point);\n" +
            "}\n"
        },
        c: {
            id: 'c',
            name: 'C',
            monaco: 'c',
            extensions: ['.c', '.h'],
            example: "// Type your code here, or load an example.\n" +
            "int square(int num) {\n" +
            "    return num * num;\n" +
            "}\n"
        },
        rust: {
            id: 'rust',
            name: 'Rust',
            monaco: 'rust',
            extensions: ['.rs'],
            example: "// Type your code here, or load an example.\n" +
            "pub fn square(num: i32) -> i32 {\n" +
            "  num * num\n" +
            "}\n"
        },
        d: {
            id: 'd',
            name: 'D',
            monaco: 'd',
            extensions: ['.d'],
            example: "// Type your code here, or load an example.\n" +
            "int square(int num) {\n" +
            "  return num * num;\n" +
            "}\n"
        },
        go: {
            id: 'go',
            name: 'Go',
            monaco: 'go',
            extensions: ['.go'],
            example: "// Type your code here, or load an example.\n" +
            "// Your function name should start with a capital letter.\n" +
            "package main\n" +
            "\n" +
            "func Square(x int) int {\n" +
            "  return x * x\n" +
            "}\n" +
            "\n" +
            "func main() {}\n"
        },
        ispc: {
            id: 'ispc',
            name: 'ispc',
            monaco: 'ispc',
            extensions: ['.ispc'],
            example: "// Type your code here, or load an example.\n" +
            "uniform int square(uniform int num) {\n" +
            "    return num * num;\n" +
            "}\n"
        },
        haskell: {
            id: 'haskell',
            name: 'Haskell',
            monaco: 'haskell',
            extensions: ['.haskell'],
            example: "module Example where\n" +
            "\n" +
            "sumOverArray :: [Int] -> Int\n" +
            "sumOverArray (x:xs) = x + sumOverArray xs\n" +
            "sumOverArray [] =  0\n"
        },
        swift: {
            id: 'swift',
            name: 'Swift',
            monaco: 'swift',
            extensions: ['.swift'],
            example: "// Type your code here, or load an example.\n" +
            "func square(n: Int) -> Int {\n" +
            "    return n * n\n" +
            "}"
        },
        pascal: {
            id: 'pascal',
            name: 'Pascal',
            monaco: 'pascal',
            extenions: ['.pas'],
            example: "unit output;\n" +
            "\n" +
            "interface\n" +
            "\n" +
            "function Square(const num: Integer): Integer;\n" +
            "\n" +
            "implementation\n" +
            "\n" +
            "// Type your code here, or load an example.\n" +
            "\n" +
            "function Square(const num: Integer): Integer;\n" +
            "begin\n" +
            "  Square := num * num;\n" +
            "end;\n" +
            "\n" +
            "end.\n"
        }
    };
}

var _ = require('underscore-node');

function asArray() {
    return _.map(languages(), _.identity);
}
module.exports = {
    list: languages,
    toArray: asArray
};
