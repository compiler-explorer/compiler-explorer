<?php declare(strict_types=1);
// Copyright (c) 2017, Mike Cochrane
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

if (!extension_loaded('ast')) {
    echo "PHP AST extension required to generate ASTs";
    exit(1);
}

use ast\flags;

// Cache flag names for each AST node kind
$exclusiveFlags = $combinableFlags = [];
foreach (ast\get_metadata() as $metadata) {
    if (!$metadata->flags) {
        continue;
    }

    $flagNames = [];
    foreach ($metadata->flags as $flag) {
        // Just the last part of the namespaced costant name for display
        $flagNames[constant($flag)] = str_replace('ast\\flags\\', '', $flag);
    }

    if ($metadata->flagsCombinable) {
        $combinableFlags[$metadata->kind] = $flagNames;
    } else {
        $exclusiveFlags[$metadata->kind] = $flagNames;
    }
}

// Recursive function to dump a node as a string
$dump_ast_node = function ($node) use (&$dump_ast_node, $exclusiveFlags, $combinableFlags) : string {
    if ($node === null) {
        return 'null';
    } else if (is_string($node)) {
        return "\"$node\"";
    } elseif ($node instanceof ast\Node) {
        $result = ast\get_kind_name($node->kind);

        // Add line number
        $result .= " <line:$node->lineno";
        if (isset($node->endLineno)) {
            $result .= "-$node->endLineno";
        }
        $result .= '>';

        // Any flags set?
        if (ast\kind_uses_flags($node->kind) && $node->flags != 0) {
            $result .= PHP_EOL . '    flags: ';
            if (isset($exclusiveFlags[$node->kind]) && isset($exclusiveFlags[$node->kind][$node->flags])) {
                // This flag can only have one value
                $result .= $exclusiveFlags[$node->kind][$node->flags] . " ($node->flags)";
            } elseif (isset($combinableFlags[$node->kind])) {
                // This flag can have multiple values (bitwise OR)
                $result .= implode(
                    ' | ',
                    array_filter(
                        $combinableFlags[$node->kind],
                        function($v, $k) use ($node) {
                            return $k & $node->flags;
                        },
                        ARRAY_FILTER_USE_BOTH
                    )
                ) . " ($node->flags)";
            } else {
                $result .= $node->flags;
            }
        }

        if (isset($node->name) && $node->name) {
            $result .= PHP_EOL . "    name: $node->name";
        }
        if (isset($node->docComment) && $node->docComment) {
            $result .= PHP_EOL . "    docComment: $node->docComment";
        }

        // Recursively add any children
        foreach ($node->children as $i => $child) {
            $result .= PHP_EOL . "    $i: ";
            $result .= str_replace(PHP_EOL, PHP_EOL . '    ', $dump_ast_node($child));
        }
        return $result;
    }

    return (string) $node;
};

// Parse the file and dump the AST
echo $dump_ast_node(ast\parse_file($argv[1], 50)) . PHP_EOL;
