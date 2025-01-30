// Copyright (c) 2022, Compiler Explorer Authors
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

import $ from 'jquery';

import * as monaco from 'monaco-editor';
import * as cpp from 'monaco-editor/esm/vs/basic-languages/cpp/cpp';
import * as cppp from './cppp-mode.js';

function definition(): monaco.languages.IMonarchLanguage {
    const cppfront = $.extend(true, {}, cppp); // deep copy
    cppfront.tokenPostfix = '.herb';
    cppfront.defaultToken = 'invalid';

    // # Cpp2 token CSS class legend.
    //
    // Declaration/use of an _identifier_.
    // - 'identifier.definition'
    // - 'identifier.use'
    // - 'keyword.identifier.definition'
    // - 'keyword.identifier.use'
    //
    // Identifier that is contextually a type.
    // - 'type.contextual'
    // - 'keyword.type.contextual'
    //
    // Grammar term.
    // - 'keyword.contract-kind'
    // - 'keyword.this-specifier'
    // - 'keyword.parameter-direction'

    // Generic parsers.

    function parseCpp2Balanced(delimiters, delimiter, opener, closer) {
        return (cppfront.tokenizer['parse_cpp2_balanced_' + delimiters] = [
            {include: '@whitespace'},
            [opener, 'delimiter.' + delimiter, '$S2.$S3.$S4'],
            [closer, 'delimiter.' + delimiter, '@pop'],
            [/./, 'invalid', '@pop'],
        ]);
    }
    parseCpp2Balanced('angles', 'angle', /</, />/);
    parseCpp2Balanced('parentheses', 'parenthesis', /\(/, /\)/);
    parseCpp2Balanced('squares', 'square', /\[/, /\]/);
    parseCpp2Balanced('curlies', 'curly', /{/, /}/);
    cppfront.tokenizer.parse_cpp2_balanced_punctuators = [
        // `$S2` is the parser.
        [/</, {token: '@rematch', switchTo: 'parse_cpp2_balanced_angles.$S2.$S3'}],
        [/\(/, {token: '@rematch', switchTo: 'parse_cpp2_balanced_parentheses.$S2'}],
        [/\[/, {token: '@rematch', switchTo: 'parse_cpp2_balanced_squares.$S2.$S3'}],
        [/{/, {token: '@rematch', switchTo: 'parse_cpp2_balanced_curlies.$S2.$S3.$S4'}],
    ];

    // Until the single character `$S2`, parses `$S3`.
    cppfront.tokenizer.parse_cpp2_until = [
        {include: '@whitespace'},
        [
            /(.).*/,
            {
                cases: {
                    '$1==$S2': {token: '@rematch', next: '@pop'},
                    '@': {token: '@rematch', next: '$S3'},
                },
            },
        ],
    ];

    function parseCommaSeparated(state) {
        return [{include: '@whitespace'}, state, [/,/, 'delimiter'], [/./, '@rematch', '@pop']];
    }

    function setupLiteralParsers() {
        cppfront.at_cpp2_interpolation = /\([^"]+\)\$/;
        cppfront.tokenizer.parse_cpp2_interpolation = [
            [/(\()(.)/, ['delimiter.parenthesis', {token: '@rematch', next: 'parse_cpp2_expression'}]],
            [/:[^)]*/, 'string'],
            [/\)/, 'delimiter.parenthesis'],
            [/\$/, 'delimiter.interpolation', '@pop'],
        ];

        cppfront.at_cpp2_string_literal = /@encoding?(?:\$?R)?"/;
        cppfront.tokenizer.parse_cpp2_string_literal = [
            // Rules adapted from [Monaco's C++][].
            [/@encoding?\$?R"(?:([^ ()\\\t]*))\(/, {token: 'string.raw.begin', switchTo: '@raw.$1.cpp2'}],
            [/@encoding?"([^"\\]|\\.)*$/, 'string.invalid', '@pop'],
            [/@encoding?"/, {token: 'string', switchTo: '@string..cpp2'}],
        ];

        function parseCpp2Interpolation(state, prefix_token_regex, token_class) {
            cppfront.tokenizer[state].unshift([
                prefix_token_regex.source + /(@at_cpp2_interpolation)/.source,
                {
                    cases: {
                        '$S3==cpp2': [token_class, {token: '@rematch', next: 'parse_cpp2_interpolation'}],
                        '@': [token_class, token_class],
                    },
                },
            ]);
        }
        // Highlight interpolation also in non-interpolated raw string literal as it's most likely used for reflection.
        parseCpp2Interpolation('string', /([^\\"]*?)/, 'string');
        parseCpp2Interpolation('raw', /(.*?)/, 'string.raw');

        // Rules adapted from [Monaco's C++][]:
        cppfront.tokenizer.parse_cpp2_number_literal = [
            [/\d*\d+[eE]([-+]?\d+)?(@floatsuffix)/, 'number.float', '@pop'],
            [/\d*\.\d+([eE][-+]?\d+)?(@floatsuffix)/, 'number.float', '@pop'],
            [/0[xX][0-9a-fA-F']*[0-9a-fA-F](@integersuffix)/, 'number.hex', '@pop'],
            [/0[0-7']*[0-7](@integersuffix)/, 'number.octal', '@pop'],
            [/0[bB][0-1']*[0-1](@integersuffix)/, 'number.binary', '@pop'],
            [/\d[\d']*\d(@integersuffix)/, 'number', '@pop'],
            [/\d(@integersuffix)/, 'number', '@pop'],
            [/./, 'invalid', '@pop'],
        ];
        cppfront.tokenizer.parse_cpp2_character_literal = [
            [/'[^\\']'/, 'string', '@pop'],
            [/(')(@escapes)(')/, ['string', 'string.escape', {token: 'string', next: '@pop'}]],
            [/'/, 'string.invalid', '@pop'],
        ];
        cppfront.at_cpp2_literal_keyword = /(?:nullptr|true|false)\b/;
        cppfront.at_cpp2_literal = /\d|'|@at_cpp2_literal_keyword|@at_cpp2_string_literal/; // No `.`; `.0` isn't Cpp2.
        cppfront.tokenizer.parse_cpp2_literal = [
            [/\d/, {token: '@rematch', switchTo: 'parse_cpp2_number_literal'}],
            [/'/, {token: '@rematch', switchTo: 'parse_cpp2_character_literal'}],
            [/@at_cpp2_literal_keyword/, 'keyword.$0', '@pop'],
            [/@at_cpp2_string_literal/, {token: '@rematch', switchTo: 'parse_cpp2_string_literal'}],
            [/./, 'invalid', '@pop'],
        ];
    }
    setupLiteralParsers();

    function setupIdExpressionParsers() {
        cppfront.at_cpp2_is_as_operator = /(?:is|as)\b/;
        cppfront.at_cpp2_overloaded_operator_keyword = cppfront.at_cpp2_is_as_operator;
        cppfront.at_cpp2_overloaded_operator =
            /\/=|\/|<<=|<<|<=>|<=|<|>>=|>>|>=|>|\+\+|\+=|\+|--|-=|->|-|/.source +
            /\|\||\|=|\||&&|&=|&|\*=|\*|%=|%|\^=|\^|~|==|=|!=|!|\(\s*\)|\[\s*\]|/.source +
            cppfront.at_cpp2_overloaded_operator_keyword.source;

        cppfront.at_cpp2_non_operator_identifier = /[a-zA-Z_]\w*/;
        cppfront.at_cpp2_operator_identifier = /operator\b\s*@at_cpp2_overloaded_operator/;
        cppfront.at_cpp2_identifier = /@at_cpp2_non_operator_identifier|@at_cpp2_operator_identifier/;

        cppfront.tokenizer.parse_cpp2_identifier = [
            [
                /(operator)(\s+)(@at_cpp2_overloaded_operator_keyword)/,
                [{token: 'keyword.identifier.$S2'}, '', {token: 'keyword.identifier.$S2', next: '@pop'}],
            ],
            [
                /(operator\b)(\s*)(@at_cpp2_overloaded_operator)/,
                [{token: 'keyword.identifier.$S2'}, '', {token: 'delimiter', next: '@pop'}],
            ],
            [
                /(?:this|that)\b/,
                {
                    cases: {
                        '$S2==parameter': {token: 'keyword.identifier.definition', next: '@pop'},
                        '@': {token: 'keyword.identifier.$S2', next: '@pop'},
                    },
                },
            ],
            [/_(?=\s*=[^=])/, 'keyword.identifier.use', '@pop'],
            [
                /_\b/,
                {
                    cases: {
                        '$S2~definition|parameter': {token: 'keyword.identifier.definition', next: '@pop'}, // Anonymous definition.
                        '$S2==using_statement': {token: 'keyword', next: '@pop'},
                        '@': {token: '@rematch', switchTo: 'parse_cpp2_non_operator_identifier.$S2'},
                    },
                },
            ],
            [/./, {token: '@rematch', switchTo: 'parse_cpp2_non_operator_identifier.$S2'}],
        ];
        cppfront.tokenizer.parse_cpp2_non_operator_identifier = [
            [
                /@at_cpp2_non_operator_identifier/,
                {
                    cases: {
                        '$S2~definition|parameter': {
                            token: 'identifier.definition',
                            switchTo: 'parse_cpp2_parameter_ellipsis.$S2',
                        },
                        '$S2==type': {token: 'type.contextual', next: '@pop'},
                        '$S2==contract_kind': {token: 'keyword.contract-kind', next: '@pop'},
                        '$S2==builtin_meta_function': {token: 'keyword', next: '@pop'},
                        '@': {token: 'identifier.use', next: '@pop'},
                    },
                },
            ],
        ];
        cppfront.tokenizer.parse_cpp2_parameter_ellipsis = [
            [
                /\.\.\./,
                {
                    cases: {
                        '$S2==parameter': {token: 'delimiter.ellipsis', next: '@pop'},
                        '@': {token: '@rematch', next: '@pop'},
                    },
                },
            ],
            [/./, '@rematch', '@pop'],
        ];

        cppfront.at_cpp2_unqualified_id_template_type_keyword = /(?:finally|cpp1_ref|cpp1_rvalue_ref)\b/;
        cppfront.at_cpp2_unqualified_id_template_expression_keyword = /(?:new|unchecked_(?:narrow|cast))\b/;
        cppfront.at_cpp2_unqualified_id_type_function_keyword = /(?:type_of|static_assert)\b(?=\s*\()/;
        cppfront.at_cpp2_unqualified_id_keywords = /(unique|shared)(\.)(new(?=<))/;
        cppfront.at_cpp2_unqualified_id_keyword =
            /@at_cpp2_unqualified_id_template_type_keyword|@at_cpp2_unqualified_id_template_expression_keyword|@at_cpp2_unqualified_id_type_function_keyword|@at_cpp2_unqualified_id_keywords/;
        cppfront.tokenizer.parse_cpp2_unqualified_id_keywords = [
            [
                /@at_cpp2_unqualified_id_keywords/,
                ['keyword', 'delimiter', {token: 'keyword.$3', switchTo: 'parse_cpp2_template_argument_list'}],
            ],
            [
                /@at_cpp2_unqualified_id_template_type_keyword(?=<)/,
                {token: 'keyword.type', switchTo: 'parse_cpp2_template_argument_list'},
            ],
            [
                /@at_cpp2_unqualified_id_template_expression_keyword(?=<)/,
                {
                    cases: {
                        '$S2==expression': {token: 'keyword', switchTo: 'parse_cpp2_template_argument_list'},
                        '@': {token: 'invalid', switchTo: 'parse_cpp2_template_argument_list'},
                    },
                },
            ],
            [
                /@at_cpp2_unqualified_id_type_function_keyword(?=\s*\()/,
                {token: 'keyword.type', switchTo: 'parse_cpp2_expression_list'},
            ],
            [
                /@at_cpp2_unqualified_id_template_type_keyword(?=\s*\()/,
                {
                    cases: {
                        '$S2==expression': {token: 'keyword.type', switchTo: 'parse_cpp2_expression_list'}, // CTAD in _expression_.
                        '@': {token: 'invalid', switchTo: 'parse_cpp2_expression_list'},
                    },
                },
            ],
            [
                /@at_cpp2_unqualified_id_template_type_keyword/,
                {
                    cases: {
                        '$S2==type': {token: 'keyword.type', next: '@pop'}, // CTAD in _type-id_.
                        '@': {token: 'invalid', next: '@pop'},
                    },
                },
            ],
        ];

        cppfront.at_cpp2_type_qualifier = /const\b|\*/;
        cppfront.tokenizer.parse_cpp2_type_qualifier_seq = [
            {include: '@whitespace'},
            [/const\b/, 'keyword'],
            [/\*/, 'delimiter'],
            [/./, '@rematch', '@pop'],
        ];

        cppfront.at_cpp2_keyword_type =
            /(?:[iu](?:8|16|32|64)|u?(?:short|int|long|longlong)|(?:un)?signed|void|bool|(?:_[su])?char|(?:long)?double|float)\b/;

        cppfront.at_cpp2_type_id =
            /@at_cpp2_type_qualifier|@at_cpp2_non_operator_id_expression|@at_cpp2_function_type_id/;
        cppfront.at_cpp2_function_type_id =
            /\(\s*\)|\(\s*(?:(?:@at_cpp2_parameter_direction\s+)?@at_cpp2_non_operator_identifier\s*)?@at_cpp2_unnamed_declaration_head/;
        cppfront.tokenizer.parse_cpp2_type_id = [
            [/@at_cpp2_type_qualifier/, '@rematch', 'parse_cpp2_type_qualifier_seq'],
            [/@at_cpp2_keyword_type\s*@at_cpp2_keyword_type/, 'invalid', '@pop'],
            [/@at_cpp2_keyword_type|_\b/, 'keyword.type.contextual', '@pop'],
            [
                /@at_cpp2_unqualified_id_keyword/,
                {token: '@rematch', switchTo: 'parse_cpp2_unqualified_id_keywords.type'},
            ],
            [/=/, '@rematch', '@pop'],
            [/@at_cpp2_non_operator_id_expression/, {token: '@rematch', switchTo: 'parse_cpp2_id_expression.type'}],
            [/\(/, {token: '@rematch', switchTo: 'parse_cpp2_function_type'}],
        ];

        cppfront.at_cpp2_builtin_contract_group = /(?:default|type_safety|bounds_safety|null_safety|testing)\b/;
        cppfront.at_cpp2_template_argument = /@at_cpp2_string_literal|@at_cpp2_expression|@at_cpp2_type_id/;
        cppfront.tokenizer.parse_cpp2_template_argument = [
            [/@at_cpp2_keyword_type/, 'keyword.type', '@pop'],
            [
                /@at_cpp2_builtin_contract_group/,
                {
                    cases: {
                        '$S2==contract_kind': {token: 'keyword.contract-group', next: '@pop'},
                        '@': {token: '@rematch', switchTo: 'parse_cpp2_template_argument_rest'},
                    },
                },
            ],
            [/./, {token: '@rematch', switchTo: 'parse_cpp2_template_argument_rest'}],
        ];
        cppfront.tokenizer.parse_cpp2_template_argument_rest = [
            [/@at_cpp2_function_type_id/, {token: '@rematch', switchTo: 'parse_cpp2_type_id'}],
            [/@at_cpp2_type_qualifier/, {token: '@rematch', switchTo: 'parse_cpp2_type_id'}],
            [/@at_cpp2_expression/, {token: '@rematch', switchTo: 'parse_cpp2_expression.template_argument'}],
            [/@at_cpp2_type_id/, {token: '@rematch', switchTo: 'parse_cpp2_type_id'}],
        ];
        cppfront.tokenizer.parse_cpp2_template_argument_seq = parseCommaSeparated([
            /@at_cpp2_template_argument/,
            '@rematch',
            'parse_cpp2_template_argument.$S2',
        ]);
        cppfront.tokenizer.parse_cpp2_template_argument_list = [
            [/</, {token: '@rematch', switchTo: 'parse_cpp2_balanced_angles.parse_cpp2_template_argument_seq.$S2'}],
        ];

        cppfront.at_cpp2_id_expression = /::|@at_cpp2_identifier/;
        cppfront.at_cpp2_non_operator_id_expression = /::|@at_cpp2_non_operator_identifier/;
        cppfront.tokenizer.parse_cpp2_template_id = [
            [/@at_cpp2_identifier/, '@rematch', 'parse_cpp2_identifier.$S2'],
            [/</, '@rematch', 'parse_cpp2_template_argument_list.$S2'],
            [/\s*(?:\.\.?|::)/, {token: '@rematch', switchTo: 'parse_cpp2_id_expression.$S2'}],
            [/./, '@rematch', '@pop'],
        ];
        cppfront.tokenizer.parse_cpp2_id_expression = [
            {include: '@whitespace'},
            [/::/, ''],
            [/\.\.[.<=]/, '@rematch', '@pop'],
            [/\.\.?/, 'delimiter'],
            [/@at_cpp2_identifier</, {token: '@rematch', switchTo: 'parse_cpp2_template_id.$S2'}],
            [/@at_cpp2_non_operator_identifier(?=\s*(?:\.\.?|::))/, '@rematch', 'parse_cpp2_identifier.use'],
            [/@at_cpp2_identifier/, {token: '@rematch', switchTo: 'parse_cpp2_identifier.$S2'}],
        ];
    }
    setupIdExpressionParsers();

    function setupExpressionParsers() {
        cppfront.tokenizer.parse_cpp2_primary_expression_id_expression = [
            [
                /@at_cpp2_unqualified_id_keyword/,
                {token: '@rematch', switchTo: 'parse_cpp2_unqualified_id_keywords.expression'},
            ],
            [/@at_cpp2_keyword_type/, 'keyword.type', '@pop'],
            [/./, {token: '@rematch', switchTo: 'parse_cpp2_id_expression.use'}],
        ];

        cppfront.tokenizer.parse_cpp2_expression_seq = parseCommaSeparated([
            /@at_cpp2_expression/,
            '@rematch',
            'parse_cpp2_expression',
        ]);
        cppfront.tokenizer.parse_cpp2_expression_list = [
            [/./, {token: '@rematch', switchTo: 'parse_cpp2_balanced_punctuators.parse_cpp2_expression_seq'}],
        ];

        cppfront.at_cpp2_primary_expression =
            /@at_cpp2_literal|@at_cpp2_id_expression|\.\.\.|\(|@at_cpp2_unnamed_declaration_head/;
        // Can't ensure sequential parsing.
        cppfront.tokenizer.parse_cpp2_primary_expression = [
            [/inspect\b/, '@rematch', 'parse_cpp2_inspect_expression'],
            // These two happen to parse UDLs:
            [/@at_cpp2_literal/, {token: '@rematch', switchTo: 'parse_cpp2_primary_expression_literal_.$S2.$S3'}],
            [
                /@at_cpp2_id_expression/,
                {token: '@rematch', switchTo: 'parse_cpp2_primary_expression_id_expression_.$S2.$S3'},
            ],
            [/\.\.\./, 'delimiter.ellipsis'],
            // Handle `(` later to workaround `(0)is` being parsed as two adjacent primary expressions.
            [/@at_cpp2_unnamed_declaration_head/, '@rematch', 'parse_cpp2_declaration.expression'],
            [/./, {token: '@rematch', switchTo: 'parse_cpp2_postfix_expression.$S2.$S3'}],
        ];
        cppfront.tokenizer.parse_cpp2_primary_expression_id_expression_ = [
            [/@at_cpp2_id_expression/, '@rematch', 'parse_cpp2_primary_expression_id_expression'],
            [/\.\.\./, 'delimiter.ellipsis'],
            [/./, {token: '@rematch', switchTo: 'parse_cpp2_postfix_expression.$S2.$S3'}],
        ];
        cppfront.tokenizer.parse_cpp2_primary_expression_literal_ = [
            [/@at_cpp2_literal/, '@rematch', 'parse_cpp2_literal'],
            [/./, {token: '@rematch', switchTo: 'parse_cpp2_postfix_expression.$S2.$S3'}],
        ];

        cppfront.at_cpp2_postfix_operator =
            /\+\+|--|~|\$|[*&](?!\s*(?=\(|@at_cpp2_non_operator_identifier|@at_cpp2_literal))/;
        cppfront.tokenizer.parse_cpp2_postfix_expression = [
            {include: '@whitespace'},
            [/@at_cpp2_postfix_operator/, 'delimiter.postfix-operator'],
            [/[[(]/, {token: '@rematch', next: 'parse_cpp2_expression_list'}],
            [
                /(\.\.?)(\s*)(@at_cpp2_id_expression)/,
                ['delimiter', '', {token: '@rematch', next: 'parse_cpp2_id_expression'}],
            ],
            [
                /(\.\.[<=])(\s*)/,
                ['delimiter.range_operator', {token: '', switchTo: 'parse_cpp2_primary_expression.stop_at_postfix'}],
            ],
            [
                /./,
                {
                    cases: {
                        '$S2==stop_at_postfix': {token: '@rematch', next: '@pop'},
                        '@': {
                            token: '@rematch',
                            switchTo:
                                'parse_cpp2_is_as_expression_target.parse_cpp2_expression.' +
                                'parse_cpp2_binary_expression_tail.$S2.$S3',
                        },
                    },
                },
            ],
        ];

        cppfront.at_cpp2_prefix_expression = /@at_cpp2_prefix_operator_delimiter|@at_cpp2_primary_expression/;
        cppfront.at_cpp2_parameter_direction = /(?:in(?:_ref)?|inout|copy|out|move|forward(?:_ref)?)\b/;
        cppfront.at_cpp2_prefix_operator_delimiter = /!|-|\+/;
        cppfront.tokenizer.parse_cpp2_prefix_expression = [
            {include: '@whitespace'},
            [
                /@at_cpp2_parameter_direction(?=\s*@at_cpp2_primary_expression)/,
                'keyword.parameter-direction.prefix-operator',
            ],
            [/@at_cpp2_prefix_operator_delimiter/, 'delimiter.prefix-operator'],
            [/@at_cpp2_primary_expression/, {token: '@rematch', switchTo: 'parse_cpp2_primary_expression.$S2.$S3'}],
            [/./, 'invalid', '@pop'],
        ];

        cppfront.tokenizer.parse_cpp2_is_as_expression_target = [
            // `@$S2` is the expression parser.
            // `@$S3.$S4.$S5` is the continuation parser.
            {include: '@whitespace'},
            [
                /(@at_cpp2_is_as_operator)(\s+)(@at_cpp2_type_id)/,
                ['keyword', '', {token: '@rematch', switchTo: 'parse_cpp2_type_id'}],
            ],
            [/(is\b)(\s*)(@at_cpp2_expression)/, ['keyword', '', {token: '@rematch', switchTo: '@$S2'}]],
            [/./, {token: '@rematch', switchTo: '@$S3.$S4.$S5'}],
        ];

        cppfront.at_cpp2_logical_or_operator = /\*|\/|%|\+|-|<<|>>|<=>|<|>|<=|>=|==|!=|&|\^|\||&&|\|\|/;
        cppfront.at_cpp2_assignment_operator = /=|\*=|\/=|%=|\+=|-=|>>=|<<=|&=|\^=|\|=/;
        cppfront.tokenizer.parse_cpp2_binary_expression_tail = [
            [/./, {token: '@rematch', switchTo: 'parse_cpp2_binary_expression_tail_preface.$S2.$S3'}],
        ];
        cppfront.tokenizer.parse_cpp2_binary_expression_tail_preface = [
            {include: '@whitespace'},
            [
                />>?(?!=)/,
                {
                    cases: {
                        '$S3==template_argument': {token: '@rematch', next: '@pop'},
                        '@': {token: '@rematch', switchTo: '@parse_cpp2_binary_expression_tail_body.$S2.$S3'},
                    },
                },
            ],
            [/./, {token: '@rematch', switchTo: '@parse_cpp2_binary_expression_tail_body.$S2.$S3'}],
        ];
        cppfront.tokenizer.parse_cpp2_binary_expression_tail_body = [
            {include: '@whitespace'},
            [
                /(@at_cpp2_logical_or_operator)(\s*)(@at_cpp2_prefix_expression)/,
                ['delimiter', '', {token: '@rematch', next: 'parse_cpp2_prefix_expression.$S2.$S3'}],
            ],
            [
                /@at_cpp2_assignment_operator/,
                {
                    cases: {
                        '$S2==assignment': {token: 'delimiter', switchTo: 'parse_cpp2_prefix_expression.$S2.$S3'},
                        '@': {token: '@rematch', next: '@pop'},
                    },
                },
            ],
            [/./, '@rematch', '@pop'],
        ];

        cppfront.tokenizer.parse_cpp2_logical_or_expression = [
            [/./, {token: '@rematch', switchTo: 'parse_cpp2_prefix_expression'}],
        ];
        cppfront.tokenizer.parse_cpp2_assignment_expression = [
            [/./, {token: '@rematch', switchTo: 'parse_cpp2_prefix_expression.assignment.$S2'}],
        ];

        cppfront.at_cpp2_expression = cppfront.at_cpp2_prefix_expression;
        cppfront.tokenizer.parse_cpp2_expression = [
            [/./, {token: '@rematch', switchTo: 'parse_cpp2_assignment_expression.$S2'}],
        ];
    }
    setupExpressionParsers();

    function setupStatementParsers() {
        cppfront.tokenizer.parse_cpp2_selection_statement = [
            {include: '@whitespace'},
            [
                /(if)(\s+)(constexpr\b)/,
                ['keyword.if', '', {token: 'keyword.constexpr', next: 'parse_cpp2_logical_or_expression'}],
            ],
            [/if\b/, 'keyword.if', 'parse_cpp2_logical_or_expression'],
            [/{/, '@rematch', 'parse_cpp2_compound_statement'],
            [/else\b/, 'keyword.else'],
            [/./, '@rematch', '@pop'],
        ];

        cppfront.tokenizer.parse_cpp2_using_statement = [
            {include: '@whitespace'},
            [/using\b/, {token: 'keyword.using', switchTo: 'parse_cpp2_id_expression.using_statement'}],
        ];

        cppfront.tokenizer.parse_cpp2_alternative = [
            {include: '@whitespace'},
            [
                /@at_cpp2_is_as_operator/,
                '@rematch',
                'parse_cpp2_is_as_expression_target.parse_cpp2_logical_or_expression.pop',
            ],
            [/@at_cpp2_non_operator_identifier/, '@rematch', 'parse_cpp2_identifier.definition'],
            [/@at_cpp2_unnamed_declaration_head/, 'identifier.definition'],
            [/=/, {token: 'delimiter', switchTo: 'parse_cpp2_statement'}],
            [/./, 'invalid', '@pop'],
        ];
        cppfront.tokenizer.parse_cpp2_inspect_expression = [
            {include: '@whitespace'},
            [
                /(inspect)(\s+)(constexpr\b)/,
                ['keyword.inspect', '', {token: 'keyword.constexpr', next: 'parse_cpp2_expression'}],
            ],
            [/inspect\b/, 'keyword.inspect', 'parse_cpp2_expression'],
            [/->/, '@rematch', 'parse_cpp2_return_list'],
            [
                /{/,
                {token: '@rematch', switchTo: 'parse_cpp2_balanced_curlies.parse_cpp2_until.}.parse_cpp2_alternative'},
            ],
            [/./, 'invalid', '@pop'],
        ];

        cppfront.tokenizer.parse_cpp2_return_statement = [
            [/(return)(\s*)(;)/, ['keyword.return', '', {token: 'delimiter', next: '@pop'}]],
            [/return\b/, {token: 'keyword.return', switchTo: 'parse_cpp2_expression_statement'}],
        ];

        cppfront.at_cpp2_jump_statement = /(?:break|continue)\b/;
        cppfront.tokenizer.parse_cpp2_jump_statement = [
            {include: '@whitespace'},
            [/@at_cpp2_jump_statement/, {token: 'keyword.$0'}],
            [/@at_cpp2_non_operator_identifier/, '@rematch', 'parse_cpp2_identifier'],
            [/;/, 'delimiter', '@pop'],
        ];

        cppfront.tokenizer.parse_cpp2_next_clause = [[/next\b/, 'keyword.next', 'parse_cpp2_assignment_expression']];
        cppfront.tokenizer.parse_cpp2_while_statement = [
            {include: '@whitespace'},
            [/while\b/, 'keyword.while', 'parse_cpp2_logical_or_expression'],
            {include: '@parse_cpp2_next_clause'},
            [/{/, {token: '@rematch', switchTo: 'parse_cpp2_compound_statement'}],
        ];
        cppfront.tokenizer.parse_cpp2_do_statement = [
            {include: '@whitespace'},
            [/do\b/, 'keyword.do', 'parse_cpp2_compound_statement'],
            [/while\b/, 'keyword.while', 'parse_cpp2_logical_or_expression'],
            {include: '@parse_cpp2_next_clause'},
            [/;/, 'delimiter', '@pop'],
        ];
        cppfront.tokenizer.parse_cpp2_for_statement = [
            {include: '@whitespace'},
            [/for\b/, 'keyword.for', 'parse_cpp2_expression'],
            {include: '@parse_cpp2_next_clause'},
            [/(do\b)(\s*)/, ['keyword.do', {token: '', switchTo: 'parse_cpp2_parameterized_statement'}]],
        ];
        cppfront.at_cpp2_iteration_statement_head = /(?:while|do|for)\b/;
        cppfront.tokenizer.parse_cpp2_iteration_statement = [
            [/while\b/, {token: '@rematch', switchTo: 'parse_cpp2_while_statement'}],
            [/do\b/, {token: '@rematch', switchTo: 'parse_cpp2_do_statement'}],
            [/for\b/, {token: '@rematch', switchTo: 'parse_cpp2_for_statement'}],
        ];

        cppfront.tokenizer.parse_cpp2_compound_statement = [
            [/{/, {token: '@rematch', switchTo: 'parse_cpp2_balanced_curlies.parse_cpp2_until.}.parse_cpp2_statement'}],
        ];

        cppfront.at_cpp2_parameterized_statement =
            /\(\s*(?:@at_cpp2_parameter_direction\s+)?@at_cpp2_identifier_definition/;
        cppfront.tokenizer.parse_cpp2_parameterized_statement = [
            [/\(/, '@rematch', 'parse_cpp2_parameter_declaration_list'],
            [/./, {token: '@rematch', switchTo: 'parse_cpp2_statement'}],
        ];

        cppfront.tokenizer.parse_cpp2_expression_statement = [
            {include: '@whitespace'},
            [/@at_cpp2_expression/, '@rematch', 'parse_cpp2_expression'],
            // Handle optional `;` after unbraced statement of unnamed declaration:
            [/;(?=\s*(?:[()\],]|@at_cpp2_is_as_operator))/, 'delimiter', '@pop'],
            [
                /;/,
                {
                    cases: {
                        '$S2==expression': {token: '@rematch', next: '@pop'},
                        '@': {token: 'delimiter', next: '@pop'},
                    },
                },
            ],
            [
                /./,
                {
                    cases: {
                        '$S2~expression|parameter': {token: '@rematch', next: '@pop'},
                        '@': {token: 'invalid', next: '@pop'},
                    },
                },
            ],
        ];

        cppfront.at_cpp2_contract_kind = /(?:pre|post|assert)\b/;
        cppfront.tokenizer.parse_cpp2_contract = [
            [/@at_cpp2_contract_kind/, '@rematch', 'parse_cpp2_id_expression.contract_kind'],
            [/\(/, {token: '@rematch', switchTo: 'parse_cpp2_expression_list'}],
            [/./, '@rematch', '@pop'],
        ];

        cppfront.tokenizer.parse_cpp2_statement = [
            {include: '@whitespace'},
            [/if\b/, {token: '@rematch', switchTo: 'parse_cpp2_selection_statement'}],
            [/using\b/, {token: '@rematch', switchTo: 'parse_cpp2_using_statement'}],
            [/inspect\b/, {token: '@rematch', switchTo: 'parse_cpp2_inspect_expression'}],
            [/return\b/, {token: '@rematch', switchTo: 'parse_cpp2_return_statement'}],
            [/@at_cpp2_jump_statement/, {token: '@rematch', switchTo: 'parse_cpp2_jump_statement'}],
            [/@at_cpp2_iteration_statement_head/, {token: '@rematch', switchTo: 'parse_cpp2_iteration_statement'}],
            [/{/, {token: '@rematch', switchTo: 'parse_cpp2_compound_statement'}],
            [/@at_cpp2_declaration_head/, {token: '@rematch', switchTo: 'parse_cpp2_declaration.definition'}],
            [/@at_cpp2_parameterized_statement/, {token: '@rematch', switchTo: 'parse_cpp2_parameterized_statement'}],
            [/@at_cpp2_contract_kind/, {token: '@rematch', switchTo: 'parse_cpp2_contract'}],
            [/./, {token: '@rematch', switchTo: 'parse_cpp2_expression_statement.$S2'}],
        ];
    }
    setupStatementParsers();

    function setupDeclarationParsers() {
        cppfront.at_cpp2_builtin_meta_function =
            /(?:ordered|weakly_ordered|partially_ordered|copyable|basic_value|value|weakly_ordered_value|partially_ordered_value|struct|hashable|interface|polymorphic_base|enum|flag_enum|union|regex|cpp1_rule_of_zero|print)\b/;
        cppfront.tokenizer.parse_cpp2_meta_functions_list = [
            [/[^@]/, '@rematch', '@pop'],
            [
                /(@)(\s*)(@at_cpp2_builtin_meta_function)/,
                ['delimiter', '', {token: '@rematch', next: 'parse_cpp2_id_expression.builtin_meta_function'}],
            ],
            [
                /(@)(\s*)(@at_cpp2_non_operator_id_expression)/,
                ['delimiter', '', {token: '@rematch', next: 'parse_cpp2_id_expression'}],
            ],
        ];

        cppfront.at_cpp2_parameter_declaration = /@at_cpp2_non_operator_identifier|@at_cpp2_unnamed_declaration_head/;
        cppfront.at_cpp2_this_specifier = /(?:implicit|virtual|override|final)\b/;
        cppfront.tokenizer.parse_cpp2_parameter_declaration = [
            [
                /(@at_cpp2_parameter_direction)(\s*)(@at_cpp2_unnamed_declaration_head)/,
                ['keyword.parameter-direction', '', {token: '@rematch', switchTo: 'parse_cpp2_declaration.parameter'}],
            ],
            [
                /@at_cpp2_declaration_head|@at_cpp2_unnamed_declaration_head/,
                {token: '@rematch', switchTo: 'parse_cpp2_declaration.parameter.$S2'},
            ],
            [
                /(@at_cpp2_parameter_direction)(\s+)(@at_cpp2_declaration_head)/,
                ['keyword.parameter-direction', '', {token: '@rematch', switchTo: 'parse_cpp2_declaration.parameter'}],
            ],
            [
                /(@at_cpp2_this_specifier)(\s+)(@at_cpp2_declaration_head)/,
                ['keyword.this-specifier', '', {token: '@rematch', switchTo: 'parse_cpp2_declaration.parameter'}],
            ],
            [
                /(@at_cpp2_this_specifier)(\s+)(@at_cpp2_parameter_direction)(\s+)(@at_cpp2_declaration_head)/,
                [
                    'keyword.this-specifier',
                    '',
                    'keyword.parameter-direction',
                    '',
                    {token: '@rematch', switchTo: 'parse_cpp2_declaration.parameter'},
                ],
            ],
            [
                /(@at_cpp2_parameter_direction)(\s+)(@at_cpp2_non_operator_identifier)/,
                ['keyword.parameter-direction', '', {token: '@rematch', switchTo: 'parse_cpp2_identifier.parameter'}],
            ],
            [
                /(@at_cpp2_this_specifier)(\s+)(@at_cpp2_parameter_direction)(\s+)(@at_cpp2_non_operator_identifier)/,
                [
                    'keyword.this-specifier',
                    '',
                    'keyword.parameter-direction',
                    '',
                    {token: '@rematch', switchTo: 'parse_cpp2_identifier.parameter'},
                ],
            ],
            [
                /(@at_cpp2_this_specifier)(\s+)(@at_cpp2_non_operator_identifier)/,
                ['keyword.this-specifier', '', {token: '@rematch', switchTo: 'parse_cpp2_identifier.parameter'}],
            ],
            [/@at_cpp2_non_operator_identifier/, {token: '@rematch', switchTo: 'parse_cpp2_identifier.parameter'}],
        ];
        cppfront.tokenizer.parse_cpp2_parameter_declaration_seq = parseCommaSeparated([
            /@at_cpp2_parameter_declaration/,
            '@rematch',
            'parse_cpp2_parameter_declaration.$S2',
        ]);
        cppfront.tokenizer.parse_cpp2_parameter_declaration_list = [
            [
                /./,
                {
                    token: '@rematch',
                    switchTo: 'parse_cpp2_balanced_punctuators.parse_cpp2_parameter_declaration_seq.$S2',
                },
            ],
        ];

        cppfront.tokenizer.parse_cpp2_return_list = [
            [
                /(->)(\s*)(@at_cpp2_parameter_direction)(\s+)(@at_cpp2_type_id)/,
                [
                    'delimiter',
                    '',
                    'keyword.parameter-direction',
                    '',
                    {token: '@rematch', switchTo: 'parse_cpp2_type_id'},
                ],
            ],
            [/(->)(\s*)(@at_cpp2_type_id)/, ['delimiter', '', {token: '@rematch', switchTo: 'parse_cpp2_type_id'}]],
            [/->/, 'invalid', '@pop'],
        ];

        cppfront.tokenizer.parse_cpp2_function_type = [
            {include: '@whitespace'},
            [/\(/, '@rematch', 'parse_cpp2_parameter_declaration_list'],
            [/throws\b/, 'keyword'],
            [/->/, '@rematch', 'parse_cpp2_return_list'],
            [/@at_cpp2_contract_kind/, '@rematch', 'parse_cpp2_contract'],
            [/./, '@rematch', '@pop'],
        ];

        cppfront.tokenizer.parse_cpp2_declaration_initializer = [
            [/./, {token: '@rematch', switchTo: 'parse_cpp2_statement.$S2'}],
        ];

        cppfront.tokenizer.parse_cpp2_declaration_signature = [
            {include: '@whitespace'},
            [/@/, '@rematch', 'parse_cpp2_meta_functions_list'],
            [/</, '@rematch', 'parse_cpp2_parameter_declaration_list.template'],
            [/\(/, '@rematch', 'parse_cpp2_function_type'],
            [/requires\b/, 'keyword', 'parse_cpp2_logical_or_expression'],
            [/final\b/, 'keyword'],
            [/type\b/, {token: 'keyword', switchTo: 'parse_cpp2_declaration_signature.type.$S3'}],
            [/namespace\b/, 'keyword'],
            [/@at_cpp2_iteration_statement_head/, {token: '@rematch', switchTo: 'parse_cpp2_iteration_statement'}],
            [/@at_cpp2_is_as_operator/, '@rematch', 'parse_cpp2_is_as_expression_target.pop.pop'],
            [/@at_cpp2_type_id/, '@rematch', 'parse_cpp2_type_id'],
            [
                /==/,
                {
                    cases: {
                        '$S2==type': {token: 'delimiter', next: 'parse_cpp2_type_id'},
                        '@': {token: 'delimiter', switchTo: 'parse_cpp2_declaration_initializer.$S2'},
                    },
                },
            ],
            [
                /=/,
                {
                    cases: {
                        '$S3==template': {token: 'delimiter', next: 'parse_cpp2_type_id'},
                        '@': {token: 'delimiter', switchTo: 'parse_cpp2_declaration_initializer.$S2'},
                    },
                },
            ],
            [
                /;/,
                {
                    cases: {
                        '$S2==expression': {token: '@rematch', next: '@pop'},
                        '@': {token: 'delimiter', next: '@pop'},
                    },
                },
            ],
            [/./, '@rematch', '@pop'],
        ];

        cppfront.at_cpp2_unnamed_declaration_head = /:(?!:)/;
        cppfront.at_cpp2_identifier_definition =
            /@at_cpp2_identifier\s*(?:\.\.\.)?\s*@at_cpp2_unnamed_declaration_head/;
        cppfront.at_cpp2_top_level_declaration_head =
            /(?:@at_cpp2_access_specifier\s+)?(?!@at_cpp2_access_specifier)@at_cpp2_identifier_definition/;
        cppfront.at_cpp2_declaration_head = /(?:@at_cpp2_access_specifier\s+)?@at_cpp2_identifier_definition/;
        cppfront.at_cpp2_access_specifier = /(?:export|public|protected|private)\b/;
        cppfront.tokenizer.parse_cpp2_declaration_head = [
            {include: '@whitespace'},
            [/@at_cpp2_access_specifier/, 'keyword'],
            [/@at_cpp2_identifier/, '@rematch', 'parse_cpp2_identifier.$S2'],
            [/\.\.\./, 'delimiter.ellipsis'],
            [
                /@at_cpp2_unnamed_declaration_head/,
                {token: 'identifier.definition', switchTo: 'parse_cpp2_declaration_signature.$S2.$S3'},
            ],
        ];
        cppfront.tokenizer.parse_cpp2_declaration = [
            [/./, {token: '@rematch', switchTo: 'parse_cpp2_declaration_head.$S2.$S3'}],
        ];
    }
    setupDeclarationParsers();

    // To not parse a Cpp1 label as a Cpp2 declaration, within a Cpp1 block, don't parse Cpp2.
    cppfront.tokenizer.root.unshift(
        [
            /^(\s*)(@at_cpp2_identifier(?=@at_cpp2_unnamed_declaration_head))/,
            {
                cases: {
                    '$S2!=cpp1': ['', {token: '@rematch', next: 'parse_cpp2_declaration.definition'}],
                    '$2~(public|protected|private|default)': ['', {token: 'keyword.$2'}],
                    '@': ['', 'identifier'],
                },
            },
        ],
        [
            /^\s*@at_cpp2_top_level_declaration_head/,
            {
                cases: {
                    '$S2!=cpp1': {token: '@rematch', next: 'parse_cpp2_declaration.definition'},
                    '@': 'label.cpp1',
                },
            },
        ],
        [/{/, 'delimiter.curly', 'root.cpp1'],
        [/}/, 'delimiter.curly', '@pop'],
    );

    return cppfront;

    // [Monaco's C++]: https://github.com/microsoft/monaco-editor/blob/main/src/basic-languages/cpp/cpp.ts
}

monaco.languages.register({id: 'cpp2-cppfront'});
monaco.languages.setLanguageConfiguration('cpp2-cppfront', cpp.conf);
monaco.languages.setMonarchTokensProvider('cpp2-cppfront', definition());

export {};
