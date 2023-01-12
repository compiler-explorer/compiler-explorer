// Copyright (c) 2021, Compiler Explorer Authors
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

'use strict';
const monaco = require('monaco-editor');

function definition() {
    return {
        symbols: /[=><!~?&|+\-*/^;.,]+/,
        escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,
        tokenizer: {
            root: [
                [
                    /(add_compile_options|add_custom_command|add_custom_target|add_definitions|add_dependencies|add_executable|add_library|add_subdirectory|add_test|aux_source_directory|break|build_command|build_name|cmake_host_system_information|cmake_minimum_required|cmake_policy|configure_file|continue|create_test_sourcelist|ctest_build|ctest_configure|ctest_coverage|ctest_empty_binary_directory|ctest_memcheck|ctest_read_custom_files|ctest_run_script|ctest_sleep|ctest_start|ctest_submit|ctest_test|ctest_update|ctest_upload|define_property|else|elseif|enable_language|enable_testing|endforeach|endfunction|endif|endmacro|endwhile|exec_program|execute_process|export|export_library_dependencies|file|find_file|find_library|find_package|find_path|find_program|fltk_wrap_ui|foreach|function|get_cmake_property|get_directory_property|get_filename_component|get_property|get_source_file_property|get_target_property|get_test_property|if|include|include_directories|include_external_msproject|include_regular_expression|install|install_files|install_programs|install_targets|link_directories|link_libraries|list|load_cache|load_command|macro|make_directory|mark_as_advanced|math|message|option|output_required_files|project|qt_wrap_cpp|qt_wrap_ui|remove|remove_definitions|return|separate_arguments|set|set_directory_properties|set_property|set_source_files_properties|set_target_properties|set_tests_properties|site_name|source_group|string|subdir_depends|subdirs|target_compile_definitions|target_compile_features|target_compile_options|target_include_directories|target_link_libraries|target_sources|try_compile|try_run|unset|use_mangled_mesa|utility_source|variable_requires|variable_watch|while|write_file)/,
                    'keyword',
                ],
                [
                    /\b((CMAKE_)?(CXX_STANDARD|CXX_FLAGS|CMAKE_CXX_FLAGS_DEBUG|CMAKE_CXX_FLAGS_MINSIZEREL|CMAKE_CXX_FLAGS_RELEASE|CMAKE_CXX_FLAGS_RELWITHDEBINFO))\b/,
                    'variable',
                ],
                [
                    /\b(ABSOLUTE|AND|BOOL|CACHE|COMMAND|COMMENT|DEFINED|DOC|EQUAL|EXISTS|EXT|FALSE|GREATER|GREATER_EQUAL|INTERNAL|IN_LIST|IS_ABSOLUTE|IS_DIRECTORY|IS_NEWER_THAN|IS_SYMLINK|LESS|LESS_EQUAL|MATCHES|NAME|NAMES|NAME_WE|NOT|OFF|ON|OR|PATH|PATHS|POLICY|PROGRAM|STREQUAL|STRGREATER|STRGREATER_EQUAL|STRING|STRLESS|STRLESS_EQUAL|TARGET|TEST|TRUE|VERSION_EQUAL|VERSION_GREATER|VERSION_GREATER_EQUAL|VERSION_LESS)\b/,
                    'keyword',
                ],
                [
                    /\b((APPLE|BORLAND|(CMAKE_)?(CL_64|COMPILER_2005|HOST_APPLE|HOST_SYSTEM|HOST_SYSTEM_NAME|HOST_SYSTEM_PROCESSOR|HOST_SYSTEM_VERSION|HOST_UNIX|HOST_WIN32|LIBRARY_ARCHITECTURE|LIBRARY_ARCHITECTURE_REGEX|OBJECT_PATH_MAX|SYSTEM|SYSTEM_NAME|SYSTEM_PROCESSOR|SYSTEM_VERSION)|CYGWIN|MSVC|MSVC80|MSVC_IDE|MSVC_VERSION|UNIX|WIN32|XCODE_VERSION|MSVC60|MSVC70|MSVC90|MSVC71))\b/,
                    'variable',
                ],
                [
                    /\b(BUILD_SHARED_LIBS|(CMAKE_)?(ABSOLUTE_DESTINATION_FILES|AUTOMOC_RELAXED_MODE|BACKWARDS_COMPATIBILITY|BUILD_TYPE|COLOR_MAKEFILE|CONFIGURATION_TYPES|DEBUG_TARGET_PROPERTIES|DISABLE_FIND_PACKAGE_\w+|FIND_LIBRARY_PREFIXES|FIND_LIBRARY_SUFFIXES|IGNORE_PATH|INCLUDE_PATH|INSTALL_DEFAULT_COMPONENT_NAME|INSTALL_PREFIX|LIBRARY_PATH|MFC_FLAG|MODULE_PATH|NOT_USING_CONFIG_FLAGS|POLICY_DEFAULT_CMP\w+|PREFIX_PATH|PROGRAM_PATH|SKIP_INSTALL_ALL_DEPENDENCY|SYSTEM_IGNORE_PATH|SYSTEM_INCLUDE_PATH|SYSTEM_LIBRARY_PATH|SYSTEM_PREFIX_PATH|SYSTEM_PROGRAM_PATH|USER_MAKE_RULES_OVERRIDE|WARN_ON_ABSOLUTE_INSTALL_DESTINATION))\b/,
                    'keyword',
                ],
                {include: '@whitespace'},
                {include: '@builtins'},
            ],
            builtins: [
                [/\$\{\w+\}/, 'variable'],
                [/\d*\.\d+([eE][-+]?\d+)?/, 'number.float'],
                [/0[xX][0-9a-fA-F_]*[0-9a-fA-F]/, 'number.hex'],
                [/\d+/, 'number'],
                [/"/, 'string', '@string."'],
                [/'/, 'string', "@string.'"],
            ],
            whitespace: [
                [/[ \t\r\n]+/, ''],
                [/#.*$/, 'comment'],
            ],
            string: [
                [/[^\\"'%]+/, {cases: {'@eos': {token: 'string', next: '@popall'}, '@default': 'string'}}],
                [/@escapes/, 'string.escape'],
                [/\\./, 'string.escape.invalid'],
                [/\$\{[\w ]+\}/, 'variable'],
                [/["']/, {cases: {'$#==$S2': {token: 'string', next: '@pop'}, '@default': 'string'}}],
                [/$/, 'string', '@popall'],
            ],
        },
    };
}

function configuration() {
    return {
        indentationRules: {
            decreaseIndentPattern: /^(.*\*\/)?\s*\}.*$/,
            increaseIndentPattern: /^.*\{[^}"']*$/,
        },
        wordPattern: /(-?\d*\.\d\w*)|([^`~!@#%^&*()\-=+[{\]}\\|;:'",.<>/?\s]+)/g,
        comments: {
            lineComment: '#',
        },
        brackets: [
            ['{', '}'],
            ['(', ')'],
        ],
        __electricCharacterSupport: {
            brackets: [
                {tokenType: 'delimiter.curly.ts', open: '{', close: '}', isElectric: true},
                {tokenType: 'delimiter.square.ts', open: '[', close: ']', isElectric: true},
                {tokenType: 'delimiter.paren.ts', open: '(', close: ')', isElectric: true},
            ],
        },
        __characterPairSupport: {
            autoClosingPairs: [
                {open: '{', close: '}'},
                {open: '(', close: ')'},
                {open: '"', close: '"', notIn: ['string']},
            ],
        },
    };
}

monaco.languages.register({id: 'cmake'});
monaco.languages.setMonarchTokensProvider('cmake', definition());
monaco.languages.setLanguageConfiguration('cmake', configuration());

export {};
