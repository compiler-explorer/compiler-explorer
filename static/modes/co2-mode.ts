import $ from 'jquery';
import * as monaco from 'monaco-editor';
import * as cpp from 'monaco-editor/esm/vs/basic-languages/cpp/cpp';

function definition(): monaco.languages.IMonarchLanguage {
    const co2 = $.extend(true, {}, cpp.language);

    co2.keywords = [
        'alignas',
        'alignof',
        'auto',
        'bool',
        'break',
        'case',
        'char',
        'const',
        'constexpr',
        'continue',
        'default',
        'do',
        'double',
        'else',
        'enum',
        'extern',
        'false',
        'float',
        'fn',
        'for',
        'goto',
        'if',
        'inline',
        'int',
        'long',
        'mod',
        'nullptr',
        'pub',
        'register',
        'restrict',
        'return',
        'short',
        'signed',
        'sizeof',
        'static',
        'static_assert',
        'struct',
        'switch',
        'thread_local',
        'true',
        'type',
        'typedef',
        'typeof',
        'typeof_unqual',
        'union',
        'unsafe',
        'unsigned',
        'use',
        'void',
        'volatile',
        'while',
        '_Alignas',
        '_Alignof',
        '_Atomic',
        '_BitInt',
        '_Bool',
        '_Complex',
        '_Decimal128',
        '_Decimal32',
        '_Decimal64',
        '_Generic',
        '_Imaginary',
        '_Noreturn',
        '_Pragma',
        '_Static_assert',
        '_Thread_local',
    ];

    // Add CO2-specific patterns before original tokenizer rules
    const origToken = co2.tokenizer.root;
    co2.tokenizer.root = [
        // CO2 attributes
        [/^#!\[.*?\]/, 'annotation'],
        [/#\[.*?\]/, 'annotation'],

        // CO2 paths: foo::bar::baz
        [/[a-zA-Z_]\w*(?:::[a-zA-Z_]\w*)+/, 'type.identifier'],

        ...origToken,
    ];

    return co2;
}

const def = definition();

monaco.languages.register({id: 'co2'});
monaco.languages.setLanguageConfiguration('co2', cpp.conf);
monaco.languages.setMonarchTokensProvider('co2', def);

export default def;
