
'use strict';
var $ = require('jquery');
var cpp = require('./vs/basic-languages/src/cpp');

// We need to create a new definition for cpp so we can remove invalid keywords

function definition() {
    var cppp = $.extend(true, {}, cpp.language); // deep copy

    function removeKeyword(keyword) {
        var index = cppp.keywords.indexOf(keyword);
        if (index > -1) {
            cppp.keywords.splice(index, 1);
        }
    }

    function removeKeywords(keywords) {
        for (var i = 0; i < keywords.length; ++i) {
            removeKeyword(keywords[i]);
        }
    }

    function addKeywords(keywords) {
        // (Ruben) Done one by one as if you just push them all, Monaco complains that they're not strings, but as
        // far as I can tell, they indeed are all strings. This somehow fixes it. If you know how to fix it, plz go
        for (var i = 0; i < keywords.length; ++i) {
            cppp.keywords.push(keywords[i]);
        }
    }

    // We remove everything that's not an identifier, underscore reserved name and not an official C++ keyword...
    // Regarding #617, final is a identifier with special meaning, not a fully qualified keyword
    removeKeywords(["abstract", "amp", "array", "cpu", "delegate", "each", "event", "finally", "gcnew",
        "generic", "in", "initonly", "interface", "interior_ptr", "internal", "literal", "partial", "pascal",
        "pin_ptr", "property", "ref", "restrict", "safe_cast", "sealed", "title_static", "where"]);

    addKeywords(["alignas", "alignof", "and", "and_eq", "asm", "bitand", "bitor", "char16_t", "char32_t", "compl",
        "not", "not_eq", "or", "or_eq", "xor", "xor_eq"]);

    return cppp;
}

monaco.languages.register({id: 'cppp'});
monaco.languages.setLanguageConfiguration('cppp', cpp.conf);
monaco.languages.setMonarchTokensProvider('cppp', definition());
