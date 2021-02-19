// Copyright (c) 2016, Compiler Explorer Authors
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
var $ = require('jquery');
var _ = require('underscore');
var colour = require('./colour');
var themes = require('./themes').themes;
var options = require('./options');

function Setting(elem, name, Control, param) {
    this.elem = elem;
    this.name = name;
    this.control = new Control(elem, param);
}

Setting.prototype.getUi = function () {
    return this.control.getUi(this.elem);
};
Setting.prototype.putUi = function (value) {
    this.control.putUi(this.elem, value);
};

function Checkbox() {
}

Checkbox.prototype.getUi = function (elem) {
    return !!elem.prop('checked');
};
Checkbox.prototype.putUi = function (elem, value) {
    elem.prop('checked', !!value);
};

function Select(elem, populate) {
    elem.empty();
    _.each(populate, function (e) {
        elem.append($('<option value="' + e.label + '">' + e.desc + '</option>'));
    });
}

Select.prototype.getUi = function (elem) {
    return elem.val();
};
Select.prototype.putUi = function (elem, value) {
    elem.val(value);
};

function Slider(elem, sliderSettings) {
    elem.slider(sliderSettings);
}

Slider.prototype.getUi = function (elem) {
    return elem.slider('getValue');
};

Slider.prototype.putUi = function (elem, value) {
    elem.slider('setValue', value);
};

function Textbox() {
}

Textbox.prototype.getUi = function (elem) {
    return elem.val();
};

Textbox.prototype.putUi = function (elem, value) {
    elem.val(value);
};

function Numeric(elem, params) {
    this.min = params.min;
    this.max = params.max;
    elem.attr('min', params.min)
        .attr('max', params.max);
}

Numeric.prototype.getUi = function (elem) {
    var val = parseInt(elem.val());
    if (val < this.min) return this.min;
    if (val > this.max) return this.max;
    return val;
};

Numeric.prototype.putUi = function (elem, value) {
    if (value < this.min) value = this.min;
    if (value > this.max) value = this.max;
    elem.val(value);
};

// Ignore max statements, there's no limit as to how many settings we need
// eslint-disable-next-line max-statements
function setupSettings(root, settings, onChange, subLangId) {
    settings = settings || {};
    // Ensure the default language is not "null" but undefined. Temporary patch for a previous bug :(
    settings.defaultLanguage = settings.defaultLanguage === null ? undefined : settings.defaultLanguage;
    var settingsObjs = [];

    function onUiChange() {
        var settings = {};
        _.each(settingsObjs, function (s) {
            settings[s.name] = s.getUi();
        });
        onChange(settings);
    }

    function onSettingsChange(settings) {
        _.each(settingsObjs, function (s) {
            s.putUi(settings[s.name]);
        });
    }

    function add(elem, key, defaultValue, Type, param) {
        if (settings[key] === undefined)
            settings[key] = defaultValue;
        settingsObjs.push(new Setting(elem, key, Type, param));
        elem.change(onUiChange);
    }

    add(root.find('.colourise'), 'colouriseAsm', true, Checkbox);
    add(root.find('.autoCloseBrackets'), 'autoCloseBrackets', true, Checkbox);
    var colourSchemeSelect = root.find('.colourScheme');
    add(colourSchemeSelect, 'colourScheme', colour.schemes[0].name, Select,
        _.map(colour.schemes, function (scheme) {
            return {label: scheme.name, desc: scheme.desc};
        })
    );
    // Handle older settings
    if (settings.delayAfterChange === 0) {
        settings.delayAfterChange = 750;
        settings.compileOnChange = false;
    }
    add(root.find('.compileOnChange'), 'compileOnChange', true, Checkbox);
    add(root.find('.delay'), 'delayAfterChange', 750, Slider, {
        max: 3000,
        step: 250,
        min: 250,
        formatter: function (x) {
            return (x / 1000.0).toFixed(2) + 's';
        },
    });
    add(root.find('.enableCommunityAds'), 'enableCommunityAds', true, Checkbox);
    add(root.find('.hoverShowSource'), 'hoverShowSource', true, Checkbox);
    add(root.find('.hoverShowAsmDoc'), 'hoverShowAsmDoc', true, Checkbox);

    var themeSelect = root.find('.theme');

    var defaultThemeId = themes.default.id;
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        defaultThemeId = themes.dark.id;
    }

    add(themeSelect, 'theme', defaultThemeId, Select,
        _.map(themes, function (theme) {
            return {label: theme.id, desc: theme.name};
        })
    );
    add(root.find('.showQuickSuggestions'), 'showQuickSuggestions', false, Checkbox);
    add(root.find('.useCustomContextMenu'), 'useCustomContextMenu', true, Checkbox);
    add(root.find('.showMinimap'), 'showMinimap', true, Checkbox);

    function handleThemes() {
        var newTheme = themeSelect.val();
        // Store the scheme of the old theme
        $.data(themeSelect, 'theme-' + $.data(themeSelect, 'last-theme'), colourSchemeSelect.val());
        // Get the scheme of the new theme
        var newThemeStoredScheme = $.data(themeSelect, 'theme-' + newTheme);
        var isStoredUsable = false;
        colourSchemeSelect.empty();
        _.each(colour.schemes, function (scheme) {
            if (!scheme.themes || scheme.themes.length === 0 || scheme.themes.indexOf(newTheme) !== -1 ||
                scheme.themes.indexOf('all') !== -1) {

                colourSchemeSelect.append($('<option value="' + scheme.name + '">' + scheme.desc + '</option>'));
                if (newThemeStoredScheme === scheme.name) {
                    isStoredUsable = true;
                }
            }
        });
        if (colourSchemeSelect.children().length >= 1) {
            colourSchemeSelect.val(isStoredUsable ? newThemeStoredScheme : colourSchemeSelect.first().val());
        } else {
            // This should never happen. In case it does, lets use the default one
            colourSchemeSelect.append(
                $('<option value="' + colour.schemes[0].name + '">' + colour.schemes[0].desc + '</option>'));
            colourSchemeSelect.val(colourSchemeSelect.first().val());
        }
        colourSchemeSelect.trigger('change');
    }

    var langs = options.languages;

    var defaultLanguageSelector = root.find('.defaultLanguage');
    var defLang = settings.defaultLanguage || _.keys(langs)[0] || 'c++';
    add(defaultLanguageSelector, 'defaultLanguage', defLang, Select,
        _.map(langs, function (lang) {
            return {label: lang.id, desc: lang.name};
        })
    );
    if (subLangId) {
        defaultLanguageSelector
            .prop('disabled', true)
            .prop('title', 'Default language inherited from subdomain')
            .css('cursor', 'not-allowed');
    }

    add(root.find('.newEditorLastLang'), 'newEditorLastLang', true, Checkbox);

    var formats = ['Google', 'LLVM', 'Mozilla', 'Chromium', 'WebKit'];
    add(root.find('.formatBase'), 'formatBase', formats[0], Select,
        _.map(formats, function (format) {
            return {label: format, desc: format};
        }));
    //add(root.find('.formatOverrides'), 'formatOverrides', "", TextAreaInput);
    add(root.find('.wordWrap'), 'wordWrap', false, Checkbox);

    function setSettings(settings) {
        onSettingsChange(settings);
        onChange(settings);
    }
    add(root.find('.useSpaces'), 'useSpaces', true, Checkbox);
    add(root.find('.tabWidth'), 'tabWidth', 4, Numeric, {min: 1, max: 80});
    add(root.find('.enableCtrlS'), 'enableCtrlS', true, Checkbox);
    add(root.find('.editorsFFont'), 'editorsFFont', 'Consolas, "Liberation Mono", Courier, monospace', Textbox);
    add(root.find('.editorsFLigatures'), 'editorsFLigatures', false, Checkbox);
    add(root.find('.allowStoreCodeDebug'), 'allowStoreCodeDebug', true, Checkbox);
    add(root.find('.useVim'), 'useVim', false, Checkbox);
    add(root.find('.autoIndent'), 'autoIndent', true, Checkbox);
    add(root.find('.keepSourcesOnLangChange'), 'keepSourcesOnLangChange', false, Checkbox);

    setSettings(settings);
    handleThemes();
    themeSelect.change(function () {
        handleThemes();
        $.data(themeSelect, 'last-theme', themeSelect.val());
    });
    $.data(themeSelect, 'last-theme', themeSelect.val());
    return setSettings;
}

module.exports = setupSettings;
