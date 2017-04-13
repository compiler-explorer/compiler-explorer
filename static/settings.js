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

define(function (require) {
    var _ = require('underscore');
    var colour = require('./colour');

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
            elem.append($('<option value="' + e.label + '">' + e.desc + "</option>"));
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

    function setupSettings(root, settings, onChange) {
        settings = settings || {};

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
        add(root.find('.colourScheme'), 'colourScheme', colour.schemes[0].name, Select,
            _.map(colour.schemes, function (scheme) {
                return {label: scheme.name, desc: scheme.desc};
            }));
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
                return (x / 1000.0).toFixed(2) + "s";
            }
        });
        add(root.find('.hoverShowSource'), 'hoverShowSource', true, Checkbox);
        add(root.find('.hoverShowAsmDoc'), 'hoverShowAsmDoc', true, Checkbox);

        function setSettings(settings) {
            onSettingsChange(settings);
            onChange(settings);
        }

        add(root.find('.tabWidth'), 'tabWidth', 4, Select,
            _.map([2, 4, 8], function (size) {
                return {label: size, desc: size};
            })
        );

        setSettings(settings);
        return setSettings;
    }

    return setupSettings;
});
