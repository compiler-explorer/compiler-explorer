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

    function setupSettings(root, settings, onChange) {
        settings = settings || {};
        if (settings.delayAfterChange === undefined)
            settings.delayAfterChange = 750;
        if (settings.colouriseAsm === undefined)
            settings.colouriseAsm = true;

        root.find('.slider').slider({
            value: settings.delayAfterChange,
            max: 3000,
            step: 250,
            formatter: function (x) {
                if (x === 0) return "Disabled";
                return (x / 1000.0).toFixed(2) + "s";
            }
        });

        function onSettingsChange(settings) {
            root.find('.colourise').prop('checked', settings.colouriseAsm);
            root.find('.slider').slider('setValue', settings.delayAfterChange);
            onChange(settings);
        }

        function onUiChange() {
            var settings = {};
            settings.colouriseAsm = !!root.find('.colourise').prop('checked');
            settings.delayAfterChange = root.find('.slider').slider('getValue');
            onChange(settings);
        }

        root.find('.colourise').change(onUiChange);
        root.find('.slider').change(onUiChange);

        onSettingsChange(settings);
    }

    return setupSettings;
});
