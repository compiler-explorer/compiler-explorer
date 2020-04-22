// Copyright (c) 2020, Compiler Explorer Authors
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

"use strict";

var
    $ = require('jquery'),
    _ = require('underscore'),
    local = require('./local'),
    sharing = require('./sharing');

var _currentPresentation = null;

function Presentation() {
    this.settings = null;
    this.source = null;
    this.currentSlide = 0;

    this.settingsDialog = $('#presentationsettings');
    this.settingsSourcelist = this.settingsDialog.find('select.sourcesessions');
    this.settingsOrderedlist = this.settingsDialog.find('select.ordered');
}

Presentation.prototype.init = function (callback) {
    this.loadPresentationSettings();
    this.loadPresentationSource(_.bind(function () {
        this.currentSlide = parseInt(local.get('presentationCurrentSlide', 0));
        this.initSettingsDialog();
        callback();
    }, this));
};

Presentation.prototype.createSettingsOption = function (order, session, compiler) {
    var option = $("<option />");
    option.val(JSON.stringify(order));

    if (session.description) {
        option.html(
            session.description + " - " +
            compiler.id + ", " + compiler.options);
    } else {
        option.html(
            order.session + ":" + order.compiler + " - " +
            session.language + ", " + compiler.id + ", " + compiler.options);
    }

    return option;
};

Presentation.prototype.loadOrderFromSettingsDialog = function () {
    this.settings = {order: []};
    this.settingsOrderedlist.find("option").each(_.bind(function (idx, option) {
        this.settings.order.push(JSON.parse($(option).val()));
    }, this));

    local.set('presentationSettings', JSON.stringify(this.settings));
};

Presentation.prototype.initSettingsDialogButtons = function () {
    this.settingsDialog.find(".add-to-order").click(_.bind(function () {
        this.settingsSourcelist.find("option:selected").each(_.bind(function (idx, option) {
            this.settingsOrderedlist.append($(option).clone());
        }, this));

        this.loadOrderFromSettingsDialog();
    }, this));

    this.settingsDialog.find(".remove-from-order").click(_.bind(function () {
        this.settingsOrderedlist.find("option:selected").each(function (idx, option) {
            option.remove();
        });

        this.loadOrderFromSettingsDialog();
    }, this));
};

Presentation.prototype.initSettingsDialog = function () {
    this.settingsDialog.find("select option").remove();

    var order = {},
        session = null,
        compiler = null;

    for (var idxSession = 0; idxSession < this.source.sessions.length; idxSession++) {
        session = this.source.sessions[idxSession];

        for (var idxCompiler = 0; idxCompiler < session.compilers.length; idxCompiler++) {
            compiler = session.compilers[idxCompiler];

            order = {
                session: idxSession,
                compiler: idxCompiler
            };

            this.settingsSourcelist.append(this.createSettingsOption(order, session, compiler));
        }
    }

    for (var idxOrder = 0; idxOrder < this.settings.order.length; idxOrder++) {
        order = this.settings.order[idxOrder];
        session = this.source.sessions[order.session];
        compiler = session.compilers[order.compiler];

        this.settingsOrderedlist.append(this.createSettingsOption(order, session, compiler));
    }

    this.initSettingsDialogButtons();
};

Presentation.prototype.first = function () {
    this.currentSlide = 0;
    local.set('presentationCurrentSlide', this.currentSlide);
    this.show();
};

Presentation.prototype.next = function () {
    if (this.settings && (this.currentSlide + 2 < this.settings.order.length)) {
        this.currentSlide++;
        local.set('presentationCurrentSlide', this.currentSlide);
        this.show();
    }
};

Presentation.prototype.prev = function () {
    if (this.currentSlide > 0) {
        this.currentSlide--;
        local.set('presentationCurrentSlide', this.currentSlide);
        this.show();
    }
};

Presentation.prototype.createSourceContentArray = function (left, right) {
    if (left.session === right.session) {
        return [this.createComponents(this.source.sessions[left.session], 1, 100)];
    } else {
        return [
            this.createComponents(this.source.sessions[left.session], 1),
            this.createComponents(this.source.sessions[right.session], 2)
        ];
    }
};

Presentation.prototype.show = function () {
    if (this.settings && (this.currentSlide + 1 < this.settings.order.length)) {
        var left = this.settings.order[this.currentSlide];
        var right = this.settings.order[this.currentSlide + 1];

        var gl =
        {
            settings: {
                hasHeaders: true,
                constrainDragToContainer: false,
                reorderEnabled: true,
                selectionEnabled: false,
                popoutWholeStack: false,
                blockedPopoutsThrowError: true,
                closePopoutsOnUnload: true,
                showPopoutIcon: false,
                showMaximiseIcon: true,
                showCloseIcon: false,
                responsiveMode: "onload",
                tabOverlapAllowance: 0,
                reorderOnTabMenuClick: true,
                tabControlOffset: 10
            },
            dimensions:
            {
                borderWidth: 5,
                borderGrabWidth: 15,
                minItemHeight: 10,
                minItemWidth: 10,
                headerHeight: 20,
                dragProxyWidth: 300,
                dragProxyHeight: 200
            },
            labels:
            {
                close: "close",
                maximise: "maximise",
                minimise: "minimise",
                popout: "open in new window",
                popin: "pop in",
                tabDropdown: "additional tabs"
            },
            content: [
                {
                    type: "column",
                    content: [
                        {
                            type: "row",
                            height: 50,
                            content: this.createSourceContentArray(left, right)
                        },
                        {
                            type: "row",
                            height: 50,
                            content: [
                                {
                                    type: "stack",
                                    width: 100,
                                    content: [
                                        this.createDiffComponent(left.compiler + 1, right.compiler + 1)
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        };

        local.set('gl', JSON.stringify(gl));
        window.hasUIBeenReset = true;
        window.history.replaceState(null, null, window.httpRoot);
        window.location.reload();
    }
};

Presentation.prototype.createDiffComponent = function (left, right) {
    return {
        type: "component",
        componentName: "diff",
        componentState:
        {
            lhs: left,
            rhs: right,
            lhsdifftype: 0,
            rhsdifftype: 0,
            fontScale: 14
        }
    };
};

Presentation.prototype.createCompilerComponent = function (session, compiler, sessionId) {
    return {
        type: "component",
        componentName: "compiler",
        componentState: {
            compiler: compiler.id,
            source: sessionId,
            options: compiler.options,
            filters: {
                binary: compiler.filters.binary,
                execute: compiler.filters.execute,
                labels: compiler.filters.labels,
                directives: compiler.filters.directives,
                commentOnly: compiler.filters.commentOnly,
                trim: compiler.filters.trim,
                intel: compiler.filters.intel,
                demangle: compiler.filters.demangle
            },
            libs: compiler.libs,
            lang: session.language
        }
    };
};

Presentation.prototype.createComponents = function (session, sessionId, customWidth) {
    var stack = {
        type: "stack",
        width: customWidth ? customWidth : 50,
        activeItemIndex: 0,
        content: [
            {
                type: "component",
                componentName: "codeEditor",
                componentState: {
                    id: sessionId,
                    source: session.source,
                    lang: session.language
                }
            }
        ]
    };

    for (var idx = 0; idx < session.compilers.length; idx++) {
        stack.content.push(
            this.createCompilerComponent(session, session.compilers[idx], sessionId)
        );
    }

    return stack;
};

Presentation.prototype.loadPresentationSource = function (onLoaded) {
    var presentationSource = local.get('presentationSource', null);
    if (!presentationSource) {
        this.requestPresentationSource(onLoaded);
    } else {
        this.source = JSON.parse(presentationSource);
        onLoaded();
    }
};

Presentation.prototype.savePresentationSource = function (source) {
    this.source = source;
    local.set('presentationSource', JSON.stringify(source));
};

Presentation.prototype.loadPresentationSettings = function () {
    var presentationSettings = local.get('presentationSettings', null);
    if (!presentationSettings) {
        this.settings = {order: []};

        local.set('presentationSettings', JSON.stringify(this.settings));
    } else {
        this.settings = JSON.parse(presentationSettings);
    }
};

Presentation.prototype.requestPresentationSource = function (onLoaded) {
    var gl = local.get('glSaved', null);
    sharing.getClientstate(JSON.parse(gl), window.httpRoot, _.bind(function (error, state) {
        if (error) {
            throw error;
        } else {
            this.savePresentationSource(state);
            onLoaded();
        }
    }, this));
};


function init(callback) {
    if (!_currentPresentation) {
        _currentPresentation = new Presentation();
        _currentPresentation.init(callback);
    } else {
        callback();
    }
}

function first() {
    if (!_currentPresentation) throw "Presentation hasn't been initialized";

    _currentPresentation.first();
}

function next() {
    if (!_currentPresentation) throw "Presentation hasn't been initialized";

    _currentPresentation.next();
}

function prev() {
    if (!_currentPresentation) throw "Presentation hasn't been initialized";

    _currentPresentation.prev();
}

function isActive() {
    var saved = local.get('glSaved', null);
    return saved !== null;
}

function isConfigured() {
    return _currentPresentation.settings.order.length > 1;
}

module.exports = {
    init: init,
    first: first,
    next: next,
    prev: prev,
    isActive: isActive,
    isConfigured: isConfigured
};
