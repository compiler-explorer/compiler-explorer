// Copyright (c) 2019, Compiler Explorer Authors
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
var ga = require('../analytics');
var Toggles = require('../toggles');
var FontScale = require('../fontscale');
var options = require('../options');
var Alert = require('../alert');
var local = require('../local');
var Libraries = require('../libs-widget');
var AnsiToHtml = require('../ansi-to-html');
require('../modes/asm-mode');
require('../modes/ptx-mode');

require('selectize');


var languages = options.languages;

function makeAnsiToHtml(color) {
    return new AnsiToHtml({
        fg: color ? color : '#333',
        bg: '#f5f5f5',
        stream: true,
        escapeXML: true,
    });
}


function Executor(hub, container, state) {
    this.container = container;
    this.hub = hub;
    this.eventHub = hub.createEventHub();
    this.compilerService = hub.compilerService;
    this.domRoot = container.getElement();
    this.domRoot.html($('#executor').html());
    this.sourceEditorId = state.source || 1;
    this.id = state.id || hub.nextExecutorId();
    this.settings = JSON.parse(local.get('settings', '{}'));
    this.initLangAndCompiler(state);
    this.infoByLang = {};
    this.deferCompiles = hub.deferred;
    this.needsCompile = false;
    this.options = state.options || options.compileOptions[this.currentLangId];
    this.executionArguments = state.execArgs || '';
    this.executionStdin = state.execStdin || '';
    this.source = '';
    this.lastResult = {};
    this.pendingRequestSentAt = 0;
    this.nextRequest = null;

    this.alertSystem = new Alert();
    this.alertSystem.prefixMessage = 'Executor #' + this.id + ': ';

    this.normalAnsiToHtml = makeAnsiToHtml();
    this.errorAnsiToHtml = makeAnsiToHtml('red');

    this.initButtons(state);

    this.fontScale = new FontScale(this.domRoot, state, 'pre.content');

    this.compilerPicker.selectize({
        sortField: this.compilerService.getSelectizerOrder(),
        valueField: 'id',
        labelField: 'name',
        searchField: ['name'],
        optgroupField: 'group',
        optgroups: this.compilerService.getGroupsInUse(this.currentLangId),
        lockOptgroupOrder: true,
        options: _.map(this.getCurrentLangCompilers(), _.identity),
        items: this.compiler ? [this.compiler.id] : [],
        dropdownParent: 'body',
        closeAfterSelect: true,
    }).on('change', _.bind(function (e) {
        var val = $(e.target).val();
        if (val) {
            ga.proxy('send', {
                hitType: 'event',
                eventCategory: 'SelectCompiler',
                eventAction: val,
            });
            this.onCompilerChange(val);
        }
    }, this));

    this.compilerSelectizer = this.compilerPicker[0].selectize;

    this.initLibraries(state);
    this.initCallbacks();
    // Handle initial settings
    this.onSettingsChange(this.settings);
    this.updateCompilerInfo();
    this.saveState();
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'OpenViewPane',
        eventAction: 'Executor',
    });
}

Executor.prototype.initLangAndCompiler = function (state) {
    var langId = state.lang;
    var compilerId = state.compiler;
    var result = this.compilerService.processFromLangAndCompiler(langId, compilerId);
    this.compiler = result.compiler;
    this.currentLangId = result.langId;
    this.updateLibraries();
};

Executor.prototype.close = function () {
    this.eventHub.unsubscribe();
    this.eventHub.emit('executorClose', this.id);
};

Executor.prototype.undefer = function () {
    this.deferCompiles = false;
    if (this.needsCompile) this.compile();
};

Executor.prototype.updateAndCalcTopBarHeight = function () {
    // If we save vertical space by hiding stuff that's OK to hide
    // when thin, then hide that stuff.
    this.hideable.show();
    var topBarHeightMax = this.topBar.outerHeight(true);
    this.hideable.hide();
    var topBarHeightMin = this.topBar.outerHeight(true);
    var topBarHeight = topBarHeightMin;
    if (topBarHeightMin === topBarHeightMax) {
        this.hideable.show();
    }

    if (!this.panelCompilation.hasClass('d-none')) {
        topBarHeight += this.panelCompilation.outerHeight(true);
    }
    if (!this.panelArgs.hasClass('d-none')) {
        topBarHeight += this.panelArgs.outerHeight(true);
    }
    if (!this.panelStdin.hasClass('d-none')) {
        topBarHeight += this.panelStdin.outerHeight(true);
    }

    return topBarHeight;
};

Executor.prototype.resize = function () {
    var topBarHeight = this.updateAndCalcTopBarHeight();
    var bottomBarHeight = this.bottomBar.outerHeight(true);
    this.outputContentRoot.outerHeight(this.domRoot.height() - topBarHeight - bottomBarHeight);

};

function errorResult(message) {
    return {code: -1, stderr: message};
}

Executor.prototype.compile = function (bypassCache) {
    if (this.deferCompiles) {
        this.needsCompile = true;
        return;
    }
    this.needsCompile = false;
    this.compileTimeLabel.text(' - Compiling...');
    var options = {
        userArguments: this.options,
        executeParameters: {
            args: this.executionArguments,
            stdin: this.executionStdin,
        },
        compilerOptions: {
            executorRequest: true,
        },
        filters: {execute: true},
        tools: [],
        libraries: [],
    };

    _.each(this.libsWidget.getLibsInUse(), function (item) {
        options.libraries.push({
            id: item.libId,
            version: item.versionId,
        });
    });

    this.compilerService.expand(this.source).then(_.bind(function (expanded) {
        var request = {
            source: expanded || '',
            compiler: this.compiler ? this.compiler.id : '',
            options: options,
            lang: this.currentLangId,
        };
        if (bypassCache) request.bypassCache = true;
        if (!this.compiler) {
            this.onCompileResponse(request, errorResult('<Please select a compiler>'), false);
        } else {
            this.sendCompile(request);
        }
    }, this));
};

Executor.prototype.sendCompile = function (request) {
    var onCompilerResponse = _.bind(this.onCompileResponse, this);

    if (this.pendingRequestSentAt) {
        // If we have a request pending, then just store this request to do once the
        // previous request completes.
        this.nextRequest = request;
        return;
    }
    // this.eventHub.emit('compiling', this.id, this.compiler);
    // Display the spinner
    this.handleCompilationStatus({code: 4});
    this.pendingRequestSentAt = Date.now();
    // After a short delay, give the user some indication that we're working on their
    // compilation.
    this.compilerService.submit(request)
        .then(function (x) {
            onCompilerResponse(request, x.result, x.localCacheHit);
        })
        .catch(function (x) {
            var message = 'Unknown error';
            if (_.isString(x)) {
                message = x;
            } else if (x) {
                message = x.error || x.code;
            }
            onCompilerResponse(request, errorResult(message), false);
        });
};

Executor.prototype.addCompilerOutputLine = function (msg, container, lineNum, column) {
    var elem = $('<p></p>').appendTo(container);
    if (lineNum) {
        elem.html(
            $('<span class="linked-compiler-output-line"></span>')
                .html(msg)
                .click(_.bind(function (e) {
                    this.eventHub.emit('editorLinkLine', this.sourceEditorId, lineNum, column, true);
                    // do not bring user to the top of index.html
                    // http://stackoverflow.com/questions/3252730
                    e.preventDefault();
                    return false;
                }, this))
                .on('mouseover', _.bind(function () {
                    this.eventHub.emit('editorLinkLine', this.sourceEditorId, lineNum, column, false);
                }, this))
        );
    } else {
        elem.html(msg);
    }
};

Executor.prototype.clearPreviousOutput = function () {
    this.executionStatusSection.empty();
    this.compilerOutputSection.empty();
    this.executionOutputSection.empty();
};

Executor.prototype.handleOutput = function (output, element, ansiParser) {
    var outElem = $('<pre class="card"></pre>').appendTo(element);
    _.each(output, function (obj) {
        if (obj.text === '') {
            this.addCompilerOutputLine('<br/>', outElem);
        } else {
            var lineNumber = obj.tag ? obj.tag.line : obj.line;
            var columnNumber = obj.tag ? obj.tag.column : -1;
            this.addCompilerOutputLine(ansiParser.toHtml(obj.text), outElem, lineNumber, columnNumber);
        }
    }, this);
    return outElem;
};

Executor.prototype.onCompileResponse = function (request, result, cached) {
    // Save which source produced this change. It should probably be saved earlier though
    result.source = this.source;
    this.lastResult = result;
    var timeTaken = Math.max(0, Date.now() - this.pendingRequestSentAt);
    var wasRealReply = this.pendingRequestSentAt > 0;
    this.pendingRequestSentAt = 0;
    ga.proxy('send', {
        hitType: 'event',
        eventCategory: 'Compile',
        eventAction: request.compiler,
        eventLabel: request.options.userArguments,
        eventValue: cached ? 1 : 0,
    });
    ga.proxy('send', {
        hitType: 'timing',
        timingCategory: 'Compile',
        timingVar: request.compiler,
        timingValue: timeTaken,
    });

    this.clearPreviousOutput();
    var compileStdout = result.buildResult.stdout || [];
    var compileStderr = result.buildResult.stderr || [];
    var execStdout = result.stdout || [];
    var execStderr = result.stderr || [];
    if (!result.didExecute) {
        this.executionStatusSection.append($('<p></p>').text('Could not execute the program'));
        this.executionStatusSection.append($('<p></p>').text('Compiler returned: ' + result.buildResult.code));
    }

    if (compileStdout.length > 0) {
        this.compilerOutputSection.append($('<p></p>').text('Compiler stdout'));
        this.handleOutput(compileStdout, this.compilerOutputSection, this.normalAnsiToHtml);
    }
    if (compileStderr.length > 0) {
        this.compilerOutputSection.append($('<p></p>').text('Compiler stderr'));
        this.handleOutput(compileStderr, this.compilerOutputSection, this.errorAnsiToHtml);
    }
    if (result.didExecute) {
        this.executionOutputSection.append($('<p></p>').text('Program returned: ' + result.code));
        if (execStdout.length > 0) {
            this.executionOutputSection.append($('<p></p>').text('Program stdout'));
            var outElem = this.handleOutput(execStdout, this.executionOutputSection, this.normalAnsiToHtml);
            outElem.addClass('execution-stdout');
        }
        if (execStderr.length > 0) {
            this.executionOutputSection.append($('<p></p>').text('Program stderr'));
            this.handleOutput(execStderr, this.executionOutputSection, this.normalAnsiToHtml);
        }
    }

    this.handleCompilationStatus({code: 1, didExecute: result.didExecute});
    var timeLabelText = '';
    if (cached) {
        timeLabelText = ' - cached';
    } else if (wasRealReply) {
        timeLabelText = ' - ' + timeTaken + 'ms';
    }
    this.compileTimeLabel.text(timeLabelText);

    this.setCompilationOptionsPopover(result.buildResult &&
        result.buildResult.compilationOptions ? result.buildResult.compilationOptions.join(' ') : '');

    this.eventHub.emit('executeResult', this.id, this.compiler, result, languages[this.currentLangId]);

    if (this.nextRequest) {
        var next = this.nextRequest;
        this.nextRequest = null;
        this.sendCompile(next);
    }
};

Executor.prototype.resendResult = function () {
    if (!$.isEmptyObject(this.lastResult)) {
        this.eventHub.emit('executeResult', this.id, this.compiler, this.lastResult);
        return true;
    }
    return false;
};

Executor.prototype.onResendExecutionResult = function (id) {
    if (id === this.id) {
        this.resendResult();
    }
};

Executor.prototype.onEditorChange = function (editor, source, langId, compilerId) {
    if (editor === this.sourceEditorId && langId === this.currentLangId &&
        (compilerId === undefined)) {
        this.source = source;
        if (this.settings.compileOnChange) {
            this.compile();
        }
    }
};

Executor.prototype.initButtons = function (state) {
    this.filters = new Toggles(this.domRoot.find('.filters'), state.filters);

    this.compileClearCache = this.domRoot.find('.clear-cache');
    this.outputContentRoot = this.domRoot.find('pre.content');
    this.executionStatusSection = this.outputContentRoot.find('.execution-status');
    this.compilerOutputSection = this.outputContentRoot.find('.compiler-output');
    this.executionOutputSection = this.outputContentRoot.find('.execution-output');

    this.optionsField = this.domRoot.find('.compilation-options');
    this.execArgsField = this.domRoot.find('.execution-arguments');
    this.execStdinField = this.domRoot.find('.execution-stdin');
    this.prependOptions = this.domRoot.find('.prepend-options');
    this.fullCompilerName = this.domRoot.find('.full-compiler-name');
    this.setCompilationOptionsPopover(this.compiler ? this.compiler.options : null);

    this.compileTimeLabel = this.domRoot.find('.compile-time');
    this.libsButton = this.domRoot.find('.btn.show-libs');

    // Dismiss on any click that isn't either in the opening element, inside
    // the popover or on any alert
    $(document).on('mouseup', _.bind(function (e) {
        var target = $(e.target);
        if (!target.is(this.prependOptions) && this.prependOptions.has(target).length === 0 &&
            target.closest('.popover').length === 0)
            this.prependOptions.popover('hide');

        if (!target.is(this.fullCompilerName) && this.fullCompilerName.has(target).length === 0 &&
            target.closest('.popover').length === 0)
            this.fullCompilerName.popover('hide');
    }, this));

    this.optionsField.val(this.options);
    this.execArgsField.val(this.executionArguments);
    this.execStdinField.val(this.executionStdin);

    this.shortCompilerName = this.domRoot.find('.short-compiler-name');
    this.compilerPicker = this.domRoot.find('.compiler-picker');
    this.setCompilerVersionPopover('');

    this.topBar = this.domRoot.find('.top-bar');
    this.bottomBar = this.domRoot.find('.bottom-bar');
    this.statusLabel = this.domRoot.find('.status-text');

    this.hideable = this.domRoot.find('.hideable');
    this.statusIcon = this.domRoot.find('.status-icon');

    this.panelCompilation = this.domRoot.find('.panel-compilation');
    this.panelArgs = this.domRoot.find('.panel-args');
    this.panelStdin = this.domRoot.find('.panel-stdin');

    this.initToggleButtons(state);
};

Executor.prototype.initToggleButtons = function (state) {
    this.toggleCompilation = this.domRoot.find('.toggle-compilation');
    this.toggleArgs = this.domRoot.find('.toggle-args');
    this.toggleStdin = this.domRoot.find('.toggle-stdin');
    this.toggleCompilerOut = this.domRoot.find('.toggle-compilerout');

    if (state.compilationPanelShown === false) {
        this.hidePanel(this.toggleCompilation, this.panelCompilation);
    }

    if (state.argsPanelShown === true) {
        this.showPanel(this.toggleArgs, this.panelArgs);
    }

    if (state.stdinPanelShown === true) {
        this.showPanel(this.toggleStdin, this.panelStdin);
    }

    if (state.compilerOutShown === false) {
        this.hidePanel(this.toggleCompilerOut, this.compilerOutputSection);
    }
};

Executor.prototype.onLibsChanged = function () {
    this.saveState();
    this.compile();
};

Executor.prototype.initLibraries = function (state) {
    this.libsWidget = new Libraries.Widget(this.currentLangId, this.compiler,
        this.libsButton, state, _.bind(this.onLibsChanged, this));
};

Executor.prototype.onFontScale = function () {
    this.saveState();
};

Executor.prototype.initListeners = function () {
    // this.filters.on('change', _.bind(this.onFilterChange, this));
    this.fontScale.on('change', _.bind(this.onFontScale, this));

    this.container.on('destroy', this.close, this);
    this.container.on('resize', this.resize, this);
    this.container.on('shown', this.resize, this);
    this.container.on('open', function () {
        this.eventHub.emit('executorOpen', this.id, this.sourceEditorId);
    }, this);
    this.eventHub.on('editorChange', this.onEditorChange, this);
    this.eventHub.on('editorClose', this.onEditorClose, this);
    this.eventHub.on('settingsChange', this.onSettingsChange, this);
    this.eventHub.on('requestCompilation', this.onRequestCompilation, this);
    this.eventHub.on('resendExecution', this.onResendExecutionResult, this);
    this.eventHub.on('resize', this.resize, this);
    this.eventHub.on('findExecutors', this.sendExecutor, this);

    this.eventHub.on('languageChange', this.onLanguageChange, this);
};

Executor.prototype.showPanel = function (button, panel) {
    panel.removeClass('d-none');
    button.addClass('active');
    this.resize();
};

Executor.prototype.hidePanel = function (button, panel) {
    panel.addClass('d-none');
    button.removeClass('active');
    this.resize();
};

Executor.prototype.togglePanel = function (button, panel) {
    if (panel.hasClass('d-none')) {
        this.showPanel(button, panel);
    } else {
        this.hidePanel(button, panel);
    }
    this.saveState();
};

Executor.prototype.initCallbacks = function () {
    this.initListeners();

    var optionsChange = _.debounce(_.bind(function (e) {
        this.onOptionsChange($(e.target).val());
    }, this), 800);

    var execArgsChange = _.debounce(_.bind(function (e) {
        this.onExecArgsChange($(e.target).val());
    }, this), 800);

    var execStdinChange = _.debounce(_.bind(function (e) {
        this.onExecStdinChange($(e.target).val());
    }, this), 800);

    this.optionsField
        .on('change', optionsChange)
        .on('keyup', optionsChange);

    this.execArgsField
        .on('change', execArgsChange)
        .on('keyup', execArgsChange);

    this.execStdinField
        .on('change', execStdinChange)
        .on('keyup', execStdinChange);

    this.compileClearCache.on('click', _.bind(function () {
        this.compilerService.cache.reset();
        this.compile(true);
    }, this));

    // Dismiss the popover on escape.
    $(document).on('keyup.editable', _.bind(function (e) {
        if (e.which === 27) {
            this.libsButton.popover('hide');
        }
    }, this));

    this.toggleCompilation.on('click', _.bind(function () {
        this.togglePanel(this.toggleCompilation, this.panelCompilation);
    }, this));

    this.toggleArgs.on('click', _.bind(function () {
        this.togglePanel(this.toggleArgs, this.panelArgs);
    }, this));

    this.toggleStdin.on('click', _.bind(function () {
        this.togglePanel(this.toggleStdin, this.panelStdin);
    }, this));

    this.toggleCompilerOut.on('click', _.bind(function () {
        this.togglePanel(this.toggleCompilerOut, this.compilerOutputSection);
    }, this));

    // Dismiss on any click that isn't either in the opening element, inside
    // the popover or on any alert
    $(document).on('click', _.bind(function (e) {
        var elem = this.libsButton;
        var target = $(e.target);
        if (!target.is(elem) && elem.has(target).length === 0 && target.closest('.popover').length === 0) {
            elem.popover('hide');
        }
    }, this));

    this.eventHub.on('initialised', this.undefer, this);

    if (MutationObserver !== undefined) {
        new MutationObserver(_.bind(this.resize, this)).observe(this.execStdinField[0], {
            attributes: true, attributeFilter: ['style'],
        });
    }
};

Executor.prototype.onOptionsChange = function (options) {
    this.options = options;
    this.saveState();
    this.compile();
};

Executor.prototype.onExecArgsChange = function (args) {
    this.executionArguments = args;
    this.saveState();
    this.compile();
};

Executor.prototype.onExecStdinChange = function (newStdin) {
    this.executionStdin = newStdin;
    this.saveState();
    this.compile();
};

Executor.prototype.onRequestCompilation = function (editorId) {
    if (editorId === this.sourceEditorId) {
        this.compile();
    }
};

Executor.prototype.updateCompilerInfo = function () {
    this.updateCompilerName();
    if (this.compiler) {
        if (this.compiler.notification) {
            this.alertSystem.notify(this.compiler.notification, {
                group: 'compilerwarning',
                alertClass: 'notification-info',
                dismissTime: 5000,
            });
        }
        this.prependOptions.data('content', this.compiler.options);
    }
    this.sendExecutor();
};

Executor.prototype.updateCompilerUI = function () {
    this.updateCompilerInfo();
    // Resize in case the new compiler name is too big
    this.resize();
};

Executor.prototype.onCompilerChange = function (value) {
    this.compiler = this.compilerService.findCompiler(this.currentLangId, value);
    this.updateLibraries();
    this.saveState();
    this.compile();
    this.updateCompilerUI();
};

Executor.prototype.sendExecutor = function () {
    this.eventHub.emit('executor', this.id, this.compiler, this.options, this.sourceEditorId);
};

Executor.prototype.onEditorClose = function (editor) {
    if (editor === this.sourceEditorId) {
        // We can't immediately close as an outer loop somewhere in GoldenLayout is iterating over
        // the hierarchy. We can't modify while it's being iterated over.
        this.close();
        _.defer(function (self) {
            self.container.close();
        }, this);
    }
};

Executor.prototype.currentState = function () {
    var state = {
        id: this.id,
        compiler: this.compiler ? this.compiler.id : '',
        source: this.sourceEditorId,
        options: this.options,
        execArgs: this.executionArguments,
        execStdin: this.executionStdin,
        libs: this.libsWidget.get(),
        lang: this.currentLangId,
        compilationPanelShown: !this.panelCompilation.hasClass('d-none'),
        compilerOutShown: !this.compilerOutputSection.hasClass('d-none'),
        argsPanelShown: !this.panelArgs.hasClass('d-none'),
        stdinPanelShown: !this.panelStdin.hasClass('d-none'),
    };
    this.fontScale.addState(state);
    return state;
};

Executor.prototype.saveState = function () {
    this.container.setState(this.currentState());
};

Executor.prototype.getCompilerName = function () {
    return this.compiler ? this.compiler.name : 'No compiler set';
};


Executor.prototype.getLanguageName = function () {
    var lang = options.languages[this.currentLangId];
    return lang ? lang.name : '?';
};

Executor.prototype.getPaneName = function () {
    var langName = this.getLanguageName();
    var compName = this.getCompilerName();
    return compName + ' Executor (Editor #' + this.sourceEditorId + ') ' + langName;
};

Executor.prototype.updateCompilerName = function () {
    var compilerName = this.getCompilerName();
    var compilerVersion = this.compiler ? this.compiler.version : '';
    this.container.setTitle(this.getPaneName());
    this.shortCompilerName.text(compilerName);
    this.setCompilerVersionPopover(compilerVersion);
};

Executor.prototype.setCompilationOptionsPopover = function (content) {
    this.prependOptions.popover('dispose');
    this.prependOptions.popover({
        content: content || 'No options in use',
        template: '<div class="popover' +
            (content ? ' compiler-options-popover' : '') +
            '" role="tooltip"><div class="arrow"></div>' +
            '<h3 class="popover-header"></h3><div class="popover-body"></div></div>',
    });
};

Executor.prototype.setCompilerVersionPopover = function (version) {
    this.fullCompilerName.popover('dispose');
    this.fullCompilerName.popover({
        content: version || '',
        template: '<div class="popover' +
            (version ? ' compiler-options-popover' : '') +
            '" role="tooltip"><div class="arrow"></div>' +
            '<h3 class="popover-header"></h3><div class="popover-body"></div></div>',
    });
};

Executor.prototype.onSettingsChange = function (newSettings) {
    this.settings = _.clone(newSettings);
};

Executor.prototype.handleCompilationStatus = function (status) {
    if (!this.statusLabel || !this.statusIcon) return;

    function ariaLabel() {
        // Compiling...
        if (status.code === 4) return 'Compiling';
        if (status.didExecute) {
            return 'Program compiled & executed';
        } else {
            return 'Program could not be executed';
        }
    }

    function color() {
        // Compiling...
        if (status.code === 4) return 'black';
        if (status.didExecute) return '#12BB12';
        return '#FF1212';
    }

    this.statusIcon
        .removeClass()
        .addClass('status-icon fas')
        .css('color', color())
        .toggle(status.code !== 0)
        .prop('aria-label', ariaLabel())
        .prop('data-status', status.code)
        .toggleClass('fa-spinner', status.code === 4)
        .toggleClass('fa-times-circle', status.code !== 4 && !status.didExecute)
        .toggleClass('fa-check-circle', status.code !== 4 && status.didExecute);
};

Executor.prototype.updateLibraries = function () {
    if (this.libsWidget) this.libsWidget.setNewLangId(this.currentLangId, this.compiler.id, this.compiler.libs);
};

Executor.prototype.onLanguageChange = function (editorId, newLangId) {
    if (this.sourceEditorId === editorId) {
        var oldLangId = this.currentLangId;
        this.currentLangId = newLangId;
        // Store the current selected stuff to come back to it later in the same session (Not state stored!)
        this.infoByLang[oldLangId] = {
            compiler: this.compiler && this.compiler.id ? this.compiler.id : options.defaultCompiler[oldLangId],
            options: this.options,
            execArgs: this.executionArguments,
            execStdin: this.executionStdin,
        };
        var info = this.infoByLang[this.currentLangId] || {};
        this.initLangAndCompiler({lang: newLangId, compiler: info.compiler});
        this.updateCompilersSelector(info);
        this.updateCompilerUI();
        this.saveState();
    }
};

Executor.prototype.getCurrentLangCompilers = function () {
    var allCompilers = this.compilerService.getCompilersForLang(this.currentLangId);
    var hasAtLeastOneExecuteSupported = _.any(allCompilers, function (compiler) {
        return (compiler.supportsExecute !== false);
    });

    if (!hasAtLeastOneExecuteSupported) {
        this.compiler = null;
        return [];
    }

    return _.filter(
        allCompilers,
        _.bind(function (compiler) {
            return ((compiler.hidden !== true) && (compiler.supportsExecute !== false)) ||
                   (this.compiler && compiler.id === this.compiler.id);
        }, this));
};

Executor.prototype.updateCompilersSelector = function (info) {
    this.compilerSelectizer.clearOptions(true);
    this.compilerSelectizer.clearOptionGroups();
    _.each(this.compilerService.getGroupsInUse(this.currentLangId), function (group) {
        this.compilerSelectizer.addOptionGroup(group.value, {label: group.label});
    }, this);
    this.compilerSelectizer.load(_.bind(function (callback) {
        callback(_.map(this.getCurrentLangCompilers(), _.identity));
    }, this));
    this.compilerSelectizer.setValue([this.compiler ? this.compiler.id : null], true);
    this.options = info.options || '';
    this.optionsField.val(this.options);
    this.executionArguments = info.execArgs || '';
    this.execArgsField.val(this.executionArguments);
    this.executionStdin = info.execStdin || '';
    this.execStdinField.val(this.executionStdin);
};

module.exports = {
    Executor: Executor,
};
