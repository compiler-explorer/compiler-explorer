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
var ga = require('../analytics').ga;
var Toggles = require('../widgets/toggles').Toggles;
var FontScale = require('../widgets/fontscale').FontScale;
var options = require('../options').options;
var Alert = require('../alert').Alert;
var LibsWidget = require('../widgets/libs-widget').LibsWidget;
var AnsiToHtml = require('../ansi-to-html').Filter;
var TimingWidget = require('../widgets/timing-info-widget');
var CompilerPicker = require('../compiler-picker').CompilerPicker;
var Settings = require('../settings').Settings;
var utils = require('../utils');
var LibUtils = require('../lib-utils');
var PaneRenaming = require('../widgets/pane-renaming').PaneRenaming;

require('../modes/asm-mode');
require('../modes/ptx-mode');

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
    this.contentRoot = this.domRoot.find('.content');
    this.sourceTreeId = state.tree ? state.tree : false;
    if (this.sourceTreeId) {
        this.sourceEditorId = false;
    } else {
        this.sourceEditorId = state.source || 1;
    }
    this.id = state.id || hub.nextExecutorId();
    this.settings = Settings.getStoredSettings();
    this.initLangAndCompiler(state);
    this.infoByLang = {};
    this.deferCompiles = hub.deferred;
    this.needsCompile = false;
    this.options = state.options || options.compileOptions[this.currentLangId];
    this.executionArguments = state.execArgs || '';
    this.executionStdin = state.execStdin || '';
    this.source = '';
    this.lastResult = {};
    this.lastTimeTaken = 0;
    this.pendingRequestSentAt = 0;
    this.pendingCMakeRequestSentAt = 0;
    this.nextRequest = null;
    this.nextCMakeRequest = null;

    this.alertSystem = new Alert();
    this.alertSystem.prefixMessage = 'Executor #' + this.id;

    this.normalAnsiToHtml = makeAnsiToHtml();
    this.errorAnsiToHtml = makeAnsiToHtml('red');

    this.initButtons(state);

    this.fontScale = new FontScale(this.domRoot, state, 'pre.content');
    this.compilerPicker = new CompilerPicker(
        this.domRoot,
        this.hub,
        this.currentLangId,
        this.compiler ? this.compiler.id : null,
        _.bind(this.onCompilerChange, this),
        this.compilerIsVisible
    );

    this.paneRenaming = new PaneRenaming(this, state);

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

    if (this.sourceTreeId) {
        this.compile();
    }
}

Executor.prototype.compilerIsVisible = function (compiler) {
    return compiler.supportsExecute;
};

Executor.prototype.getEditorIdBySourcefile = function (sourcefile) {
    if (this.sourceTreeId) {
        var tree = this.hub.getTreeById(this.sourceTreeId);
        if (tree) {
            return tree.multifileService.getEditorIdByFilename(sourcefile.file);
        }
    } else {
        if (sourcefile !== null && (sourcefile.file === null || sourcefile.mainsource)) {
            return this.sourceEditorId;
        }
    }

    return false;
};

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
    this.compilerPicker.close();
    this.eventHub.emit('executorClose', this.id);
};

Executor.prototype.undefer = function () {
    this.deferCompiles = false;
    if (this.needsCompile) this.compile();
};

Executor.prototype.resize = function () {
    var topBarHeight = utils.updateAndCalcTopBarHeight(this.domRoot, $(this.topBar[0]), this.hideable);

    // We have some more elements that modify the topBarHeight
    if (!this.panelCompilation.hasClass('d-none')) {
        topBarHeight += this.panelCompilation.outerHeight(true);
    }
    if (!this.panelArgs.hasClass('d-none')) {
        topBarHeight += this.panelArgs.outerHeight(true);
    }
    if (!this.panelStdin.hasClass('d-none')) {
        topBarHeight += this.panelStdin.outerHeight(true);
    }

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
            skipAsm: true,
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

    if (this.sourceTreeId) {
        this.compileFromTree(options, bypassCache);
    } else {
        this.compileFromEditorSource(options, bypassCache);
    }
};

Executor.prototype.compileFromEditorSource = function (options, bypassCache) {
    if (!this.compiler.supportsExecute) {
        this.alertSystem.notify('This compiler (' + this.compiler.name + ') does not support execution', {
            group: 'execution',
        });
        return;
    }
    this.compilerService.expand(this.source).then(_.bind(function (expanded) {
        var request = {
            source: expanded || '',
            compiler: this.compiler ? this.compiler.id : '',
            options: options,
            lang: this.currentLangId,
            files: [],
        };
        if (bypassCache) request.bypassCache = true;
        if (!this.compiler) {
            this.onCompileResponse(request, errorResult('<Please select a compiler>'), false);
        } else {
            this.sendCompile(request);
        }
    }, this));
};

Executor.prototype.compileFromTree = function (options, bypassCache) {
    var tree = this.hub.getTreeById(this.sourceTreeId);
    if (!tree) {
        this.sourceTreeId = false;
        this.compileFromEditorSource(options, bypassCache);
        return;
    }

    var mainsource = tree.multifileService.getMainSource();

    var request = {
        source: mainsource,
        compiler: this.compiler ? this.compiler.id : '',
        options: options,
        lang: this.currentLangId,
        files: tree.multifileService.getFiles(),
    };

    var treeState = tree.currentState();
    var cmakeProject = tree.multifileService.isACMakeProject();

    if (bypassCache) request.bypassCache = true;
    if (!this.compiler) {
        this.onCompileResponse(request, errorResult('<Please select a compiler>'), false);
    } else if (cmakeProject && request.source === '') {
        this.onCompileResponse(request, errorResult('<Please supply a CMakeLists.txt>'), false);
    } else {
        if (cmakeProject) {
            request.options.compilerOptions.cmakeArgs = treeState.cmakeArgs;
            request.options.compilerOptions.customOutputFilename = treeState.customOutputFilename;
            this.sendCMakeCompile(request);
        } else {
            this.sendCompile(request);
        }
    }
};

Executor.prototype.sendCMakeCompile = function (request) {
    var onCompilerResponse = _.bind(this.onCMakeResponse, this);

    if (this.pendingCMakeRequestSentAt) {
        // If we have a request pending, then just store this request to do once the
        // previous request completes.
        this.nextCMakeRequest = request;
        return;
    }
    // this.eventHub.emit('compiling', this.id, this.compiler);
    // Display the spinner
    this.handleCompilationStatus({code: 4});
    this.pendingCMakeRequestSentAt = Date.now();
    // After a short delay, give the user some indication that we're working on their
    // compilation.
    this.compilerService.submitCMake(request)
        .then(function (x) {
            onCompilerResponse(request, x.result, x.localCacheHit);
        })
        .catch(function (x) {
            var message = 'Unknown error';
            if (_.isString(x)) {
                message = x;
            } else if (x) {
                message = x.error || x.code || x.message || x;
            }
            onCompilerResponse(request, errorResult(message), false);
        });
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
                message = x.error || x.code || x.message || x;
            }
            onCompilerResponse(request, errorResult(message), false);
        });
};

Executor.prototype.addCompilerOutputLine = function (msg, container, lineNum/*, column*/) {
    var elem = $('<div/>').appendTo(container);
    if (lineNum) {
        elem.html(
            $('<span class="linked-compiler-output-line"></span>')
                .html(msg)
                .click(_.bind(function (e) {
                    // var editorId = this.getEditorIdBySourcefile(source);
                    // if (editorId) {
                    //     this.eventHub.emit('editorLinkLine', editorId, lineNum, column, column + 1, true);
                    // }
                    // do not bring user to the top of index.html
                    // http://stackoverflow.com/questions/3252730
                    e.preventDefault();
                    return false;
                }, this))
                .on('mouseover', _.bind(function () {
                    // var editorId = this.getEditorIdBySourcefile(source);
                    // if (editorId) {
                    //     this.eventHub.emit('editorLinkLine', editorId, lineNum, column, column + 1, false);
                    // }
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

Executor.prototype.getBuildStdoutFromResult = function (result) {
    var arr = [];

    if (result.buildResult && result.buildResult.stdout !== undefined) {
        arr = arr.concat(result.buildResult.stdout);
    }

    if (result.buildsteps) {
        _.each(result.buildsteps, function (step) {
            arr = arr.concat(step.stdout);
        });
    }

    return arr;
};

Executor.prototype.getBuildStderrFromResult = function (result) {
    var arr = [];

    if (result.buildResult && result.buildResult.stderr !== undefined) {
        arr = arr.concat(result.buildResult.stderr);
    }

    if (result.buildsteps) {
        _.each(result.buildsteps, function (step) {
            arr = arr.concat(step.stderr);
        });
    }

    return arr;
};

Executor.prototype.getExecutionStdoutfromResult = function (result) {
    if (result.execResult && result.execResult.stdout !== undefined) {
        return result.execResult.stdout;
    }

    return result.stdout || [];
};

Executor.prototype.getExecutionStderrfromResult = function (result) {
    if (result.execResult) {
        return result.execResult.stderr;
    }

    return result.stderr || [];
};

Executor.prototype.onCMakeResponse = function (request, result, cached) {
    result.source = this.source;
    this.lastResult = result;
    var timeTaken = Math.max(0, Date.now() - this.pendingCMakeRequestSentAt);
    this.lastTimeTaken = timeTaken;
    var wasRealReply = this.pendingCMakeRequestSentAt > 0;
    this.pendingCMakeRequestSentAt = 0;

    this.handleCompileRequestAndResponse(request, result, cached, wasRealReply, timeTaken);

    this.doNextCMakeRequest();
};

Executor.prototype.doNextCompileRequest = function () {
    if (this.nextRequest) {
        var next = this.nextRequest;
        this.nextRequest = null;
        this.sendCompile(next);
    }
};

Executor.prototype.doNextCMakeRequest = function () {
    if (this.nextCMakeRequest) {
        var next = this.nextCMakeRequest;
        this.nextCMakeRequest = null;
        this.sendCMakeCompile(next);
    }
};

Executor.prototype.handleCompileRequestAndResponse = function (request, result, cached, wasRealReply, timeTaken) {
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
    var compileStdout = this.getBuildStdoutFromResult(result);
    var compileStderr = this.getBuildStderrFromResult(result);
    var execStdout = this.getExecutionStdoutfromResult(result);
    var execStderr = this.getExecutionStderrfromResult(result);

    var buildResultCode = 0;

    if (result.buildResult) {
        buildResultCode = result.buildResult.code;
    } else if (result.buildsteps) {
        _.each(result.buildsteps, function (step) {
            buildResultCode = step.code;
        });
    }

    if (!result.didExecute) {
        this.executionStatusSection.append($('<div/>').text('Could not execute the program'));
        this.executionStatusSection.append($('<div/>').text('Compiler returned: ' + buildResultCode));
    }
    if (compileStdout.length > 0) {
        this.compilerOutputSection.append($('<div/>').text('Compiler stdout'));
        this.handleOutput(compileStdout, this.compilerOutputSection, this.normalAnsiToHtml);
    }
    if (compileStderr.length > 0) {
        this.compilerOutputSection.append($('<div/>').text('Compiler stderr'));
        this.handleOutput(compileStderr, this.compilerOutputSection, this.errorAnsiToHtml);
    }
    if (result.didExecute) {
        var exitCode = result.execResult ? result.execResult.code : result.code;
        this.executionOutputSection.append($('<div/>').text('Program returned: ' + exitCode));
        if (execStdout.length > 0) {
            this.executionOutputSection.append($('<div/>').text('Program stdout'));
            var outElem = this.handleOutput(execStdout, this.executionOutputSection, this.normalAnsiToHtml);
            outElem.addClass('execution-stdout');
        }
        if (execStderr.length > 0) {
            this.executionOutputSection.append($('<div/>').text('Program stderr'));
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
};

Executor.prototype.onCompileResponse = function (request, result, cached) {
    // Save which source produced this change. It should probably be saved earlier though
    result.source = this.source;
    this.lastResult = result;
    var timeTaken = Math.max(0, Date.now() - this.pendingRequestSentAt);
    this.lastTimeTaken = timeTaken;
    var wasRealReply = this.pendingRequestSentAt > 0;
    this.pendingRequestSentAt = 0;

    this.handleCompileRequestAndResponse(request, result, cached, wasRealReply, timeTaken);

    this.doNextCompileRequest();
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
    if (this.sourceTreeId) {
        var tree = this.hub.getTreeById(this.sourceTreeId);
        if (tree) {
            if (tree.multifileService.isEditorPartOfProject(editor)) {
                if (this.settings.compileOnChange) {
                    this.compile();

                    return;
                }
            }
        }
    }

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
    this.toggleWrapButton = new Toggles(this.domRoot.find('.options'), state);
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
    this.fullTimingInfo = this.domRoot.find('.full-timing-info');
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
    this.setCompilerVersionPopover({version: '', fullVersion: ''}, '');

    this.topBar = this.domRoot.find('.top-bar');
    this.bottomBar = this.domRoot.find('.bottom-bar');
    this.statusLabel = this.domRoot.find('.status-text');

    this.hideable = this.domRoot.find('.hideable');
    this.statusIcon = this.domRoot.find('.status-icon');

    this.panelCompilation = this.domRoot.find('.panel-compilation');
    this.panelArgs = this.domRoot.find('.panel-args');
    this.panelStdin = this.domRoot.find('.panel-stdin');

    this.wrapButton = this.domRoot.find('.wrap-lines');
    this.wrapTitle = this.wrapButton.prop('title');

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

    if (state.wrap === true) {
        this.contentRoot.addClass('wrap');
        this.wrapButton.prop('title', '[ON] ' + this.wrapTitle);
    } else {
        this.contentRoot.removeClass('wrap');
        this.wrapButton.prop('title', '[OFF] ' + this.wrapTitle);
    }
};

Executor.prototype.onLibsChanged = function () {
    this.saveState();
    this.compile();
};

Executor.prototype.initLibraries = function (state) {
    this.libsWidget = new LibsWidget(
        this.currentLangId,
        this.compiler,
        this.libsButton,
        state,
        _.bind(this.onLibsChanged, this),
        LibUtils.getSupportedLibraries(this.compiler ? this.compiler.libsArr : [], this.currentLangId)
    );
};

Executor.prototype.onFontScale = function () {
    this.saveState();
};

Executor.prototype.initListeners = function () {
    // this.filters.on('change', _.bind(this.onFilterChange, this));
    this.fontScale.on('change', _.bind(this.onFontScale, this));
    this.paneRenaming.on('renamePane', this.saveState.bind(this));
    this.toggleWrapButton.on('change', _.bind(this.onToggleWrapChange, this));

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

    this.fullTimingInfo
        .off('click')
        .click(_.bind(function () {
            TimingWidget.displayCompilationTiming(this.lastResult, this.lastTimeTaken);
        }, this));
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

Executor.prototype.onRequestCompilation = function (editorId, treeId) {
    if ((editorId === this.sourceEditorId) || (treeId && treeId === this.sourceTreeId)) {
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

Executor.prototype.onToggleWrapChange = function () {
    var state = this.currentState();
    this.contentRoot.toggleClass('wrap', state.wrap);
    this.wrapButton.prop('title', '[' + (state.wrap ? 'ON' : 'OFF') + '] ' + this.wrapTitle);
    this.saveState();
};

Executor.prototype.sendExecutor = function () {
    this.eventHub.emit('executor', this.id, this.compiler, this.options, this.sourceEditorId, this.sourceTreeId);
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
        tree: this.sourceTreeId,
        options: this.options,
        execArgs: this.executionArguments,
        execStdin: this.executionStdin,
        libs: this.libsWidget.get(),
        lang: this.currentLangId,
        compilationPanelShown: !this.panelCompilation.hasClass('d-none'),
        compilerOutShown: !this.compilerOutputSection.hasClass('d-none'),
        argsPanelShown: !this.panelArgs.hasClass('d-none'),
        stdinPanelShown: !this.panelStdin.hasClass('d-none'),
        wrap: this.toggleWrapButton.get().wrap,
    };
    this.paneRenaming.addState(state);
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

Executor.prototype.getLinkHint = function () {
    var linkhint = '';
    if (this.sourceTreeId) {
        linkhint = 'Tree #' + this.sourceTreeId;
    } else {
        linkhint = 'Editor #' + this.sourceEditorId;
    }
    return linkhint;
};

Executor.prototype.getPaneName = function () {
    var langName = this.getLanguageName();
    var compName = this.getCompilerName();
    return 'Executor ' + compName + ' (' + langName + ', ' + this.getLinkHint() + ')';
};

Executor.prototype.updateTitle = function () {
    var name = this.paneName ? this.paneName : this.getPaneName();
    this.container.setTitle(_.escape(name));
};

Executor.prototype.updateCompilerName = function () {
    this.updateTitle();
    var compilerName = this.getCompilerName();
    var compilerVersion = this.compiler ? this.compiler.version : '';
    var compilerFullVersion = this.compiler && this.compiler.fullVersion ? this.compiler.fullVersion : compilerVersion;
    var compilerNotification = this.compiler ? this.compiler.notification : '';
    this.shortCompilerName.text(compilerName);
    this.setCompilerVersionPopover({version: compilerVersion, fullVersion: compilerFullVersion}, compilerNotification);
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

Executor.prototype.setCompilerVersionPopover = function (version, notification) {
    this.fullCompilerName.popover('dispose');
    // `notification` contains HTML from a config file, so is 'safe'.
    // `version` comes from compiler output, so isn't, and is escaped.
    var bodyContent = $('<div>');
    var versionContent = $('<div>')
        .html(_.escape(version.version));
    bodyContent.append(versionContent);
    if (version.fullVersion) {
        var hiddenSection = $('<div>');
        var hiddenVersionText = $('<div>')
            .html(_.escape(version.fullVersion))
            .hide();
        var clickToExpandContent = $('<a>')
            .attr('href', 'javascript:;')
            .text('Toggle full version output')
            .on('click', _.bind(function () {
                versionContent.toggle();
                hiddenVersionText.toggle();
                this.fullCompilerName.popover('update');
            }, this));
        hiddenSection.append(hiddenVersionText).append(clickToExpandContent);
        bodyContent.append(hiddenSection);
    }
    this.fullCompilerName.popover({
        html: true,
        title: notification ?
            $.parseHTML('<span>Compiler Version: ' + notification + '</span>')[0] :
            'Full compiler version',
        content: bodyContent,
        template: '<div class="popover' + (version ? ' compiler-options-popover' : '') + '" role="tooltip">' +
            '<div class="arrow"></div>' +
            '<h3 class="popover-header"></h3><div class="popover-body"></div>' +
            '</div>',
    });
};

Executor.prototype.onSettingsChange = function (newSettings) {
    this.settings = _.clone(newSettings);
};

function ariaLabel(status) {
    // Compiling...
    if (status.code === 4) return 'Compiling';
    if (status.didExecute) {
        return 'Program compiled & executed';
    } else {
        return 'Program could not be executed';
    }
}

function color(status) {
    // Compiling...
    if (status.code === 4) return '#888888';
    if (status.didExecute) return '#12BB12';
    return '#FF1212';
}

Executor.prototype.handleCompilationStatus = function (status) {
    // We want to do some custom styles for the icon, so we don't pass it here and instead do it later
    this.compilerService.handleCompilationStatus(this.statusLabel, null, status);

    if (this.statusIcon != null) {
        this.statusIcon
            .removeClass()
            .addClass('status-icon fas')
            .css('color', color(status))
            .toggle(status.code !== 0)
            .prop('aria-label', ariaLabel(status))
            .prop('data-status', status.code)
            .toggleClass('fa-spinner fa-spin', status.code === 4)
            .toggleClass('fa-times-circle', status.code !== 4 && !status.didExecute)
            .toggleClass('fa-check-circle', status.code !== 4 && status.didExecute);
    }
};

Executor.prototype.updateLibraries = function () {
    if (this.libsWidget) {
        var filteredLibraries = {};
        if (this.compiler) {
            filteredLibraries = LibUtils.getSupportedLibraries(this.compiler.libsArr, this.currentLangId);
        }

        this.libsWidget.setNewLangId(this.currentLangId,
            this.compiler ? this.compiler.id : false,
            filteredLibraries);
    }
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
    this.compilerPicker.update(this.currentLangId, this.compiler ? this.compiler.id : null);
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
