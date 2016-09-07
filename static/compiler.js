define(function (require) {
    "use strict";
    var CodeMirror = require('codemirror');
    var $ = require('jquery');
    var _ = require('underscore');
    var ga = require('analytics').ga;
    var colour = require('colour');
    var Toggles = require('toggles')
    require('asm-mode');
    require('selectize');

    var options = require('options');
    var compilers = options.compilers;
    var compilersById = _.object(_.pluck(compilers, "id"), compilers);

    function Compiler(hub, container, state) {
        var self = this;
        this.container = container;
        this.eventHub = container.layoutManager.eventHub;
        this.domRoot = container.getElement();
        this.domRoot.html($('#compiler').html());

        this.id = state.id || hub.nextId();
        this.sourceEditorId = state.source || 1;
        this.compiler = compilersById[state.compiler] || options.defaultCompiler;
        this.options = state.options || options.compileOptions;
        this.filters = new Toggles(this.domRoot.find(".filters"), state.filters);
        this.source = "";
        this.assembly = [];

        this.debouncedAjax = _.debounce($.ajax, 250);

        this.domRoot.find(".compiler").selectize({
            sortField: 'name',
            valueField: 'id',
            labelField: 'name',
            searchField: ['name'],
            options: compilers,
            items: [this.compiler.id],
            openOnFocus: true
        }).on('change', function () {
            self.onCompilerChange($(this).val());
        });
        var optionsChange = function () {
            self.onOptionsChange($(this).val());
        };
        this.domRoot.find(".options")
            .val(this.options)
            .on("change", optionsChange)
            .on("keyup", optionsChange);

        var outputEditor = CodeMirror.fromTextArea(this.domRoot.find("textarea")[0], {
            lineNumbers: true,
            mode: "text/x-asm",
            readOnly: true,
            gutters: ['CodeMirror-linenumbers'],
            lineWrapping: true
        });
        this.outputEditor = outputEditor;

        function resize() {
            var topBarHeight = self.domRoot.find(".top-bar").outerHeight(true);
            outputEditor.setSize(self.domRoot.width(), self.domRoot.height() - topBarHeight);
            outputEditor.refresh();
        }

        this.filters.on('change', _.bind(this.onFilterChange, this));

        container.on('resize', resize);
        container.on('open', function () {
            self.eventHub.emit('compilerOpen', self.id);
            resize();
        });
        container.setTitle("Compiled");
        container.on('close', function () {
            self.eventHub.emit('compilerClose', self.id);
        });
        self.eventHub.on('editorChange', this.onEditorChange, this);
        self.eventHub.on('colours', this.onColours, this);
    }

    Compiler.prototype.compile = function (fromEditor) {
        var self = this;
        if (!this.source || !this.compiler) return;  // TODO blank out the output?
        var request = {
            fromEditor: fromEditor,
            source: this.source,
            compiler: this.compiler.id,
            options: this.options,
            filters: this.filters.get()
        };
        // TODO: caching (pre-timestamp)
        request.timestamp = Date.now();
        this.debouncedAjax({
            type: 'POST',
            url: '/compile',
            dataType: 'json',
            contentType: 'application/json',
            data: JSON.stringify(request),
            success: function (result) {
                self.onCompileResponse(request, result);
            },
            error: function (xhr, e_status, error) {
                self.onCompileResponse(request, errorResult("Remote compilation failed: " + error));
            },
            cache: false
        });
    };

    Compiler.prototype.setAssembly = function (assembly) {
        this.assembly = assembly;
        this.outputEditor.setValue(_.pluck(assembly, 'text').join("\n"));

        var addrToAddrDiv = {};
        _.each(this.assembly, _.bind(function (obj, line) {
            var address = obj.address ? obj.address.toString(16) : "";
            var div = $("<div class='address cm-number'>" + address + "</div>");
            addrToAddrDiv[address] = {div: div, line: line};
            this.outputEditor.setGutterMarker(line, 'address', div[0]);
        }, this));

        _.each(this.assembly, _.bind(function (obj, line) {
            var opcodes = $("<div class='opcodes'></div>");
            if (obj.opcodes) {
                var title = [];
                _.each(obj.opcodes, function (op) {
                    var opcodeNum = "00" + op.toString(16);
                    opcodeNum = opcodeNum.substr(opcodeNum.length - 2);
                    title.push(opcodeNum);
                    var opcode = $("<span class='opcode'>" + opcodeNum + "</span>");
                    opcodes.append(opcode);
                });
                opcodes.attr('title', title.join(" "));
            }
            this.outputEditor.setGutterMarker(line, 'opcodes', opcodes[0]);
            if (obj.links) {
                _.each(obj.links, _.bind(function (link) {
                    var from = {line: line, ch: link.offset};
                    var to = {line: line, ch: link.offset + link.length};
                    var address = link.to.toString(16);
                    var thing = $("<a href='#' class='cm-number'>" + address + "</a>");
                    this.outputEditor.markText(
                        from, to, {replacedWith: thing[0], handleMouseEvents: false});
                    var dest = addrToAddrDiv[address];
                    if (dest) {
                        var editor = this.outputEditor;
                        thing.on('hover', function (e) {
                            var entered = e.type == "mouseenter";
                            dest.div.toggleClass("highlighted", entered);
                            thing.toggleClass("highlighted", entered);
                        });
                        thing.on('click', function (e) {
                            editor.scrollIntoView({line: dest.line, ch: 0}, 30);
                            dest.div.toggleClass("highlighted", false);
                            thing.toggleClass("highlighted", false);
                            e.preventDefault();
                        });
                    }
                }, this));
            }
        }, this));
    };

    function errorResult(text) {
        return {asm: fakeAsm(text)};
    }

    function fakeAsm(text) {
        return [{text: text, source: null, fake: true}];
    }

    Compiler.prototype.onCompileResponse = function (request, result) {
        ga('send', 'event', 'Compile', request.compiler, request.options, result.code);
        ga('send', 'timing', 'Compile', 'Timing', Date.now() - request.timestamp)
        this.outputEditor.operation(_.bind(function () {
            this.setAssembly(result.asm || fakeAsm("[no output]"));
            if (request.filters.binary) {
                this.outputEditor.setOption('lineNumbers', false);
                this.outputEditor.setOption('gutters', ['address', 'opcodes']);
            } else {
                this.outputEditor.setOption('lineNumbers', true);
                this.outputEditor.setOption('gutters', ['CodeMirror-linenumbers']);
            }
        }, this));
        this.eventHub.emit('compileResult', this.id, this.compiler, result);
    };

    Compiler.prototype.onEditorListChange = function () {
        // TODO: if we can't find our source, select none?
        // TODO: Update dropdown of source
        // TODO: remember if we change editor source we must detach and re-attach
        //this.sourceEditorId = ...
    };

    Compiler.prototype.onEditorChange = function (editor, source) {
        if (editor == this.sourceEditorId) {
            this.source = source;
            this.compile();
        }
    };
    Compiler.prototype.onOptionsChange = function (options) {
        this.options = options;
        this.saveState();
        this.compile();
    };
    Compiler.prototype.onCompilerChange = function (value) {
        this.compiler = compilersById[value];  // TODO check validity?
        this.saveState();
        this.compile();
    };

    Compiler.prototype.onFilterChange = function () {
        this.saveState();
        this.compile();
    };

    Compiler.prototype.saveState = function () {
        this.container.setState({
            compiler: this.compiler.id,
            options: this.options,
            source: this.editor,
            filters: this.filters.get()
        });
    };

    Compiler.prototype.onColours = function (editor, colours) {
        if (editor == this.sourceEditorId) {
            var asmColours = {};
            this.assembly.forEach(function (x, index) {
                if (x.source) asmColours[index] = colours[x.source - 1];
            });
            colour.applyColours(this.outputEditor, asmColours);
        }
    };

    return Compiler;
});