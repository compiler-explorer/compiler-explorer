// Copyright (c) 2012-2016, Matt Godbolt
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

function parseLines(lines, callback) {
    var re = /^\/tmp\/[^:]+:([0-9]+)(:([0-9]+))?:\s+(.*)/;
    $.each(lines.split('\n'), function (_, line) {
        line = line.trim();
        if (line !== "") {
            var match = line.match(re);
            if (match) {
                callback(parseInt(match[1]), match[4].trim());
            } else {
                callback(null, line);
            }
        }
    });
}

// Function called on the editor's window, or on a slot's window
function clearBackground(cm) {
    for (var i = 0; i < cm.lineCount(); ++i) {
        cm.removeLineClass(i, "background", null);
    }
}

function Editor(container, lang) {
    var domRoot = container.getElement();
    var cmMode;
    switch (lang.toLowerCase()) {
        default:
            cmMode = "text/x-c++src";
            break;
        case "c":
            cmMode = "text/x-c";
            break;
        case "rust":
            cmMode = "text/x-rustsrc";
            break;
        case "d":
            cmMode = "text/x-d";
            break;
        case "go":
            cmMode = "text/x-go";
            break;
    }

    var cppEditor = CodeMirror.fromTextArea(domRoot.find("textarea")[0], {
        lineNumbers: true,
        matchBrackets: true,
        useCPP: true,
        mode: cmMode
    });

    // With reference to "fix typing '#' in emacs mode"
    // https://github.com/mattgodbolt/gcc-explorer/pull/131
    cppEditor.setOption("extraKeys", {
        "Alt-F": false
    });
    // cppEditor.on("change", function () {
    //     if ($('.autocompile').hasClass('active')) {
    //         onEditorChange();
    //     }
    // });
    function resize() {
        cppEditor.setSize(domRoot.width(), domRoot.height());
        cppEditor.refresh();
    }
    container.on('resize', resize);
    container.on('open', resize);
    container.setTitle(lang + " source");
}

function CompileToAsm(container) {
    var domRoot = container.getElement();
    var asmCodeMirror = CodeMirror.fromTextArea(domRoot.find("textarea")[0], {
        lineNumbers: true,
        mode: "text/x-asm",
        readOnly: true,
        gutters: ['CodeMirror-linenumbers'],
        lineWrapping: true
    });
    function resize() {
        asmCodeMirror.setSize(domRoot.width(), domRoot.height());
        asmCodeMirror.refresh();
    }
    container.on('resize', resize);
    container.on('open', resize);
    container.setTitle("monkey");
}

const NumRainbowColours = 12;

// This function is called in function initialise in static/gcc.js
function Compiler(domRoot, origFilters, windowLocalPrefix,
                  onEditorChangeCallback, lang, compilers, defaultCompiler) {
    console.log("[TRACE] Entering function Compiler()");
    // Global array that will contain the slots.
    var slots = [];
    var Slot = function (id) {
        // this id will be used in the DOM, ex : class="slot"+id
        // the getAvailableId could be placed here but that could
        // lead to confusion if the Slot is not pushed in slots
        this.id = id;
        this.asmCodeMirror = null;
        this.currentAssembly = null;
        // lastRequest contains the previous request in the slot
        // it can be used to prevent useless re-compilation
        // if no parameters were really changed (e.g. spaces moved)
        this.lastRequest = null;
        this.pendingDiffs = [];
        this.node = null; // Will be initialized in slotDomCtor
        this.description = function () {
            var options = this.node.find('.compiler-options').val();
            var compiler = currentCompiler(this);
            return compiler.name + " " + options;
        };
        this.shortDescription = function () {
            var description = this.description();
            if (description.length >= 19) {
                return description.substring(0, 13) + "[...]"
            } else {
                return description;
            }
        }
    };

    var diffs = [];
    var Diff = function (id) {
        this.id = id;
        this.beforeSlot = null;
        this.afterSlot = null;
        this.currentDiff = null;
        this.asmCodeMirror = null;
        this.zones = null;

        // used only if the editor's code is modified,
        // to prevent two diff generation by waiting for the second one.
        this.remainingTriggers = 2;
        this.node = null; // Will be initialized in diffDomCtor
    };

    function contains(array, object) {
        for (var i = 0; i < array.length; i++) {
            if (array[i] === object) {
                return true;
            }
        }
        return false;
    }

    // adds diff to the set (array) slot.pendingDiffs
    function addToPendings(slot, diff) {
        if (!contains(slot.pendingDiffs, diff)) {
            slot.pendingDiffs.push(diff);
        }
    }

    // returns the smallest Natural that is not used as an id
    // (suppose that ids are Naturals, but any other way of getting 
    // an unique id usable in HTML's classes should work)
    function getAvailableId(array) {
        if (array.length == 0) return 0;
        var usedIds = [];
        var i;
        for (i = 0; i < array.length; i++) {
            usedIds.push(array[i].id)
        }
        usedIds.sort();
        var prev = -1;
        for (i = 0; i < usedIds.length; i++) {
            if (usedIds[i] - prev > 1) return prev + 1;
            prev = usedIds[i];
        }
        return usedIds.length;
    }


    // setSetting('leaderSlot', null); // TODO : remove ?
    var leaderSlot = null; // is set up correctly at the end of this function.
    function isLeader(slot) {
        return (slot == leaderSlot);
    }

    var compilersById = {};
    var compilersByAlias = {};

    var pendingTimeoutInEditor = null;

    var cppEditor = null;

    // default filters
    var filters_ = $.extend({}, origFilters);
    var ignoreChanges = true; // Horrible hack to avoid onEditorChange doing anything on first starting, ie before we've set anything up.

    function setCompilerById(compilerId, slot) {
        var compilerNode = slot.node.find('.compiler');
        compilerNode.text(compilersById[compilerId].name);
        compilerNode.attr('data', compilerId);
    }

    function currentCompilerId(slot) {
        return slot.node.find('.compiler').attr('data');
    }

    function currentCompiler(slot) {
        return compilersById[currentCompilerId(slot)];
    }

    // set the autocompile button on static/index.html
    var autocompile = $('.autocompile');
    autocompile.click(function (e) {
        autocompile.toggleClass('active');
        onEditorChange();
        setSetting('autocompile', autocompile.hasClass('active'));
    });
    autocompile.toggleClass('active', getSetting("autocompile") !== "false");

    // handle filter options that are specific to a compiler
    function patchUpFilters(filters) {
        filters = $.extend({}, filters);
        // TODO : correct the fact that filters should be slot specific !!
        var compiler = currentCompiler(slots[0]);
        var compilerSupportsBinary = compiler ? compiler.supportsBinary : true;
        if (filters.binary && !(OPTIONS.supportsBinary && compilerSupportsBinary)) {
            filters.binary = false;
        }
        return filters;
    }

    var cmMode;
    switch (lang.toLowerCase()) {
        default:
            cmMode = "text/x-c++src";
            break;
        case "c":
            cmMode = "text/x-c";
            break;
        case "rust":
            cmMode = "text/x-rustsrc";
            break;
        case "d":
            cmMode = "text/x-d";
            break;
        case "go":
            cmMode = "text/x-go";
            break;
    }

    // Set up the editor's window
    // [0] is mandatory here because domRoot.find() returns an array of all
    // matching elements. It is not a problem for jquery which applies it's
    // functions to all elements, whereas CodeMirror works on a single DOM node.
    cppEditor = CodeMirror.fromTextArea(domRoot.find("textarea.editor")[0], {
        lineNumbers: true,
        matchBrackets: true,
        useCPP: true,
        mode: cmMode
    });

    // With reference to "fix typing '#' in emacs mode"
    // https://github.com/mattgodbolt/gcc-explorer/pull/131
    cppEditor.setOption("extraKeys", {
        "Alt-F": false
    });
    cppEditor.on("change", function () {
        if ($('.autocompile').hasClass('active')) {
            onEditorChange();
        }
    });

    // Load/Save/Remove setting from the browser
    function getSetting(name) {
        return window.localStorage[windowLocalPrefix + "." + name];
    }

    function setSetting(name, value) {
        window.localStorage[windowLocalPrefix + "." + name] = value;
    }

    function removeSetting(name) {
        // source: http://stackoverflow.com/questions/9943220/how-to-delete-a-localstorage-item-when-the-browser-window-tab-is-closed
        window.localStorage.removeItem(windowLocalPrefix + "." + name);
    }

    var codeText = getSetting('code');
    if (!codeText) codeText = $(".template.lang." + lang.replace(/[^a-zA-Z]/g, '').toLowerCase()).text();
    if (codeText) cppEditor.setValue(codeText);

    ignoreChanges = false;

    // auxiliary function to onCompileResponse(),
    // used to display the compiler error messages in the editor's window
    function makeErrNode(text) {
        var clazz = "error";
        if (text.match(/^warning/)) clazz = "warning";
        if (text.match(/^note/)) clazz = "note";
        var node = $('<div class="' + clazz + ' inline-msg"><span class="icon">!!</span><span class="msg"></span></div>');
        node.find(".msg").text(text);
        return node[0];
    }

    // keep track of error widgets to delete them if need be
    var errorWidgets = [];

    // Compilation's callback
    function onCompileResponse(slot, request, data) {
        var leaderSlot = getSetting('leaderSlot');

        console.log("[CALLBACK] onCompileResponse() with slot = " + slot.id + ", leaderSlot = ", +leaderSlot);
        var stdout = data.stdout || "";
        var stderr = data.stderr || "";
        if (data.code === 0) {
            stdout += "Compiled ok in slot " + slot.id + "\n";
        } else {
            stderr += "Compilation failed in slot " + slot.id + "\n";
        }
        if (_gaq) {
            // TODO : to modify to handle the slots ?
            _gaq.push(['_trackEvent', 'Compile', request.compiler, request.options, data.code]);
            _gaq.push(['_trackTiming', 'Compile', 'Timing', new Date() - request.timestamp]);
        }
        slot.node.find('.result .output :visible').remove();
        // only show in Editor messages comming from the leaderSlot
        var i;
        if (slot.id == leaderSlot) {
            for (i = 0; i < errorWidgets.length; ++i)
                cppEditor.removeLineWidget(errorWidgets[i]);
            errorWidgets.length = 0;
        }
        var numLines = 0;
        parseLines(stderr + stdout, function (lineNum, msg) {
            if (numLines > 50) return;
            if (numLines === 50) {
                lineNum = null;
                msg = "Too many output lines...truncated";
            }
            numLines++;
            var elem = slot.node.find('.result .output .template').clone().appendTo(slot.node.find(' .result .output')).removeClass('template');
            if (lineNum) {
                // only show messages coming from the leaderSlot in the Editor
                if (slot.id == leaderSlot) {
                    errorWidgets.push(cppEditor.addLineWidget(lineNum - 1, makeErrNode(msg), {
                        coverGutter: false, noHScroll: true
                    }));
                }
                elem.html($('<a href="#">').text(lineNum + " : " + msg)).click(function (e) {
                    cppEditor.setSelection({line: lineNum - 1, ch: 0}, {line: lineNum, ch: 0});

                    // do not bring user to the top of index.html
                    // http://stackoverflow.com/questions/3252730
                    e.preventDefault();
                    return false;
                });
            } else {
                elem.text(msg);
            }
        });
        slot.currentAssembly = data.asm || fakeAsm("[no output]");
        updateAsm(slot);
        for (i = 0; i < slot.pendingDiffs.length; i++) {
            onDiffChange(slot.pendingDiffs[i], request.fromEditor);
        }
    }

    function numberUsedLines() {
        var result = {source: {}, pending: false, asm: {}};
        _.each(slots, function (slot) {
            if (slot.currentAssembly) {
                slot.currentAssembly.forEach(function (x) {
                    if (x.source) result.source[x.source - 1] = true;
                });
            }
        });
        var ordinal = 0;
        _.each(result.source, function (v, k) {
            result.source[k] = ordinal++;
        });
        _.each(slots, function (slot) {
            var asmLines = {};
            if (slot.currentAssembly) {
                slot.currentAssembly.forEach(function (x, index) {
                    if (x.source) asmLines[index] = result.source[x.source - 1];
                    if (x.fake) result.pending = true;
                });
            }
            result.asm[slot.id] = asmLines;
        });
        return result;
    }

    var colourise = function colourise() {
        var numberedLines = numberUsedLines();
        cppEditor.operation(function () {
            clearBackground(cppEditor);
            if (numberedLines.pending) return; // don't colourise until all results are in
            _.each(numberedLines.source, function (ordinal, line) {
                cppEditor.addLineClass(parseInt(line),
                    "background", "rainbow-" + (ordinal % NumRainbowColours));
            });
        });
        // colourise the assembly in slots
        _.each(slots, function (slot) {
            slot.asmCodeMirror.operation(function () {
                clearBackground(slot.asmCodeMirror);
                if (numberedLines.pending) return; // don't colourise until all results are in
                _.each(numberedLines.asm[slot.id], function (ordinal, line) {
                    slot.asmCodeMirror.addLineClass(parseInt(line),
                        "background", "rainbow-" + (ordinal % NumRainbowColours));
                });
            });
        });
    };

    function updateAsm(slot, forceUpdate) {
        console.log("[CALLBACK] updateAsm() with slot = " + slot.id + ", forceUpdate = " + forceUpdate);
        if (!slot.currentAssembly) return;
        var hashedUpdate = JSON.stringify(slot.currentAssembly);
        if (!forceUpdate && JSON.stringify(slot.lastUpdatedAsm) == hashedUpdate) {
            return;
        }
        slot.lastUpdatedAsm = hashedUpdate;

        var asmText = $.map(slot.currentAssembly, function (x) {
            return x.text;
        }).join("\n");

        var filters = currentFilters();
        slot.asmCodeMirror.operation(function () {
            slot.asmCodeMirror.setValue(asmText);
            clearBackground(slot.asmCodeMirror);
            var addrToAddrDiv = {};
            $.each(slot.currentAssembly, function (line, obj) {
                var address = obj.address ? obj.address.toString(16) : "";
                var div = $("<div class='address cm-number'>" + address + "</div>");
                addrToAddrDiv[address] = {div: div, line: line};
                slot.asmCodeMirror.setGutterMarker(line, 'address', div[0]);
            });
            $.each(slot.currentAssembly, function (line, obj) {
                var opcodes = $("<div class='opcodes'></div>");
                if (obj.opcodes) {
                    var title = [];
                    $.each(obj.opcodes, function (_, op) {
                        var opcodeNum = "00" + op.toString(16);
                        opcodeNum = opcodeNum.substr(opcodeNum.length - 2);
                        title.push(opcodeNum);
                        var opcode = $("<span class='opcode'>" + opcodeNum + "</span>");
                        opcodes.append(opcode);
                    });
                    opcodes.attr('title', title.join(" "));
                }
                slot.asmCodeMirror.setGutterMarker(line, 'opcodes', opcodes[0]);
                if (obj.links) {
                    $.each(obj.links, function (_, link) {
                        var from = {line: line, ch: link.offset};
                        var to = {line: line, ch: link.offset + link.length};
                        var address = link.to.toString(16);
                        var thing = $("<a href='#' class='cm-number'>" + address + "</a>");
                        slot.asmCodeMirror.markText(
                            from, to, {replacedWith: thing[0], handleMouseEvents: false});
                        var dest = addrToAddrDiv[address];
                        if (dest) {
                            thing.on('hover', function (e) {
                                var entered = e.type == "mouseenter";
                                dest.div.toggleClass("highlighted", entered);
                                thing.toggleClass("highlighted", entered);
                            });
                            thing.on('click', function (e) {
                                slot.asmCodeMirror.scrollIntoView({line: dest.line, ch: 0}, 30);
                                dest.div.toggleClass("highlighted", false);
                                thing.toggleClass("highlighted", false);

                                // do not bring user to the top of index.html
                                e.preventDefault();
                            });
                        }
                    });
                }
            });
            if (filters.binary) {
                slot.asmCodeMirror.setOption('lineNumbers', false);
                slot.asmCodeMirror.setOption('gutters', ['address', 'opcodes']);
            } else {
                slot.asmCodeMirror.setOption('lineNumbers', true);
                slot.asmCodeMirror.setOption('gutters', ['CodeMirror-linenumbers']);
            }
        });

        // colourise the editor
        if (filters.colouriseAsm) {
            colourise();
        }
    }

    function fakeAsm(text) {
        return [{text: text, source: null, fake: true}];
    }

    // slot's parameters callback
    // TODO : refactor with onEditorChange : those functions could call an updateSlot(slot)
    function onParamChange(slot, force) {
        console.log("[CALLBACK] onParamChange() on slot " + slot.id);

        // update the UI of the diff that depends of this slot
        for (var i = 0; i < slot.pendingDiffs.length; i++) {
            updateDiffUI(slot.pendingDiffs[i]);
        }

        if (ignoreChanges) return;  // Ugly hack during startup.
        if (slot.pendingTimeoutInSlot) {
            console.log("[TIME] Clearing time out in slot " + slot.id);
            clearTimeout(slot.pendingTimeoutInSlot);
        }

        console.log("[TIME] Setting time out in slot " + slot.id);
        slot.pendingTimeoutInSlot = setTimeout(function () {
            console.log("[TIME] Timed out : compiling in slot " + slot.id + " triggered by modification of params...");
            (function (slot) {
                var data = {
                    fromEditor: false,
                    source: cppEditor.getValue(),
                    compiler: currentCompilerId(slot),
                    options: slot.node.find('.compiler-options').val(),
                    filters: currentFilters()
                };
                setSetting('compiler' + slot.id, data.compiler);
                setSetting('compilerOptions' + slot.id, data.options);
                var stringifiedReq = JSON.stringify(data);
                if (!force && stringifiedReq == slot.lastRequest) return;
                slot.lastRequest = stringifiedReq;
                data.timestamp = new Date();
                $.ajax({
                    type: 'POST',
                    url: '/compile',
                    dataType: 'json',
                    contentType: 'application/json',
                    data: JSON.stringify(data),
                    success: function (result) {
                        onCompileResponse(slot, data, result);
                    },
                    error: function (xhr, e_status, error) {
                        console.log("AJAX request failed, reason : " + error);
                    },
                    cache: false
                });
                setSetting('code', cppEditor.getValue());
                slot.currentAssembly = fakeAsm("[Processing...]");
                updateAsm(slot);
            })(slot);
        }, 750); // Time in ms after which action is taken (if inactivity)

        // (maybe redundant) execute the callback passed to Compiler()
        onEditorChangeCallback();
    }

    function onEditorChange(force) {
        console.log("[CALLBACK] onEditorChange()");
        if (ignoreChanges) return;  // Ugly hack during startup.
        if (pendingTimeoutInEditor) {
            console.log("[TIME] Clearing time out in editor");
            clearTimeout(pendingTimeoutInEditor);
        }

        console.log("[TIME] Setting time out in editor");
        pendingTimeoutInEditor = setTimeout(function () {
            console.log("[TIME] Timed out in editor : compiling in all " + slots.length + " slots...");
            for (var i = 0; i < slots.length; i++) {
                (function (slot) {
                    var data = {
                        fromEditor: true,
                        source: cppEditor.getValue(),
                        compiler: currentCompilerId(slot),
                        options: slot.node.find('.compiler-options').val(),
                        filters: currentFilters()
                    };
                    setSetting('compiler' + slot.id, data.compiler);
                    setSetting('compilerOptions' + slot.id, data.options);
                    var stringifiedReq = JSON.stringify(data);
                    if (!force && stringifiedReq == slot.lastRequest) return;
                    slot.lastRequest = stringifiedReq;
                    data.timestamp = new Date();
                    $.ajax({
                        type: 'POST',
                        url: '/compile',
                        dataType: 'json',
                        contentType: 'application/json',
                        data: JSON.stringify(data),
                        success: function (result) {
                            onCompileResponse(slot, data, result);
                        },
                        error: function (xhr, e_status, error) {
                            console.log("AJAX request failed, reason : " + error);
                        },
                        cache: false
                    });
                    slot.currentAssembly = fakeAsm("[Processing...]");
                    updateAsm(slot);
                })(slots[i]);
            }
        }, 750); // Time in ms after which action is taken (if inactivity)
        setSetting('code', cppEditor.getValue());
        for (var i = 0; i < slots.length; i++) {
            (function (slot) {
                updateAsm(slot);
            })(slots[i]);
        }

        // execute the callback passed to Compiler()
        onEditorChangeCallback();
    }

    function setSource(code) {
        cppEditor.setValue(code);
    }

    function getSource() {
        return cppEditor.getValue();
    }

    function serialiseState(compress) {
        console.log("[WINDOW] Serialising state...");
        // Beware: we do not serialise the whole objects Slots / Diffs !
        // (they are to big to be passed by a URL)

        // Memorize informations on slots
        var slotIds = []; // necessary only to link with iiffs
        var leaderSlotId = leaderSlot.id;
        var compilersInSlots = [];
        var optionsInSlots = [];
        slots.forEach(function (slot) {
            slotIds.push(slot.id);
            compilersInSlots.push(currentCompilerId(slot));
            optionsInSlots.push(slot.node.find('.compiler-options').val());
        });

        // Memorize informations on diffs
        var diffIds = [];
        var slotsInDiffs = [];
        diffs.forEach(function (diff) {
            diffIds.push(diff.id);
            slotsInDiffs.push({
                before: diff.beforeSlot.id,
                after: diff.afterSlot.id
            });
        });

        var state = {
            slotCount: slots.length,
            slotIds: slotIds,
            leaderSlotId: leaderSlotId,
            compilersInSlots: compilersInSlots,
            optionsInSlots: optionsInSlots,

            diffCount: diffs.length,
            diffIds: diffIds,
            slotsInDiffs: slotsInDiffs
        };
        if (compress) {
            state.sourcez = LZString.compressToBase64(cppEditor.getValue());
        } else {
            state.source = cppEditor.getValue();
        }
        return state;
    }

    function deserialiseState(state, compilers, defaultCompiler) {
        console.log("[WINDOW] Deserialising state ...");
        if (state.hasOwnProperty('sourcez')) {
            cppEditor.setValue(LZString.decompressFromBase64(state.sourcez));
        } else {
            cppEditor.setValue(state.source);
        }

        if (slots.length != 0) {
            console.log("[WINDOW] Deserialisation : deleting existing slots ...");
            while (slots.length != 0) {
                deleteAndUnplaceSlot(slots[slots.length - 1]);
            }
        }

        if (diffs.length != 0) {
            console.log("[WINDOW] Deserialisation : deleting existing diffs ...");
            while (diffs.length != 0) {
                deleteAndUnplaceDiff(diffs[diffs.length - 1]);
            }
        }

        // Deserialise slots
        console.log("[WINDOW] Deserialisation : deserializing slots...");
        var i;
        for (i = 0; i < state.slotCount; i++) {
            var newSlot = createAndPlaceSlot(compilers,
                defaultCompiler,
                state.slotIds[i]);
            setCompilerById(state.compilersInSlots[i], newSlot);
            newSlot.node.find('.compiler-options').val(state.optionsInSlots[i]);
            // Somewhat hackily persist compiler into local storage else when the ajax response comes in
            // with the list of compilers it can splat over the deserialized version.
            // The whole serialize/hash/localStorage code is a mess! TODO(mg): fix

            setSetting('compiler' + newSlot.id, state.compilersInSlots[i]);
        }

        // Deserialise leaderSlot
        setLeaderSlotIcon(getSlotById(state.leaderSlotId));
        leaderSlot = getSlotById(state.leaderSlotId);
        setSetting('leaderSlot', leaderSlot.id);

        // Deserialise diffs
        console.log("[WINDOW] Deserialisation : deserializing diffs...");
        for (i = 0; i < state.diffCount; i++) {
            var newDiff = createAndPlaceDiff(state.diffIds[i]);
            setSlotInDiff(newDiff,
                "before",
                getSlotById(state.slotsInDiffs[i]["before"]));
            setSlotInDiff(newDiff,
                "after",
                getSlotById(state.slotsInDiffs[i]["after"]));
            onDiffChange(newDiff, false);
        }

        resizeEditors();
        return true;
    }

    // TODO : split in two functions : one is slot dependant, the other set parameters common to all slots
    function updateCompilerAndButtons(slot) {
        var compiler = currentCompiler(slot);  // TODO handle compiler being null (if e.g. a compiler is removed from the site)
        slot.node.find('.compilerVersion').text(compiler.name + " (" + compiler.version + ")");
        var filters = currentFilters();
        var supportsIntel = compiler.intelAsm || filters.binary;
        domRoot.find('#commonParams .filter button.btn[value="intel"]').toggleClass("disabled", !supportsIntel);
        domRoot.find('#commonParams .filter button.btn[value="binary"]').toggleClass("disabled", !compiler.supportsBinary).toggle(OPTIONS.supportsBinary);
        domRoot.find('#commonParams .filter .nonbinary').toggleClass("disabled", !!filters.binary);
    }

    function onCompilerChange(slot) {
        console.log("[DEBUG] onCompilerChange called with slot.id = " + slot.id);
        onParamChange(slot);
        updateCompilerAndButtons(slot);
        setAllDiffSlotsMenus();
    }

    function onDiffChange(diff, fromEditor) {
        console.log("[DEBUG] inside onDiffChange with diff id = " + diff.id +
            ", seen fromEditor = " + fromEditor);
        if (fromEditor == false) {
            diff.remainingTriggers = 2;
        } else {
            diff.remainingTriggers = diff.remainingTriggers - 1;
            if (diff.remainingTriggers == 0) {
                diff.remainingTriggers = 2;
            } else {
                return null;
            }
        }

        // If one slot is not mentioned, stop before making the ajax request
        if (diff.beforeSlot == null || diff.afterSlot == null) {
            return null;
        }
        var data = {
            // it should also be possible to use .currentAsembly
            before: diff.beforeSlot.asmCodeMirror.getValue(),
            after: diff.afterSlot.asmCodeMirror.getValue()
        };

        $.ajax({
            type: 'POST',
            url: '/diff',
            dataType: 'json',
            contentType: 'application/json',
            data: JSON.stringify(data),
            success: function (result) {
                //console.log("Success : "+JSON.stringify(result));
                onDiffResponse(diff, data, result);
            },
            error: function (xhr, e_status, error) {
                console.log("AJAX request for diff failed, reason : " + error);
            },
            cache: false
        });
    }

    function onDiffResponse(diff, data, result) {
        console.log("[CALLBACK] onDiffResponse() with diff = " + diff.id);
        // console.log("[DEBUG] result: "+result);
        diff.currentDiff = result.computedDiff;
        diff.zones = result.zones;
        updateDiff(diff);
    }

    function updateDiff(diff) {
        console.log("[CALLBACK] updateDiff() with diff = " + diff.id);
        // console.log("[DEBUG] currentDiff: " + JSON.stringify(diff.currentDiff));
        if (!diff.currentDiff) {
            return;
        }

        diff.asmCodeMirror.operation(function () {
            diff.asmCodeMirror.setValue(diff.currentDiff);
            clearBackground(diff.asmCodeMirror);
        });
        if (!diff.zones) return;
        var doc = diff.asmCodeMirror.getDoc();
        var computeLineChCoord = buildComputeLineChCoord(diff.currentDiff);
        // Same colors as in phabricator's diffs
        var cssStyles = ["background-color: rgba(151,234,151,.6);",
            "background-color: rgba(251,175,175,.7);"];
        var colorMarkedZones = [];
        for (var i = 0; i < diff.zones.length; i++) {
            for (var j = 0; j < diff.zones[i].length; j++) {
                colorMarkedZones.push(
                    doc.markText(computeLineChCoord(diff.zones[i][j].begin),
                        computeLineChCoord(diff.zones[i][j].end + 1),
                        {css: cssStyles[i]}));
            }
        }
    }

    // This function is required to place multiline marks in a CodeMirror
    // windows: markText accepts only coordinates in the form (line, column)
    // with line and column starting at 1.
    function buildComputeLineChCoord(text) {
        // assume text is 1 line containing '\n' to break lines
        // below calculations are placed outside the function to speed up
        var splitedStr = text.split("\n");
        var i;
        for (i = 0; i < splitedStr.length; i++) {
            splitedStr[i] = splitedStr[i] + "\n";
        }
        var lastPosInLine = [];
        var currentSum = splitedStr[0].length - 1;
        // console.log("Last pos in line "+0+": "+currentSum);
        lastPosInLine.push(currentSum);
        for (i = 1; i < splitedStr.length; i++) {
            currentSum = currentSum + splitedStr[i].length;
            lastPosInLine.push(currentSum);
            // console.log("Last pos in line "+i+": "+currentSum);
        }
        return function (pos) {
            var line = 0;
            while (lastPosInLine[line] < pos) {
                line = line + 1;
            }
            var ch = (line === 0) ? pos : pos - lastPosInLine[line - 1] - 1;
            return {line: line, ch: ch};
        }
    }

    function mapCompiler(compiler) {
        if (!compilersById[compiler]) {
            // Handle old settings and try the alias table.
            compiler = compilersByAlias[compiler];
            if (compiler) compiler = compiler.id;
        }
        return compiler;
    }

    // added has auxiliary to setCompilers, in order not to break interface
    // TODO : consider refactoring as some tasks are repeated
    function setCompilersInSlot(compilers, defaultCompiler, slot) {
        console.log("[INIT] inside setCompilersInSlot()");
        slot.node.find('.compilers li').remove();
        compilersById = {};
        compilersByAlias = {};
        // fills the compiler list
        $.each(compilers, function (index, arg) {
            compilersById[arg.id] = arg;
            if (arg.alias) compilersByAlias[arg.alias] = arg;
            var elem = $('<li><a href="#">' + arg.name + '</a></li>');
            slot.node.find('.compilers').append(elem);
            (function () {
                elem.click(function (e) {
                    setCompilerById(arg.id, slot);
                    onCompilerChange(slot);

                    // do not bring user to the top of index.html
                    e.preventDefault();
                });
            })(elem.find("a"), arg.id);
        });
        var compiler = getSetting('compiler' + slot.id);
        if (!compiler) {
            compiler = defaultCompiler;
            compiler = mapCompiler(compiler);
            if (!compiler)
                console.log("Could not map the default compiler id. " +
                    "Please double check your configuration file.");
        } else {
            compiler = mapCompiler(compiler);
            if (!compiler)
                console.log("Could not map the compiler found in settings. " +
                    "Please clear your browser cache.");
        }
        if (compiler) {
            setCompilerById(compiler, slot);
        }
        onCompilerChange(slot);
    }

    function setCompilers(compilers, defaultCompiler) {
        console.log("[INIT] setCompilers() was called with compilers = " +
            JSON.stringify(compilers) + ", defaultCompiler = " + defaultCompiler);
        for (var i = 0; i < slots.length; i++) {
            (function (slot) {
                setCompilersInSlot(compilers, defaultCompiler, slot);
            })(slots[i]);
        }
    }

    function currentFilters() {
        return patchUpFilters(filters_);
    }

    function setFilters(f) {
        filters_ = $.extend({}, f);
        slots.forEach(function (slot) {
            onParamChange(slot);
            updateCompilerAndButtons(slot);
        });
    }

    function setEditorHeight(height) {
        const MinHeight = 80;
        if (height < MinHeight) height = MinHeight;
        cppEditor.setSize(null, height);
        var i;
        for (i = 0; i < slots.length; i++) {
            slots[i].asmCodeMirror.setSize(null, height);
        }
        for (i = 0; i < diffs.length; i++) {
            diffs[i].asmCodeMirror.setSize(null, height);
        }
    }

    // External wrapper used by gcc.js only
    function refreshSlot(slot) {
        onCompilerChange(slot);
    }

    // Auxiliary function to slotUseDom
    function generateChangeCallback(slot) {
        return function callback() {
            onParamChange(slot);
        }
    }

    // must be called *before* doing slot = leaderSlot;
    function setLeaderSlotIcon(slot) {
        console.log("[UI] Toggling icon(s)...");

        // source for javascript unicode escape codes:
        // http://www.fileformat.info/info/unicode/char/search.htm
        // to convert a character to JS/HTML/CSS/URIs... escape code:
        // https://r12a.github.io/apps/conversion/

        // accessKey property : www.w3schools.com/jsref/prop_html_accesskey.asp

        var selectedIcon = "\u2605"; // Unicode character: full star
        var unselectedIcon = "\u2606"; // Unicode character: empty star
        // you must keep those characters in sync with the template 
        // slotTemplate in index.html

        // if there was a leaderSlot,
        if (leaderSlot != null) {
            // toggle icon in previous leaderSlot:
            var prevIcon = leaderSlot.node.find('.leaderSlotIcon');
            prevIcon.text(unselectedIcon);
            prevIcon.addClass("unselectedCharacterIcon");
            prevIcon.removeClass("selectedCharacterIcon");

            // removes access key in previous leaderSlot
            leaderSlot.node.find('.compiler-selection').prop("accessKey", "");
            leaderSlot.node.find('.compiler-options').prop("accessKey", "");
        }

        // toggles the new icon
        var newIcon = slot.node.find('.leaderSlotIcon');
        newIcon.text(selectedIcon);
        newIcon.addClass("selectedCharacterIcon");
        newIcon.removeClass("unselectedCharacterIcon");

        // enables accesskeys in the leaderSlot
        slot.node.find('.compiler-selection').prop("accessKey", "c");
        slot.node.find('.compiler-options').prop("accessKey", "o");
    }

    // Function to call each time a slot is added to the page.
    // This function requires that the slot's DOM object already exists.
    function slotUseDom(slot) {
        slot.asmCodeMirror = CodeMirror.fromTextArea(
            slot.node.find(".asm textarea")[0], {
                lineNumbers: true,
                mode: "text/x-asm",
                readOnly: true,
                gutters: ['CodeMirror-linenumbers'],
                lineWrapping: true
            });
        // handle compiler option (slot specific) such as '-O1'
        slot.node.find('.compiler-options').change(
            generateChangeCallback(slot)).keyup(
            generateChangeCallback(slot));

        slot.node.find('.leaderSlotIcon').on('click', function (e) {
            console.log("[UI] Clicked on leaderSlotIcon in slot " + slot.id);
            if (slot != leaderSlot) {
                clearBackground(leaderSlot.asmCodeMirror);

                setLeaderSlotIcon(slot);
                leaderSlot = slot;
                setSetting('leaderSlot', slot.id);
                onParamChange(slot, true);
            }

            // do not bring user to the top of index.html
            e.preventDefault();
        });

        if (slots.length == 1) {
            // "force" menu update if this is the first slot added
            var leaderSlotMenuNode = domRoot.find('#commonParams .leaderSlot');
            leaderSlotMenuNode.text('leader slot : ' + slots[0].id);
        }
    }

    function diffUseDom(diff) {
        diff.asmCodeMirror = CodeMirror.fromTextArea(
            diff.node.find('.diffText textarea')[0], {
                lineNumbers: true,
                mode: "text/x-asm",
                readOnly: true,
                gutters: ['CodeMirror-linenumbers'],
                lineWrapping: true
            });
        var reverseDiffButton = diff.node.find('.reverse-diff');
        reverseDiffButton.on('click', function (e) {
            console.log("[UI] User clicked on reverse-diff button in diff " + diff.id);
            var tmp = diff.beforeSlot;
            diff.beforeSlot = diff.afterSlot;
            diff.afterSlot = tmp;
            setSlotInDiff(diff, "before", diff.beforeSlot);
            setSlotInDiff(diff, "after", diff.afterSlot);
            setDiffSlotsMenus(diff);
            onDiffChange(diff, false);

            // do not bring user to the top of index.html
            e.preventDefault();
        });
    }

    function getSlotsIds() {
        var ids = [];
        for (var i = 0; i < slots.length; i++) {
            ids.push(slots[i].id);
        }
        return ids;
    }

    // TODO : refactor !
    function getDiffsIds() {
        var ids = [];
        for (var i = 0; i < diffs.length; i++) {
            ids.push(diffs[i].id);
        }
        return ids;
    }

    function slotCtor(optionalId) {
        var newSlot = new Slot();
        if (optionalId) {
            newSlot.id = optionalId;
        } else {
            newSlot.id = getAvailableId(slots);
        }
        slots.push(newSlot);
        setSetting('slotIds', JSON.stringify(getSlotsIds()));
        return newSlot;
    }

    function diffCtor(optionalId) {
        var newDiff = new Diff();
        if (optionalId) {
            newDiff.id = optionalId;
        } else {
            newDiff.id = getAvailableId(diffs);
        }
        diffs.push(newDiff);
        setSetting('diffIds', JSON.stringify(getDiffsIds()));
        return newDiff;
    }

    // Array Remove - By John Resig (MIT Licensed)
    Array.prototype.remove = function (from, to) {
        var rest = this.slice((to || from) + 1 || this.length);
        this.length = from < 0 ? this.length + from : from;
        return this.push.apply(this, rest);
    };

    function removeIdInArray(id, array) {
        for (var i = 0; i < array.length; i++) {
            if (array[i].id == id) {
                array.remove(i);
                break;
            }
        }
    }

    function anotherSlot(slot) {
        for (var i = 0; i < slots.length; i++) {
            if (slots[i] != slot) {
                return slots[i];
            }
        }
        return null; // should not happen (at least 1 slot is open)
    }

    function slotDtor(slot) {
        // if slot is the leader, find a new leader and change the icon
        if (slots.length > 1) {
            leaderSlot = anotherSlot(slot);
            setLeaderSlotIcon(leaderSlot);
            setSetting('leaderSlot', leaderSlot.id);
        }

        // hide the close-slot icon if there will remain 1 slot
        if (slots.length <= 2) {
            domRoot.find('.slot .closeButton').fadeOut(550);
        }

        // now safely delete:
        removeSetting('compiler' + slot.id);
        removeSetting('compilerOptions' + slot.id);
        removeIdInArray(slot.id, slots);
        // after the deletion, update the browser's settings :
        setSetting('slotIds', JSON.stringify(getSlotsIds()));
    }

    function diffDtor(diff) {
        removeIdInArray(diff.id, diffs);
        // after the deletion, update the browser's settings :
        setSetting('diffIds', JSON.stringify(getDiffsIds()));
    }

    function getSlotById(slotId) {
        for (var i = 0; i < slots.length; i++) {
            if (slots[i].id == slotId) return slots[i];
        }
        return null;
    }

    function getDiffById(diffId) {
        for (var i = 0; i < diffs.length; i++) {
            if (diffs[i].id == diffId) return diffs[i];
        }
        return null;
    }

    function setPanelListSortable() {
        // source : JQuery UI examples
        var panelList = $('#draggablePanelList');
        panelList.sortable({
            // Do not set the new-slot button sortable (nor a place to set windows)
            items: "li:not(.panel-sortable-disabled)", // source : JQuery examples
            // Omit this to make then entire <li>...</li> draggable.
            handle: '.panel-heading',
            update: function () {
                $('.panel', panelList).each(function (index, elem) {
                    var $listItem = $(elem),
                        newIndex = $listItem.index();

                    // TODO Persist the new indices.
                });
            }
        });
    }

    function slotDomCtor(slot, onUserAction) {
        // source : http://stackoverflow.com/questions/10126395/how-to-jquery-clone-and-change-id
        var slotTemplate = domRoot.find('#slotTemplate');
        var clone = slotTemplate.clone().prop('id', 'slot' + slot.id);

        // Insert the slot in the list of panels
        // domRoot.find('#slotTemplate').after(clone);
        domRoot.find('#draggablePanelList').append(clone);

        slot.node = domRoot.find('#slot' + slot.id);

        slot.node.find('.title').text("Slot " + slot.id + " (drag me)  ");

        if (onUserAction) {
            slot.node.fadeIn(550);
        } else {
            // auto spawn of the panel: no need to fade in
            slot.node.show();
        }

        slot.node.find('.closeButton').on('click', function (e) {
            console.log("[UI] User clicked on closeButton in slot " + slot.id);
            var slotToDelete = getSlotById(slot.id);
            deleteAndUnplaceSlot(slotToDelete, true);

            // do not bring user to the top of index.html
            e.preventDefault();
        });

        // show the close-slot icons if there will be more than 1 slot
        if (slots.length == 1) {
            domRoot.find('.slot .closeButton').fadeOut(550);
        }
        if (slots.length == 2) {
            domRoot.find('.slot .closeButton').fadeIn(550);
        }

        setPanelListSortable();
    }

    function diffDomCtor(diff, onUserAction) {
        var diffTemplate = domRoot.find('#diffTemplate');
        var clone = diffTemplate.clone().prop('id', 'diff' + diff.id);

        // Insert the diff in the list of panels
        // domRoot.find('#diffTemplate').after(clone);
        domRoot.find('#draggablePanelList').append(clone);

        diff.node = domRoot.find('#diff' + diff.id);

        diff.node.find('.title').text("Diff " + diff.id + " (drag me)  ");

        if (onUserAction) {
            diff.node.fadeIn(550);
        } else {
            // auto spawn of the panel: no need to fade in
            diff.node.show();
        }

        diff.node.find('.closeButton').on('click', function (e) {
            console.log("[UI] User clicked on closeButton in diff " + diff.id);
            var diffToDelete = getDiffById(diff.id);
            deleteAndUnplaceDiff(diffToDelete, true);
            // do not bring user to the top of index.html
            e.preventDefault();
        });

        setPanelListSortable();
        setDiffSlotsMenus(diff);
    }

    function slotDomDtor(slot, onUserAction) {
        if (onUserAction) {
            slot.node.fadeOut(550, function () {
                slot.node.remove();
            });
        } else {
            slot.node.remove();
        }
    }

    function diffDomDtor(diff, onUserAction) {
        if (onUserAction) {
            diff.node.fadeOut(550, function () {
                diff.node.remove();
            });
        } else {
            diff.node.remove();
        }
    }

    function createAndPlaceSlot(compilers, defaultCompiler, optionalId, onUserAction) {
        var newSlot = slotCtor(optionalId);
        slotDomCtor(newSlot, onUserAction);
        slotUseDom(newSlot);
        setCompilersInSlot(compilers, defaultCompiler, newSlot);
        return newSlot;
    }

    function createAndPlaceDiff(optionalId, onUserAction) {
        var newDiff = diffCtor(optionalId);
        diffDomCtor(newDiff, onUserAction);
        diffUseDom(newDiff);
        return newDiff;
    }

    function createAndPlaceDiffUI(optionalId, onUserAction) {
        var newDiff = createAndPlaceDiff(optionalId, onUserAction);
        if (slots.length > 0) {
            setSlotInDiff(newDiff, "before", slots[0]);
            if (slots.length > 1) {
                setSlotInDiff(newDiff, "after", slots[1]);
            } else {
                setSlotInDiff(newDiff, "after", slots[0]);
            }
            onDiffChange(newDiff, false);
        }
        return newDiff;
    }

    function deleteAndUnplaceSlot(slot, onUserAction) {
        slotDomDtor(slot, onUserAction);
        slotDtor(slot);
    }

    function deleteAndUnplaceDiff(diff, onUserAction) {
        diffDomDtor(diff, onUserAction);
        diffDtor(diff);
    }

    // refresh (or place) the two drop-down lists of the diff panel containging 
    // descriptions of the slots that can be used to make a diff

    function updateDiffButton(diff, className) {
        var slot = diff[className + 'Slot'];
        var diffSlotMenuNode = diff.node.find('.' + className + ' .slotName');
        diffSlotMenuNode.text(slot.shortDescription());
    }

    // this function is called if the name or option of a compiler is changed.
    function updateDiffUI(diff) {
        updateDiffButton(diff, "before");
        updateDiffButton(diff, "after");
        setDiffSlotsMenus(diff);
    }

    // Auxilary to setDiffSlotsMenus and deserialisation
    function setSlotInDiff(diff, className, slot) {
        // className can be "before" or "after"
        if (slot == null) {
            console.log("[DEBUG] setSlotInDiff: diff.id = " +
                diff.id + ", " + className + " -> " + "null.id");
        } else {
            console.log("[DEBUG] setSlotInDiff: diff.id = " +
                diff.id + ", " + className + " -> " + slot.id);
        }
        diff[className + 'Slot'] = slot;
        if (slot != null) {
            updateDiffButton(diff, className);

            setSetting('diff' + diff.id + className, slot.id);

            addToPendings(slot, diff);
        }
    }

    function setDiffSlotsMenus(diff) {
        // className can be "before" or "after"
        function setSlotMenu(className) {
            console.log("[DEBUG] : setSlotMenu with " + className +
                " and diff.id = " + diff.id);
            diff.node.find('.' + className + ' li').remove();
            for (var i = 0; i < slots.length; i++) {
                var elem = $('<li><a href="#">' + slots[i].description() + '</a></li>');
                diff.node.find('.' + className + ' .slotNameList').append(elem);
                (function (i) {
                    elem.click(function (e) {
                        // TODO: check if modifying diff with [] will survive to
                        // minifying http://stackoverflow.com/questions/4244896/
                        console.log("[UI] user set " + slots[i].id + " as " + className +
                            " slot in diff with id " + diff.id);
                        setSlotInDiff(diff, className, slots[i]);
                        onDiffChange(diff, false);

                        // do not bring user to the top of index.html
                        e.preventDefault();
                    });
                })(i);
            }
        }

        setSlotMenu("before");
        setSlotMenu("after");
    }

    // refresh (or place) all of the drop-down lists containging descriptions of
    // the slots that can be used to make a diff
    function setAllDiffSlotsMenus() {
        console.log("[DEBUG] : inside setAllDiffSlotsMenus, diffs.length = " + diffs.length);
        for (var i = 0; i < diffs.length; i++) {
            setDiffSlotsMenus(diffs[i]);
        }
    }

    // on startup, for each slot,
    // if a setting is defined, set it on static/index.html page
    var slotIds = getSetting('slotIds');
    if (slotIds != null) {
        slotIds = JSON.parse(slotIds);
    } else {
        slotIds = [];
    }
    var i, newSlot;
    if (slotIds.length > 0) {
        console.log("[STARTUP] found slot data : restoring from previous session");
        console.log("[DEBUG] slotIds: " + JSON.stringify(slotIds));
        for (i = 0; i < slotIds.length; i++) {
            newSlot = slotCtor(slotIds[i]);
            slotDomCtor(newSlot);
            slotUseDom(newSlot);
            if (getSetting('compilerOptions' + slotIds[i]) === undefined) {
                console.log("[STARTUP] There was a problem while restoring slots from previous session.");
            } else {
                newSlot.node.find('.compiler-options').val(getSetting('compilerOptions' + newSlot.id));
            }
        }
        var leaderSlotSetting = getSetting('leaderSlot');
        if (!leaderSlotSetting) {
            console.log("Leader slot was missing; picking first");
            leaderSlot = getSlotById(0);
        } else {
            leaderSlot = getSlotById(leaderSlotSetting);
        }
        if (leaderSlot) setLeaderSlotIcon(leaderSlot);
    } else {
        // not slot data found. It is probably the first time the user come to
        // visit (or to debug a wipeSetting(); was done in the browser console)
        // therefore it seems logical to open at least 1 slot
        leaderSlot = createAndPlaceSlot(compilers, defaultCompiler);
        setSetting('leaderSlot', leaderSlot.id);
        setLeaderSlotIcon(leaderSlot);
    }

    // This part of the initialisation must (currently) occur 
    // after the creation of slots, and before the creation of diffs.
    // Previously it was located in gcc.js, in Initialize()
    setCompilers(compilers, defaultCompiler);

    // on startup, for each diff,
    // if a setting is defined, set it on static/index.html page
    var diffIds = getSetting('diffIds');
    if (diffIds != null) {
        diffIds = JSON.parse(diffIds);
    } else {
        diffIds = [];
    }
    console.log("[DEBUG] diffIds: " + JSON.stringify(diffIds));
    if (diffIds.length > 0) {
        console.log("[STARTUP] found diff data : restoring diffs from previous session");
        for (i = 0; i < diffIds.length; i++) {
            var newDiff = createAndPlaceDiff(diffIds[i]);

            newDiff.beforeSlot = getSlotById(getSetting('diff' + newDiff.id + "before"));
            setSlotInDiff(newDiff, "before", newDiff.beforeSlot);
            addToPendings(newDiff.beforeSlot, newDiff);

            newDiff.afterSlot = getSlotById(getSetting('diff' + newDiff.id + "after"));
            setSlotInDiff(newDiff, "after", newDiff.afterSlot);
            addToPendings(newDiff.afterSlot, newDiff);
            //onDiffChange(newDiff);
        }
    }

    return {
        serialiseState: serialiseState,
        deserialiseState: deserialiseState,
        getSource: getSource,
        setSource: setSource,
        setFilters: setFilters,
        setEditorHeight: setEditorHeight,
        createAndPlaceSlot: createAndPlaceSlot,
        createAndPlaceDiffUI: createAndPlaceDiffUI,
        refreshSlot: refreshSlot
    };
}
