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

const NumRainbowColours = 12;

// This function is called in function initialise in static/gcc.js
function Compiler(domRoot, origFilters, windowLocalPrefix, onEditorChangeCallback, lang) {
    console.log("[TRACE] Entering function Compiler()");
    // TODO : allow user to dynamically change the number of slots
    var slotsCount = 2;
    setSetting('leaderSlot', 0);
    var compilersById = {};
    var compilersByAlias = {};

    // timeout : 
    // correponds to the time in ms after which resend code to compiler 
    var pendingTimeoutInSlots = [];
    for (var i = 0; i<slotsCount; i++) {
        pendingTimeoutInSlots.push(null);
    }

    var pendingTimeoutInEditor = null;

    var asmCodeMirrors = [];
    for (var i = 0; i<slotsCount; i++) {
        asmCodeMirrors.push(null);
    }

    var cppEditor = null;

    // lastRequest[i] contains the previous request in slot i
    // it can be used to prevent useless re-compilation
    // if no parameters were changed
    var lastRequest = [];
    for (var i = 0; i<slotsCount; i++) {
        lastRequest.push(null);
    }

    // currentAssembly[i] contains the assembly text
    // located in slot i
    var currentAssembly = [];
    for (var i = 0; i<slotsCount; i++) {
        currentAssembly.push(null);
    }

    // default filters
    var filters_ = $.extend({}, origFilters);
    var ignoreChanges = true; // Horrible hack to avoid onEditorChange doing anything on first starting, ie before we've set anything up.

    function setCompilerById(id,slot) {
        var compilerNode = domRoot.find('#params'+slot+' .compiler');
        compilerNode.text(compilersById[id].name);
        compilerNode.attr('data', id);
    }

    function currentCompilerId(slot) {
        return domRoot.find('#params'+slot+' .compiler').attr('data');
    }

    function currentCompiler(slot) {
        return compilersById[currentCompilerId(slot)];
    }

    // set the autocompile button on static/index.html
    $('.autocompile').click(function () {
        $('.autocompile').toggleClass('active');
        onEditorChange();
        setSetting('autocompile', $('.autocompile').hasClass('active'));
    });
    $('.autocompile').toggleClass('active', getSetting("autocompile") !== "false");

    // handle filter options that are specific to a compiler
    function patchUpFilters(filters) {
        filters = $.extend({}, filters);
        var compiler = currentCompiler();
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
    cppEditor = CodeMirror.fromTextArea(domRoot.find(".editor textarea")[0], {
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

    // Set up all slot's windows
    for (var i = 0; i < slotsCount; i++) {
        // TODO : explain why [0] is mandatory here
        asmCodeMirrors[i] = CodeMirror.fromTextArea(domRoot.find("#asm"+i+" textarea")[0], {
            lineNumbers: true,
            mode: "text/x-asm",
            readOnly: true,
            gutters: ['CodeMirror-linenumbers'],
            lineWrapping: true
        });
    }

    // Load/Save setting from the browser
    function getSetting(name) {
        return window.localStorage[windowLocalPrefix + "." + name];
    }
    function setSetting(name, value) {
        window.localStorage[windowLocalPrefix + "." + name] = value;
    }

    var codeText = getSetting('code');
    if (!codeText) codeText = $(".template.lang." + lang.replace(/[^a-zA-Z]/g, '').toLowerCase()).text();
    if (codeText) cppEditor.setValue(codeText);

    // handle compiler option (slot specific) such as '-O1'
    for (var slot = 0; slot < slotsCount; slot++) {
        (function (slot) {
            domRoot.find('#params'+slot+' .compiler_options').change(generate_change_callback(slot)).keyup(generate_change_callback(slot));
        })(slot);
    }
    // auxiliary function to the previous for loop
    function generate_change_callback(slot) {
        return function callback() {
            onParamChange(slot);
        }
    }

    ignoreChanges = false;


    // on startup, for each slot,
    // if a setting is defined, set it on static/index.html page
    for (var slot = 0; slot < slotsCount; slot++) {
        if (getSetting('compilerOptions'+slot)) {
            domRoot.find('#params'+slot+' .compiler_options').val(getSetting('compilerOptions'+slot));
        }
    }

    // function similar to setCompilersInSlot
    // TODO : should be returned by Compiler() !
    // (not executed in setCompilers)
    function setLeaderSlotMenu() {
        domRoot.find('#commonParams .slots li').remove();
        // fills the leader-slot list
        for (var n = 0; n < slotsCount; n++) {
            var elem = $('<li><a href="#">' + n + '</a></li>');
            domRoot.find('#commonParams .slots').append(elem);
            (function (n) {
                elem.click(function () {
                    var leaderSlotMenuNode = domRoot.find('#commonParams .leaderSlot');
                    leaderSlotMenuNode.text('leader slot : '+n);
                    setSetting('leaderSlot', n);
                    onEditorChange(true);
                });
            })(n);
        }
    }

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
    function onCompileResponse(request, data) {
        var leaderSlot = getSetting('leaderSlot');
        console.log("[CALLBACK] onCompileResponse() with slot = "+request.slot+", leaderSlot = ",+leaderSlot);
        var stdout = data.stdout || "";
        var stderr = data.stderr || "";
        if (data.code === 0) {
            stdout += "Compiled ok in slot " + request.slot + "\n";
        } else {
            stderr += "Compilation failed in slot " + request.slot + "\n";
        }
        if (_gaq) {
            // TODO : to modify to handle the slots ?
            _gaq.push(['_trackEvent', 'Compile', request.compiler, request.options, data.code]);
            _gaq.push(['_trackTiming', 'Compile', 'Timing', new Date() - request.timestamp]);
        }
        domRoot.find('#output'+request.slot+' .result .output :visible').remove();
        // only show in Editor messages comming from the leaderSlot
        if (request.slot == leaderSlot) {
            for (var i = 0; i < errorWidgets.length; ++i)
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
            var elem = domRoot.find('#output'+request.slot+' .result .output .template').clone().appendTo(domRoot.find('#output'+request.slot+' .result .output')).removeClass('template');
            if (lineNum) {
                // only show in Editor messages comming from the leaderSlot
                if  (request.slot == leaderSlot) {
                    errorWidgets.push(cppEditor.addLineWidget(lineNum - 1, makeErrNode(msg), {
                        coverGutter: false, noHScroll: true
                    }));
                }
                elem.html($('<a href="#">').text(lineNum + " : " + msg)).click(function () {
                    cppEditor.setSelection({line: lineNum - 1, ch: 0}, {line: lineNum, ch: 0});
                    return false;
                });
            } else {
                elem.text(msg);
            }
        });
        currentAssembly[request.slot] = data.asm || fakeAsm("[no output]");
        updateAsm(request.slot);
    }

    function numberUsedLines(asm) {
        var sourceLines = {};
        $.each(asm, function (_, x) {
            if (x.source) sourceLines[x.source - 1] = true;
        });
        var ordinal = 0;
        $.each(sourceLines, function (k, _) {
            sourceLines[k] = ordinal++;
        });
        var asmLines = {};
        $.each(asm, function (index, x) {
            if (x.source) asmLines[index] = sourceLines[x.source - 1];
        });
        return {source: sourceLines, asm: asmLines};
    }

    var lastUpdatedAsm = [];
    for (var i = 0; i < slotsCount; i++) {
        lastUpdatedAsm.push(null);
    }

    function updateAsm(slot,forceUpdate) {
        console.log("[CALLBACK] updateAsm() with slot = " + slot + ", forceUpdate = " + forceUpdate);
        if (!currentAssembly[slot]) return;
        var hashedUpdate = JSON.stringify(currentAssembly[slot]);
        // TODO : real hash here ?
        if (!forceUpdate && lastUpdatedAsm[slot] == hashedUpdate) {
            return;
        }
        lastUpdatedAsm[slot] = hashedUpdate;

        var asmText = $.map(currentAssembly[slot], function (x) {
            return x.text;
        }).join("\n");
        var numberedLines = numberUsedLines(currentAssembly[slot]);

        cppEditor.operation(function () {
            clearBackground(cppEditor);
        });
        var filters = currentFilters();
        asmCodeMirrors[slot].operation(function () {
            asmCodeMirrors[slot].setValue(asmText);
            clearBackground(asmCodeMirrors[slot]);
            var addrToAddrDiv = {};
            $.each(currentAssembly[slot], function (line, obj) {
                var address = obj.address ? obj.address.toString(16) : "";
                var div = $("<div class='address cm-number'>" + address + "</div>");
                addrToAddrDiv[address] = {div: div, line: line};
                asmCodeMirrors[slot].setGutterMarker(line, 'address', div[0]);
            });
            $.each(currentAssembly[slot], function (line, obj) {
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
                asmCodeMirrors[slot].setGutterMarker(line, 'opcodes', opcodes[0]);
                if (obj.links) {
                    $.each(obj.links, function (_, link) {
                        var from = {line: line, ch: link.offset};
                        var to = {line: line, ch: link.offset + link.length};
                        var address = link.to.toString(16);
                        var thing = $("<a href='#' class='cm-number'>" + address + "</a>");
                        asmCodeMirrors[slot].markText(
                            from, to, {replacedWith: thing[0], handleMouseEvents: false});
                        var dest = addrToAddrDiv[address];
                        if (dest) {
                            thing.on('hover', function (e) {
                                var entered = e.type == "mouseenter";
                                dest.div.toggleClass("highlighted", entered);
                                thing.toggleClass("highlighted", entered);
                            });
                            thing.on('click', function (e) {
                                asmCodeMirrors[slot].scrollIntoView({line: dest.line, ch: 0}, 30);
                                dest.div.toggleClass("highlighted", false);
                                thing.toggleClass("highlighted", false);
                            });
                        }
                    });
                }
            });
            if (filters.binary) {
                asmCodeMirrors[slot].setOption('lineNumbers', false);
                asmCodeMirrors[slot].setOption('gutters', ['address', 'opcodes']);
            } else {
                asmCodeMirrors[slot].setOption('lineNumbers', true);
                asmCodeMirrors[slot].setOption('gutters', ['CodeMirror-linenumbers']);
            }
        });

        if (filters.colouriseAsm) {
            // colorise the editor
            cppEditor.operation(function () {
                $.each(numberedLines.source, function (line, ordinal) {
                    cppEditor.addLineClass(parseInt(line),
                        "background", "rainbow-" + (ordinal % NumRainbowColours));
                });
            });
            // colorise the assembly in slot
                asmCodeMirrors[slot].operation(function () {
                    $.each(numberedLines.asm, function (line, ordinal) {
                        asmCodeMirrors[slot].addLineClass(parseInt(line),
                            "background", "rainbow-" + (ordinal % NumRainbowColours));
                    });
                });
        }
    }

    function fakeAsm(text) {
        return [{text: text, source: null}];
    }

    // slot's parameters callback
    // TODO : refactor with onEditorChange : those functions could call an updateSlot(slot)
    function onParamChange(slot) { 
        console.log("[CALLBACK] onParamChange() on slot "+slot);
        if (ignoreChanges) return;  // Ugly hack during startup.
        if (pendingTimeoutInSlots[slot]) {
            console.log("[TIME] Clearing time out in slot " + slot);
            clearTimeout(pendingTimeoutInSlots[slot]);
        }

        console.log("[TIME] Setting time out in slot " + slot);
        pendingTimeoutInSlots[slot] = setTimeout(function () {
            console.log("[TIME] Timed out : compiling in slot " + slot + " triggered by modification of params...");
            (function(slot) {
                var data = {
                    slot: slot, // TO DECIDE : probably better not put it here
                    source: cppEditor.getValue(),
                    compiler: currentCompilerId(slot),
                    options: $('#params'+slot+' .compiler_options').val(),
                    filters: currentFilters()
                };
                setSetting('compiler'+slot, data.compiler);
                setSetting('compilerOptions'+slot, data.options);
                var stringifiedReq = JSON.stringify(data);
                if (stringifiedReq == lastRequest[slot]) return;
                lastRequest[slot] = stringifiedReq;
                data.timestamp = new Date();
                $.ajax({
                    type: 'POST',
                    url: '/compile',
                    dataType: 'json',
                    contentType: 'application/json',
                    data: JSON.stringify(data),
                    success: function (result) {
                        onCompileResponse(data, result);
                    },
                    error: function (xhr, e_status, error) {
                        console.log("AJAX request failed, reason : " + error);
                    },
                    cache: false
                });
                setSetting('code', cppEditor.getValue());
                currentAssembly[slot] = fakeAsm("[Processing...]");
                updateAsm(slot);
            }) (slot);
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
            console.log("[TIME] Timed out in editor : compiling in all "+slotsCount+ " slots...");
            for (var i = 0; i < slotsCount; i++) {
                //console.log("Compiling for slot " + i + "...");
                (function(slot) {
                    var data = {
                        slot: slot, // TO DECIDE : probably better not put it here
                        source: cppEditor.getValue(),
                        compiler: currentCompilerId(slot),
                        options: $('#params'+slot+' .compiler_options').val(),
                        filters: currentFilters()
                    };
                    setSetting('compiler'+slot, data.compiler);
                    setSetting('compilerOptions'+slot, data.options);
                    var stringifiedReq = JSON.stringify(data);
                    if (!force && stringifiedReq == lastRequest[slot]) return;
                    lastRequest[slot] = stringifiedReq;
                    data.timestamp = new Date();
                    $.ajax({
                        type: 'POST',
                        url: '/compile',
                        dataType: 'json',
                        contentType: 'application/json',
                        data: JSON.stringify(data),
                        success: function (result) {
                            onCompileResponse(data, result);
                        },
                        error: function (xhr, e_status, error) {
                            console.log("AJAX request failed, reason : " + error);
                        },
                        cache: false
                    });
                    currentAssembly[slot] = fakeAsm("[Processing...]");
                    updateAsm(slot);
                }) (i);
            }
        }, 750); // Time in ms after which action is taken (if inactivity)
        setSetting('code', cppEditor.getValue());
        for (var i = 0; i < slotsCount; i++) {
            (function(slot) {
                updateAsm(slot);
            }) (i);
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

        var compilersInSlots = [];
        for (var slot = 0; slot < slotCount; slot++) {
            compilersInSlots.push(currentCompilerId(slot));
        }
        var optionsInSlots = [];
        for (var slot = 0; slot < slotCount; slot++) {
            optionsInSlots.push(domRoot.find('#params'+slot+' .compiler_options').val());
        }
        // TODO : add slotCount in state
        var state = {
            // compiler: currentCompilerId(), became :
            compilersInSlots: compilersInSlots,
            optionsInSlots: optionsInSlots
        };
        if (compress) {
            state.sourcez = LZString.compressToBase64(cppEditor.getValue());
        } else {
            state.source = cppEditor.getValue();
        }
        return state;
    }

    function deserialiseState(state) {
        if (state.hasOwnProperty('sourcez')) {
            cppEditor.setValue(LZString.decompressFromBase64(state.sourcez));
        } else {
            cppEditor.setValue(state.source);
        }

        // Deserealise Compilers id
        state.compilersInSlots = mapCompiler(state.compilersInSlots);
        for (var slot = 0; slot < slotsCount; slot ++) {
            (function(slot) {
                setCompilerById(state.compilersInSlots[slot],slot);
            }) (slot);
        }

        // Deserealise Compilers options
        for (var slot = 0; slot < slotsCount; slot ++) {
            domRoot.find('#params'+slot+' .compiler_options').val(state.optionsInSlots[slot]);
        }
        // Somewhat hackily persist compiler into local storage else when the ajax response comes in
        // with the list of compilers it can splat over the deserialized version.
        // The whole serialize/hash/localStorage code is a mess! TODO(mg): fix
        
        // compiler -> compilerX
        for (var slot = 0; slot < slotsCount; slot ++) {
            setSetting('compiler'+slot, state.compilersInSlots[slot]);
        }

        for (var slot = 0; slot < slotsCount; slot++) {
            (function(i) {
                updateAsm(i,true);  // Force the update to reset colours after calling cppEditor.setValue
            })(slot);
        }
        return true;
    }

    // TODO : split in two functions : one is slot dependant, the other set parameters common to all slots
    function updateCompilerAndButtons(slot) {
        var compiler = currentCompiler(slot);
        domRoot.find('#output'+slot+' .compilerVersion').text(compiler.name + " (" + compiler.version + ")");
        var filters = currentFilters();
        var supportsIntel = compiler.intelAsm || filters.binary;
        domRoot.find('#commonParams .filter button.btn[value="intel"]').toggleClass("disabled", !supportsIntel);
        domRoot.find('#commonParams .filter button.btn[value="binary"]').toggleClass("disabled", !compiler.supportsBinary).toggle(OPTIONS.supportsBinary);
        domRoot.find('#commonParams .filter .nonbinary').toggleClass("disabled", !!filters.binary);
    }

    function onCompilerChange(slot) {
        onParamChange(slot);
        updateCompilerAndButtons(slot);
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
        console.log("[INIT] in setCompilersInSlot(), compilers = "+JSON.stringify(compilers)+", slot = "+slot);
        domRoot.find('#params'+slot+' .compilers li').remove();
        compilersById = {};
        compilersByAlias = {};
        // fills the compiler list
        $.each(compilers, function (index, arg) {
            compilersById[arg.id] = arg;
            if (arg.alias) compilersByAlias[arg.alias] = arg;
            var elem = $('<li><a href="#">' + arg.name + '</a></li>');
            domRoot.find('#params'+slot+' .compilers').append(elem);
            (function () {
                elem.click(function () {
                    setCompilerById(arg.id,slot);
                    onCompilerChange(slot);
                });
            })(elem.find("a"), arg.id);
        });
        var compiler = getSetting('compiler'+slot);
        if (!compiler) {
            compiler = defaultCompiler;
            compiler = mapCompiler(compiler);
            if (!compiler)
                console.log("Could not map the default compiler id. Please double check your configuration file.");
        } else {
            compiler = mapCompiler(compiler);
            if (!compiler)
                console.log("Could not map the compiler found in settings. Please clear your browser cache.");
        }
        if (compiler) {
            setCompilerById(compiler,slot);
        }
        onCompilerChange(slot);
    }

    function setCompilers(compilers, defaultCompiler) {
        console.log("[INIT] setCompilers() was called with compilers = "+JSON.stringify(compilers)+", defaultCompiler = "+defaultCompiler);
        setLeaderSlotMenu();
        for (var slot = 0; slot < slotsCount; slot++) {
            (function(slot){
                setCompilersInSlot(compilers,defaultCompiler,slot);
            })(slot);
        }
    }

    function currentFilters() {
        return patchUpFilters(filters_);
    }

    function setFilters(f) {
        filters_ = $.extend({}, f);
        for (var slot = 0; slot < slotsCount; slot++) {
            onParamChange(slot);
            updateCompilerAndButtons(slot);
        }
    }

    function setEditorHeight(height) {
        const MinHeight = 80;
        if (height < MinHeight) height = MinHeight;
        cppEditor.setSize(null, height);

        for (var i = 0; i < slotsCount; i++) {
            asmCodeMirrors[i].setSize(null, height);
        }
    }

    return {
        serialiseState: serialiseState,
        deserialiseState: deserialiseState,
        setCompilers: setCompilers,
        getSource: getSource,
        setSource: setSource,
        setFilters: setFilters,
        setEditorHeight: setEditorHeight
    };
}
