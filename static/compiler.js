// Copyright (c) 2012, Matt Godbolt
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
    $.each(lines.split('\n'), function(_, line) {
        line = line.trim();
        if (line != "") {
            var match = line.match(re);
            if (match) {
                callback(parseInt(match[1]), match[4]);
            } else {
                callback(null, line);
            }
        }
    });
}

function clearBackground(cm) {
    for (var i = 0; i < cm.lineCount(); ++i) {
        cm.removeLineClass(i, "background", null);
    }
}

const NumRainbowColours = 12;

function Compiler(domRoot, origFilters, windowLocalPrefix, onChangeCallback) {
    var compilersByExe = {};
    var pendingTimeout = null;
    var asmCodeMirror = null;
    var cppEditor = null;
    var lastRequest = null;
    var currentAssembly = null;
    var filters = origFilters;
    var ignoreChanges = true; // Horrible hack to avoid onChange doing anything on first starting, ie before we've set anything up.

    cppEditor = CodeMirror.fromTextArea(domRoot.find(".editor textarea")[0], {
        lineNumbers: true,
        gutters: ["CodeMirror-linenumbers", "info-margin"],
        matchBrackets: true,
        useCPP: true,
        mode: "text/x-c++src"
    });
    cppEditor.on("change", onChange);
    asmCodeMirror = CodeMirror.fromTextArea(domRoot.find(".asm textarea")[0], {
        lineNumbers: true,
                  matchBrackets: true,
                  mode: "text/x-asm",
                  readOnly: true
    });

    function getSetting(name) {
        return window.localStorage[windowLocalPrefix + "." + name];
    }
    function setSetting(name, value) {
        window.localStorage[windowLocalPrefix + "." + name] = value;
    }

    if (getSetting('code')) cppEditor.setValue(getSetting('code'));
    domRoot.find('.compiler').change(onCompilerChange);
    domRoot.find('.compiler_options').change(onChange).keyup(onChange);
    ignoreChanges = false;

    if (getSetting('compilerOptions')) {
        domRoot.find('.compiler_options').val(getSetting('compilerOptions'));
    }

    function makeErrNode(tooltip) {
        return $('<div></div>').text(">").addClass("error").attr("title", tooltip)[0];
    }

    function onCompileResponse(data) {
        var stdout = data.stdout || "";
        var stderr = data.stderr || "";
        if (data.code == 0) {
            stdout += "\nCompiled ok";
        } else {
            stderr += "\nCompilation failed";
        }
        $('.result .output :visible').remove();
        var highlightLine = (data.asm == null);
        for (var i = 0; i < cppEditor.lineCount(); ++i) cppEditor.setGutterMarker(i, "info-margin", null);
        parseLines(stderr + stdout, function(lineNum, msg) {
            var elem = $('.result .output .template').clone().appendTo('.result .output').removeClass('template');
            if (lineNum) {
                cppEditor.setGutterMarker(lineNum - 1, "info-margin", makeErrNode(msg));
                elem.html($('<a href="#">').append(lineNum + " : " + msg)).click(function() {
                    cppEditor.setSelection({line: lineNum - 1, ch: 0}, {line: lineNum, ch: 0});
                    return false;
                });
            } else {
                elem.text(msg);
            }
        });
        currentAssembly = data.asm || "[no output]";
        updateAsm();
    }

    function numberUsedLines(asm) {
        var sourceLines = {};
        $.each(asm, function(_, x) { if (x.source) sourceLines[x.source - 1] = true; });
        var ordinal = 0;
        $.each(sourceLines, function(k, _) { sourceLines[k] = ordinal++; });
        var asmLines = {};
        $.each(asm, function(index, x) { if (x.source) asmLines[index] = sourceLines[x.source - 1]; });
        return { source: sourceLines, asm: asmLines };
    }

    var lastUpdatedAsm = null;
    function updateAsm(forceUpdate) {
        if (!currentAssembly) return;
        var hashedUpdate = JSON.stringify({
            asm: currentAssembly, 
            filters: filters
        });
        if (!forceUpdate && lastUpdatedAsm == hashedUpdate) { return; }
        lastUpdatedAsm = hashedUpdate;

        var asm = processAsm(currentAssembly, filters);
        var asmText = $.map(asm, function(x){ return x.text; }).join("\n");
        var numberedLines = numberUsedLines(asm);
        asmCodeMirror.setValue(asmText);
        
        clearBackground(cppEditor);
        clearBackground(asmCodeMirror);
        if (filters.colouriseAsm) {
            $.each(numberedLines.source, function(line, ordinal) {
                cppEditor.addLineClass(parseInt(line), "background", "rainbow-" + (ordinal % NumRainbowColours));
            });
            $.each(numberedLines.asm, function(line, ordinal) {
                asmCodeMirror.addLineClass(parseInt(line), "background", "rainbow-" + (ordinal % NumRainbowColours));
            });
        }
    }

    function pickOnlyRequestFilters(filters) {
        return {intel: !!filters.intel };
    }

    function onChange() {
        if (ignoreChanges) return;  // Ugly hack during startup.
        if (pendingTimeout) clearTimeout(pendingTimeout);
        pendingTimeout = setTimeout(function() {
            var data = { 
                source: cppEditor.getValue(),
                compiler: $('.compiler').val(),
                options: $('.compiler_options').val(),
                filters: pickOnlyRequestFilters(filters)
            };
            setSetting('compiler', data.compiler);
            setSetting('compilerOptions', data.options);
            if (JSON.stringify(data) == JSON.stringify(lastRequest)) return;
            lastRequest = data;
            $.ajax({
                type: 'POST',
                url: '/compile',
                dataType: 'json',
                data: data,
                success: onCompileResponse});
        }, 750);
        setSetting('code', cppEditor.getValue());
        updateAsm();
        onChangeCallback();
    }

    function setSource(code) {
        cppEditor.setValue(code);
    }

    function getSource() {
        return cppEditor.getValue();
    }

    function serialiseState() {
        var state = {
            source: cppEditor.getValue(),
            compiler: domRoot.find('.compiler').val(),
            options: domRoot.find('.compiler_options').val(),
        };
        return state;
    }

    function deserialiseState(state) {
        cppEditor.setValue(state.source);
        domRoot.find('.compiler').val(state.compiler);
        domRoot.find('.compiler_options').val(state.options);
        // Somewhat hackily persist compiler into local storage else when the ajax response comes in
        // with the list of compilers it can splat over the deserialized version.
        // The whole serialize/hash/localStorage code is a mess! TODO(mg): fix
        setSetting('compiler', state.compiler);
        updateAsm(true);  // Force the update to reset colours after calling cppEditor.setValue
        return true;
    }

    function onCompilerChange() {
        onChange();
        var compiler = compilersByExe[$('.compiler').val()];
        domRoot.find('.filter button.btn[value="intel"]').toggleClass("disabled", !compiler.supportedOpts["-masm"]);
    }
    
    function setCompilers(compilers) {
        domRoot.find('.compiler option').remove();
        compilersByExe = {};
        $.each(compilers, function(index, arg) {
            compilersByExe[arg.exe] = arg;
            domRoot.find('.compiler').append($('<option value="' + arg.exe + '">' + arg.version + '</option>'));
        });
        if (getSetting('compiler')) {
            domRoot.find('.compiler').val(getSetting('compiler'));
        }
        onCompilerChange();
    }

    function setFilters(f) {
        filters = f;
        onChange();  // used to just update ASM, but things like "Intel syntax" need a new request
    }

    return {
        serialiseState: serialiseState,
        deserialiseState: deserialiseState,
        setCompilers: setCompilers,
        getSource: getSource,
        setSource: setSource,
        setFilters: setFilters
    };
}
