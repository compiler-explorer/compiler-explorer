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

var pendingTimeout = null;
var asmCodeMirror = null;
var cppEditor = null;
var lastRequest = null;
var currentAssembly = null;
var ignoreChanges = false;

function parseLines(lines, callback) {
    var re = /^\<stdin\>:([0-9]+):([0-9]+):\s+(.*)/;
    $.each(lines.split('\n'), function(_, line) {
        line = line.trim();
        if (line != "") {
            var match = line.match(re);
            if (match) {
                callback(parseInt(match[1]), match[3]);
            } else {
                callback(null, line);
            }
        }
    });
}

var errorLines = [];
function onCompileResponse(data) {
    var stdout = data.stdout || "";
    var stderr = data.stderr || "";
    if (data.code == 0) {
        stdout += "\nCompiled ok";
    } else {
        stderr += "\nCompilation failed";
    }
    $.each(errorLines, function(_, line) { 
        if (line) cppEditor.setLineClass(line, null, null);
    });
    errorLines = [];
    $('.result .output :visible').remove();
    parseLines(stderr + stdout, function(lineNum, msg) {
        var elem = $('.result .output .template').clone().appendTo('.result .output').removeClass('template');
        if (lineNum) {
            errorLines.push(cppEditor.setLineClass(lineNum - 1, null, "error"));
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

function updateAsm() {
    if (!currentAssembly) return;
    asmCodeMirror.setValue(filterAsm(currentAssembly, getAsmFilters()));
}

function onChange() {
    if (ignoreChanges) return;  // Ugly hack during startup.
    if (pendingTimeout) clearTimeout(pendingTimeout);
    pendingTimeout = setTimeout(function() {
        var data = { 
            source: cppEditor.getValue(),
            compiler: $('.compiler').val(),
            options: $('.compiler_options').val()
        };
        window.localStorage['compiler'] = data.compiler;
        window.localStorage['compilerOptions'] = data.options;
        if (data == lastRequest) return;
        lastRequest = data;
        $.ajax({
            type: 'POST',
            url: '/compile',
            dataType: 'json',
            data: data,
            success: onCompileResponse});
    }, 750);
    window.localStorage['code'] = cppEditor.getValue();
    window.localStorage['filter'] = JSON.stringify(getAsmFilters());
    updateAsm();
    $('a.permalink').attr('href', '#' + serialiseState());
}

function getSource() {
    var source = $('.source').val();
    if (source == "browser") {
        if (window.localStorage['files'] == undefined) window.localStorage['files'] = "{}";
        return {
            list: function(callback) {
                var files = JSON.parse(window.localStorage['files']);
                callback($.map(files, function(val, key) { return val; }));
            },
            load: function(name, callback) {
                var files = JSON.parse(window.localStorage['files']);
                callback(files[name]);
            },
            save: function(obj, callback) {
                var files = JSON.parse(window.localStorage['files']);
                files[obj.name] = obj;
                window.localStorage['files'] = JSON.stringify(files);
                callback(true);
            }
        };
    } else {
        var base = "/source/" + source;
        return {
            list: function(callback) { $.getJSON(base + "/list", callback); },
            load: function(name, callback) { $.getJSON(base + "/load/" + name, callback); },
            save: function(obj, callback) { alert("Coming soon..."); }
        };
    }
}

var currentFileList = {};
function updateFileList() {
    getSource().list(function(results) {
        currentFileList = {};
        $('.filename option').remove();
        $.each(results, function(index, arg) {
            currentFileList[arg.name] = arg;
            $('.filename').append($('<option value="' + arg.urlpart + '">' + arg.name + '</option>'));
            if (window.localStorage['filename'] == arg.urlpart) $('.filename').val(arg.urlpart);
        });
    });
}

function onSourceChange() {
    updateFileList();
    window.localStorage['source'] = $('.source').val();
}

function loadFile() {
    var name = $('.filename').val();
    window.localStorage['filename'] = name;
    getSource().load(name, function(results) {
        if (results.file) {
            cppEditor.setValue(results.file);
        } else {
            // TODO: error?
            console.log(results);
        }
    });
}

function saveFile() {
    saveAs($('.files .filename').val());
}

function saveAs(filename) {
    var prevFilename = window.localStorage['filename'] || "";
    if (filename != prevFilename && currentFileList[filename]) {
        // TODO!
        alert("Coming soon - overwriting files");
        return;
    }
    var obj = { urlpart: filename, name: filename, file: cppEditor.getValue() };
    getSource().save(obj, function(ok) {
        if (ok) {
            window.localStorage['filename'] = filename;
            updateFileList();
        }
    });
}

function saveFileAs() {
    $('#saveDialog').modal();
    $('#saveDialog .save-filename').val($('.files .filename').val());
    $('#saveDialog .save-filename').focus();
    function onSave() {
        $('#saveDialog').modal('hide');
        saveAs($('#saveDialog .save-filename').val());
    };
    $('#saveDialog .save').click(onSave);
    $('#saveDialog .save-filename').keyup(function(event) {
        if (event.keyCode == 13) onSave();
    });
}

function serialiseState() {
    var state = {
        version: 2,
        source: cppEditor.getValue(),
        compiler: $('.compiler').val(),
        options: $('.compiler_options').val(),
        filterAsm: getAsmFilters()
    };
    return encodeURIComponent(JSON.stringify(state));
}

function deserialiseState(state) {
    try {
        var state = $.parseJSON(decodeURIComponent(state));
        if (state.version == 1) { 
            state.filterAsm = {};
        }
        else if (state.version != 2) return false;
    } catch (ignored) { return false; }
    cppEditor.setValue(state.source);
    $('.compiler').val(state.compiler);
    $('.compiler_options').val(state.options);
    setFilterUi(state.filterAsm);
    // Somewhat hackily persist compiler into local storage else when the ajax response comes in
    // with the list of compilers it can splat over the deserialized version.
    // The whole serialize/hash/localStorage code is a mess! TODO(mg): fix
    window.localStorage['compiler'] = state.compiler;
    return true;
}

function initialise() {
    ignoreChanges = true; // Horrible hack to avoid onChange being called on first starting, ie before we've set anything up.
    cppEditor = CodeMirror.fromTextArea($("#c")[0], {
        lineNumbers: true,
              matchBrackets: true,
              useCPP: true,
              mode: "text/x-c++src",
              onChange: onChange
    });
    asmCodeMirror = CodeMirror.fromTextArea($(".asm textarea")[0], {
        lineNumbers: true,
                  matchBrackets: true,
                  mode: "text/x-asm",
                  readOnly: true
    });

    if (window.localStorage['code']) cppEditor.setValue(window.localStorage['code']);
    if (window.localStorage['compilerOptions']) $('.compiler_options').val(window.localStorage['compilerOptions']);
    setFilterUi($.parseJSON(window.localStorage['filter'] || "{}"));

    $('form').submit(function() { return false; });
    $('.compiler').change(onChange);
    $('.compiler_options').change(onChange).keyup(onChange);
    $.getJSON("/compilers", function(results) {
        $('.compiler option').remove();
        $.each(results, function(index, arg) {
            $('.compiler').append($('<option value="' + arg.exe + '">' + arg.version + '</option>'));
            if (window.localStorage['compiler'] == arg.exe) {
                $('.compiler').val(arg.exe);
            }
        });
        onChange();
    });
    $('.files .source').change(onSourceChange);
    $.getJSON("/sources", function(results) {
        $('.source option').remove();
        $.each(results, function(index, arg) {
            $('.files .source').append($('<option value="' + arg.urlpart + '">' + arg.name + '</option>'));
            if (window.localStorage['source'] == arg.urlpart) {
                $('.files .source').val(arg.urlpart);
            }
        });
        onSourceChange();
    });
    $('.files .load').click(function() {
        loadFile();
        return false;
    });
    $('.files .save').click(function() {
        saveFile();
        return false;
    });
    $('.files .saveas').click(function() {
        saveFileAs();
        return false;
    });
    
    $('.filter button.btn').click(function(e) {
        $(e.target).toggleClass('active');
        onChange();
    });

    function loadFromHash() {
        deserialiseState(window.location.hash.substr(1));
    }

    $(window).bind('hashchange', function() {
        loadFromHash();
    });
    loadFromHash();
    ignoreChanges = false;
}

function getAsmFilters() {
    var asmFilters = {};
    $('.filter button.btn.active').each(function() {
        asmFilters[$(this).val()] = true;
    });
    return asmFilters;
}

function setFilterUi(asmFilters) {
    $('.filter button.btn').each(function() {
        $(this).toggleClass('active', !!asmFilters[$(this).val()]);
    });
}

$(initialise);
