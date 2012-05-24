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
var lastRequest = null;

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
    asmCodeMirror.setValue(data.asm || "[no output]");
}

function onChange() {
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
