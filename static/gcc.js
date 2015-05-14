// Copyright (c) 2012-2015, Matt Godbold
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

var currentCompiler = null;
var allCompilers = [];

function getSource() {
    var source = $('.source').val();
    if (source == "browser") {
        if (window.localStorage.files === undefined) window.localStorage.files = "{}";
        return {
            list: function (callback) {
                var files = JSON.parse(window.localStorage.files);
                callback($.map(files, function (val, key) {
                    return val;
                }));
            },
            load: function (name, callback) {
                var files = JSON.parse(window.localStorage.files);
                callback(files[name]);
            },
            save: function (obj, callback) {
                var files = JSON.parse(window.localStorage.files);
                files[obj.name] = obj;
                window.localStorage.files = JSON.stringify(files);
                callback(true);
            }
        };
    } else {
        var base = "/source/" + source;
        return {
            list: function (callback) {
                $.getJSON(base + "/list", callback);
            },
            load: function (name, callback) {
                $.getJSON(base + "/load/" + name, callback);
            },
            save: function (obj, callback) {
                alert("Coming soon...");
            }
        };
    }
}

var currentFileList = {};
function updateFileList() {
    getSource().list(function (results) {
        currentFileList = {};
        $('.filename option').remove();
        $.each(results, function (index, arg) {
            currentFileList[arg.name] = arg;
            $('.filename').append($('<option value="' + arg.urlpart + '">' + arg.name + '</option>'));
            if (window.localStorage.filename == arg.urlpart) $('.filename').val(arg.urlpart);
        });
    });
}

function onSourceChange() {
    updateFileList();
    window.localStorage.source = $('.source').val();
}

function loadFile() {
    var name = $('.filename').val();
    window.localStorage.filename = name;
    getSource().load(name, function (results) {
        if (results.file) {
            currentCompiler.setSource(results.file);
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
    var prevFilename = window.localStorage.filename || "";
    if (filename != prevFilename && currentFileList[filename]) {
        // TODO!
        alert("Coming soon - overwriting files");
        return;
    }
    var obj = {urlpart: filename, name: filename, file: currentCompiler.getSource()};
    getSource().save(obj, function (ok) {
        if (ok) {
            window.localStorage.filename = filename;
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
    }

    $('#saveDialog .save').click(onSave);
    $('#saveDialog .save-filename').keyup(function (event) {
        if (event.keyCode == 13) onSave();
    });
}

function makePermalink() {
    $('#permalink').val('');

    shortenURL(window.location.href.split('#')[0] + '#' + serialiseState(),
        function (shorturl) {
            $('#permalink').val(shorturl);
        });
}

function hidePermalink() {
    if ($('.files .permalink').hasClass('active')) {  // do nothing if already hidden.
        togglePermalink();
    }
}

function togglePermalink() {
    if (!$('.files .permalink').hasClass('active')) {
        $('.files .permalink').addClass('active');
        $('.files .permalink-collapse').collapse('show');
        makePermalink();
    } else {
        $('.files .permalink-collapse').collapse('hide');
        $('.files .permalink').removeClass('active');
    }
}

function serialiseState() {
    var state = {
        version: 3,
        filterAsm: getAsmFilters(),
        compilers: $.map(allCompilers, function (compiler) {
            return compiler.serialiseState();
        })
    };
    return encodeURIComponent(JSON.stringify(state));
}

function deserialiseState(state) {
    try {
        state = $.parseJSON(decodeURIComponent(state));
        switch (state.version) {
            case 1:
                state.filterAsm = {};
            /* falls through */
            case 2:
                state.compilers = [state];
            /* falls through */
            case 3:
                break;
            default:
                return false;
        }
    } catch (ignored) {
        return false;
    }
    setFilterUi(state.filterAsm);
    for (var i = 0; i < Math.min(allCompilers.length, state.compilers.length); i++) {
        allCompilers[i].setFilters(state.filterAsm);
        allCompilers[i].deserialiseState(state.compilers[i]);
    }
    return true;
}

function initialise(options) {
    var defaultFilters = JSON.stringify(getAsmFilters());
    var actualFilters = $.parseJSON(window.localStorage.filter || defaultFilters);
    setFilterUi(actualFilters);

    $(".compiler_options").val(options.compileoptions);
    $(".language-name").text(options.language);

    var compiler = new Compiler($('body'), actualFilters, "a", function () {
        hidePermalink();
    }, options.language);
    allCompilers.push(compiler);
    currentCompiler = compiler;

    $('form').submit(function () {
        return false;
    });
    $('.files .source').change(onSourceChange);
    compiler.setCompilers(options.compilers, options.defaultCompiler);
    function setSources(sources, defaultSource) {
        $('.source option').remove();
        $.each(sources, function (index, arg) {
            $('.files .source').append($('<option value="' + arg.urlpart + '">' + arg.name + '</option>'));
            if (defaultSource == arg.urlpart) {
                $('.files .source').val(arg.urlpart);
            }
        });
        onSourceChange();
    }

    setSources(options.sources, window.localStorage.source || options.defaultSource);
    $('.files .load').click(function () {
        loadFile();
        return false;
    });
    $('.files .save').click(function () {
        saveFile();
        return false;
    });
    $('.files .saveas').click(function () {
        saveFileAs();
        return false;
    });
    $('.files .permalink').click(function (e) {
        togglePermalink(e);
        return false;
    });

    $('.filter button.btn').click(function (e) {
        $(e.target).toggleClass('active');
        var filters = getAsmFilters();
        window.localStorage.filter = JSON.stringify(filters);
        currentCompiler.setFilters(filters);
    });

    function loadFromHash() {
        deserialiseState(window.location.hash.substr(1));
    }

    $(window).bind('hashchange', function () {
        loadFromHash();
    });
    loadFromHash();

    function resizeEditors() {
        var codeMirrors = $('.CodeMirror');
        var top = codeMirrors.position().top;
        var windowHeight = $(window).height();
        var compOutputSize = Math.max(100, windowHeight * 0.05);
        $('.output').height(compOutputSize);
        var resultHeight = $('.result').height();
        var height = windowHeight - top - resultHeight - 40;
        currentCompiler.setEditorHeight(height);
    }

    $(window).on("resize", resizeEditors);
    resizeEditors();
}

function getAsmFilters() {
    var asmFilters = {};
    $('.filter button.btn.active').each(function () {
        asmFilters[$(this).val()] = true;
    });
    return asmFilters;
}

function setFilterUi(asmFilters) {
    $('.filter button.btn').each(function () {
        $(this).toggleClass('active', !!asmFilters[$(this).val()]);
    });
}

$(function () {
    initialise(OPTIONS);
});
