// Copyright (c) 2012-2016, Matt Godbolt
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

function hidePermalink() {
    if ($('.files .permalink-collapse').hasClass('in')) {  // do nothing if already hidden.
        $('.files .permalink-collapse').collapse('hide');
    }
}

function showPermalink(short) {
    if (!$('.files .permalink-collapse').hasClass('in')) {
        $('.files .permalink-collapse').collapse('show');
    }
    $('#permalink').val('');

    var fullUrl = window.location.href.split('#')[0] + '#' + serialiseState();
    if (short) {
        shortenURL(fullUrl,
            function (shortUrl) {
                $('#permalink').val(shortUrl);
            });
    } else {
        $('#permalink').val(fullUrl);
    }
}

function serialiseState() {
    var compressed = rison.quote(rison.encode_object(getState(true)));
    var uncompressed = rison.quote(rison.encode_object(getState(false)));
    var MinimalSavings = 0.20;  // at least this ratio smaller
    if (compressed.length < uncompressed.length * (1.0 - MinimalSavings)) {
        return compressed;
    } else {
        return uncompressed;
    }
}

function getState(compress) {
    return {
        version: 3,
        filterAsm: getAsmFilters(),
        compilers: $.map(allCompilers, function (compiler) {
            return compiler.serialiseState(compress);
        })
    };
}

function toGist(state) {
    files = {};
    function nameFor(compiler) {
        var addNum = 0;
        var name, add;
        for (; ;) {
            add = addNum ? addNum.toString() : "";
            name = compiler + add + '.' + OPTIONS.sourceExtension;
            if (files[name] === undefined) return name;
            addNum++;
        }
    };
    state.compilers.forEach(function (s) {
        var name = nameFor(s.compiler);
        files[name] = {
            content: s.source,
            language: OPTIONS.language
        };
        s.source = name;
    });
    files['state.json'] = {content: JSON.stringify(state)};
    return JSON.stringify({
        description: "Compiler Explorer automatically generated files",
        'public': false,
        files: files
    });
}

function isGithubLimitError(request) {
    var remaining = parseInt(request.getResponseHeader('X-RateLimit-Remaining'));
    var reset = parseInt(request.getResponseHeader('X-RateLimit-Reset'));
    var limit = parseInt(request.getResponseHeader('X-RateLimit-Limit'));
    if (remaining !== 0) return null;
    var left = (new Date(reset * 1000) - Date.now()) / 1000;
    return "Rate limit of " + limit + " exceeded: " + Math.round(left / 60) + " mins til reset";
}

function makeGist(onDone, onFail) {
    var req = $.ajax('https://api.github.com/gists', {
        type: 'POST',
        accepts: 'application/vnd.github.v3+json',
        dataType: 'json',
        contentType: 'application/json',
        data: toGist(getState())
    });
    req.done(function (msg) {
        onDone(msg);
    });
    req.fail(function (jqXHR, textStatus) {
        var rateLimited = isGithubLimitError(jqXHR);
        if (rateLimited)
            onFail(rateLimited);
        else
            onFail(textStatus + " (" + jqXHR.statusText + ")");
    });
}

function fromGist(msg) {
    var state = JSON.parse(msg.files['state.json'].content);
    state.compilers.forEach(function (s) {
        s.source = msg.files[s.source].content;
    });
    return state;
}
function loadGist(gist) {
    var req = $.ajax('https://api.github.com/gists/' + gist);
    req.done(function (msg) {
        loadState(fromGist(msg));
    });
    req.fail(function (jqXHR, textStatus) {
        var err = isGithubLimitError(jqXHR);
        if (!err) {
            err = textStatus + " (" + jqXHR.statusText + ")";
        }
        alert("Unable to load gist: " + err);
    });
}

function deserialiseState(stateText) {
    var state = null;
    if (stateText.substr(0, 2) == "g=") {
        loadGist(stateText.substr(2));
        return;
    }

    try {
        state = rison.decode_object(decodeURIComponent(stateText.replace(/\+/g, '%20')));
    } catch (ignored) {
    }

    if (!state) {
        try {
            state = $.parseJSON(decodeURIComponent(stateText));
        } catch (ignored) {
        }
    }
    if (state) {
        return loadState(state);
    }
    return false;
}

function loadState(state) {
    if (!state || state['version'] === undefined) return false;
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
    $('.files .fulllink').click(function (e) {
        showPermalink(false);
        return false;
    });
    $('.files .shortlink').click(function (e) {
        showPermalink(true);
        return false;
    });

    new Clipboard('.btn.clippy');

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
        var top = codeMirrors.offset().top;
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

function setPanelListSortable() {
        var panelList = $('#draggablePanelList');

        panelList.sortable({
            // Do not set the new-slot button sortable (nor a place to set windows)
            items: "li:not(.panel-sortable-disabled)", // source : JQuery examples
            // Omit this to make then entire <li>...</li> draggable.
            handle: '.panel-heading', 
            update: function() {
                $('.panel', panelList).each(function(index, elem) {
                    var $listItem = $(elem),
                newIndex = $listItem.index();

                // Persist the new indices.
                });
            }
        });
}

$(document).ready(setPanelListSortable);

$(document).ready(function() {
    $('#new-slot').on('click', function(e)  {
        console.log("[UI] User clicked on new-slot button. slotCount was : "+ currentCompiler.getSlotsCount());
        var newSlotCode = currentCompiler.getSlotsCount();
        currentCompiler.setSlotsCount(currentCompiler.getSlotsCount() + 1);
        // source : http://stackoverflow.com/questions/10126395/how-to-jquery-clone-and-change-id
        var slotTemplate = $('#slotTemplate');
        var clone = slotTemplate.clone().prop('id', 'slot'+newSlotCode);
        var last = $('#new-slot');
        last.before(clone); // insert right before the "+" button

        $('#slot'+newSlotCode+' .title').text("Slot " + newSlotCode+" (drag me) ");
        $('#slot'+newSlotCode).show();
        $('#slot'+newSlotCode+' .closeButton').on('click', function(e)  {
            console.log("[UI] User clicked on closeButton in slot "+newSlotCode);
            // demo code : 
            $('#slot'+newSlotCode).remove();
            currentCompiler.setSlotsCount(currentCompiler.getSlotsCount() - 1);
        });

        setPanelListSortable();
        currentCompiler.oneMoreSlot();
        currentCompiler.setCompilersInSlot(OPTIONS.compilers, OPTIONS.defaultCompiler, newSlotCode);
        currentCompiler.refreshSlot(newSlotCode);
    });
});

$(document).ready(function() {
    $('#slotTemplate').hide();
});

$(function () {
    initialise(OPTIONS);
});
