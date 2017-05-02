// Copyright (c) 2012-2017, Matt Godbolt
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

define(function (require) {
    "use strict";
    const $ = require('jquery');
    const _ = require('underscore');
    const Alert = require('./alert');
    const local = require('./local');
    // Estamos usando esto!
    //noinspection JSUnusedLocalSymbols
    const FileSaver = require('filesaver');

    function getLocalFiles() {
        return JSON.parse(local.get('files', "{}"));
    }

    function setLocalFile(name, file) {
        const files = getLocalFiles();
        files[name] = file;
        local.set('files', JSON.stringify(files));
    }

    function LoadSave() {
        this.modal = $('#load-save');
        this.alert = new Alert();
        this.onLoad = _.identity;
        this.editorText = '';
        this.extension = '.txt';
        this.modal.find('.local-file').change(_.bind(this.onLocalFile, this));

        this.modal.find('.save-button').click(_.bind(this.onSaveToBrowserStorage, this));
        this.modal.find('.save-file').click(_.bind(this.onSaveToFile, this));

        this.populateBuiltins();
    }

    LoadSave.prototype.populateBuiltins = function () {
        $.getJSON('source/builtin/list', _.bind(function (list) {
            this.populate(
                this.modal.find('.examples'),
                _.map(list, _.bind(function (elem) {
                    return {
                        name: elem.name, load: _.bind(function () {
                            this.doLoad(elem.urlpart);
                        }, this)
                    };
                }, this)));
        }, this));
    };

    LoadSave.prototype.populateLocalStorage = function () {
        this.populate(
            this.modal.find('.local-storage'),
            _.map(getLocalFiles(), _.bind(function (data, name) {
                return {
                    name: name,
                    load: _.bind(function () {
                        this.onLoad(data);
                        this.modal.modal('hide');
                    }, this)
                };
            }, this)));
    };

    LoadSave.prototype.populate = function (root, list) {
        root.find('li:not(.template)').remove();
        const template = root.find('.template');
        _.each(list, _.bind(function (elem) {
            template
                .clone()
                .removeClass('template')
                .appendTo(root)
                .find('a')
                .text(elem.name)
                .click(elem.load);
        }, this));
    };

    LoadSave.prototype.onLocalFile = function (event) {
        const files = event.target.files;
        const file = files[0];
        const reader = new FileReader();
        reader.onload = _.bind(function () {
            this.onLoad(reader.result);
        }, this);
        reader.readAsText(file);
        this.modal.modal('hide');
    };

    LoadSave.prototype.run = function (onLoad, editorText, extensions) {
        this.populateLocalStorage();
        this.onLoad = onLoad;
        this.editorText = editorText;
        // In case we don't send anything...
        this.extension = extensions[0] || '.txt';
        this.modal.find('.local-file').attr('accept', _.map(extensions, function (extension) {
            return extension + ', ';
        }, this));
        this.modal.modal();
    };

    LoadSave.prototype.onSaveToBrowserStorage = function () {
        const name = this.modal.find('.save-name').val();
        if (!name) {
            this.alert.alert("Save name", "Invalid save name");
            return;
        }
        const done = _.bind(function () {
            setLocalFile(name, this.editorText);
        }, this);
        if (getLocalFiles()[name] !== undefined) {
            this.modal.modal('hide');
            this.alert.ask(
                "Replace current?",
                "Do you want to replace the existing saved file '" + name + "'?",
                {yes: done});
        } else {
            done();
            this.modal.modal('hide');
        }
    };

    LoadSave.prototype.onSaveToFile = function () {
        try {
            saveAs(new Blob([this.editorText], {type: "text/plain;charset=utf-8"}), "Compiler Explorer Code" + this.extension);
        } catch (e) {
            new Alert().notify('Error while saving your code. Use the clipboard instead.', {
                group: "savelocalerror",
                alertClass: "notification-error",
                dismissTime: 5000
            });
        }
    };

    LoadSave.prototype.doLoad = function (urlpart) {
        // TODO: handle errors. consider promises...
        $.getJSON('source/builtin/load/' + urlpart, _.bind(function (response) {
            this.onLoad(response.file);
        }, this));
        this.modal.modal('hide');
    };

    return {LoadSave: LoadSave};
});
