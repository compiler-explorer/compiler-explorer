// Copyright (c) 2021, Compiler Explorer Authors
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
var ga = require('./analytics');
var local = require('./local');
var TomSelect = require('tom-select');

var favoriteGroupName = '__favorites__';
var favoriteStoreKey = 'favCompilerIds';
var nextSelectorId = 1;

function CompilerPicker(domRoot, hub, langId, compilerId, onCompilerChange) {
    this.eventHub = hub.createEventHub();
    this.id = nextSelectorId++;
    this.domNode = domRoot.find('.compiler-picker')[0];
    this.compilerService = hub.compilerService;
    this.onCompilerChange = onCompilerChange;
    this.eventHub.on('compilerFavoriteChange', this.onCompilerFavoriteChange, this);
    this.tomSelect = null;
    this.lastLangId = null;
    this.lastCompilerId = null;

    this.initialize(langId, compilerId);
}

CompilerPicker.prototype.close = function () {
    this.eventHub.unsubscribe();
    if (this.tomSelect)
        this.tomSelect.destroy();
    this.tomSelect = null;
};

CompilerPicker.prototype.initialize = function (langId, compilerId) {
    this.lastLangId = langId;
    this.lastCompilerId = compilerId;

    this.tomSelect = new TomSelect(this.domNode, {
        sortField: this.compilerService.getSelectizerOrder(),
        valueField: 'id',
        labelField: 'name',
        searchField: ['name'],
        optgroupField: '$groups',
        optgroups: this.getGroups(langId),
        lockOptgroupOrder: true,
        options: this.getOptions(langId, compilerId),
        items: compilerId ? [compilerId] : [],
        dropdownParent: 'body',
        closeAfterSelect: true,
        plugins: ['dropdown_input'],
        maxOptions: 1000,
        onChange: _.bind(function (val) {
            if (val) {
                ga.proxy('send', {
                    hitType: 'event',
                    eventCategory: 'SelectCompiler',
                    eventAction: val,
                });
                this.onCompilerChange(val);
            }
        }, this),
        duplicates: true,
        render: {
            option: function (data, escape) {
                var extraClasses = data.$groups.indexOf(favoriteGroupName) !== -1 ? ' fav' : '';
                return '<div class="d-flex' + extraClasses + '"><div>' + escape(data.name) + '</div>' +
                    '<div title="Click to mark or unmark as a favorite" class="ml-auto toggle-fav">' +
                    '<i class="fas fa-star"></i>' +
                    '</div>' +
                    '</div>';
            },
        },
    });

    $(this.tomSelect.dropdown_content).on('click', '.toggle-fav', _.bind(function (evt) {
        evt.preventDefault();
        evt.stopPropagation();

        var optionElement = evt.currentTarget.closest('.option');
        var clickedGroup = optionElement.parentElement.dataset.group;
        var value = optionElement.dataset.value;
        var data = this.tomSelect.options[value];
        var isAddingNewFavorite = data.$groups.indexOf(favoriteGroupName) === -1;
        var elemTop = optionElement.offsetTop;

        if (isAddingNewFavorite) {
            data.$groups.push(favoriteGroupName);
            this.addToFavorites(data.id);
        } else {
            data.$groups.splice(data.group.indexOf(favoriteGroupName), 1);
            this.removeFromFavorites(data.id);
        }

        this.tomSelect.updateOption(value, data);
        this.tomSelect.refreshOptions(false);

        if (clickedGroup !== favoriteGroupName) {
            // If the user clicked on an option that wasn't in the top "Favorite" group, then we just added
            // or removed a bunch of controls way up in the list. Find the new element top and adjust the scroll
            // so the element that was just clicked is back under the mouse.
            optionElement = this.tomSelect.getOption(value);
            var previousSmooth = this.tomSelect.dropdown_content.style.scrollBehavior;
            this.tomSelect.dropdown_content.style.scrollBehavior = 'auto';
            this.tomSelect.dropdown_content.scrollTop += (optionElement.offsetTop - elemTop);
            this.tomSelect.dropdown_content.style.scrollBehavior = previousSmooth;
        }
    }, this));
};

CompilerPicker.prototype.getOptions = function (langId, compilerId) {
    var favorites = this.getFavorites();
    return _.chain(this.compilerService.getCompilersForLang(langId))
        .filter(function (e) {
            return !e.hidden || e.id === compilerId;
        })
        .map(function (e) {
            e.$groups = [e.group];
            if (favorites[e.id])
                e.$groups.unshift(favoriteGroupName);
            return e;
        })
        .value();
};

CompilerPicker.prototype.getGroups = function (langId) {
    var optgroups = this.compilerService.getGroupsInUse(langId);
    optgroups.unshift({value: favoriteGroupName, label: 'Favorites'});
    return optgroups;
};

CompilerPicker.prototype.update = function (langId, compilerId) {
    this.tomSelect.destroy();
    this.initialize(langId, compilerId);
};

CompilerPicker.prototype.onCompilerFavoriteChange = function (id) {
    if (this.id !== id) {
        this.update(this.lastLangId, this.lastCompilerId);
    }
};

CompilerPicker.prototype.getFavorites = function () {
    return JSON.parse(local.get(favoriteStoreKey, '{}'));
};

CompilerPicker.prototype.setFavorites = function (faves) {
    local.set(favoriteStoreKey, JSON.stringify(faves));
};

CompilerPicker.prototype.isAFavorite = function (compilerId) {
    return !!this.getFavorites()[compilerId];
};

CompilerPicker.prototype.addToFavorites = function (compilerId) {
    var faves = this.getFavorites();
    faves[compilerId] = true;
    this.setFavorites(faves);
    this.eventHub.emit('compilerFavoriteChange', this.id);
};

CompilerPicker.prototype.removeFromFavorites = function (compilerId) {
    var faves = this.getFavorites();
    delete faves[compilerId];
    this.setFavorites(faves);
    this.eventHub.emit('compilerFavoriteChange', this.id);
};

module.exports = CompilerPicker;
