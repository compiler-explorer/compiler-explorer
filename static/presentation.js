// Copyright (c) 2020, Compiler Explorer Authors
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

"use strict";

var
    local = require('./local');

var _currentPresentation = null;

function Presentation() {
    this.maxSlides = 0;
    this.currentSlide = 0;
    this.originallocation = window.location.href;
}

Presentation.prototype.init = function (maxSlides, callback) {
    this.maxSlides = maxSlides;
    this.currentSlide = parseInt(local.get('presentationCurrentSlide', 0));
    if (callback !== undefined) callback();
};

Presentation.prototype.first = function () {
    this.currentSlide = 0;
    local.set('presentationCurrentSlide', this.currentSlide);
    this.show();
};

Presentation.prototype.next = function () {
    if (this.currentSlide + 1 < this.maxSlides) {
        this.currentSlide++;
        local.set('presentationCurrentSlide', this.currentSlide);
        this.show();
    }
};

Presentation.prototype.prev = function () {
    if (this.currentSlide > 0) {
        this.currentSlide--;
        local.set('presentationCurrentSlide', this.currentSlide);
        this.show();
    }
};

Presentation.prototype.show = function () {
    window.hasUIBeenReset = true;
    if (window.location.href === this.originallocation) {
        window.location.reload();
    } else {
        window.location.href = this.originallocation;
    }
};

function init(maxSlides, callback) {
    if (!_currentPresentation) {
        _currentPresentation = new Presentation();
        _currentPresentation.init(maxSlides, callback);
    } else {
        callback();
    }
}

function first() {
    if (!_currentPresentation) throw "Presentation hasn't been initialized";

    _currentPresentation.first();
}

function next() {
    if (!_currentPresentation) throw "Presentation hasn't been initialized";

    _currentPresentation.next();
}

function prev() {
    if (!_currentPresentation) throw "Presentation hasn't been initialized";

    _currentPresentation.prev();
}

function getCurrentSlide() {
    if (!_currentPresentation) throw "Presentation hasn't been initialized";

    return _currentPresentation.currentSlide;
}

function setCurrentSlide(idx) {
    if (!_currentPresentation) throw "Presentation hasn't been initialized";

    _currentPresentation.currentSlide = idx;
}

module.exports = {
    init: init,
    first: first,
    next: next,
    prev: prev,
    getCurrentSlide: getCurrentSlide,
    setCurrentSlide: setCurrentSlide
};
