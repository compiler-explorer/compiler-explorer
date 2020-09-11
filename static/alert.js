// Copyright (c) 2017, Matt Godbolt & Rubén Rincón
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

var _ = require('underscore');
var $ = require('jquery');

function Alert() {
    this.yesHandler = null;
    this.noHandler = null;
    this.prefixMessage = '';
    var yesNo = $('#yes-no');
    yesNo.find('button.yes').click(_.bind(function () {
        if (this.yesHandler) this.yesHandler();
    }, this));
    yesNo.find('button.no').click(_.bind(function () {
        if (this.noHandler) this.noHandler();
    }, this));
}

Alert.prototype.alert = function (title, body, onClose) {
    var modal = $('#alert');
    modal.find('.modal-title').html(title);
    modal.find('.modal-body').html(body);
    modal.modal();
    if (onClose) {
        modal.off('hidden.bs.modal');
        modal.on('hidden.bs.modal', onClose);
    }
};

/***
 * Asks the user a two choice question, where the title, content & buttons are customizable
 *
 * @param title
 * @param question
 * @param handlers
 * @param handlers.yes {function?} Function to execute on yes press
 * @param handlers.no {function?} Function to execute on no press
 * @param handlers.yesHtml {HTML?} HTMl markup of yes button
 * @param handlers.yesClass {string?} Custom class to add to yes button
 * @param handlers.noHtml {HTML?} HTMl markup of no button
 * @param handlers.noClass {string?} Custom class to add to no button
 * @param handlers.onClose {function?} Function to execute on pane closure
 */
Alert.prototype.ask = function (title, question, handlers) {
    var modal = $('#yes-no');
    this.yesHandler = handlers ? handlers.yes : function () {};
    this.noHandler = handlers ? handlers.no : function () {};
    modal.find('.modal-title').html(title);
    modal.find('.modal-body').html(question);
    if (handlers.yesHtml) {
        modal.find('.modal-footer .yes').html(handlers.yesHtml);
    }
    if (handlers.yesClass) {
        modal.find('.modal-footer .yes').addClass(handlers.yesClass);
    }
    if (handlers.noHtml) {
        modal.find('.modal-footer .no').html(handlers.noHtml);
    }
    if (handlers.noClass) {
        modal.find('.modal-footer .no').addClass(handlers.noClass);
    }
    if (handlers.onClose) {
        modal.off('hidden.bs.modal');
        modal.on('hidden.bs.modal', handlers.onClose);
    }
    return modal.modal();
};

/***
 * @typedef {number} Milliseconds
 */

/***
 * Notifies the user of something by a popup which can be stacked, autodismissed etc... based on options
 *
 * @param body {string} Inner message html
 * @param options {object} Object containing some extra settings
 * @param options.group {string} What group this notification is from. Sets data-group's value.
 *  Default: ""
 * @param options.collapseSimilar {boolean} If set to true, other notifications with the same group
 *  will be removed before sending this one. (Note that this only has any effect if options.group is set).
 *  Default: true
 * @param options.alertClass {string} Space separated classes to give to the notification div element.
 *  Default: ""
 * @param options.autoDismiss {boolean} If set to true, the notification will fade out and be removed automatically.
 *  Default: true
 * @param options.dismissTime {Milliseconds} If allowed by options.autoDismiss, controls how long in the notification
 *  will be visible before being automatically removed.
 *  Default: 5000
 *  Min: 1000
 */
Alert.prototype.notify = function (body, options) {
    var container = $('#notifications');
    if (!container) return;
    var newElement = $('<div class="alert notification" tabindex="-1" role="dialog">' +
        '<button type="button" class="close" style="float: left;margin-right: 5px;" data-dismiss="alert">' +
            '&times;' +
        '</button>' +
        '<span id="msg">' + this.prefixMessage + body + '</span>' +
        '</div>');
    if (!options) options = {};
    // Set defaults
    // Collapse similar by default
    options.collapseSimilar = ('collapseSimilar' in options) ? options.collapseSimilar : true;
    // autoDismiss by default
    options.autoDismiss = ('autoDismiss' in options) ? options.autoDismiss : true;
    // Dismiss this after 5000ms by default
    options.dismissTime = options.dismissTime ? Math.max(1000, options.dismissTime) : 5000;
    if (options.group) {
        if (options.collapseSimilar) {
            // Only collapsing if a group has been specified
            container.find('[data-group="' + options.group + '"]').remove();
        }
        newElement.attr('data-group', options.group);  // Add the group to the data-group
    }
    if (options.alertClass) {  // If we want a custom class, apply it
        newElement.addClass(options.alertClass);
    }
    if (options.autoDismiss) {  // Dismiss this after dismissTime
        setTimeout(function () {
            newElement.fadeOut('slow', function () {
                newElement.remove();
            });
        }, options.dismissTime);
    }
    container.append(newElement);  // Add the new notification to the container
};

module.exports = Alert;
