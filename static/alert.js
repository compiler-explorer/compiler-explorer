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
    var $ = require('jquery');

    function Alert() {
        this.yesHandler = null;
        this.noHandler = null;
        $('#yes-no button.yes').click(_.bind(function () {
            if (this.yesHandler) this.yesHandler();
        }, this));
        $('#yes-no button.no').click(_.bind(function () {
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

    Alert.prototype.ask = function (title, question, handlers) {
        var modal = $('#yes-no');
        this.yesHandler = handlers.yes;
        this.noHandler = handlers.no;
        modal.find('.modal-title').html(title);
        modal.find('.modal-body').html(question);
        modal.modal();
    };

    /* Options parameter:
     *  group: What group this notification is from. Sets data-group's value. Default: none
     *  noCollapse: If set to true, other notifications with the same group will not be removed before sending this. (Note that this only has any effect if group is set). Default: false
     *  alertClass: Space separated classes to give to the notification div element. Default: none
     *  noAutoDismiss: If set to true, the notification will not fade out nor be removed automatically. Default: false (This element will be dissmised)
     *  dismissTime: If allowed by noAutoDismiss, controls how long the notification will be visible before being automatically removed. Default: 5000ms
     */
    Alert.prototype.notify = function (body, options) {
        var modal = $('#notifications');
        if (!modal) return;
        var newElement = $('<div class="alert notification" tabindex="-1" role="dialog"><button type="button" class="close" data-dismiss="alert">&times;</button><span id="msg">' + body + '</span></div>');
        if (options) {
            if (options.group) {
                if (!options.noCollapse) {
                    modal.find('[data-group="' + options.group + '"]').remove();
                }
                newElement.attr('data-group', options.group);
            }
            if (options.alertClass) {
                newElement.addClass(options.alertClass);
            }
        }
        modal.append(newElement);
        if (!options || !options.noAutoDismiss) {
            // Dismiss this after dismissTime
            setTimeout(function () {
                newElement.fadeOut('slow', function() {
                    newElement.remove();
                });
            }, options && options.dismissTime ? options.dismissTime : 5000);
        }
    };

    Alert.prototype.onYesNoHide = function (evt) {
        console.log(evt);
    };

    return Alert;
});