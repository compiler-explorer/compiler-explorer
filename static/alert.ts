// Copyright (c) 2017, Compiler Explorer Authors
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

import $ from 'jquery';

export interface AlertAskOptions {
    /** Function to execute on "yes" button press */
    yes?: () => void;
    /** Function to execute on "no" button press */
    no?: () => void;
    /** HTML markup of "yes" button */
    yesHtml?: string;
    /** Custom HTML class to add to "yes" button */
    yesClass?: string;
    /** HTML markup of "no" button */
    noHtml?: string;
    /** Custom HTML class to add to "no" button */
    noClass?: string;
    /** Function to execute on pane closure */
    onClose?: () => void;
}

export interface AlertNotifyOptions {
    /**
     * Which group this notification is from. Sets data-group attribute value
     * Default: ""
     */
    group?: string
    /** If set to true, other notifications within the same group will be removed before sending this one. (Note that
     * this only has any effect if options.group is set).
     * Default: true
     */
    collapseSimilar?: boolean
    /**
     * Space separated list of HTML classes to give to the notification's div element.
     * Default: ""
     */
    alertClass?: string
    /**
     * If set to true, the notification will fade out and be removed automatically.
     * Default: true
     */
    autoDismiss?: boolean
    /**
     * If allow by autoDismiss, controls how long the notification will be visible (in milliseconds) before
     * automatically removed
     * Default: 5000
     */
    dismissTime?: number
}

export class Alert {
    yesHandler: () => void | null = null;
    noHandler: () => void | null = null;
    prefixMessage: string = '';

    constructor() {
        const yesNoModal = $('#yes-no');
        yesNoModal.find('button.yes').on('click', () => {
            this.yesHandler?.();
        });
        yesNoModal.find('button.no').on('click', () => {
            this.noHandler?.();
        });
    }

    /**
     * Display an alert with a title and a body
     */
    alert(title: string, body: string, onClose?: () => void) {
        const modal = $('#alert');
        modal.find('.modal-title').html(title);
        modal.find('.modal-body').html(body);
        modal.modal();
        if (onClose) {
            modal.off('hidden.bs.modal');
            modal.on('hidden.bs.modal', onClose);
        }
    }

    /**
     * Asks the user a two choice question, where the title, content and buttons are customizable
     */
    ask(title: string, question: string, handlers: AlertAskOptions) {
        const modal = $('#yes-no');
        this.yesHandler = handlers?.yes ?? (() => undefined);
        this.noHandler = handlers?.no ?? (() => undefined);
        modal.find('.modal-title').html(title);
        modal.find('.modal-body').html(question);
        if (handlers.yesHtml) modal.find('.modal-footer .yes').html(handlers.yesHtml);
        if (handlers.yesClass) modal.find('.modal-footer .yes').addClass(handlers.yesClass);
        if (handlers.noHtml) modal.find('.modal-footer .no').html(handlers.noHtml);
        if (handlers.noClass) modal.find('.modal-footer .no').addClass(handlers.noClass);
        if (handlers.onClose) {
            modal.off('hidden.bs.modal');
            modal.on('hidden.bs.modal', handlers.onClose);
        }
        modal.modal();
    }

    /**
     * Notifes the user of something by a popup which can be stacked, auto-dismissed, etc... based on options
     */
    notify(body: string, {
        group = "",
        collapseSimilar = true,
        alertClass = "",
        autoDismiss = true,
        dismissTime = 5000
    }: AlertNotifyOptions) {
        const container = $('#notifications');
        if (!container) return;
        const newElement = $(`
            <div class="alert notification ${alertClass}" tabindex="-1" role="dialog">
                <button type="button" class="close" style="float: left; margin-right: 5px;" data-dismiss="alert">
                    &times;
                </button>
                <span id="msg">${this.prefixMessage}${body}</span>
            </div>
        `);
        if (group !== "") {
            if (collapseSimilar) {
                // Only collapsing if a group has been specified
                container.find(`[data-group="${group}"]`).remove();
            }
            newElement.attr('data-group', group)
        }
        if (autoDismiss) {
            setTimeout(() => {
                newElement.fadeOut('slow', () => {
                    newElement.remove();
                });
            }, dismissTime);
        }
        // Append the newly created element to the container
        container.append(newElement);
    }
}
