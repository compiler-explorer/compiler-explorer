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

import $ from 'jquery';

import { AlertAskOptions, AlertNotifyOptions } from "./alert.interfaces";

export class Alert {
    yesHandler: (answer?: string | string[] | number) => void | null = null;
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

        const enterSomething = $('#enter-something');
        enterSomething.find('button.yes').on('click', () => {
            const answer = enterSomething.find('.question-answer');
            this.yesHandler?.(answer.val());
        });
        enterSomething.find('button.no').on('click', () => {
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
        return modal;
    }

    /**
     * Asks the user a two choice question, where the title, content and buttons are customizable
     */
    ask(title: string, question: string, askOptions: AlertAskOptions) {
        const modal = $('#yes-no');
        this.yesHandler = askOptions?.yes ?? (() => undefined);
        this.noHandler = askOptions?.no ?? (() => undefined);
        modal.find('.modal-title').html(title);
        modal.find('.modal-body')
            .css("min-height", "inherit")
            .html(question);
        if (askOptions.yesHtml) modal.find('.modal-footer .yes').html(askOptions.yesHtml);
        if (askOptions.yesClass) {
            modal.find('.modal-footer .yes')
                .removeClass('btn-link')
                .addClass(askOptions.yesClass);
        }
        if (askOptions.noHtml) modal.find('.modal-footer .no').html(askOptions.noHtml);
        if (askOptions.noClass) {
            modal.find('.modal-footer .no')
                .removeClass('btn-link')
                .addClass(askOptions.noClass);
        }
        if (askOptions.onClose) {
            modal.off('hidden.bs.modal');
            modal.on('hidden.bs.modal', askOptions.onClose);
        }
        modal.modal();
        return modal;
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

    /**
     * Asks the user a two choice question, where the title, content and buttons are customizable
     */
    enterSomething(title: string, question: string, defaultValue: string, askOptions: AlertAskOptions) {
        const modal = $('#enter-something');
        this.yesHandler = askOptions?.yes ?? (() => undefined);
        this.noHandler = askOptions?.no ?? (() => undefined);
        modal.find('.modal-title').html(title);
        modal.find('.modal-body .question').html(question);

        const yesButton = modal.find('.modal-footer .yes');
        const noButton = modal.find('.modal-footer .no');
        const answerEdit = modal.find('.modal-body .question-answer');
        answerEdit.val(defaultValue);
        answerEdit.on('keyup', (e) => {
            if (e.keyCode === 13 || e.which === 13) {
                yesButton.trigger('click');
            }
        });

        if (askOptions.yesHtml) yesButton.html(askOptions.yesHtml);
        if (askOptions.yesClass) {
            yesButton.removeClass('btn-light').addClass(askOptions.yesClass);
        }
        if (askOptions.noHtml) noButton.html(askOptions.noHtml);
        if (askOptions.noClass) {
            noButton.removeClass('btn-light').addClass(askOptions.noClass);
        }
        if (askOptions.onClose) {
            modal.off('hidden.bs.modal');
            modal.on('hidden.bs.modal', askOptions.onClose);
        }

        modal.on('shown.bs.modal', () => {
            answerEdit.trigger('focus');
        });
        modal.modal();
        return modal;
    }
}
