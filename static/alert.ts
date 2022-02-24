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

import { AlertAskOptions, AlertEnterTextOptions, AlertNotifyOptions } from './alert.interfaces';
import { toggleEventListener } from './utils';

export class Alert {
    yesHandler: ((answer?: string | string[] | number) => void) | null = null;
    noHandler: (() => void) | null = null;
    prefixMessage = '';

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
            .css('min-height', 'inherit')
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
     * Notifies the user of something by a popup which can be stacked, auto-dismissed, etc... based on options
     */
    notify(body: string, {
        group = '',
        collapseSimilar = true,
        alertClass = '',
        autoDismiss = true,
        dismissTime = 5000,
    }: AlertNotifyOptions) {
        const container = $('#notifications');
        if (!container) return;
        const newElement = $(`
            <div class="toast" tabindex="-1" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="toast-header ${alertClass}">
                    <strong class="mr-auto">${this.prefixMessage}</strong>
                    <button type="button" class="ml-2 mb-1 close" data-dismiss="toast" aria-label="Close">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
                <div class="toast-body ${alertClass}">
                    <span id="msg">${body}</span>
               </div>
            </div>
        `);
        container.append(newElement);
        newElement.toast({
            autohide: autoDismiss,
            delay: dismissTime,
        });
        if (group !== '') {
            if (collapseSimilar) {
                // Only collapsing if a group has been specified
                const old = container.find(`[data-group="${group}"]`);
                old.toast('hide');
                old.remove();
            }
            newElement.attr('data-group', group);
        }
        newElement.toast('show');
    }

    /**
     * Asks the user a two choice question, where the title, content and buttons are customizable
     */
    enterSomething(title: string, question: string, defaultValue: string, askOptions: AlertEnterTextOptions) {
        const modal = $('#enter-something');
        this.yesHandler = askOptions?.yes ?? (() => undefined);
        this.noHandler = askOptions?.no ?? (() => undefined);
        modal.find('.modal-title').html(title);
        modal.find('.modal-body .question').html(question);

        const yesButton = modal.find('.modal-footer .yes');
        toggleEventListener(yesButton, 'click', () => {
            const answer = modal.find('.question-answer');
            this.yesHandler?.(answer.val());
        });

        const noButton = modal.find('.modal-footer .no');
        toggleEventListener(noButton, 'click', () => {
            this.noHandler?.();
        });

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
