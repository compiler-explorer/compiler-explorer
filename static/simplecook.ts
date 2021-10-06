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

type SimpleCookieCallback = () => void;

export class SimpleCook {
    private onDoConsent: SimpleCookieCallback = () => undefined;
    private onDontConsent: SimpleCookieCallback = () => undefined;
    private onHide: SimpleCookieCallback = () => undefined;
    private readonly elem: JQuery;

    public constructor() {
        this.elem = $('#simplecook');
        this.elem.hide();

        this.elem.find('.cookies').on('click', () => {
            $('#cookies').trigger('click');
        });
        this.elem.find('.cook-do-consent').on('click', this.callDoConsent.bind(this));
        this.elem.find('.cook-dont-consent').on('click', this.callDontConsent.bind(this));
    }

    public show(): void {
        this.elem.show();
    };

    public hide(): void {
        this.elem.hide();
        this.onHide();
    };

    public callDoConsent(): void {
        this.hide();
        this.onDoConsent();
    };

    public callDontConsent(): void {
        this.hide();
        this.onDontConsent();
    };

    public setOnDoConsent(callback: SimpleCookieCallback): void {
        this.onDoConsent = callback;
    }

    public setOnDontConsent(callback: SimpleCookieCallback): void {
        this.onDontConsent = callback;
    }

    public setOnHide(callback: SimpleCookieCallback): void {
        this.onHide = callback;
    }
}
