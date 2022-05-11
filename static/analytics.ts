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

import {options} from './options';
import * as Sentry from '@sentry/browser';

if (options.statusTrackingEnabled && options.sentryDsn) {
    Sentry.init({
        dsn: options.sentryDsn,
        release: options.release,
        environment: options.sentryEnvironment,
    });
}

class GAProxy {
    private hasBeenEnabled = false;
    private isEnabled = false;
    private _proxy: (...args) => void = () => {};

    initialise() {
        if (!this.isEnabled && options.statusTrackingEnabled && options.googleAnalyticsEnabled) {
            // Check if this is a re-enable, as the script is already there in this case
            if (!this.hasBeenEnabled) {
                (function (i, s, o, g, r, a, m) {
                    i.GoogleAnalyticsObject = r;
                    i[r] =
                        i[r] ||
                        function () {
                            // We could push the unexpanded args, but better might have some unintended side-effects
                            // eslint-disable-next-line prefer-rest-params
                            (i[r].q = i[r].q || []).push(arguments);
                        };
                    // @ts-ignore
                    i[r].l = 1 * new Date();
                    // @ts-ignore
                    a = s.createElement(o);
                    // @ts-ignore
                    m = s.getElementsByTagName(o)[0];
                    // @ts-ignore
                    a.async = 1;
                    // @ts-ignore
                    a.src = g;
                    // @ts-ignore
                    m.parentNode.insertBefore(a, m);
                })(window, document, 'script', '//www.google-analytics.com/analytics.js', 'ga');
                window.ga('set', 'anonymizeIp', true);
                window.ga('create', options.googleAnalyticsAccount, {
                    cookieDomain: 'auto',
                    cookieFlags: 'SameSite=None; Secure',
                });
                window.ga('send', 'pageview');
            }
            this._proxy = function () {
                // eslint-disable-next-line prefer-rest-params
                window.ga.apply(window.ga, arguments);
            };
            this.isEnabled = true;
            this.hasBeenEnabled = true;
        } else {
            this.isEnabled = false;
            this._proxy = () => {};
        }
    }

    toggle(doEnable) {
        if (doEnable) {
            if (!this.isEnabled) this.initialise();
        } else {
            this.isEnabled = false;
            this._proxy = () => {};
        }
    }

    proxy(...args) {
        this._proxy(args);
    }
}

const ga = new GAProxy();
export {ga};
