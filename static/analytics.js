// Copyright (c) 2012-2018, Matt Godbolt
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

"use strict";
var options = require('options');
var Raven = require('raven-js');
var $ = require('jquery');

if (options.raven) {
    Raven.config(options.raven, {
        release: options.release,
        environment: options.environment.join("/")
    }).install();
}

var gaProxy;
/* eslint-disable */
if (options.googleAnalyticsEnabled) {
    (function (i, s, o, g, r, a, m) {
        i.GoogleAnalyticsObject = r;
        i[r] = i[r] || function () {
            (i[r].q = i[r].q || []).push(arguments);
        };
        i[r].l = 1 * new Date();
        a = s.createElement(o);
        m = s.getElementsByTagName(o)[0];
        a.async = 1;
        a.src = g;
        m.parentNode.insertBefore(a, m);
    })(window, document, 'script', '//www.google-analytics.com/analytics.js', 'ga');
    ga('create', options.googleAnalyticsAccount, 'auto');
    ga('send', 'pageview');
    gaProxy = function () {
        window.ga.apply(window.ga, arguments);
    };
    // fullstory stuff:
    window['_fs_debug'] = false;
    window['_fs_host'] = 'www.fullstory.com';
    window['_fs_org'] = '2F4NV';
    window['_fs_namespace'] = 'FS';
    (function (m, n, e, t, l, o, g, y) {
        if (e in m && m.console && m.console.log) {
            m.console.log('FullStory namespace conflict. Please set window["_fs_namespace"].');
            return;
        }
        g = m[e] = function (a, b) {
            g.q ? g.q.push([a, b]) : g._api(a, b);
        };
        g.q = [];
        o = n.createElement(t);
        o.async = 1;
        o.src = 'https://' + _fs_host + '/s/fs.js';
        y = n.getElementsByTagName(t)[0];
        y.parentNode.insertBefore(o, y);
        g.identify = function (i, v) {
            g(l, {uid: i});
            if (v) g(l, v);
        };
        g.setUserVars = function (v) {
            g(l, v);
        };
        g.identifyAccount = function (i, v) {
            o = 'account';
            v = v || {};
            v.acctId = i;
            g(o, v);
        };
        g.clearUserCookie = function (c, d, i) {
            if (!c || document.cookie.match('fs_uid=[`;`]*`[`;`]*`[`;`]*`')) {
                d = n.domain;
                while (1) {
                    n.cookie = 'fs_uid=;domain=' + d +
                        ';path=/;expires=' + new Date(0).toUTCString();
                    i = d.indexOf('.');
                    if (i < 0) break;
                    d = d.slice(i + 1);
                }
            }
        };
    })(window, document, window['_fs_namespace'], 'script', 'user');
} else {
    gaProxy = function () {};
}
/* eslint-enable */

function initialise() {
    if (options.embedded) return;
    $(function () {
        function create_script_element(id, url) {
            var el = document.createElement('script');
            el.type = 'text/javascript';
            el.async = true;
            el.id = id;
            el.src = url;
            var s = document.getElementsByTagName('script')[0];
            s.parentNode.insertBefore(el, s);
        }

        if (options.sharingEnabled) {
            create_script_element('twitter-wjs', 'https://platform.twitter.com/widgets.js');
        }
    });
}

module.exports = {
    ga: gaProxy,
    initialise: initialise
};
