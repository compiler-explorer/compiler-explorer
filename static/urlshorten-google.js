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
    "use strict";
    const options = require('options');
    const Alert = require('alert');

    function googleJSClientLoaded() {
        gapi.client.setApiKey(options.gapiKey);
        gapi.client.load('urlshortener', 'v1', googleJSClientLoaded.done);
    }

    function shortenURL(url, done) {
        if (!window.gapi || !gapi.client) {
            // Load the Google APIs client library asynchronously, then the
            // urlshortener API, and finally come back here.
            window.googleJSClientLoaded = googleJSClientLoaded;
            googleJSClientLoaded.done = function () {
                shortenURL(url, done);
            };
            $(document.body).append('<script src="https://apis.google.com/js/client.js?onload=googleJSClientLoaded">');
            return;
        }
        const request = gapi.client.urlshortener.url.insert({resource: {longUrl: url}});
        request.then(function (resp) {
            let id = resp.result.id;
            if (options.googleShortLinkRewrite.length === 2) {
                id = id.replace(new RegExp(options.googleShortLinkRewrite[0]), options.googleShortLinkRewrite[1]);
            }
            done(id);
        }, function () {
            new Alert().notify("The URL could not be shortened. It probaly exceeds the Google URL Shortener length limits.", {
                group: "urltoolong",
                alertClass: "notification-error"
            });
            done(url);
        });
    }

    return shortenURL;
});
