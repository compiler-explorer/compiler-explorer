// Copyright (c) 2018, Compiler Explorer Authors
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
var $ = require('jquery'),
    Sentry = require('@sentry/browser'),
    _ = require('underscore'),
    ga = require('analytics');

function handleMotd(motd, motdNode, subLang, adsEnabled, onHide) {
    if (motd.motd) {
        motdNode.find('.content').html(motd.motd);
        motdNode.removeClass('d-none');
        motdNode.find('.close')
            .on('click', function () {
                motdNode.addClass('d-none');
            })
            .prop('title', 'Hide message');
    } else if (adsEnabled) {
        var applicableAds = _.filter(motd.ads, function (ad) {
            return !subLang || !ad.filter || ad.filter.length === 0 || ad.filter.indexOf(subLang) >= 0;
        });
        var randomAd = applicableAds[_.random(applicableAds.length - 1)];
        if (randomAd) {
            motdNode.find('.content').html(randomAd.html);
            motdNode.find('.close').on('click', function () {
                ga.proxy('send', {
                    hitType: 'event',
                    eventCategory: 'Ads',
                    eventLabel: 'Visibility',
                    eventAction: 'Hide',
                });
                motdNode.addClass('d-none');
                onHide();
            });
            motdNode.find('a').on('click', function () {
                ga.proxy('send', {
                    hitType: 'event',
                    eventCategory: 'Ads',
                    eventAction: 'Click',
                    eventLabel: this.href,
                });
            });
            motdNode.removeClass('d-none');
        }
    }
}

function initialise(url, motdNode, defaultLanguage, adsEnabled, onMotd, onHide) {
    if (!url) return;
    $.getJSON(url)
        .then(function (res) {
            onMotd(res);
            handleMotd(res, motdNode, defaultLanguage, adsEnabled, onHide);
        })
        .catch(function (jqXHR, textStatus, errorThrown) {
            var message = 'MOTD error for ' + url + ' - ' + textStatus + ' - ' + (errorThrown || 'no error') +
                ' - readyState=' + jqXHR.readyState + ' - status=' + jqXHR.status;
            Sentry.captureMessage(message, 'warning');
        });
}

module.exports = {
    initialise: initialise,
};
