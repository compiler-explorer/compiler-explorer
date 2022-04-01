// Copyright (c) 2022, Compiler Explorer Authors
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
import {ga} from './analytics';

import {Motd} from './motd.interfaces';


function ensureShownMessage(message: string, motdNode: JQuery) {
    motdNode.find('.content').html(message);
    motdNode.removeClass('d-none');
    motdNode.find('.close')
        .on('click', () => {
            motdNode.addClass('d-none');
        })
        .prop('title', 'Hide message');
}

function handleMotd(motd: Motd, motdNode: JQuery, subLang: string, adsEnabled: boolean, onHide: () => void) {
    if (motd.update) {
        ensureShownMessage(motd.update, motdNode);
    } else if (motd.motd) {
        ensureShownMessage(motd.motd, motdNode);
    } else if (adsEnabled) {
        const applicableAds = motd.ads?.filter((ad) => {
            return !subLang || !ad.filter || ad.filter.length === 0 || ad.filter.indexOf(subLang) >= 0;
        });

        if (applicableAds != null && applicableAds.length > 0) {
            const randomAd = applicableAds[Math.floor(Math.random() * applicableAds.length)];
            motdNode.find('.content').html(randomAd.html);
            motdNode.find('.close').on('click', () => {
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

export function initialise(
    url: string,
    motdNode: JQuery,
    defaultLanguage: string,
    adsEnabled: boolean,
    onMotd: (res?: Motd) => void,
    onHide: () => void)
{
    if (!url) return;
    $.getJSON(url)
        .then((res: Motd) => {
            onMotd(res);
            handleMotd(res, motdNode, defaultLanguage, adsEnabled, onHide);
        })
        .catch(() => {
            // do nothing! we've long tried to find out why this might fail, and it seems page load cancels or ad
            // blockers might reasonably cause a failure here, and it's no big deal.
            // Some history at https://github.com/compiler-explorer/compiler-explorer/issues/1057
        });
}
