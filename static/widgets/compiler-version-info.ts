// Copyright (c) 2023, Compiler Explorer Authors
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
import {escapeHTML} from '../../shared/common-utils.js';
import * as BootstrapUtils from '../bootstrap-utils.js';
import {options} from '../options.js';

export type CompilerVersionInfo = {version: string; fullVersion?: string};

async function getVersionInfo(compilerId: string): Promise<CompilerVersionInfo> {
    let response: any;

    if (window.location.protocol === 'http:') {
        // use jsonp for testing
        response = await new Promise((resolve, reject) => {
            $.getJSON(options.compilerVersionsUrl + '?id=' + encodeURIComponent(compilerId) + '&jsonp=?', resolve).fail(
                reject,
            );
        });
    } else {
        response = await $.getJSON(options.compilerVersionsUrl + '?id=' + encodeURIComponent(compilerId));
    }

    return {
        version: response.version,
        fullVersion: response.full_version,
    };
}

function reallySetCompilerVersionPopover(
    pane: any, // Compiler | Executor
    version?: CompilerVersionInfo,
    notification?: string,
    compilerId?: string,
) {
    // `notification` contains HTML from a config file, so is 'safe'.
    // `version` comes from compiler output, so isn't, and is escaped.
    const bodyContent = $('<div>');
    const versionContent = $('<div>').html(escapeHTML(version?.version ?? ''));
    bodyContent.append(versionContent);
    if (version?.fullVersion && version.fullVersion.trim() !== version.version.trim()) {
        const hiddenSection = $('<div>');
        const lines =
            version.fullVersion
                .split('\n')
                .map(line => {
                    return escapeHTML(line);
                })
                .join('<br/>') +
            'Internal compiler ID: ' +
            compilerId +
            '<br/>';
        const hiddenVersionText = $('<div>').html(lines).hide();
        const clickToExpandContent = $('<a>')
            .text('Toggle full version output')
            .addClass('link-primary')
            .on('click', () => {
                versionContent.toggle();
                hiddenVersionText.toggle();
                const popover = BootstrapUtils.getPopoverInstance(pane.fullCompilerName);
                if (popover) popover.update();
            });
        hiddenSection.append(hiddenVersionText).append(clickToExpandContent);
        bodyContent.append(hiddenSection);
    }

    // Dispose of existing popover
    const existingPopover = BootstrapUtils.getPopoverInstance(pane.fullCompilerName);
    if (existingPopover) existingPopover.dispose();

    // Initialize a new popover; may not exist in embedded links.
    BootstrapUtils.initPopoverIfExists(pane.fullCompilerName, {
        html: true,
        title: notification
            ? ($.parseHTML('<span>Compiler Version: ' + notification + '</span>')[0] as Element)
            : 'Full compiler version',
        content: bodyContent,
        template:
            '<div class="popover' +
            (version ? ' compiler-options-popover' : '') +
            '" role="tooltip">' +
            '<div class="arrow"></div>' +
            '<h3 class="popover-header"></h3><div class="popover-body"></div>' +
            '</div>',
    });
}

export function setCompilerVersionPopoverForPane(
    pane: any, // Compiler | Executor
    version?: CompilerVersionInfo,
    notification?: string,
    compilerId?: string,
) {
    if (options.compilerVersionsUrl && compilerId && pane.compiler?.isNightly) {
        getVersionInfo(compilerId)
            .then(updatedVersion => {
                reallySetCompilerVersionPopover(pane, updatedVersion, notification, compilerId);
            })
            .catch(() => {
                reallySetCompilerVersionPopover(pane, version, notification, compilerId);
            });
    } else {
        reallySetCompilerVersionPopover(pane, version, notification, compilerId);
    }
}
