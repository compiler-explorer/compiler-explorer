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

// This jQuery import needs to be here, because noscript.ts is a different entrypoint than the rest of the code.
// See webpack.config.esm.ts -> entry for more details.
import $ from 'jquery';
import 'bootstrap';
import 'popper.js';

import {Toggles} from './widgets/toggles.js';
import './styles/noscript.scss';

$(document).on('ready', () => {
    $('.button-checkbox').each(function () {
        const container = $(this);
        const span = container.find('span');
        span.remove();
        const option = container.find('input');
        option.addClass('d-none');

        const button = $('<button />');
        button.addClass('dropdown-item btn btn-sm btn-light');
        button.attr('type', 'button');
        button.attr('title', option.attr('title') ?? '');
        button.data('bind', option.attr('name') ?? '');
        button.attr('aria-pressed', option.attr('checked') === 'checked' ? 'true' : 'false');
        button.append(span);

        container.prepend(button);
        const parent = container.parent();
        parent.removeClass('noscriptdropdown');
        parent.addClass('dropdown-menu');
    });

    $('.noscriptdropdown').removeClass('noscriptdropdown').addClass('dropdown-menu');
    $('.nodropdown-toggle').removeClass('nodropdown-toggle').addClass('dropdown-toggle');

    new Toggles($('.output'), {});
    new Toggles($('.filters'), {});
});
