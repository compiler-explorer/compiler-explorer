require('./noscript.css');

var $ = require('jquery');

// eslint-disable-next-line requirejs/no-js-extension
require('popper.js');
require('bootstrap');

var Toggles = require('./toggles');

function initOptionMenus() {
    $('.button-checkbox').each(function () {
        var container = $(this);

        var span = container.find('span');
        span.remove();

        var option = container.find('input');
        option.addClass('d-none');

        var button = $('<button />');
        button.addClass('dropdown-item btn btn-sm btn-light');
        button.attr('type', 'button');
        button.attr('title', option.attr('title'));
        button.data('bind', option.attr('name'));
        button.attr('aria-pressed', option.attr('checked') === 'checked' ? 'true' : 'false');
        button.append(span);
        container.prepend(button);

        var parent = container.parent();
        parent.removeClass('noscriptdropdown');
        parent.addClass('dropdown-menu');
    });

    new Toggles($('.output'));
    new Toggles($('.filters'));
}

function initLanguageMenu() {
    $('.noscriptdropdown').removeClass('noscriptdropdown').addClass('dropdown-menu');
    $('.nodropdown-toggle').removeClass('nodropdown-toggle').addClass('dropdown-toggle');
}

$(document).ready(function () {
    initOptionMenus();
    initLanguageMenu();
});
