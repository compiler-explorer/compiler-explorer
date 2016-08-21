define(function (require, exports) {
    "use strict";
    var options = require('options');

    if (options.googleAnalyticsEnabled) {
        (function (i, s, o, g, r, a, m) {
            i['GoogleAnalyticsObject'] = r;
            i[r] = i[r] || function () {
                    (i[r].q = i[r].q || []).push(arguments)
                }, i[r].l = 1 * new Date();
            a = s.createElement(o),
                m = s.getElementsByTagName(o)[0];
            a.async = 1;
            a.src = g;
            m.parentNode.insertBefore(a, m)
        })(window, document, 'script', '//www.google-analytics.com/analytics.js', 'ga');
        exports.ga = window.ga;
        ga('create', options.googleAnalyticsAccount, 'auto');
        ga('send', 'pageview');
    } else {
        exports.ga = function () {
        };
    }

    exports.initialise = function () {
        setTimeout(function () {
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
                create_script_element('gp', 'https://apis.google.com/js/plusone.js');
                create_script_element('twitter-wjs', '//platform.twitter.com/widgets.js');
                (function (document, i) {
                    var f, s = document.getElementById(i);
                    f = document.createElement('iframe');
                    f.src = '//api.flattr.com/button/view/?uid=mattgodbolt&button=compact&url=' + encodeURIComponent(document.URL);
                    f.title = 'Flattr';
                    f.height = 20;
                    f.width = 110;
                    f.style.borderWidth = 0;
                    s.appendChild(f);
                }(document, 'flattr_button'));
            }
        }, 0);
    };
});