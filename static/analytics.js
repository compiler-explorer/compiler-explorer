define(function (require, exports) {
    "use strict";
    var options = require('options');
    exports.initialise = function () {
        var _gaq = _gaq || [];
        _gaq.push(['_setAccount', options.googleAnalyticsAccount]);
        _gaq.push(['_trackPageview']);

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

            if (options.googleAnalyticsEnabled)
                create_script_element('ga', ('https:' == document.location.protocol ? 'https://ssl' : 'http://www') + '.google-analytics.com/ga.js');
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