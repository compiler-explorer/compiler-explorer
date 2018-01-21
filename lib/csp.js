// Copyright (c) 2012-2018, Rubén Rincón
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

const _ = require('underscore-node');

const data = {
    godbolt: {
        'default-src': ["'self'", 'https://*.godbolt.org', "'report-sample'"],
        'style-src': ["'self'", 'https://*.godbolt.org', "'unsafe-inline'", "'report-sample'"],
        'script-src': ["'self'", 'https://*.godbolt.org', "'unsafe-inline'", 'https://*.twitter.com', 'https://www.fullstory.com',
            'https://www.google-analytics.com', 'https://apis.google.com', 'https://ssl.google-analytics.com', "'report-sample'"],
        'img-src': ["'self'", 'https://*.godbolt.org', 'data:', 'https://www.google-analytics.com/' , 'https://syndication.twitter.com',
            'https://ssl.google-analytics.com', 'https://csi.gstatic.com'],
        'font-src': ["'self'", 'https://*.godbolt.org', 'data:', 'https://fonts.gstatic.com'],
        'frame-src': ["'self'", 'https://*.godbolt.org', 'https://www.google-analytics.com', ' https://rs.fullstory.com/',
            'https://sentry.io', 'https://platform.twitter.com/', 'https://syndication.twitter.com/'],
        'report-uri': ['https://sentry.io/api/102028/csp-report/?sentry_key=849826dce97d4d1eae26df64061ec4bb'],
        'connect-src': ['*'],
        'media-src': ['https://ssl.gstatic.com']
    }
};
const policies = {};

_.each(data, (policy, key) => {
    policies[key] = _.map(policy, (value, policyKey) => `${policyKey} ${value.join(' ')}`).join(';');
});

module.exports = {
    policies: policies
};
