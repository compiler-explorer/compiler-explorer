// Copyright (c) 2017, Matt Godbolt
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

const chai = require('chai'),
    chaiAsPromised = require("chai-as-promised"),
    nock = require('nock'),
    google = require('../lib/google');

chai.use(chaiAsPromised);
chai.should();

const googleApiUrl = 'https://www.googleapis.com';
const shortUrlEndpoint = '/urlshortener/v1/url';

describe('Google short URL resolver tests', () => {
    const resolver = new google.ShortLinkResolver('GoogleApiKey');

    it('Resolves simple URLs', () => {
        const resultObj = {longUrl: "http://long.url/", shortUrl: "badger"};
        nock(googleApiUrl)
            .get(shortUrlEndpoint)
            .query({
                key: 'GoogleApiKey',
                shortUrl: 'https://goo.gl/short'
            })
            .reply(200, JSON.stringify(resultObj));
        return resolver
            .resolve('https://goo.gl/short')
            .should.eventually.deep.equal(resultObj);
    });
    it('Handles missing long urls', () => {
        const resultObj = {missing: "no long url"};
        nock(googleApiUrl)
            .get(shortUrlEndpoint)
            .query({
                key: 'GoogleApiKey',
                shortUrl: 'https://goo.gl/broken'
            })
            .reply(200, JSON.stringify(resultObj));
        return resolver
            .resolve('https://goo.gl/broken')
            .should.be.rejectedWith("Missing longUrl");
    });
    it('Handles unsuccessful requests', () => {
        nock(googleApiUrl)
            .get(shortUrlEndpoint)
            .query({
                key: 'GoogleApiKey',
                shortUrl: 'https://goo.gl/broken'
            })
            .reply(404, "Google says no");
        return resolver
            .resolve('https://goo.gl/broken')
            .should.be.rejectedWith("Got response 404");
    });
    it('Handles failed requests', () => {
        nock(googleApiUrl)
            .get(shortUrlEndpoint)
            .query({
                key: 'GoogleApiKey',
                shortUrl: 'https://goo.gl/broken'
            })
            .replyWithError("Something went wrong");
        return resolver
            .resolve('https://goo.gl/broken')
            .should.be.rejectedWith("Something went wrong");
    });
});
