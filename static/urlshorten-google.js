function googleJSClientLoaded() {
    gapi.client.setApiKey(OPTIONS.gapiKey);
    gapi.client.load('urlshortener', 'v1', function () {
        shortenURL(googleJSClientLoaded.url, googleJSClientLoaded.done);
    });
}

function shortenURL(url, done) {
    if (!window.gapi || !gapi.client) {
        // Load the Google APIs client library asynchronously, then the
        // urlshortener API, and finally come back here.
        googleJSClientLoaded.url = url;
        googleJSClientLoaded.done = done;
        $(document.body).append('<script src="https://apis.google.com/js/client.js?onload=googleJSClientLoaded">');
        return;
    }
    var request = gapi.client.urlshortener.url.insert({
        resource: {
            longUrl: url
        }
    });
    request.then(function (resp) {
        var id = resp.result.id;
        if (OPTIONS.googleShortLinkRewrite.length === 2) {
            id = id.replace(new RegExp(OPTIONS.googleShortLinkRewrite[0]), OPTIONS.googleShortLinkRewrite[1]);
        }
        done(id);
    }, function () {
        done(url);
    });
}
