function googleJSClientLoaded() {
    gapi.client.setApiKey(OPTIONS.gapiKey);
    gapi.client.load('urlshortener', 'v1', makePermalink);
}

function shortenURL(url, done) {
    if (!window.gapi || !gapi.client) {
        // Load the Google APIs client library asynchronously, then the
        // urlshortener API, and finally come back here.
        $(document.body).append('<script src="https://apis.google.com/js/client.js?onload=googleJSClientLoaded">');
        return;
    }
    var request = gapi.client.urlshortener.url.insert({
        resource: {
            longUrl: url
        }
    });
    request.then(function (resp) {
        done(resp.result.id);
    }, function () {
        done(url);
    });
}
