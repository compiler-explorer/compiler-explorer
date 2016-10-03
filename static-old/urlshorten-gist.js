function shortenURL(url, done) {
    makeGist(function (msg) {
            done(document.location.origin + "#g=" + msg.id);
        },
        function (failure) {
            done("Failed: " + failure);
        }
    );
}
