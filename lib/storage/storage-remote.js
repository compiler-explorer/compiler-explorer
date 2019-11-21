const logger = require('../logger').logger,
    request = require('request'),
    StorageBase = require('./storage').StorageBase,
    util = require('util');

class StorageRemote extends StorageBase {
    constructor(httpRootDir, compilerProps) {
        super(httpRootDir, compilerProps);

        this.baseUrl = compilerProps.ceProps('remoteStorageServer');

        const req = request.defaults({
            baseUrl: this.baseUrl
        });

        this.get = util.promisify(req.get);
        this.post = util.promisify(req.post);
    }

    async handler(req, res) {
        let resp;
        try {
            resp = await this.post('/shortener', {
                json: true,
                body: req.body
            });
        } catch (err) {
            logger.error(err);
            res.status(500);
            res.end(err.message);
            return;
        }

        const url = resp.body.url;
        if (!url) {
            res.status(resp.statusCode);
            res.end(resp.body);
            return;
        }

        const relativeUrl = url.substring(url.lastIndexOf('/z/') + 1);
        const shortlink = `${req.protocol}://${req.get('host')}${this.httpRootDir}${relativeUrl}`;

        res.set('Content-Type', 'application/json');
        res.send(JSON.stringify({url: shortlink}));
    }

    async expandId(id) {
        const resp = await this.get(`/api/shortlinkinfo/${id}`);

        if (resp.statusCode !== 200) throw new Error(`ID ${id} not present in remote storage`);

        return {
            config: resp.body,
            specialMetadata: null
        };
    }

    incrementViewCount() {
        return Promise.resolve();
    }
}

module.exports = StorageRemote;
