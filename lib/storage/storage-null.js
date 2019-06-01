const StorageBase = require('./storage').StorageBase;

class StorageNull extends StorageBase {
    constructor(httpRootDir, compilerProps) {
        super(httpRootDir, compilerProps);
    }

    storeItem(item) {
        return Promise.resolve(item);
    }

    findUniqueSubhash() {
        return Promise.resolve({
            prefix: "null",
            uniqueSubHash: "null",
            alreadyPresent: true
        });
    }

    expandId() {
        return Promise.resolve({
            config: "{}",
            specialMetadata: null
        });
    }

    incrementViewCount() {
        return Promise.resolve();
    }
}
module.exports = StorageNull;
