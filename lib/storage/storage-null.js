const StorageBase = require('./storage').StorageBase;

class StorageNull extends StorageBase {
    constructor(compilerProps) {
        super(compilerProps);
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
