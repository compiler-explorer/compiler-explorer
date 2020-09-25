module.exports = {
    StorageLocal: require('./local'),
    StorageNull: require('./null'),
    StorageRemote: require('./remote'),
    StorageS3: require('./s3'),
};
