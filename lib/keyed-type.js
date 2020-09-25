const logger = require('./logger').logger;

function makeKeyMap(typeName, objects) {
    const keyToNameMap = {};
    const keyToTypeMap = {};
    let haveErrors = false;

    for (const name in objects) {
        const type = objects[name];
        const key = type.key;

        if (key === undefined) {
            logger.error(`${typeName} ${name} does not provide a key value`);
            haveErrors = true;
        } else if (!key) {
            logger.error(`${typeName} ${name} provides empty key value`);
            haveErrors = true;
        } else if (keyToTypeMap[key] !== undefined) {
            logger.error(`${typeName} ${name} key conflicts with ${keyToNameMap[key]}`);
            haveErrors = true;
        } else {
            keyToTypeMap[key] = type;
            keyToNameMap[key] = name;
        }
    }

    /*
     * If there are any errors we just log them and continue above.
     *
     * Once done, we throw to halt instance startup so the logs don't
     * get lost in a wall of text.
     */
    if (haveErrors)
        throw new Error(`${typeName} KeyedType configuration error`);

    return keyToTypeMap;
}

function makeKeyedTypeGetter(typeName, objects) {
    const keyMap = makeKeyMap(typeName, objects);

    return function getFromKey(key) {
        const obj = keyMap[key];
        if (obj === undefined) {
            throw new Error(`No ${typeName} named '${key}' found`);
        }

        return obj;
    };
}

module.exports = {
    makeKeyedTypeGetter,
};
