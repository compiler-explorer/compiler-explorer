// Copyright (c) 2020, Compiler Explorer Authors
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

import { logger } from './logger';

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

export function makeKeyedTypeGetter(typeName, objects) {
    const keyMap = makeKeyMap(typeName, objects);

    return function getFromKey(key) {
        const obj = keyMap[key];
        if (obj === undefined) {
            throw new Error(`No ${typeName} named '${key}' found`);
        }

        return obj;
    };
}
