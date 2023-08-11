import {makeKeyedTypeGetter} from '../keyed-type.js';

import * as all from './_all.js';

export * from './_all.js';

export const getExternalParserByKey = makeKeyedTypeGetter('externalParser', all);
