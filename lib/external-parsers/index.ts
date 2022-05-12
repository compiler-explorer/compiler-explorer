import {makeKeyedTypeGetter} from '../keyed-type';

import * as all from './_all';

export * from './_all';

export const getExternalParserByKey = makeKeyedTypeGetter('externalParser', all);
