import { makeKeyedTypeGetter } from '../keyed-type';

import * as all from './_all';

export { StorageBase } from './base';
export * from './_all';

export const getStorageTypeByKey = makeKeyedTypeGetter('storage', all);
