import { makeKeyedTypeGetter } from '../keyed-type';

import * as all from './_all';

export { BaseDemangler } from './base';
export * from './_all';

export const getDemanglerTypeByKey = makeKeyedTypeGetter('demangler', all);
