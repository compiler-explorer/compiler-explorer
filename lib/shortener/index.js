import { makeKeyedTypeGetter } from '../keyed-type';

import * as all from './_all';

export { BaseShortener } from './base';
export * from './_all';

export const getShortenerTypeByKey = makeKeyedTypeGetter('shortener', all);
