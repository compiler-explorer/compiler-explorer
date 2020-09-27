import { makeKeyedTypeGetter } from '../keyed-type';

import * as all from './_all';

export { BaseTool } from './base-tool';
export * from './_all';

export const getToolTypeByKey = makeKeyedTypeGetter('tool', all);
