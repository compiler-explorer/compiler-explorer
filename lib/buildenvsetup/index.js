import { makeKeyedTypeGetter } from '../keyed-type';

import * as all from './_all';

export { BuildEnvSetupBase } from './base';
export * from './_all';

export const getBuildEnvTypeByKey = makeKeyedTypeGetter('buildenv', all);
