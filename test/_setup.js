import chai from 'chai';
import chaiAsPromised from 'chai-as-promised';
import chaiHttp from 'chai-http';
import deepEqualInAnyOrder from 'deep-equal-in-any-order';

import { suppressConsoleLog } from '../lib/logger';

// this hook will run once before any tests are executed
before(() => {
    suppressConsoleLog();
});

chai.use(chaiAsPromised);
chai.use(chaiHttp);
chai.use(deepEqualInAnyOrder);
