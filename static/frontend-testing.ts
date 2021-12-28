import { ICETestable } from './frontend-testing.interfaces';

type describedTestfunc = () => Promise<void>;
type itTestfunc = () => Promise<void>;

const frontendTestingResults: any = {};

export class CEDescribedTest implements ICETestable {
    public Name: string;
    public TestFunc: describedTestfunc;

    constructor(name: string, func: describedTestfunc) {
        this.Name = name;
        this.TestFunc = func;
    }

    public run() {
        console.log(this.Name);

        this.TestFunc();
    }
}

export class CEItTest implements ICETestable {
    public Name: string;
    public TestFunc: describedTestfunc;
    public Result: Boolean;

    constructor(name: string, func: describedTestfunc) {
        this.Name = name;
        this.TestFunc = func;
    }

    public run() {
        this.TestFunc();
    }
}

export class CEExpectedFrom {
    public something: any;

    constructor(something: any) {
        this.something = something;
    }

    public equal(expected: any) {
        if (this.something !== expected) {
            return this.fail();
        } else {
            return this.pass();
        }
    }

    public fail() {
        throw Error('failed');
    }

    public pass() {
    }

    public includes(expected: any) {
        if (this.something.includes(expected)) {
            this.pass();
        } else {
            this.fail();
        }
    }
}

class CEFrontendTesting {
    private testSuites: Array<ICETestable>;

    constructor() {
        this.testSuites = [];
    }

    public add(test: ICETestable) {
        this.testSuites.push(test);
    }

    public async run() {
        for (let testSuite of this.testSuites) {
            testSuite.run();
        }
    }
}

export const frontendTesting = new CEFrontendTesting();

export function describe(name: string, func: describedTestfunc) {
    frontendTesting.add(new CEDescribedTest(name, func));
}

export async function it(desc: string, func: itTestfunc) {
    try {
        await func();

        console.log('- ' + desc + ': pass');
    } catch (e) {
        console.log('- ' + desc + ': ' + e);
    }
}

export function expect(something: any): CEExpectedFrom {
    return new CEExpectedFrom(something);
}
