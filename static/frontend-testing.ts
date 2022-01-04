import { ICEFrontendTesting, ICETestable } from './frontend-testing.interfaces';

class CEFrontendTesting implements ICEFrontendTesting {
    private testSuites: Array<ICETestable>;

    constructor() {
        this.testSuites = [];
    }

    public add(test: ICETestable) {
        this.testSuites.push(test);
    }

    private findTest(name: string) {
        for (const suite of this.testSuites) {
            if (suite.description === name) { 
                return suite;
            }
        }

        throw new Error(`Can't find test ${name}`);
    }

    public async run(testToRun: string) {
        const testSuite = this.findTest(testToRun);
        await testSuite.run();
    }
}

window.frontendTesting = new CEFrontendTesting();
