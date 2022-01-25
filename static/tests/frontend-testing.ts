import { IFrontendTesting, ITestable } from './frontend-testing.interfaces';

class FrontendTesting implements IFrontendTesting {
    private testSuites: Array<ITestable> = [];

    public add(test: ITestable) {
        this.testSuites.push(test);
    }

    public getAllTestNames(): string[] {
        return this.testSuites.map((val) => val.description);
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

window.compilerExplorerFrontendTesting = new FrontendTesting();
