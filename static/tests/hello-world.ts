import { ITestable } from './frontend-testing.interfaces';
import { assert } from 'chai';

class HelloWorldTests implements ITestable {
    public readonly description: string = 'HelloWorld';

    public async run() {
        const person = true;
        assert.equal(person, true);
    }
}

window.compilerExplorerFrontendTesting.add(new HelloWorldTests());
