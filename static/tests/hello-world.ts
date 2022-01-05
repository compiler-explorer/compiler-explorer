import { ICETestable } from "../frontend-testing.interfaces";
import { assert } from 'chai';

class HelloWorldTests implements ICETestable {
    public readonly description: string = 'HelloWorld';

    public async run() {
        const person = true;
        assert.equal(person, true);
    }
}

window.frontendTesting.add(new HelloWorldTests());
