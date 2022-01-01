import { expect } from "../frontend-testing";
import { ICETestable } from "../frontend-testing.interfaces";

class HelloWorldTests implements ICETestable {
    public run() {
        const hello = 'Hello, World!';
        expect(hello).includes('Hello');

        const person = false;
        expect(person).equal(true);
    }
}

window.frontendTesting.add(new HelloWorldTests());
