import { frontendTesting, describe, it, expect } from "../frontend-testing";
import { ICETestable } from "../frontend-testing.interfaces";

class HelloWorldTests implements ICETestable {
    public run() {
        describe('Hello-world testing', async () => {
            it('should be polite', async () => {
                const hello = 'Hello, World!';

                expect(hello).includes('Hello');
            });
        });
    }
}

frontendTesting.add(new HelloWorldTests());
