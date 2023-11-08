# Frontend testing

We have a mixture of typescript in the main website's code (located in `static/tests`) and Cypress (located in
`cypress/integration`) to test and report on the workings of that code.

But there's always the possibility to use Cypress code to do UI checks and testing.

## Recommended

The recommended way of testing is to use typescript to test the inner workings of the various interfaces that are
available.

This has the advantage of having types and being able to verify your code is consistent with the rest of the website and
probably going to run correctly - without having to startup the website and Cypress.

## Adding a test

Steps to add a test:

- Create a new file in `static/tests` (copy paste from `static/tests/hello-world.ts`)
- Make sure to change the `description` as well as the test
- Add the file to the imports of `static/tests/_all.ts`
- Add a test file with a `runFrontendTest()` call in `cypress/e2e`

## Starting tests locally

You don't need to install an entire X server to actually run cypress (just xfvb).

You can find a complete list at https://docs.cypress.io/guides/getting-started/installing-cypress#System-requirements

If you have the prerequisites installed, you should be able to run `npx cypress run` - however, you will need to start
the CE website separately in another terminal before that.

Some extra tips can be found [here](../UsingCypress.md)
