# Frontend testing

We have a mixture of unit tests (located in `static/tests`) and Cypress UI tests(located in `cypress/e2e`).

If possible, create unit tests in `static/tests` for the code you are working on. If you can get away with simple state
tests and don't need to do any real DOM manipulation, this is the way to go. Testing "does this filter correctly" or "do
we parse this right" are perfect examples of this. These tests use `happy-dom` for _extremely minimal_ DOM mocking just
enough to get the code running at all.

If you need to check actual behaviour or rely on the pug loading/HTML rendering, you should use the Cypress tests.

To run the cypress tests, see [this document](../UsingCypress.md).
