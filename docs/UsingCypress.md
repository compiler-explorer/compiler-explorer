### Using Cypress

Our frontend testing is done with cypress.

To run the tests locally:

- start a server with `npm run dev -- --language c++ --noLocal` - this configuration ensures your setup is clean of any
  local properties.
- in another terminal run `npx cypress open`, then choose "end to end" and then you should be able to run tests
  interactively.
