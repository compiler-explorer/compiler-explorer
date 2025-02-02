## `vitest` crib sheet

We just moved to a new testing framework, `vitest`. Here are some notes to help you get started.

### Running tests

- `npm test` will run the tests
- `npm run test:watch` will run the watcher; which runs all the tests and then watches for changes and then runs on the
  changed tests. This is much quicker as a lot of the work is cached.
- `npm run test:watch <path>` will do the same with just a path

### Writing tests

In general use `describe`, `it` and `expect` as you would with jest. Import these all from `vitest`. If you need async
things then use `await expect(someAsyncThing()).resolves.toEqual(someValue)`.

The `expect` is pretty rich and supports a lot of matchers, like `expect(x).toContain("moo")` or
`expect(y).not.toHaveProperty("badger")`. You can see the full list in the
[vitest documentation](https://vitest.dev/api/expect.html).
