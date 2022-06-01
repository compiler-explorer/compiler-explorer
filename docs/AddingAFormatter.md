# Adding a new formatter

- Add a `etc/config/compiler-explorer.local.properties` file

  - Add a new formatter under the `formatters` key
  - The new formatter can have the following keys: name, exe, styles, type, explicitVersion (to override version
    parsing), version (argument to get version info), versionRe (regex to filter out the right version info)
  - Add a `lib/formatters/<formatter>.ts` file using the template below, replacing `Type` and `type` as appropriate

    ```js
    import {UnprocessedExecResult} from '../../types/execution/execution.interfaces';
    import {BaseFormatter} from './base';
    import {FormatOptions} from './base.interfaces';

    export class TypeFormatter extends BaseFormatter {
      static get key() {
        return 'type';
      }
    }
    ```

  - The value returned by `key` above corresponds to the `type` property you set in the compiler-explorer properties
    configuration file.
  - Tweak `format(source: string, options: FormatOptions): Promise<UnprocessedExecResult>` and
    `isValidStyle(style: string): boolean` as necessary. See the JSDoc for `format` and the implementations for other
    formatters to get a further understanding of how to implement `format(source, options)`.

- Add your `TypeFormatter` to `lib/formatters/_all.ts` in alphabetical order

- You can check the output of http://localhost:10240/api/formats to be sure your formatter is there.

- Make an installer in the [infra](https://github.com/compiler-explorer/infra) repository. An example patch for adding
  an installer can be found [here](https://github.com/compiler-explorer/infra/pull/560)
