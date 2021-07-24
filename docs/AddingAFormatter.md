# Adding a new formatter

* Add a `etc/config/local.compiler-explorer.properties` file
  - Add a new formatter under the `formatters` key
  - The new formatter can have the following keys: name, exe, styles, type,
     version (argument to get version info), versionRe (regex to filter out the right version info)
  - Add a `lib/formatters/<formatter>.js` file using the template below, replacing `Type` and `type` as
     appropriate
    ```js
    import { BaseFormatter } from '../base-formatter';

    export class TypeFormatter extends BaseFormatter {
        static get key() { return 'type'; }
    }
    ```
  - The value returned by `key` above corrosponds to the `type` property you set in the compiler-explorer properties
     configuration file.
  - Tweak `format(args, source)`, `getDefaultArguments()`, `getStyleArguments(style)` and `isValidStyle(style)` as
     necessary
* Add your `TypeFormatter` to `lib/formatters/_all.js` in alphabetical order

* You can check http://127.0.0.1/api/formats to be sure your formatter is there.

* Make an installer in the [infra](https://github.com/compiler-explorer/infra) repository
