# Adding a new language

If you want to add a new language to the site, you should follow this steps:

- Add the new language to the exported `languages` variable in `lib/languages.ts`:

  - The syntax is as follows:

  ```js
  'language-key': {
    name: YOUR_LANGUAGE_DISPLAY_NAME,
    monaco: MONACO_MODE_ID,
    extensions: ARRAY_OF_FILE_EXTENSIONS_OF_YOUR_LANGUAGE,
    alias: [], // Leave empty unless needed,
    logoUrl: PATH_TO_LOGO,
    logoUrlDark: PATH_TO_DARK_LOGO, // Optional if not needed
  }
  ```

  - If the language is supported by Monaco Editor (You can find the list
    [here](https://github.com/microsoft/monaco-editor/tree/main/src/basic-languages)), you should add it to the list of
    languages inside the `MonacoEditorWebpackPlugin` config in `webpack.config.esm.ts`
  - If not, you should implement your own language mode; see `static/modes/asm-mode.js` as an example. Don't forget to
    _require_ your mode file in `static/modes/_all.ts`, in alphabetical order
  - `language-key` is how your language will be referred internally by the code. In the rest of this document, replace
    `{language-key}` by the corresponding value in the real files.
  - Add a logo file to the `views/resources/logos/` folder and add its path to the `logoUrl{Dark}` key(s) in the
    language object

- Add a `lib/compilers/{language-key}.js` file using the template below:

  ```js
  import {BaseCompiler} from '../base-compiler';

  export class LanguageCompiler extends BaseCompiler {
    static get key() {
      return 'language';
    }
  }
  ```

  - The value returned by `key` above corresponds to the `compilerType` value in
    `etc/config/{language-key}.defaults.properties` (Explained below). This is usually `{language-key}`, but you can use
    whatever fits best
  - Override the `OptionsForFilter` method from the base class
  - Comment out the line saying `fs.remove(result.dirPath);` in base-compiler.js, so the latest CE compile attempt
    remains on disk for you to review
    - Remember to undo this change before opening a PR!
  - For reference, the basic behaviour of BaseCompiler is:
    - make a random temporary folder
    - save example.extension to the new folder, the full path to this is the `inputFilename`
    - the `outputFilename` is determined by the `getOutputFilename()` method
    - execute the compiler.exe with the arguments from `OptionsForFilter()` and adding `inputFilename`
    - be aware that the language class is only instanced once, so storing state is not possible
  - If the compiler has problems with the defaults, you will have to override the `runCompiler()` method too. When
    overriding it, here are some ideas
    - set `execOptions.customCwd` parameter if the working directory needs to be somewhere else
    - set `execOptions.env` parameter if the compiler requires special environment variables
    - manipulate `options`, but make sure the user can still add their own arguments in CE

- Add your `LanguageCompiler` to `lib/compilers/_all.js`, in alphabetical order

- Add a `etc/config/{language-key}.local.properties` file:

  - The syntax for its basic contents is documented in [AddingACompiler.md](AddingACompiler.md)
  - This file is ignored by Git, so that you can test the config locally
  - You should add 1 compiler for the language you want to test
  - Test the command line options of the language compilers outside CE

- Add a new file `etc/config/{language-key}.defaults.properties`. This is where a default configuration will live.

  - Usually, this loads default compilers from their usual paths on a normal installation (Check
    `etc/config/c++.defaults.properties` for an example)

- Of important note, for both files, is to properly define `compilerType` property in the newly added compilers. This
  should equal the value returned by your `LanguageCompiler.key` function

- Running `make dev EXTRA_ARGS="--debug --language {language-key}"` to tell CE to only load your new language

- You can check http://127.0.0.1:10240/api/compilers to be sure your language and compilers are there

- Make an installer in the [infra](https://github.com/compiler-explorer/infra) repository

- Add your language files (`{language-key}.*.properties` and `lib/compilers/{language-key}.js`) to the list in
  `.github/labeler.yml`
