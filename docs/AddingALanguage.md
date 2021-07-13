# Adding a new language

* Add a `etc/config/language.defaults.properties` file
* Add a `etc/config/language.local.properties` file with the first compiler
  - make sure the path is correct, and during launch of CE (`make dev`) check to see if CE agrees with you
  - test the command line options of the language compilers outside CE
* Add language to `lib/languages.js`
  - if you use a built-in monaco language you must also add it to the list of languages inside the `MonacoEditorWebpackPlugin` config in `webpack.config.js`
  - if you don't use a built-in monaco language you will need to implement your own language mode; see `static/modes/asm-mode.js` as an example
* Add a `lib/compilers/language.js` file using the template below, replacing `Language` and `language` as appropriate
    ```js
    import { BaseCompiler } from '../base-compiler';

    export class LanguageCompiler extends BaseCompiler {
        static get key() { return 'language'; }
    }
    ```
  - the value returned by `key` above corresponds to the `compilerType` value in `etc/config/language.defaults.properties`
  - implement the `OptionsForFilter` method
  - comment out the line saying `fs.remove(result.dirPath);` in base-compiler.js, so the latest CE compile attempt remains on disk for you to review
     - remember to undo this change before opening a PR
  - for reference, the basic behaviour of the BaseCompiler is:
     - make a random temporary folder
     - save example.extension to the new folder, the full path to this is the **inputFilename**
     - the **outputFilename** is determined by the `getOutputFilename()` method
     - execute the compiler.exe with the arguments from `OptionsForFilter()` and adding **inputFilename**
     - be aware that the language class is only instanced once, so storing state is not possible
  - if the compiler has problems with the defaults, you will have to override the `runCompiler()` method
  - when overriding `runCompiler()`, here are some ideas:
     - set **execOptions.customCwd** parameter if the working directory needs to be somewhere else
     - set **execOptions.env** parameter if the compiler requires special environment variables
     - manipulate **options**, but make sure the user can still add their own arguments in CE
  - test with `node app.js --debug` so you see all execution arguments
* Add your `LanguageCompiler` to `lib/compilers/_all.js`, in alphabetical order
* Add `static/modes/language-mode.js` and *require* it in `static/panes/editor.js`

* You can check http://127.0.0.1:10240/api/compilers to be sure your language and compiler are there

* Make an installer in the [infra](https://github.com/compiler-explorer/infra) repository
