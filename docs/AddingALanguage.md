# Adding a new language

* Add a etc/config/language.defaults.properties file
* Add a etc/config/language.local.properties file with the first compiler
  - make sure the path is correct, and during launch of CE (node app.js) check to see if CE agrees with you
  - test the command line options of the language compilers outside of CE
* Add language to lib/languages.js
  - start with the basics, *class LanguageCompiler extends BaseCompiler* and implement the `OptionsForFilter` method
  - comment out the line saying `fs.remove(result.dirPath);` in base-compiler.js, so the latest CE compile attempt remains on disk for you to review
     - remember to undo this change before opening a PR
  - for reference, the basic behaviour of the BaseCompiler is:
     - make a random temporary folder
     - save example.extension to the new folder, the full path to this is the **inputFilename**
     - the **outputFilename** is determined by the `getOutputFilename()` method
     - execute the compiler.exe with the arguments from `OptionsForFilter()` and adding **inputFilename**
     - be aware that the the language class is only instanced once, so storing state is not possible
  - if the compiler has problems with the defaults, you will have to override the `runCompiler()` method
  - when overriding `runCompiler()`, here are some ideas:
     - set **execOptions.customCwd** parameter if the working directory needs to be somewhere else
     - set **execOptions.env** parameter if the compiler requires special environment variables
     - manipulate **options**, but make sure the user can still add their own arguments in CE
  - test with node app.js --debug so you see all execution arguments

* Add static/modes/language-mode.js and *require* it in static/panes/editor.js

* Add a basic lib/compilers/language.js (and reference to it in etc/config/language.defaults.properties as the **compilerType**)

* You can check http://127.0.0.1:10240/api/compilers to be sure your language and compiler are there

* Make an installer in the [infra](https://github.com/compiler-explorer/infra) repository
