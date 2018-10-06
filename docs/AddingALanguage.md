# Adding a new language

* Add a etc/config/language.defaults.properties file
* Add a etc/config/language.local.properties file with the first compiler
  - make sure the path is correct, and during launch of CE (node app.js) check to see if CE agrees with you
  - test the command line options of the language compilers outside of CE
* Add language to lib/languages.js
  - start with the basics, *class LanguageCompiler extends BaseCompiler* and implement the **OptionsForFilter** method
  - comment out the line saying `fs.remove(result.dirPath);` in base-compiler.js, so the latest CE compile attempt remains on disk for you to review
  - for reference, the basic behaviour of the BaseCompiler is:
     - make a random temporary folder
     - save example.extension to the new folder, the full path to this is the **inputFilename**
     - the **outputFilename** is determined by the `getOutputFilename()` method
     - execute the compiler.exe with the arguments -S inputFilename outputFilename
  - if the compiler has problems with various paths, you could try to:
     - override the `runCompiler()` method and add a **customCwd** parameter to execOptions
     - add an **env** parameter if it requires special environment variables
  - test with node app.js --debug so you see all execution arguments

* Add a basic lib/compilers/language.js (and reference to it in etc/config/language.defaults.properties as the **compilerType**)

* You can check http://127.0.0.1:10240/api/compilers to be sure your language and compiler are there

* Make an installer on [compiler-explorer-image](https://github.com/mattgodbolt/compiler-explorer-image)
