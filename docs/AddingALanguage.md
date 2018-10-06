# Adding a new language

* add a etc/config/language.defaults.properties file
* add a etc/config/language.local.properties file with the first compiler
  - make sure the path is correct, and pay attention if CE agrees with you
* add language to lib/languages.js
  - start with the basics, class LanguageCompiler extends BaseCompiler and implement the OptionsForFilter method
  - test the command line options outside of CE
  - Comment out the line saying "fs.remove(result.dirPath);" in base-compiler.js, so the latest CE compile attempt remains on disk for you to review
  - if the compiler has problems taking different paths, override the runCompiler method and add a customCwd parameter to execOptions
  - test with node app.js --debug so you see all execution arguments and outputs

* add a basic lib/compilers/language.js (and reference to it in etc/config/language.defaults.properties as the compilerType)

* you can check http://127.0.0.1:10240/api/compilers to be sure your language and compiler are there

* make an installer on compiler-explorer-image

