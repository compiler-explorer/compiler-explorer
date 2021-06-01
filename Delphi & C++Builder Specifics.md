```The original basis document:  https://github.com/compiler-explorer/compiler-explorer/blob/main/docs/WindowsNative.md ```

Installation:
- Install latest Node.js  ( my current = v14.17.0 )
- Install latest npm      ( my current = 7.14.0 )

- Clone the main Compiler Explorer repository into a directory : https://github.com/compiler-explorer/compiler-explorer 

In the directory, run :
- npm install
- npm install webpack -g
- npm install webpack-cli -g
- npm update webpack

Substitute the following files from this repo :  (make changes to directories and installed versions as necessary)
- \lib\languages.js   -  Comment out languages as desired.  

For Object Pascal compilers :  
- \lib\compilers\pascal.js
- \lib\compilers\pascal-win.js
- \etc\config\pascal.defaults.properties

For C++ compilers :
- \etc\config\c++.win32.properties
- \etc\config\c++.local.properties

Finally : 
- npm start 

To access Compiler Explorer, browse to : (http://localhost:10240/)

//   ---   ---   ---   ---   ---   ---   ---   ---   
![Install](https://user-images.githubusercontent.com/11953157/120332379-455d1f80-c321-11eb-85ae-e9cd31cc9814.png)
//   ---   ---   ---   ---   ---   ---   ---   ---   
![Start Server](https://user-images.githubusercontent.com/11953157/120332478-5d34a380-c321-11eb-9bd5-a2a86447e963.png)
//   ---   ---   ---   ---   ---   ---   ---   ---   
![Compiler Explorer - Object Pascal](https://user-images.githubusercontent.com/11953157/120333650-7b4ed380-c322-11eb-8042-5b1d710a5814.png)
//   ---   ---   ---   ---   ---   ---   ---   ---   
![Compiler Explorer - C++](https://user-images.githubusercontent.com/11953157/120351986-eaccbf00-c332-11eb-939c-a0338e333945.png)
