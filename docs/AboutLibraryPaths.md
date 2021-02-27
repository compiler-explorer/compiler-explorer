# About LD_LIBRARY_PATH and library paths

## Background

Libraries are hard. Libraries can be needed for user code, but also to run compilers.

For CE we use a lot of different compilers and environments that need to be separated from the OS's installation, and that makes things more complicated than just building with your standard OS's compiler.

Including header files or equivalent is usually the easy part. If there are binaries involved, things get complicated.

We have a couple of seperate stages where we use and mix different techniques to be able to produce the right assembly or executables.

* Compilation without linking
  * The `LD_LIBRARY_PATH` environment variable is used here to enable the compiler to find the `.so` files that they need to run.
  * If you're running a local installation, this is usually your own systems' `LD_LIBRARY_PATH` plus extra things that CE adds through properties.
  * On godbolt.org we always start with an empty `LD_LIBRARY_PATH` and add what is set in the properties.
* Building an executable or binary
  * We use `-Wl,-rpath=` (or equivalent `rpathFlag`) to force library paths into the executable so that they will always find the same `.so` files no matter where they are run. Usually this also includes lib64 and lib folders that the compiler offers for standard libraries that the toolchain offers.
  * Library paths supplied through `-Wl,-rpath=` for shared libraries will be able to dynamically link to the right architecture's `.so` even if multiple paths are given that contain the same `.so` file.
  * We use `-L` (or equivalent `libpathFlag`) to enable the compiler to find both static (`.a`) and shared (`.so`) libraries.
  * We always add '.' as a path as well because that's where we put libraries that are downloaded from our Conan server.
  * We use `-l` (or equivalent `linkFlag`) to say we want to statically or dynamically link to a named library binary (the compiler and linker decide if it's gonna be static or dynamic).
* Running the executable
  * We use `LD_LIBRARY_PATH` just in case these are dependencies inherited from the compiler - and for the libraries that are used (also just in case).


## Specific properties that are used in certain situations

* Compiler .ldPath
  * is used for `LD_LIBRARY_PATH` to support running the compiler
  * is used for linking (`-Wl,-rpath=` and/or `-L`) during building binaries
  * is used for `LD_LIBRARY_PATH` to enable the users's executable to find `.so` files
* Compiler .libPath
  * is used for linking (`-Wl,-rpath=` and/or `-L`) during building binaries
  * is used for `LD_LIBRARY_PATH` to enable the users's executable to find `.so` files
* Library .libPath
  * is used for linking (`-Wl,-rpath=` and/or `-L`) during building binaries
  * is used for `LD_LIBRARY_PATH` to enable the users's executable to find `.so` files (just in case)


## Example

Say we have the following things in a `c++.local.properties` file:

```
compilers=mycl
compiler.mycl.exe=/home/ubuntu/mycl/bin
compiler.mycl.ldPath=/home/ubuntu/mycl/lib/lib64
compiler.mycl.libPath=/home/ubuntu/mycl/lib/lib64:/home/ubuntu/mycl/lib/lib32
compiler.mycl.options=--gcc-toolchain=/home/ubuntu/gcc10
compiler.mycl.includeFlag=-I

libs=mylib
libs.mylib.name=My library
libs.mylib.path=/home/ubuntu/mylib/include
libs.mylib.libpath=/home/ubuntu/mylib/lib
libs.mylib.staticliblink=mylib
```

This will result in the following situations if we want to compile some code with both the mycl compiler and the mylib library:

* Compilation without linking
  * `LD_LIBRARY_PATH` is set to `/home/ubuntu/mycl/lib/lib64`
  * `-I/home/ubuntu/mylib/include` is added to the compilation arguments
* Building an executable or binary
  * `LD_LIBRARY_PATH` is set to `/home/ubuntu/mycl/lib/lib64`
  * The following are added to the compilation arguments
    * `-I/home/ubuntu/mylib/include` (library include path)
    * `-Wl,-rpath=/home/ubuntu/mycl/lib/lib64` (compiler library paths)
    * `-Wl,-rpath=/home/ubuntu/mycl/lib/lib32`
    * `-Wl,-rpath=.` (conan library path)
    * `-L.`
    * `-Wl,-rpath=/home/ubuntu/gcc10/lib/lib` (gcc toolchain library paths)
    * `-Wl,-rpath=/home/ubuntu/gcc10/lib/lib32`
    * `-Wl,-rpath=/home/ubuntu/gcc10/lib/lib64`
    * `-Wl,-rpath=/home/ubuntu/mylib/lib` (mylib library path - just in case there are `.so` files used)
    * `-L/home/ubuntu/mylib/lib` (mylib library path used to find `libmylib.a`)
    * `-lmylib` (mylib library name)
* Running the executable
  * `LD_LIBRARY_PATH` is set to `/home/ubuntu/mycl/lib/lib64:/home/ubuntu/mycl/lib/lib32:/home/ubuntu/mylib/lib`
