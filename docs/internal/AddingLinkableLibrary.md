Adding a library that is linked against:

- If the library is a C shared library, you should not use these instructions, use the `after_staging_script` instead to build the library during the installation
- Check what buildsystem is used.
  - If CMake, it's going to be relatively easy.
  - If something else, it's going to be complicated.
- If CMake
  - If the CMakeLists.txt is in the root folder, everything should be work
  - However, at least prefer to include make_targets property in the yaml to avoid any automagick confusion (it would test library name in all kinds of variants and when all fails do make all)
    - If there are multiple ways of generating libraries (shared/static), make a choice here

1. Add to bin/yaml/libraries.yaml
 - There's a difference between nightly trunks' and versions, search for "nightly:" to find the nightlies
 - Note: the name of the library in the yaml file needs to be the same as the name in the c++.amazon.properties file
 - Add basic entry
 ```
      unifex:
        type: github
        method: nightlyclone
        repo: facebookexperimental/libunifex
        build_type: cmake
        make_targets:
          - unifex
        targets:
          - trunk
```
2. Test installing
 - `bin/ce_install --enable nightly install 'unifex'`
3. Test building
 - Make sure you have a compiler installed, for example via `bin/ce_install install 'gcc 10.1.0'`
 - `bin/ce_install --enable nightly --buildfor g101 --dry build 'unifex'`
 - check one of the buildfolders that are created and see if there are .so's or .a's and otherwise check the cecmakelog.txt and cemakelog_X.txt
 - Iterate over this to make it work
4. If a static link file has been produced:
 - Add to the `c++.amazon.properties` in compiler-explorer in the libs properties for the new library a `libs.libraryname.staticliblink=libraryname`
   - libraryname here is without the 'lib' prefix of the .a file
 - Example unifex
```
libs.unifex.name=libunifex
libs.unifex.versions=trunk
libs.unifex.staticliblink=unifex
libs.unifex.versions.trunk.version=trunk
libs.unifex.versions.trunk.path=/opt/compiler-explorer/libs/unifex/trunk/include
```
5. If a config header file has been generated based on the compiler configuration, we can only maybe support this if the header does NOT include any defines about the architecture.
6. Send PR's
7. Merge the amazon.properties to main
8. Start library builder for the new library until no later than 00:00 UTC
9. Await and check libraries @ https://conan.compiler-explorer.com/libraries.html and logs @ https://conan.compiler-explorer.com/failedbuilds.html
  - These new libraries won't show up on these pages until you do https://conan.compiler-explorer.com/reinitialize and then go to https://conan.compiler-explorer.com/libraries and hit refresh

