The library build status is accessible from https://conan.compiler-explorer.com/

# build prerequisites

The build process of libraries (started by https://github.com/compiler-explorer/infra/blob/main/admin-daily-builds.sh)
has a couple of safeguards to not build libraries allday everyday, namely:

* The compiler needs to have supportBinary=true on in the current https://github.com/compiler-explorer/compiler-explorer/blob/main/etc/config/c%2B%2B.amazon.properties
* If all libraries are being built, only a commit hash change will result in a new build.
* If a particular compiler has failed to produce a valid build, it will be marked and will not build the library with this compiler anymore.
* There are a couple of hardcoded exceptions that never attempt the build, these can be found in https://github.com/compiler-explorer/infra/blob/main/bin/lib/library_builder.py#L29
* You can manually trigger a forced-build, by setting the `--build-for=compilerid` parameter, which can also be set by changing the 2nd parameter here -> https://github.com/compiler-explorer/infra/blob/main/admin-daily-builds.sh#L104
* If you want to trigger all compilers to rebuild a certain library, you will need to manually delete the logging related to this library.


# manually delete logging

You have to login to the conan instance and delete records from the sqlite database responsible for keeping the logging.

Example:
```
ce conan login
sudo -u ce sqlite3 /home/ce/.conan_server/buildslogs.db
delete from latest where library='unifex' and success=0;
```

You can find more info about the database in https://github.com/compiler-explorer/conanproxy/blob/main/build-logging.js
