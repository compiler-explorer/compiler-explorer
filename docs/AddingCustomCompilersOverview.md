This is an overview of what building and adding a compiler to the site looks like, using GCC as an example, but note
that the process is similar for all other types of compilers.

- Build GCC versions with the relevant magic. We build all our own GCCs using Docker from
  [this repo](https://github.com/compiler-explorer/gcc-builder)
- For non-trunk builds we manually run the docker image. For trunk dailies we have a
  [cron job](https://github.com/compiler-explorer/infra/blob/main/crontab.admin#L8) that runs a
  [script](https://github.com/compiler-explorer/infra/blob/main/admin-daily-builds.sh) to build them.
- Built tarballs are uploaded to S3, and installed by our
  [custom tool](https://github.com/compiler-explorer/infra/blob/main/bin/lib/ce_install.py) from a
  [YAML configuration file](https://github.com/compiler-explorer/infra/blob/main/bin/yaml/cpp.yaml)
- The installation puts compilers on a shared NFS drive at `/opt/compiler-explorer/gcc-some-version/`
- We then configure CE to look for the compiler
  [here](https://github.com/compiler-explorer/compiler-explorer/blob/main/etc/config/c%2B%2B.amazon.properties#L9).
- If we need to customise the way we execute the compiler and/or display the results, then we can change the
  "[driver](https://github.com/compiler-explorer/compiler-explorer/tree/main/lib/compilers)" for the compiler. Usually
  we can just override a few aspects of the driver, relying on the defaults from the
  [base driver](https://github.com/compiler-explorer/compiler-explorer/blob/main/lib/base-compiler.ts).
- Any UI changes are a bit more work.

More info still in
[Adding a Compiler](https://github.com/compiler-explorer/compiler-explorer/blob/main/docs/AddingACompiler.md), and if
you can bear listening to Matt, here's [a talk](https://www.youtube.com/watch?v=kIoZDUd5DKw) about some behind the
scenes stuff, with [slides online](https://www.youtube.com/watch?v=kIoZDUd5DKw).
