# Running EWAVR compilers using linux docker images

Contact: [Ethan Slattery](https://github.com/CrustyAuklet)

## Prerequisites

To run the IAR compiler for IAR on linux you will need the following things:

- A valid copy of the IAR compiler
- A license for the compiler
- wine set up for win32 and i386

### Getting a copy of IAR

Assuming you already have a valid copy of IAR installed on a Windows machine, you can simply zip the installation
directory. The installation is "portable" for all use cases needed by compiler explorer. The one thing that doesn't work
so far is building a project file. Individual file compilation works well using `iccavr.exe`.

If you installed to the default location this means creating an archive of the directory
`C:\Program Files (x86)\IAR Systems\Embedded Workbench 8.0`

### Compiler license

The machine used for compiling needs to have access to a license, or compilation will fail. Ensure the server running
compiler explorer can access the license server. If you want to use a dongle, then modify the following instructions
accordingly, and good luck.

Note: Each time a compilation happens the machine doing that compilation claims the license for 20 minutes. Each docker
instance counts as a different machine. Because of this it is a good idea to ensure that the machine running IAR is a
long-lived container/VM/machine to prevent using up all licenses and irritating your coworkers.

### Wine

There are many places online to learn how to set up wine. A very simplified setup I often use in a fresh ubuntu docker
container is:

```bash
$ apt-get update
$ dpkg --add-architecture i386
$ apt-get -y update
$ apt-get -y install wine-stable wine32 git curl bzip2 make
$ WINEARCH=win32 winecfg
```

## Setup and Configuration

### Running locally on a Windows workstation or a windows server

Refer to the readme on running [Native on Windows](WindowsNative.md) for general setup of Compiler Explorer and other
compilers. Then add EWAVR as an additional compiler in your `c++.local.properties` file. The example in the compiler
explorer [documentation folder](EWAVR.properties) is written for linux so just modify the paths as needed. The following
properties are the ones that need to be set in addition to the typical settings you do for any compiler.

- `versionRe=^IAR C\/C\+\+ Compiler.*AVR$`
- `isSemVer=true`
- `options=--eec++`
- `supportsDemangler=false`

### Running on a linux server using docker

#### Step 1: provide configuration files, compilers, and libraries

_If you are more experienced with docker this can be done with a volume, but I just use a local folder on my server_

Create a folder on the server `/opt/compiler_explorer`. We are focusing on EWAVR, but you can see we also include
several other embedded compilers, the native compilers, and many open source and internal libraries. The EWAVR specific
portions of `c++.local.properties` is provided in the [example](EWAVR.properties), the rest is generic library and
compiler settings. `compiler-explorer.local.properties` contains server specific sub-domain settings.

The EWAVR directories are just the archives of the installation folders from windows un-tarred there, as you can see
from the layout.

```bash
/opt/compiler_explorer/
├── c++.local.properties
├── compiler-explorer.local.properties
├── compilers
│   ├── avr-gcc-5.4.0-atmel
│   ├── avr-gcc-9.2.0-P0829
│   ├── bin -> /usr/bin
│   ├── ewavr_7108
│   │   ├── avr
│   │   ├── common
│   │   └── install-info
│   ├── ewavr_7205
│   │   ├── avr
│   │   ├── common
│   │   └── install-info
│   ├── gcc-arm-none-eabi-6.3.1-atmel
│   ├── gcc-arm-none-eabi-8-2018-q4-major
│   └── gcc-arm-none-eabi-9-2019-q4-major
└── libs
    ├── bitpacker
    ├── boost_1_72_0
    ├── cunpack
    ├── dyno
    ├── expected
    ├── function_ref
    ├── gsl
    ├── span
    ├── static_string
    ├── static_vector
    ├── tl-optional
    └── xmega-hal
```

#### Step 2: Make iccavr callable

To invoke `iccavr` you need to use the command `wine /opt/compiler_explorer/compilers/ewavr_7108/avr/bin/iccavr.exe`
which created some issues for me in CE. I create small bash script aliases using the following commands, for each
versions `bin` folder:

```bash
$ IAR_ROOT=/opt/compiler_explorer/compilers/ewavr_7108
$ echo -e "#\!/bin/bash\nwine $IAR_ROOT/avr/bin/iccavr.exe\n" > $IAR_ROOT/avr/bin/iccavr && chmod +x $IAR_ROOT/avr/bin/iccavr
$ IAR_ROOT=/opt/compiler_explorer/compilers/ewavr_7205
$ echo -e "#\!/bin/bash\nwine $IAR_ROOT/avr/bin/iccavr.exe\n" > $IAR_ROOT/avr/bin/iccavr && chmod +x $IAR_ROOT/avr/bin/iccavr
```

#### Step 3: Start Compiler Explorer as a docker image

A [docker image](https://hub.docker.com/repository/docker/crustyauklet/compiler_explorer) is available that bundles a
working version of Compiler Explorer and wine so that no additional setup is needed on the host server. The IAR license
needs to be run once, at system startup to register with the license server, so that can be seen in the docker command.
If you have multiple IAR versions, only one needs to be run, and it doesn't matter which one.

```bash
$ IAR_SERVER=my.liscense_server.ip.address
$ docker run --restart=always -d --name compiler_explorer \
  -v /opt/compiler_explorer:/opt/compiler_explorer \
  -v /opt/compiler_explorer/c++.local.properties:/compiler-explorer/etc/config/c++.local.properties \
  -p 80:10240 crustyauklet/compiler_explorer \
  /bin/bash -c "wine /opt/compiler_explorer/compilers/ewavr_7108/common/bin/LightLicenseManager.exe setup -s ${IAR_SERVER} && make"
```
