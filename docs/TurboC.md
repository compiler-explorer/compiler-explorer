# Running Turbo C

Instructions on how to run Turbo C using Dosbox on Linux.

## Prerequisites

To run the Turbo C compiler you will need:

- Dosbox - the easiest way to install is to use your OS's package manager, e.g. `sudo apt install dosbox`
- Turbo C installation - if you have the 3 installation disks, first install those to a single directory with dosbox
  - You will need to setup a directory to function as the `C` drive with a `TC` directory inside.
  - Note that it's assumed all files are in uppercase

## Configuration

In the `turboc.properties` file you can see an example on how to setup the compiler to work with Compiler Explorer.

Make sure the `.dosbox` path is correct, as well as the `.root` and `.exe`. The `.root` indicates the root of the `C`
drive, and the `.exe` points to the actual `TCC.EXE`

## More notes

Note that Turbo C is C only, so it belongs in your `c.local.properties`.

Also note that you will immediately get an error with the default example source code, because the compiler doesn't
support `//` comments.

Also note that in this old C version, you must declare all variables in the first few lines of your functions.
