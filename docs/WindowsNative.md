# Running on Windows

Contact: [Nicole Mazzuca](https://github.com/ubsan)

## Basic Setup

The setup on Windows should be fairly trivial:
the only prerequisite is node.
If you haven't yet installed node yet, you can grab it from
[here](https://nodejs.org/en/);
get the Windows LTS release.

Once you've done this,
and added `npm` to the path,
run the following commands from any command line,
in the directory you want the Compiler Explorer (from here on, CE)
to live:

```bat
git clone https://github.com/mattgodbolt/compiler-explorer.git
```

Then, we'll need to make a configuration file
which points at your compilers and include directories.
Copy `docs\WindowsLocal.properties` to a new file,
`etc\config\c++.local.properties`, and edit it,
following the instructions in the comments.
If you have any questions, please ping me on discord.


## Actually Running the danged thing

Once you've finished setting it up,
you can `cd` into the `compiler-explorer` directory,
then run

```bat
npm install yarn
node_modules\.bin\yarn install
node_modules\.bin\yarn start
```

Eventually, you'll see something that looks like

```
info: =======================================
info:   git release 96451ae8b92e420462137eaaec58f78d3cd6667b
info:   serving static files from 'static'
info:   Listening on http://localhost:10240/
info: =======================================
```

Now point your favorite web browser at http://localhost:10240
and you should be done!

You only have to run `yarn install` the first time;
every time after that, you should just be able to run `yarn start`.

### Current Limitations

  - Execution support doesn't yet exist
  - Binary disassembly doesn't work yet
