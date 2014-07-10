GCC Explorer
------------

GCC Explorer is an interactive compiler. The left-hand pane shows editable C/C++ code. The right, the assembly output of having compiled the code with a given compiler and settings.

Try out the [demo site][demo]!

[demo]: http://gcc.godbolt.org/

### Developing

GCC Explorer is written in node. Most of the heavy lifting is actually done on the client, which is arguably a bad decision.

Assuming you have npm and node installed, simply running `make` ought to get you up and running with a GCC explorer running on port 10240 on your local machine: http://localhost:10240

If you want to point it at your own GCC or similar binaries, either edit the `etc/config/gcc-explorer.defaults.properties` or else make a new one with the name `gcc-explorer.YOURHOSTNAME.properties`.  The config system leaves a lot to be desired, I'm working on porting [CCS](https://github.com/hellige/ccs-cpp) to javascript and then something more rational can be used.

Feel free to raise an issue on github or email me directly for more help.
