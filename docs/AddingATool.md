# Adding a new tool

Tools are a way to execute something on your code or the output of a compilation.

Adding tools requires adding configuration to a properties file for a specific language:

```
tools=rewritecpp

tools.rewritecpp.name=rewritecpp
tools.rewritecpp.exe=/opt/compiler-explorer/rewritertool/bin/rewritecpp
tools.rewritecpp.type=independent
tools.rewritecpp.exclude=
tools.rewritecpp.class=base-tool
tools.rewritecpp.stdinHint=disabled
```

The **name** and **exe** are what they say they are, this is the display name for within CE and the tool executable that will be used.

The **type** of the tool represents the stage in which the tool will run:
* independent - when running a tool on sourcecode
* postcompilation - when running a tool on the assembly or a binary

The **exclude** property is to indicate which compilers are proven to be incompatible with the tool.
You can supply the full id of the compiler or a partial id (for example 'arm' to exclude all arm compilers).

The **class** of the tool says which javascript class is needed to run the tool and process its output.

Should you want to deviate from the standard behaviour of base-tool, which runs the tool on the sourcecode filename,
you should add a new class that extends from base-tool.

The **stdinHint** is there to show the user a hint as to what the stdin field is used for in the tool. To disable stdin you can use _disabled_ here.

## compilationInfo

When writing a special class for a tool, you will probably need the `compilationInfo` parameter to pass the correct parameters to the tool.

The contents of `compilationInfo` varies slightly between the different **type**s of tools.

### compilationInfo for independent tools

```json
{
    "backendOptions": {"produceGccDump": {}, "produceCfg": false},
    "compiler": {"id": "clang_trunk", "exe": "clang++", ...},
    "filters": {"binary": false, "commentOnly": true, "demangle": true, ... },
    "inputFilename": "/tmp/ce-tmp/example.cpp",
    "dirPath": "/tmp/ce-tmp",
    "libraries": [{"id": "ctre", "version": "v2"}],
    "options": ["-O3"],
    "source": "int main() {\nreturn 1;\n}\n"
}
```

### compilationInfo for postcompilation tools

```json
{
    ... everything from the compilationInfo for independent tools
    "asm": [
        {"text": "main:", "source": null},
        {"text": "  mov eax, 1", "source": {"file": null, "line": 2}}
        {"text": "  ret", "source": {"file": null, "line": 3}}
    ],
    "asmSize": 123,
    "compilationOptions": ["-O3", "-S", "/tmp/ce-tmp/example.cpp", ...],
    "code": 0,
    "stderr": [
        {"text": "warning: 'x' is used uninitialized in this function [-Wuninitialized]", "tag": {"line": 4, "column": 16}}
    ],
    "stdout": [],
    "outputFilename": "/tmp/ce-tmp/example.o",
    "executableFilename": "/tmp/ce-tmp/a.out"
}
```
