# Adding a new tool

Tools are a way to execute something on your code or the output of a compilation.

Adding tools requires adding configuration to a properties file for a specific language:

```INI
tools=rewritecpp

tools.rewritecpp.name=rewritecpp
tools.rewritecpp.exe=/opt/compiler-explorer/rewritertool/bin/rewritecpp
tools.rewritecpp.type=independent
tools.rewritecpp.exclude=
tools.rewritecpp.class=base-tool
tools.rewritecpp.stdinHint=disabled
tools.rewritecpp.monacoStdin=false
tools.rewritecpp.languageId=cppp
tools.rewritecpp.options=--a
tools.rewritecpp.args=--b
```

The `name` and `exe` are what they say they are, this is the display name for within CE and the tool executable that
will be used.

The `type` of the tool represents the stage in which the tool will run:

- independent - when running a tool on sourcecode
- postcompilation - when running a tool on the assembly or a binary

The `exclude` property is to indicate which compilers are proven to be incompatible with the tool. You can supply the
full id of the compiler, or a partial id (for example 'arm' to exclude all arm compilers).

The `class` of the tool says which javascript class is needed to run the tool and process its output. The folder
_lib/tooling_ is used for these classes.

Should you want to deviate from the standard behaviour of `base-tool`, which runs the tool on the sourcecode filename,
you should add a new class that extends from `base-tool`.

The `stdinHint` is there to show the user a hint as to what the stdin field is used for in the tool. To disable stdin
you can use _disabled_ here.

The `monacoStdin` option makes the stdin editor a separate pane containing a monaco editor. This is useful when a tool
has complex input spanning multiple lines and it's more friendly to indent it.

The `languageId` can be used to highlight the output of the tool according to a language known within CE. For example
`cppp` will highlight c++ output. Leaving `languageId` empty will use the terminal-like output.

The `options` field is useful for tools that derive `base-tool` and want to add non-user configurable options to it

The `args` field is shown and editable by the user in the UI, and passed automatically to the tool

# compilationInfo

When writing a special class for a tool, you will probably need the `compilationInfo` parameter to pass the correct
parameters to the tool.

The contents of `compilationInfo` varies slightly between the different `type`s of tools.

## compilationInfo for independent tools

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

The `filters` can be used to assert boundary conditions or adjust the tooling process based on the filters the user
checked on or off.

The `inputFilename` contains the path to the sourcecode stored on disk. The `source` contains the sourcecode as text.

The `dirPath` can be used to write extra files to disk which the tool might need.

The `options` are the arguments the user gave for the compilation.

## compilationInfo for postcompilation tools

```json
{
    ... everything from the compilationInfo for independent tools
    "compilationOptions": ["-O3", "-S", "/tmp/ce-tmp/example.cpp", ...],
    "code": 0,
    "asm": [
        {"text": "main:", "source": null},
        {"text": "  mov eax, 1", "source": {"file": null, "line": 2}}
        {"text": "  ret", "source": {"file": null, "line": 3}}
    ],
    "asmSize": 123,
    "stderr": [
        {"text": "warning: 'x' is used uninitialized in this function [-Wuninitialized]", "tag": {"line": 4, "column": 16}}
    ],
    "stdout": [],
    "outputFilename": "/tmp/ce-tmp/example.o",
    "executableFilename": "/tmp/ce-tmp/a.out"
}
```

`code` indicates the exitcode of the compilation. Usually, 0 means everything's ok.

`asm` contains the returned assembly. This is the same assembly that is shown within compiler-explorer, including extra
information like for which sourcecode line the assembly was generated.

`stderr` and `stdout` contain the different outputs from the compilation process.

The `outputFilename` is always filled, but not guaranteed to exist, for example when the compilation has failed.

The `executableFilename` is always filled, but does not guarantee the existence of an executable.
