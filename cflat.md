> [name=Arbie, Yin, Guo, Ashima]

## Information about the project

### What is compiler explorer?

Compiler Explorer is an interactive compiler exploration website. Edit code in C, C++, C#, F#, Rust, Go, D, Haskell,
Swift, Pascal, ispc, Python, Java, or any of the other 30+ supported languages, and see how that code looks after being
compiled in real time. Multiple compilers are supported for each language, many different tools and visualizations are
available, and the UI layout is configurable (thanks to GoldenLayout).

### What is our project is about?

This project focuses on enhancing Compiler Explorer (CE) by adapting it to support in-browser compilers. The primary
objectives include restructuring the backend by removing the original CE server and integrating WebAssembly (Wasm) code
for compiler execution. Additionally, the project aims to add a new feature of running different compiler passes so that
users can better understand the compiler process. Furthermore, the project implements another functionality to load
different compilers.

## Build process

### installation

```bash=
pip install node
node --version
v18.16.0

pip install npm
npm --version
9.5.1
```

## (run the cflat part first)

(need to be edited after Cflat's part is done)

### Run the program

```bash
cd compiler-explorer/

# run
make dev EXTRA_ARGS="--language cflat"

# run the developer version
make dev EXTRA_ARGS="--debug --language cflat"
```

## Expectation

With or without debuggers, this result is expected.

```bash=
# Getting info
info: OPTIONS HASH: 38a9453ee62d473c143f3d8556649e1f56352c9834481202cebad762e62cb966
info:   using webpack dev middleware
webpack: compiling for development.
webpack: Limiting parallelism to 5
info:   Listening on http://localhost:10240/
info:   Startup duration: 5705ms
info: =======================================
```

If you get errors like below, It is normal

```bash=
error: Unable to resolve 'cflat01'
error: Unable to resolve 'cflat02'
error: Unable to resolve 'cflat'
```

- Open the [local webpage](http://localhost:10240/) in any browser
- See the wepage like this ![](https://hackmd.io/_uploads/HJhctvFo3.png)
- If you got `<Compilation failed: unreachable>` on the right hand side (as shown below), try to add a new line at the
  very end of your input. ![](https://hackmd.io/_uploads/r1ILcwFi3.png)

## Test

### Run the test

```bash=
make test
```

## What to expect

The terminal is going to run more than 520 tests, and Cflat compiler contains 2 tests among them. If you find them
paased as shown below, you are good to go.

```bash=
...
# The cflat part
  Basic compiler setup
    ✔ Should not crash on instantiation

  cflatp compiling
    ✔ Compiles a simple LIR program
...
```
