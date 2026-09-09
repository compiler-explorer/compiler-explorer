# Local Native Image assembly exploration

The `native-image` compiler adapter compiles Java or Kotlin to class files, displays their JVM bytecode in a separate pane, and extracts Native Image machine code before image creation. It does not build or execute an executable or shared library.

This integration targets **Linux x86-64 and Oracle GraalVM 25.3.4.1 (innovation)**, based on JDK 25.0.4.1. The JDK version at the start of the `native-image --version` output is distinct from the GraalVM version on the following lines. Older GraalVM releases are not currently supported. The extraction feature uses internal hosted APIs; rebuild it and the base layer with the same GraalVM installation used by the compiler.

## Build the extraction feature

Install a GraalVM distribution containing `native-image`, GNU `objdump`, and the C build toolchain required by Native Image. For Kotlin, install the Kotlin/JVM compiler and its standard library.

From the repository root:

```sh
bash etc/scripts/native-image/build.sh /path/to/graalvm "$PWD/out/native-image"
```

The output is `out/native-image/explorer-feature.jar`, with the toolchain version recorded alongside it. No Maven or GraalVM source build is required.

## Configure CE

Add the following to `etc/config/java.local.properties`, substituting absolute paths. If the file already defines compilers, append `javanativeimage25` to its `compilers` list rather than replacing it.

```properties
compilers=javanativeimage25
defaultCompiler=javanativeimage25
compiler.javanativeimage25.exe=/path/to/graalvm/bin/native-image
compiler.javanativeimage25.name=GraalVM 25 Native Image (Java, local)
compiler.javanativeimage25.compilerType=native-image
compiler.javanativeimage25.frontend=/path/to/graalvm/bin/javac
compiler.javanativeimage25.nativeImageFeature=/path/to/compiler-explorer/out/native-image/explorer-feature.jar
compiler.javanativeimage25.objdumper=/usr/bin/objdump
compiler.javanativeimage25.instructionSet=amd64
compiler.javanativeimage25.versionFlag=--version
compiler.javanativeimage25.options=-O2
compiler.javanativeimage25.supportsExecute=false
compiler.javanativeimage25.interpreted=false
compiler.javanativeimage25.nativeImageTimeoutMs=120000
```

For Kotlin, use the equivalent entries in `etc/config/kotlin.local.properties` with a distinct ID such as `kotlinnativeimage25`, and change/add:

```properties
compiler.kotlinnativeimage25.frontend=/path/to/kotlinc/bin/kotlinc
compiler.kotlinnativeimage25.nativeImageClasspath=/path/to/kotlinc/lib/kotlin-stdlib.jar
```

The adapter takes `javap` from the Native Image executable's directory by default. Override it with `compiler.ID.javap` if necessary. `nativeImageClasspath` is a platform-separated list of dependency JARs used by both frontend and Native Image compilation.

Run CE with a supported Node.js version:

```sh
npm ci
npm run dev -- --language java kotlin --host 127.0.0.1 --port 10240
```

Open <http://127.0.0.1:10240>, select the language and Native Image compiler, then select **Add new… → JVM Bytecode** in the compiler pane. The main pane displays native assembly; the bytecode pane displays the actual class files supplied to Native Image. Both outputs carry source-line information and colour highlighting. Compilation also works through the normal `/api/compiler/ID/compile` route: the response adds `jvmBytecodeOutput`, an array of assembly-style lines with `text` and optional `source` properties. The compiler advertises `supportsJvmBytecodeView`.

Java example (the usual single-file `javac` naming rules apply):

```java
class Square {
    static int square(int num) {
        return num * num;
    }
}
```

Kotlin example:

```kotlin
fun square(num: Int): Int = num * num
```

Neither example needs `main`, native annotations or a generated call with constant arguments.

## Options and output

The arguments field accepts `-O0`, `-O1`, `-O2`, `-O3`, `-Ob` and `-Os` (availability still depends on GraalVM). Put frontend flags after `--`, for example `-O2 -- -parameters` for Java or `-O2 -- -Xno-param-assertions` for Kotlin. The initial frontend allowlist consists of `-g`/`-g:…`, `-parameters`, `-nowarn`, `-Werror`, `-Xlint`/`-Xlint:…`, `-java-parameters`, `-Xno-param-assertions`, and `-Xno-call-assertions`. Flags for the wrong frontend produce that compiler's normal diagnostics.

The target is generic x86-64. The Intel syntax toggle is supported. Cosmetic filters unsupported by this adapter are disabled. Native Image runtime checks remain visible. Only methods belonging to submitted classes are displayed, including constructors and compiler-generated methods. Native Image may inline calls while retaining separately rooted method bodies.

The assembly is **unlinked code**. Direct call and runtime-branch placeholders are replaced with symbolic targets. Indirect calls retain their register/memory operands. Data patches and image-heap constants are marked unresolved; opcode bytes at relocation sites remain placeholders. Branches within a method use local labels. The bytes are for inspection and cannot be executed independently.

## Implementation and limits

`ExplorerFeature.beforeAnalysis` discovers submitted classes and registers their concrete methods and constructors as roots. Concrete submitted classes are registered as potentially allocated. This defines an exploration context: it may differ from the reachability and optimisation assumptions of a complete application.

`afterCompilation` exports method bytes, call/data patches and source-position ranges to a versioned JSON manifest. It then throws Native Image's internal `InterruptImageBuilding` exception. This occurs after method layout but before image-heap construction and image writing. The adapter requires both successful process completion and a valid manifest. Bytecode remains available if the subsequent native stage fails.

The manifest is disassembled with GNU objdump; runtime references are annotated from the metadata. Source mapping is limited to submitted classes. Kotlin SMAP interpretation and complete inline stacks are not implemented, so inlined Kotlin source mappings may be incomplete. Reflection, native methods, missing dependencies and other Native Image compatibility restrictions can still prevent compilation. This is not Kotlin/Native and does not use the HotSpot JIT backend.

The feature/JAR dependency digest and frontend version are included in compiler version metadata for caching. Restart CE after rebuilding the feature or changing toolchains/classpath. The defaults are two compilation threads and a 3 GiB builder heap; the builder JVM can use additional threads and resident memory. Small examples took roughly 40–50 seconds on the development host. The timeout covers the frontend, bytecode inspection, Native Image and disassembly stages. Disable automatic compilation while editing if desired.

**Use this configuration locally.** Native Image can execute application class initialisers during building. Production deployment needs appropriate compilation sandbox configuration for the GraalVM builder and its toolchain, worker memory/concurrency budgets, and separate operational validation. The adapter uses CE's existing compiler execution path; it does not create a separate sandbox or a persistent builder process.

## Validation

```sh
npm run ts-check
npm run lint
npm run test-min
npm run test:props
```

`test/native-image-tests.ts` covers paired output, frontend failure, native timeouts, manifest validation, option restrictions and relocation rendering without requiring GraalVM. A real local smoke test should compile the Java and Kotlin examples above and open the JVM Bytecode pane. Rebuild the feature whenever its Java source changes.

## Resource tuning

Set builder resources per compiler in the server properties, then restart CE:

```properties
compiler.javanativeimage25.nativeImageParallelism=4
compiler.javanativeimage25.nativeImageMaxHeap=6g
```

Use the Kotlin compiler ID for its settings. Parallelism must be a positive integer;
heap sizes accept a positive integer with a `k`, `m` or `g` suffix. These settings
are controlled by the server, rather than the user arguments field.

On the deployment host, compare the existing `2` / `3g` baseline with `4` / `6g`
and `8` / `6g`, keeping optimisation at `-O2`. Run Java and Kotlin snippets with
arithmetic, library calls, allocation and exceptions. Record total latency and
Native Image stage timings for repeated uncached builds, plus peak resident memory.
Repeat with simultaneous requests at the intended service concurrency; allow memory
for the builder outside its Java heap and CPU for CE and the frontend. Avoid serving
cached compilation responses when measuring. Choose production defaults from those
measurements rather than CPU count alone. `-Ob` changes optimisation and should be
measured separately as an explicit quick-build choice.

## Reusable JDK layer

Build a base layer once with the same GraalVM installation used by CE:

```sh
bash etc/scripts/native-image/build-layer.sh /path/to/graalvm "$PWD/out/native-image-layer"
```

The script compiles `java.base` at `-O2` for generic x86-64 and writes `base.nil`.
It uses four threads and a 6 GiB builder heap. The layer build must finish normally;
it does not use the extraction feature's early exit. Both Java and Kotlin snippets
can use this JDK layer; Kotlin still supplies its standard library on the classpath.

Configure a compiler with the absolute path to the layer, then restart CE:

```properties
compiler.javanativeimage25.nativeImageLayer=/path/to/compiler-explorer/out/native-image-layer/base.nil
```

Use the corresponding Kotlin compiler ID for Kotlin. The adapter passes
`-H:LayerUse` to Native Image and still extracts only submitted methods before image
creation. An empty or absent property keeps ordinary builds. The layer contents
are included in the compiler version digest for caching; restart CE after replacing
a layer. The builder must be able to read the archive inside its compilation sandbox.
GraalVM checks layer compatibility; rebuild the layer when changing toolchains or
incompatible build options. Do not disable these checks to reuse an incompatible layer.

Give layered compiler entries a distinct name such as **GraalVM 25 Native Image
(Java, JDK layer)**. Layered output can differ from ordinary output even at `-O2`:
runtime calls can become indirect and library inlining can change. For example,
the tested string-concatenation method became a call to a precompiled helper.
Keep an ordinary compiler entry available when comparing assembly.

### GCP measurements

On `ce-native-image` (8 vCPUs, 31 GiB RAM), two uncached runs per configuration
with four builder threads, a 6 GiB heap and `-O2` gave these mean times:

| Snippet | Ordinary build | With JDK layer |
| --- | ---: | ---: |
| Java | 21.55 s | 17.12 s |
| Kotlin | 23.56 s | 19.80 s |

Each snippet covered arithmetic, an array loop, string concatenation and an
exception. Times include frontend compilation, bytecode inspection and native
extraction, but exclude CE/HTTP overhead and disassembly. These are small pilot
measurements, not a concurrency or production load benchmark. The layer took
3m 8s to create and occupies 852 MiB. Layered Java code generation took 0.1 s;
initialisation, layer loading and analysis now account for most of the latency.
Both paths exported the same submitted methods and source-line sets, but generated
different machine code because of layer boundaries.
