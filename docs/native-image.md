# Local Native Image assembly exploration

The `native-image` compiler adapter compiles Java or Kotlin to class files, displays their JVM bytecode in a separate pane, and extracts Native Image machine code before image creation. It does not build or execute an executable or shared library.

This is an experimental local integration for **Linux x86-64 and GraalVM JDK 25**. It has been exercised with Oracle GraalVM 25.0.4.1 / 25.3.4.1+1.1. The extraction feature uses internal hosted APIs; rebuild it with the same GraalVM installation used by the compiler. Other releases and distributions may require changes to the feature or module exports.

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

The feature/JAR dependency digest and frontend version are included in compiler version metadata for caching. Restart CE after rebuilding the feature or changing toolchains/classpath. The initial local configuration uses two compilation threads and a 3 GiB builder heap; the builder JVM can use additional threads and resident memory. Small examples took roughly 40–50 seconds on the development host. The timeout covers the frontend, bytecode inspection, Native Image and disassembly stages. Disable automatic compilation while editing if desired.

**Use this configuration locally.** Native Image can execute application class initialisers during building. Production deployment needs appropriate compilation sandbox configuration for the GraalVM builder and its toolchain, worker memory/concurrency budgets, and separate operational validation. The adapter uses CE's existing compiler execution path; it does not create a separate sandbox or a persistent builder process.

## Validation

```sh
npm run ts-check
npm run lint
npm run test-min
npm run test:props
```

`test/native-image-tests.ts` covers paired output, frontend failure, native timeouts, manifest validation, option restrictions and relocation rendering without requiring GraalVM. A real local smoke test should compile the Java and Kotlin examples above and open the JVM Bytecode pane. Rebuild the feature whenever its Java source changes.
