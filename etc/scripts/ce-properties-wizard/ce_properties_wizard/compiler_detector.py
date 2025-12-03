"""Compiler detection logic."""

import os
import platform
import re
from pathlib import Path
from typing import Optional, Set, Tuple

from .models import CompilerInfo, LanguageConfig
from .utils import SubprocessRunner, VersionExtractor, find_ce_lib_directory


def get_supported_compiler_types() -> Set[str]:
    """Dynamically extract all supported compiler types from lib/compilers/*.ts files."""
    compiler_types = set()

    try:
        lib_dir = find_ce_lib_directory()
        lib_compilers_dir = lib_dir / "compilers"
    except FileNotFoundError:
        # Return a minimal fallback set if we can't find the directory
        return {
            "gcc",
            "clang",
            "icc",
            "icx",
            "ifx",
            "ifort",
            "nvcc",
            "rustc",
            "golang",
            "python",
            "java",
            "fpc",
            "z88dk",
            "tinygo",
            "other",
        }

    # Scan all .ts files in lib/compilers
    for ts_file in lib_compilers_dir.glob("*.ts"):
        try:
            with open(ts_file, "r", encoding="utf-8") as f:
                content = f.read()

                # Look for patterns like: static get key() { return 'compiler_type'; }
                # Handle both single-line and multi-line formats
                patterns = [
                    r'static\s+get\s+key\(\)\s*{\s*return\s+[\'"]([^\'"]+)[\'"]',
                    r'static\s+override\s+get\s+key\(\)\s*{\s*return\s+[\'"]([^\'"]+)[\'"]',
                    r'static\s+get\s+key\(\)\s*:\s*string\s*{\s*return\s+[\'"]([^\'"]+)[\'"]',
                ]

                for pattern in patterns:
                    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                    for match in matches:
                        compiler_types.add(match.strip())

        except (IOError, UnicodeDecodeError):
            # Skip files that can't be read
            continue

    return compiler_types


LANGUAGE_CONFIGS = {
    "c++": LanguageConfig(
        name="C++",
        properties_file="c++.local.properties",
        compiler_types=["gcc", "clang", "icc", "icx", "win32-vc", "win32-mingw-gcc", "win32-mingw-clang"],
        extensions=[".cpp", ".cc", ".cxx", ".c++"],
        keywords=["g++", "clang++", "icpc", "icx", "c++", "cl"],
    ),
    "c": LanguageConfig(
        name="C",
        properties_file="c.local.properties",
        compiler_types=["gcc", "clang", "icc", "icx", "win32-vc", "win32-mingw-gcc", "win32-mingw-clang"],
        extensions=[".c"],
        keywords=["gcc", "clang", "icc", "cc", "cl"],
    ),
    "cuda": LanguageConfig(
        name="CUDA",
        properties_file="cuda.local.properties",
        compiler_types=["nvcc", "clang"],
        extensions=[".cu", ".cuh"],
        keywords=["nvcc", "cuda"],
    ),
    "rust": LanguageConfig(
        name="Rust",
        properties_file="rust.local.properties",
        compiler_types=["rustc"],
        extensions=[".rs"],
        keywords=["rustc"],
    ),
    "go": LanguageConfig(
        name="Go",
        properties_file="go.local.properties",
        compiler_types=["go", "gccgo"],
        extensions=[".go"],
        keywords=["go", "gccgo"],
    ),
    "python": LanguageConfig(
        name="Python",
        properties_file="python.local.properties",
        compiler_types=["python", "pypy"],
        extensions=[".py"],
        keywords=["python", "pypy"],
    ),
    "java": LanguageConfig(
        name="Java",
        properties_file="java.local.properties",
        compiler_types=["javac"],
        extensions=[".java"],
        keywords=["javac", "java"],
    ),
    "fortran": LanguageConfig(
        name="Fortran",
        properties_file="fortran.local.properties",
        compiler_types=["gfortran", "ifort"],
        extensions=[".f90", ".f95", ".f03", ".f08", ".f", ".for"],
        keywords=["gfortran", "ifort", "fortran"],
    ),
    "pascal": LanguageConfig(
        name="Pascal",
        properties_file="pascal.local.properties",
        compiler_types=["fpc", "delphi"],
        extensions=[".pas", ".pp", ".p"],
        keywords=["fpc", "pascal", "delphi"],
    ),
    "kotlin": LanguageConfig(
        name="Kotlin",
        properties_file="kotlin.local.properties",
        compiler_types=["kotlin"],
        extensions=[".kt", ".kts"],
        keywords=["kotlin", "kotlinc"],
    ),
    "zig": LanguageConfig(
        name="Zig",
        properties_file="zig.local.properties",
        compiler_types=["zig"],
        extensions=[".zig"],
        keywords=["zig"],
    ),
    "dart": LanguageConfig(
        name="Dart",
        properties_file="dart.local.properties",
        compiler_types=["dart"],
        extensions=[".dart"],
        keywords=["dart"],
    ),
    # Popular compiled languages
    "d": LanguageConfig(
        name="D",
        properties_file="d.local.properties",
        compiler_types=["dmd", "ldc", "gdc"],
        extensions=[".d"],
        keywords=["dmd", "ldc", "gdc"],
    ),
    "swift": LanguageConfig(
        name="Swift",
        properties_file="swift.local.properties",
        compiler_types=["swiftc"],
        extensions=[".swift"],
        keywords=["swift", "swiftc"],
    ),
    "nim": LanguageConfig(
        name="Nim",
        properties_file="nim.local.properties",
        compiler_types=["nim"],
        extensions=[".nim"],
        keywords=["nim"],
    ),
    "crystal": LanguageConfig(
        name="Crystal",
        properties_file="crystal.local.properties",
        compiler_types=["crystal"],
        extensions=[".cr"],
        keywords=["crystal"],
    ),
    "v": LanguageConfig(
        name="V",
        properties_file="v.local.properties",
        compiler_types=["v"],
        extensions=[".v"],
        keywords=["v"],
    ),
    # Functional languages
    "haskell": LanguageConfig(
        name="Haskell",
        properties_file="haskell.local.properties",
        compiler_types=["ghc"],
        extensions=[".hs", ".lhs"],
        keywords=["ghc", "haskell"],
    ),
    "ocaml": LanguageConfig(
        name="OCaml",
        properties_file="ocaml.local.properties",
        compiler_types=["ocamlc", "ocamlopt"],
        extensions=[".ml", ".mli"],
        keywords=["ocaml"],
    ),
    "scala": LanguageConfig(
        name="Scala",
        properties_file="scala.local.properties",
        compiler_types=["scalac"],
        extensions=[".scala"],
        keywords=["scala", "scalac"],
    ),
    # JVM languages
    "csharp": LanguageConfig(
        name="C#",
        properties_file="csharp.local.properties",
        compiler_types=["csharp", "dotnet"],
        extensions=[".cs"],
        keywords=["csharp", "dotnet", "mcs", "csc"],
    ),
    "fsharp": LanguageConfig(
        name="F#",
        properties_file="fsharp.local.properties",
        compiler_types=["fsharp", "dotnet"],
        extensions=[".fs", ".fsi", ".fsx"],
        keywords=["fsharp", "dotnet", "fsharpc"],
    ),
    # Scripting/Dynamic languages
    "ruby": LanguageConfig(
        name="Ruby",
        properties_file="ruby.local.properties",
        compiler_types=["ruby"],
        extensions=[".rb"],
        keywords=["ruby"],
    ),
    "julia": LanguageConfig(
        name="Julia",
        properties_file="julia.local.properties",
        compiler_types=["julia"],
        extensions=[".jl"],
        keywords=["julia"],
    ),
    "elixir": LanguageConfig(
        name="Elixir",
        properties_file="elixir.local.properties",
        compiler_types=["elixir"],
        extensions=[".ex", ".exs"],
        keywords=["elixir"],
    ),
    "erlang": LanguageConfig(
        name="Erlang",
        properties_file="erlang.local.properties",
        compiler_types=["erlc"],
        extensions=[".erl", ".hrl"],
        keywords=["erlang", "erlc"],
    ),
    # Assembly and low-level
    "assembly": LanguageConfig(
        name="Assembly",
        properties_file="assembly.local.properties",
        compiler_types=["nasm", "gas", "as", "yasm"],
        extensions=[".s", ".asm"],
        keywords=["nasm", "gas", "as", "yasm", "asm"],
    ),
    # Modern systems languages
    "carbon": LanguageConfig(
        name="Carbon",
        properties_file="carbon.local.properties",
        compiler_types=["carbon"],
        extensions=[".carbon"],
        keywords=["carbon"],
    ),
    "mojo": LanguageConfig(
        name="Mojo",
        properties_file="mojo.local.properties",
        compiler_types=["mojo"],
        extensions=[".mojo", ".🔥"],
        keywords=["mojo"],
    ),
    "odin": LanguageConfig(
        name="Odin",
        properties_file="odin.local.properties",
        compiler_types=["odin"],
        extensions=[".odin"],
        keywords=["odin"],
    ),
    "ada": LanguageConfig(
        name="Ada",
        properties_file="ada.local.properties",
        compiler_types=["gnatmake", "gprbuild"],
        extensions=[".adb", ".ads"],
        keywords=["ada", "gnat"],
    ),
    "cobol": LanguageConfig(
        name="COBOL",
        properties_file="cobol.local.properties",
        compiler_types=["gnucobol", "gcobol"],
        extensions=[".cob", ".cobol"],
        keywords=["cobol", "gnucobol", "gcobol"],
    ),
}


class CompilerDetector:
    """Handles compiler detection and language inference."""

    def __init__(self, debug: bool = False):
        """Initialize the detector.
        
        Args:
            debug: Enable debug output for subprocess commands
        """
        self.debug = debug

    def detect_from_path(self, compiler_path: str) -> CompilerInfo:
        """Detect compiler information from executable path."""
        if not os.path.isfile(compiler_path):
            raise ValueError(f"Compiler not found at: {compiler_path}")

        if not os.access(compiler_path, os.X_OK):
            raise ValueError(f"File is not executable: {compiler_path}")

        compiler_name = os.path.basename(compiler_path)

        # Detect language
        language = self._detect_language(compiler_path, compiler_name)

        # Detect compiler type and version
        compiler_type, version = self._detect_compiler_type_and_version(compiler_path)

        # Detect target platform (for cross-compilers)
        target = self._detect_target_platform(compiler_path, compiler_type)
        is_cross = self._is_cross_compiler(target)

        # Generate ID based on whether it's a cross-compiler
        compiler_id = self._generate_id(compiler_type, version, compiler_name, language, target if is_cross else None)

        # Generate display name
        display_name = self._generate_display_name(compiler_type, version, compiler_name, target if is_cross else None)

        # Group will be suggested later by smart group suggestion logic
        group = None

        # Detect Java-related properties for Java-based compilers
        java_home, runtime = self._detect_java_properties(compiler_type, compiler_path)

        # Detect execution wrapper for specific compilers
        execution_wrapper = self._detect_execution_wrapper(compiler_type, compiler_path)

        # Detect MSVC include and library paths
        include_path, lib_path = self._detect_msvc_paths(compiler_type, compiler_path, language)
        
        # Check if this is an MSVC compiler that might need SDK prompting
        # We need SDK prompting if it's MSVC but no Windows SDK paths were detected from existing compilers
        needs_sdk_prompt = False
        if compiler_type == "win32-vc":
            # Quick check - do any existing compilers have Windows SDK paths?
            try:
                from .utils import find_ce_config_directory
                from .config_manager import ConfigManager
                
                config_dir = find_ce_config_directory()
                temp_config = ConfigManager(config_dir, "local", debug=self.debug)
                properties_path = temp_config.get_properties_path(language)
                
                if properties_path.exists():
                    properties = temp_config.read_properties_file(properties_path)
                    has_sdk = False
                    
                    for key, value in properties.items():
                        if key.endswith(".includePath") and isinstance(value, str):
                            if "/include/" in value and "/um" in value:
                                has_sdk = True
                                break
                                
                    needs_sdk_prompt = not has_sdk
                else:
                    needs_sdk_prompt = True  # No properties file means no SDK paths
                    
            except Exception:
                needs_sdk_prompt = True  # If we can't check, prompt to be safe

        return CompilerInfo(
            id=compiler_id,
            name=display_name,
            exe=compiler_path,
            compiler_type=compiler_type,
            version=version,
            semver=self._extract_semver(version),
            group=group,
            language=language,
            target=target,
            is_cross_compiler=is_cross,
            java_home=java_home,
            runtime=runtime,
            execution_wrapper=execution_wrapper,
            include_path=include_path,
            lib_path=lib_path,
            needs_sdk_prompt=needs_sdk_prompt,
        )

    def _detect_language(self, compiler_path: str, compiler_name: str) -> str:
        """Detect programming language from compiler path/name."""
        compiler_lower = compiler_name.lower()
        path_lower = compiler_path.lower()

        # Check each language's keywords
        for lang_key, config in LANGUAGE_CONFIGS.items():
            for keyword in config.keywords:
                if keyword in compiler_lower or keyword in path_lower:
                    # Special case: differentiate between C and C++
                    if lang_key == "c" and ("++" in compiler_lower or "plus" in compiler_lower):
                        return "c++"
                    return lang_key

        # Default to C++ if unclear
        return "c++"

    def _detect_compiler_type_and_version(self, compiler_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Detect compiler type and version by running it."""
        compiler_name = os.path.basename(compiler_path).lower()

        # Special case for Go - use 'version' subcommand instead of flag
        if compiler_name == "go" or compiler_name.endswith("/go"):
            result = SubprocessRunner.run_with_timeout([compiler_path, "version"], timeout=5)
            if result and "go version" in result.stdout.lower():
                version = VersionExtractor.extract_version("go", result.stdout)
                return "go", version

        # Special case for Zig - use 'version' subcommand
        if compiler_name == "zig" or compiler_name.endswith("/zig"):
            result = SubprocessRunner.run_with_timeout([compiler_path, "version"], timeout=5)
            if result and result.stdout.strip():
                # Zig version command just outputs the version number
                version = result.stdout.strip()
                if re.match(r"\d+\.\d+\.\d+", version):
                    return "zig", version

        # Special case for Kotlin - may need JAVA_HOME environment
        if "kotlin" in compiler_name:
            # Try to find a suitable JAVA_HOME if not set
            original_java_home = os.environ.get("JAVA_HOME")
            if not original_java_home:
                # Try to infer JAVA_HOME from nearby JDK installations
                compiler_dir = Path(compiler_path).parent.parent.parent
                for potential_jdk in compiler_dir.glob("jdk-*"):
                    if potential_jdk.is_dir() and (potential_jdk / "bin" / "java").exists():
                        os.environ["JAVA_HOME"] = str(potential_jdk)
                        break

            # Try version detection with potentially updated JAVA_HOME
            for flag in ["-version", "--version"]:
                result = SubprocessRunner.run_with_timeout([compiler_path, flag], timeout=10)
                if result and ("kotlinc" in result.stderr.lower() or "kotlin" in result.stderr.lower()):
                    version = VersionExtractor.extract_version("kotlin", result.stderr)
                    return "kotlin", version

            # Restore original JAVA_HOME if we modified it
            if not original_java_home and "JAVA_HOME" in os.environ:
                del os.environ["JAVA_HOME"]

        # Try common version flags and subcommands
        version_flags = ["--version", "-v", "--help", "-V", "/help", "/?", "version"]

        # Detect if compiler is on a network drive (common for shared compiler installations)
        is_network_drive = compiler_path[1:2] == ":" and compiler_path[0].upper() >= "X"
        
        for flag in version_flags:
            # Use longer timeout for --version on network drives (can take 15+ seconds)
            if flag == "--version" and is_network_drive:
                timeout_value = 20
            elif is_network_drive:
                timeout_value = 10
            else:
                timeout_value = 2
                
            # Try with appropriate timeout
            if self.debug:
                print(f"Running: {compiler_path} {flag} (timeout: {timeout_value}s)")
            
            result = SubprocessRunner.run_with_timeout([compiler_path, flag], timeout=timeout_value)
            if result is None:
                if self.debug:
                    print(f"  -> Command failed or timed out")
                continue
            
            if self.debug:
                print(f"  -> Command succeeded, return code: {result.returncode}")
                if result.stdout:
                    print(f"  -> stdout: {result.stdout[:200]}")
                if result.stderr:
                    print(f"  -> stderr: {result.stderr[:200]}")
            

            output = (result.stdout + result.stderr).lower()
            full_output = result.stdout + result.stderr

            # Detect z88dk first (before Clang) since z88dk mentions clang in its help
            if "z88dk" in output:
                version = VersionExtractor.extract_version("z88dk", full_output)
                return "z88dk", version

            # Detect Clang (before GCC) since clang output may contain 'gnu'
            if "clang" in output:
                version = VersionExtractor.extract_version("clang", full_output)
                
                # Check if this is MinGW Clang on Windows
                if platform.system() == "Windows":
                    # Check for MinGW indicators
                    if ("mingw" in output or "windows-gnu" in output or 
                        "mingw" in compiler_path.lower() or 
                        any(indicator in compiler_path.lower() for indicator in ["mingw", "tdm-gcc", "winlibs"])):
                        return "win32-mingw-clang", version
                
                return "clang", version

            # Detect GCC (including MinGW on Windows)
            if "gcc" in output or "g++" in output or ("gnu" in output and "clang" not in output):
                version = VersionExtractor.extract_version("gcc", full_output)
                
                # Check if this is MinGW based on version output
                if "mingw" in output:
                    return "win32-mingw-gcc", version
                
                return "gcc", version

            # Detect Intel Fortran first
            if "ifx" in output or "ifort" in output:
                version = VersionExtractor.extract_version("intel_fortran", full_output)
                if "ifx" in output:
                    return "ifx", version
                else:
                    return "ifort", version

            # Detect Intel C/C++
            if "intel" in output:
                version = VersionExtractor.extract_version("intel", full_output)
                if "icx" in output or "dpcpp" in output:
                    return "icx", version
                else:
                    return "icc", version

            # Detect MSVC
            if "microsoft" in output or "msvc" in output:
                version = VersionExtractor.extract_version("msvc", full_output)
                return "win32-vc", version

            # Detect NVCC
            if "nvidia" in output or "nvcc" in output:
                version = VersionExtractor.extract_version("nvcc", full_output)
                return "nvcc", version

            # Detect Rust
            if "rustc" in output:
                version = VersionExtractor.extract_version("rust", full_output)
                return "rustc", version

            # Detect TinyGo first (before regular Go)
            if "tinygo" in output:
                version = VersionExtractor.extract_version("tinygo", full_output)
                return "tinygo", version

            # Detect Go
            if "go version" in output or "gccgo" in output:
                version = VersionExtractor.extract_version("go", full_output)
                return "go" if "go version" in output else "gccgo", version

            # Detect Python
            if "python" in output:
                version = VersionExtractor.extract_version("python", full_output)
                return "pypy" if "pypy" in output else "python", version

            # Detect Free Pascal
            if "free pascal" in output or "fpc" in output:
                version = VersionExtractor.extract_version("fpc", full_output)
                return "fpc", version

            # Detect Kotlin
            if "kotlinc" in output or "kotlin" in output:
                version = VersionExtractor.extract_version("kotlin", full_output)
                return "kotlin", version

            # Detect Zig
            if "zig" in output:
                version = VersionExtractor.extract_version("zig", full_output)
                return "zig", version

            # Detect Dart
            if "dart" in output:
                version = VersionExtractor.extract_version("dart", full_output)
                return "dart", version

            # Detect D language compilers
            if "dmd" in output:
                version = VersionExtractor.extract_version("dmd", full_output)
                return "dmd", version
            if "ldc" in output:
                version = VersionExtractor.extract_version("ldc", full_output)
                return "ldc", version
            if "gdc" in output and "gnu d compiler" in output:
                version = VersionExtractor.extract_version("gdc", full_output)
                return "gdc", version

            # Detect Swift
            if "swift" in output:
                version = VersionExtractor.extract_version("swiftc", full_output)
                return "swiftc", version

            # Detect Nim
            if "nim" in output:
                version = VersionExtractor.extract_version("nim", full_output)
                return "nim", version

            # Detect Crystal
            if "crystal" in output:
                version = VersionExtractor.extract_version("crystal", full_output)
                return "crystal", version

            # Detect V
            if "v " in output or "vlang" in output:
                version = VersionExtractor.extract_version("v", full_output)
                return "v", version

            # Detect Haskell
            if "ghc" in output or "haskell" in output:
                version = VersionExtractor.extract_version("ghc", full_output)
                return "ghc", version

            # Detect OCaml
            if "ocaml" in output:
                if "ocamlopt" in output:
                    version = VersionExtractor.extract_version("ocamlopt", full_output)
                    return "ocamlopt", version
                else:
                    version = VersionExtractor.extract_version("ocamlc", full_output)
                    return "ocamlc", version

            # Detect Scala
            if "scala" in output:
                version = VersionExtractor.extract_version("scalac", full_output)
                return "scalac", version

            # Detect C# / .NET
            if "c# compiler" in output or "csharp" in output:
                version = VersionExtractor.extract_version("csharp", full_output)
                return "csharp", version
            if "dotnet" in output:
                version = VersionExtractor.extract_version("dotnet", full_output)
                return "dotnet", version

            # Detect F#
            if "f# compiler" in output or "fsharp" in output:
                version = VersionExtractor.extract_version("fsharp", full_output)
                return "fsharp", version

            # Detect Ruby
            if "ruby" in output:
                version = VersionExtractor.extract_version("ruby", full_output)
                return "ruby", version

            # Detect Julia
            if "julia" in output:
                version = VersionExtractor.extract_version("julia", full_output)
                return "julia", version

            # Detect Elixir
            if "elixir" in output:
                version = VersionExtractor.extract_version("elixir", full_output)
                return "elixir", version

            # Detect Erlang
            if "erlang" in output or "erlc" in output:
                version = VersionExtractor.extract_version("erlc", full_output)
                return "erlc", version

            # Detect Assembly tools
            if "nasm" in output:
                version = VersionExtractor.extract_version("nasm", full_output)
                return "nasm", version
            if "yasm" in output:
                version = VersionExtractor.extract_version("yasm", full_output)
                return "yasm", version
            if "gnu assembler" in output:
                version = VersionExtractor.extract_version("gas", full_output)
                return "gas", version

            # Detect modern systems languages
            if "carbon" in output:
                version = VersionExtractor.extract_version("carbon", full_output)
                return "carbon", version
            if "mojo" in output:
                version = VersionExtractor.extract_version("mojo", full_output)
                return "mojo", version
            if "odin" in output:
                version = VersionExtractor.extract_version("odin", full_output)
                return "odin", version

            # Detect Ada
            if "gnatmake" in output or "ada" in output:
                version = VersionExtractor.extract_version("gnatmake", full_output)
                return "gnatmake", version

            # Detect COBOL
            if "gnucobol" in output or "cobol" in output:
                version = VersionExtractor.extract_version("gnucobol", full_output)
                return "gnucobol", version

        return None, None

    def _extract_semver(self, version: Optional[str]) -> Optional[str]:
        """Extract semantic version from version string."""
        return VersionExtractor.extract_semver(version)

    def _detect_target_platform(self, compiler_path: str, compiler_type: Optional[str]) -> Optional[str]:
        """Detect the target platform of the compiler."""
        if not compiler_type:
            return None

        # Try to get target info using -v flag
        result = SubprocessRunner.run_with_timeout([compiler_path, "-v"], timeout=5)
        if result:
            # Look for Target: line in output
            for line in (result.stdout + result.stderr).split("\n"):
                if line.strip().startswith("Target:"):
                    target = line.split(":", 1)[1].strip()
                    return target

        return None

    def _is_cross_compiler(self, target: Optional[str]) -> bool:
        """Determine if this is a cross-compiler based on target."""
        if not target:
            return False

        # Get the host platform
        host_machine = platform.machine().lower()

        # Normalize host architecture names
        host_arch_map = {
            "x86_64": ["x86_64", "amd64"],
            "i386": ["i386", "i486", "i586", "i686"],
            "aarch64": ["aarch64", "arm64"],
            "armv7l": ["arm", "armv7"],
        }

        # Find normalized host arch
        normalized_host = host_machine
        for norm_arch, variants in host_arch_map.items():
            if host_machine in variants:
                normalized_host = norm_arch
                break

        # Extract target architecture
        target_parts = target.lower().split("-")
        if not target_parts:
            return False

        target_arch = target_parts[0]

        # Check if architectures match
        for norm_arch, variants in host_arch_map.items():
            if normalized_host in variants and target_arch in variants:
                return False

        # If architectures don't match, it's a cross-compiler
        return target_arch != normalized_host

    def _generate_id(
        self, compiler_type: Optional[str], version: Optional[str], compiler_name: str, language: str, target: Optional[str] = None
    ) -> str:
        """Generate a unique compiler ID."""
        parts = ["custom"]

        # Add target architecture for cross-compilers
        if target:
            arch = target.split("-")[0]
            parts.append(arch)

        # Add language prefix for C to avoid conflicts with C++
        if language == "c" and compiler_type in ["gcc", "clang", "icc", "icx"]:
            parts.append("c")

        # Add compiler type
        if compiler_type:
            parts.append(compiler_type)

        # Add version
        if version:
            version_part = version.replace(".", "-")
            parts.append(version_part)
        elif not compiler_type:
            # Use sanitized compiler name as fallback
            safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", compiler_name)
            parts.append(safe_name)

        return "-".join(parts)

    def _generate_display_name(
        self, compiler_type: Optional[str], version: Optional[str], compiler_name: str, target: Optional[str] = None
    ) -> str:
        """Generate a display name for the compiler."""
        type_display = {
            "gcc": "GCC",
            "win32-mingw-gcc": "MinGW GCC",
            "clang": "Clang",
            "win32-mingw-clang": "MinGW Clang",
            "win32-vc": "MSVC",
            "icc": "ICC",
            "icx": "Intel ICX",
            "ifx": "Intel IFX",
            "ifort": "Intel Fortran",
            "msvc": "MSVC",
            "nvcc": "NVCC",
            "rustc": "Rust",
            "go": "Go",
            "gccgo": "GCC Go",
            "tinygo": "TinyGo",
            "python": "Python",
            "pypy": "PyPy",
            "fpc": "Free Pascal",
            "z88dk": "z88dk",
            "zig": "Zig",
            "dart": "Dart",
            # Popular compiled languages
            "dmd": "DMD",
            "ldc": "LDC",
            "gdc": "GDC",
            "swiftc": "Swift",
            "nim": "Nim",
            "crystal": "Crystal",
            "v": "V",
            # Functional languages
            "ghc": "GHC",
            "ocamlc": "OCaml",
            "ocamlopt": "OCaml",
            "scalac": "Scala",
            # .NET languages
            "csharp": "C#",
            "dotnet": ".NET",
            "fsharp": "F#",
            # Scripting/Dynamic languages
            "ruby": "Ruby",
            "julia": "Julia",
            "elixir": "Elixir",
            "erlc": "Erlang",
            # Assembly and low-level
            "nasm": "NASM",
            "gas": "GAS",
            "yasm": "YASM",
            # Modern systems languages
            "carbon": "Carbon",
            "mojo": "Mojo",
            "odin": "Odin",
            "gnatmake": "Ada",
            "gnucobol": "COBOL",
        }.get(compiler_type or "", compiler_type.upper() if compiler_type else "")

        parts = []

        # Add target architecture for cross-compilers
        if target:
            arch = target.split("-")[0].upper()
            parts.append(arch)

        # Add compiler type and version
        if compiler_type and version:
            parts.append(f"{type_display} {version}")
        elif compiler_type:
            parts.append(type_display)
        else:
            parts.append(compiler_name)

        return " ".join(parts)

    def _detect_java_properties(
        self, compiler_type: Optional[str], compiler_path: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Detect JAVA_HOME and runtime for Java-based compilers.

        Args:
            compiler_type: Type of compiler (kotlin, etc.)
            compiler_path: Path to the compiler executable

        Returns:
            Tuple of (java_home, runtime) paths
        """
        if compiler_type != "kotlin":
            return None, None

        # For Kotlin, try to detect JAVA_HOME from environment or infer from common locations
        java_home = os.environ.get("JAVA_HOME")

        if not java_home:
            # Try to infer JAVA_HOME from common locations near the compiler
            compiler_dir = Path(compiler_path).parent.parent

            # Look for JDK installations in the same parent directory
            parent_dir = compiler_dir.parent
            for potential_jdk in parent_dir.glob("jdk-*"):
                if potential_jdk.is_dir() and (potential_jdk / "bin" / "java").exists():
                    java_home = str(potential_jdk)
                    break

        # Determine runtime executable
        runtime = None
        if java_home:
            java_exe = Path(java_home) / "bin" / "java"
            if java_exe.exists():
                runtime = str(java_exe)

        return java_home, runtime

    def _detect_execution_wrapper(self, compiler_type: Optional[str], compiler_path: str) -> Optional[str]:
        """Detect execution wrapper for compilers that need it.

        Args:
            compiler_type: Type of compiler (dart, etc.)
            compiler_path: Path to the compiler executable

        Returns:
            Path to execution wrapper if needed, None otherwise
        """
        if compiler_type != "dart":
            return None

        # For Dart, look for dartaotruntime in the same bin directory
        compiler_dir = Path(compiler_path).parent
        dartaotruntime_path = compiler_dir / "dartaotruntime"

        if dartaotruntime_path.exists() and dartaotruntime_path.is_file():
            return str(dartaotruntime_path)

        return None

    def _detect_msvc_paths(self, compiler_type: Optional[str], compiler_path: str, language: str) -> Tuple[Optional[str], Optional[str]]:
        """Detect include and library paths for MSVC compilers.
        
        Args:
            compiler_type: Type of compiler (should be "win32-vc" for MSVC)
            compiler_path: Path to the compiler executable
            
        Returns:
            Tuple of (include_path, lib_path) strings, or (None, None) if not MSVC
        """
        if compiler_type != "win32-vc":
            return None, None
            
        # Convert Windows backslashes to forward slashes for consistency
        normalized_path = compiler_path.replace("\\", "/")
        
        # Extract the base MSVC directory from the compiler path
        # Example: Z:/compilers/msvc/14.40.33807-14.40.33811.0/bin/Hostx64/x64/cl.exe
        # Should extract: Z:/compilers/msvc/14.40.33807-14.40.33811.0
        
        # Look for the pattern /bin/Host*/*/cl.exe and extract base directory
        import re
        match = re.search(r"^(.+)/bin/Host[^/]+/[^/]+/cl\.exe$", normalized_path, re.IGNORECASE)
        if not match:
            # Try alternative pattern for different MSVC layouts
            match = re.search(r"^(.+)/bin/cl\.exe$", normalized_path, re.IGNORECASE)
            
        if not match:
            self._debug_log(f"DEBUG: Could not extract MSVC base directory from path: {compiler_path}")
            return None, None
            
        base_dir = match.group(1)
        self._debug_log(f"DEBUG: Detected MSVC base directory: {base_dir}")
        
        # Detect architecture from the compiler path
        arch = None
        if "/hostx64/x64/" in normalized_path.lower():
            arch = "x64"
        elif "/hostx86/x86/" in normalized_path.lower():
            arch = "x86"
        elif "/hostx64/arm64/" in normalized_path.lower():
            arch = "arm64"
        elif "/hostx86/arm/" in normalized_path.lower():
            arch = "arm"
        else:
            # Default to x64 if we can't detect
            arch = "x64"
            self._debug_log(f"DEBUG: Could not detect architecture from path, defaulting to x64")
            
        self._debug_log(f"DEBUG: Detected MSVC architecture: {arch}")
        
        # Build include path
        include_path = f"{base_dir}/include"
        
        # Build library paths based on architecture
        lib_paths = [
            f"{base_dir}/lib",
            f"{base_dir}/lib/{arch}",
            f"{base_dir}/atlmfc/lib/{arch}",
            f"{base_dir}/ifc/{arch}"
        ]
        
        lib_path = ";".join(lib_paths)
        
        # Detect Windows SDK paths from existing compilers
        sdk_include_paths, sdk_lib_paths = self._detect_windows_sdk_paths(language, arch)
        
        # Combine MSVC paths with Windows SDK paths
        if sdk_include_paths:
            include_path = f"{include_path};{sdk_include_paths}"
            self._debug_log(f"DEBUG: Added Windows SDK include paths: {sdk_include_paths}")
            
        if sdk_lib_paths:
            lib_path = f"{lib_path};{sdk_lib_paths}"
            self._debug_log(f"DEBUG: Added Windows SDK library paths: {sdk_lib_paths}")
        else:
            # Store info that SDK detection failed for later interactive prompting
            self._debug_log("DEBUG: Windows SDK auto-detection failed - will prompt user in interactive mode")
        
        self._debug_log(f"DEBUG: Final MSVC include path: {include_path}")
        self._debug_log(f"DEBUG: Final MSVC library paths: {lib_path}")
        
        return include_path, lib_path

    def _detect_windows_sdk_paths(self, language: str, arch: str) -> Tuple[Optional[str], Optional[str]]:
        """Detect Windows SDK paths by scanning existing compiler configurations.
        
        Args:
            language: Programming language (e.g., "c++")
            arch: Target architecture (e.g., "x64", "x86", "arm64")
            
        Returns:
            Tuple of (sdk_include_paths, sdk_lib_paths) strings, or (None, None) if not found
        """
        try:
            from .utils import find_ce_config_directory
            from .config_manager import ConfigManager
            
            # Create a temporary config manager to read existing properties
            config_dir = find_ce_config_directory()
            temp_config = ConfigManager(config_dir, "local", debug=self.debug)
            properties_path = temp_config.get_properties_path(language)
            
            if not properties_path.exists():
                self._debug_log(f"DEBUG: Properties file not found: {properties_path}")
                return None, None
                
            properties = temp_config.read_properties_file(properties_path)
            
            # Scan all compiler includePath properties for Windows SDK patterns
            sdk_base_path = None
            sdk_version = None
            
            for key, value in properties.items():
                if key.endswith(".includePath") and isinstance(value, str):
                    self._debug_log(f"DEBUG: Scanning includePath: {key} = {value}")
                    
                    # Look for pattern ending with /include/<version>/um
                    import re
                    match = re.search(r"([^;]+)/include/([^/;]+)/um(?:;|$)", value)
                    if match:
                        sdk_base_path = match.group(1)
                        sdk_version = match.group(2)
                        self._debug_log(f"DEBUG: Found Windows SDK: base={sdk_base_path}, version={sdk_version}")
                        break
                        
            if not sdk_base_path or not sdk_version:
                self._debug_log("DEBUG: No Windows SDK path found in existing compilers")
                return None, None
                
            # Generate Windows SDK include paths
            sdk_include_dirs = [
                f"{sdk_base_path}/include/{sdk_version}/cppwinrt",
                f"{sdk_base_path}/include/{sdk_version}/shared",
                f"{sdk_base_path}/include/{sdk_version}/ucrt",
                f"{sdk_base_path}/include/{sdk_version}/um",
                f"{sdk_base_path}/include/{sdk_version}/winrt"
            ]
            
            sdk_include_paths = ";".join(sdk_include_dirs)
            
            # Generate Windows SDK library paths based on architecture
            sdk_lib_dirs = [
                f"{sdk_base_path}/lib/{sdk_version}/ucrt/{arch}",
                f"{sdk_base_path}/lib/{sdk_version}/um/{arch}"
            ]
            
            sdk_lib_paths = ";".join(sdk_lib_dirs)
            
            self._debug_log(f"DEBUG: Generated SDK include paths: {sdk_include_paths}")
            self._debug_log(f"DEBUG: Generated SDK library paths: {sdk_lib_paths}")
            
            return sdk_include_paths, sdk_lib_paths
            
        except Exception as e:
            self._debug_log(f"DEBUG: Error detecting Windows SDK paths: {e}")
            return None, None

    def set_windows_sdk_path(self, compiler_info: 'CompilerInfo', sdk_path: Optional[str]) -> 'CompilerInfo':
        """Update MSVC compiler info with Windows SDK paths.
        
        Args:
            compiler_info: CompilerInfo object for MSVC compiler
            sdk_path: Optional Windows SDK base path (e.g., "Z:/compilers/windows-kits-10")
            
        Returns:
            Updated CompilerInfo with SDK paths added
        """
        if compiler_info.compiler_type != "win32-vc" or not sdk_path:
            return compiler_info
            
        # Extract architecture from the compiler path
        normalized_path = compiler_info.exe.replace("\\", "/")
        arch = "x64"  # default
        if "/hostx64/x64/" in normalized_path.lower():
            arch = "x64"
        elif "/hostx86/x86/" in normalized_path.lower():
            arch = "x86"
        elif "/hostx64/arm64/" in normalized_path.lower():
            arch = "arm64"
        elif "/hostx86/arm/" in normalized_path.lower():
            arch = "arm"
            
        # Find the SDK version by looking for the latest version directory
        import os
        from pathlib import Path
        
        sdk_base = Path(sdk_path.replace("\\", "/"))
        sdk_version = None
        
        # Look for include directory with version subdirectories
        include_dir = sdk_base / "include"
        if include_dir.exists():
            # Find the latest version directory (highest version number)
            version_dirs = [d.name for d in include_dir.iterdir() if d.is_dir() and d.name.startswith("10.")]
            if version_dirs:
                sdk_version = sorted(version_dirs, reverse=True)[0]  # Get the latest version
                self._debug_log(f"DEBUG: Found SDK version: {sdk_version}")
                
        if not sdk_version:
            self._debug_log(f"DEBUG: No SDK version found in {include_dir}")
            return compiler_info
            
        # Generate Windows SDK include paths
        sdk_include_dirs = [
            f"{sdk_path}/include/{sdk_version}/cppwinrt",
            f"{sdk_path}/include/{sdk_version}/shared",
            f"{sdk_path}/include/{sdk_version}/ucrt",
            f"{sdk_path}/include/{sdk_version}/um",
            f"{sdk_path}/include/{sdk_version}/winrt"
        ]
        
        sdk_include_paths = ";".join(sdk_include_dirs)
        
        # Generate Windows SDK library paths based on architecture
        sdk_lib_dirs = [
            f"{sdk_path}/lib/{sdk_version}/ucrt/{arch}",
            f"{sdk_path}/lib/{sdk_version}/um/{arch}"
        ]
        
        sdk_lib_paths = ";".join(sdk_lib_dirs)
        
        # Combine with existing MSVC paths
        if compiler_info.include_path:
            compiler_info.include_path = f"{compiler_info.include_path};{sdk_include_paths}"
        else:
            compiler_info.include_path = sdk_include_paths
            
        if compiler_info.lib_path:
            compiler_info.lib_path = f"{compiler_info.lib_path};{sdk_lib_paths}"
        else:
            compiler_info.lib_path = sdk_lib_paths
            
        self._debug_log(f"DEBUG: Added user-provided SDK paths from: {sdk_path}")
        self._debug_log(f"DEBUG: SDK include paths: {sdk_include_paths}")
        self._debug_log(f"DEBUG: SDK library paths: {sdk_lib_paths}")
        
        return compiler_info

    def _debug_log(self, message: str):
        """Log debug message if debug mode is enabled."""
        if self.debug:
            print(message)
