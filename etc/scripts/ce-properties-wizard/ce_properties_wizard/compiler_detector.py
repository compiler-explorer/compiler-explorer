"""Compiler detection logic."""

import os
import platform
import re
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
        compiler_types=["gcc", "clang", "icc", "icx", "msvc"],
        extensions=[".cpp", ".cc", ".cxx", ".c++"],
        keywords=["g++", "clang++", "icpc", "icx", "c++"],
    ),
    "c": LanguageConfig(
        name="C",
        properties_file="c.local.properties",
        compiler_types=["gcc", "clang", "icc", "icx", "msvc"],
        extensions=[".c"],
        keywords=["gcc", "clang", "icc", "cc"],
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
}


class CompilerDetector:
    """Handles compiler detection and language inference."""

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
        compiler_id = self._generate_id(compiler_type, version, compiler_name, target if is_cross else None)

        # Generate display name
        display_name = self._generate_display_name(compiler_type, version, compiler_name, target if is_cross else None)

        # Group will be suggested later by smart group suggestion logic
        group = None

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

        # Try common version flags and subcommands
        version_flags = ["--version", "-v", "--help", "-V", "/help", "/?", "version"]

        for flag in version_flags:
            result = SubprocessRunner.run_with_timeout([compiler_path, flag], timeout=5)
            if result is None:
                continue

            output = (result.stdout + result.stderr).lower()
            full_output = result.stdout + result.stderr

            # Detect z88dk first (before Clang) since z88dk mentions clang in its help
            if "z88dk" in output:
                version = VersionExtractor.extract_version("z88dk", full_output)
                return "z88dk", version

            # Detect Clang (before GCC) since clang output may contain 'gnu'
            if "clang" in output:
                version = VersionExtractor.extract_version("clang", full_output)
                return "clang", version

            # Detect GCC
            if "gcc" in output or "g++" in output or ("gnu" in output and "clang" not in output):
                version = VersionExtractor.extract_version("gcc", full_output)
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
                return "msvc", version

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
        self, compiler_type: Optional[str], version: Optional[str], compiler_name: str, target: Optional[str] = None
    ) -> str:
        """Generate a unique compiler ID."""
        parts = ["custom"]

        # Add target architecture for cross-compilers
        if target:
            arch = target.split("-")[0]
            parts.append(arch)

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
            "clang": "Clang",
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
