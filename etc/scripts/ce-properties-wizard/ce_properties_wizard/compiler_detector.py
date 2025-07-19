"""Compiler detection logic."""

import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Optional, Set, Tuple

from .models import CompilerInfo, LanguageConfig


def get_supported_compiler_types() -> Set[str]:
    """Dynamically extract all supported compiler types from lib/compilers/*.ts files."""
    compiler_types = set()

    # Find the lib/compilers directory relative to the current location
    current_dir = Path(__file__).resolve().parent

    # Look for lib/compilers directory by going up the directory tree
    # The wizard is in etc/scripts/ce-properties-wizard/ce_properties_wizard/
    # So we need to go up to find the root directory containing lib/compilers
    for _ in range(6):  # Max 6 levels up
        lib_compilers_dir = current_dir / "lib" / "compilers"
        if lib_compilers_dir.exists() and lib_compilers_dir.is_dir():
            # Verify this looks like the CE lib/compilers directory
            if any(lib_compilers_dir.glob("*.ts")):
                break
        current_dir = current_dir.parent
    else:
        # Fallback: assume we're in the main CE directory
        lib_compilers_dir = Path("lib/compilers")
        if not (lib_compilers_dir.exists() and lib_compilers_dir.is_dir() and any(lib_compilers_dir.glob("*.ts"))):
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
            try:
                result = subprocess.run([compiler_path, "version"], capture_output=True, text=True, timeout=5)
                if "go version" in result.stdout.lower():
                    version = self._extract_go_version(result.stdout)
                    return "go", version
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass

        # Try common version flags and subcommands
        version_flags = ["--version", "-v", "--help", "-V", "/help", "/?", "version"]

        for flag in version_flags:
            try:
                result = subprocess.run([compiler_path, flag], capture_output=True, text=True, timeout=5)
                output = (result.stdout + result.stderr).lower()

                # Detect z88dk first (before Clang) since z88dk mentions clang in its help
                if "z88dk" in output:
                    version = self._extract_z88dk_version(result.stdout + result.stderr)
                    return "z88dk", version

                # Detect Clang (before GCC) since clang output may contain 'gnu'
                if "clang" in output:
                    version = self._extract_clang_version(result.stdout + result.stderr)
                    return "clang", version

                # Detect GCC
                if "gcc" in output or "g++" in output or ("gnu" in output and "clang" not in output):
                    version = self._extract_gcc_version(result.stdout + result.stderr)
                    return "gcc", version

                # Detect Intel Fortran first
                if "ifx" in output or "ifort" in output:
                    version = self._extract_intel_fortran_version(result.stdout + result.stderr)
                    if "ifx" in output:
                        return "ifx", version
                    else:
                        return "ifort", version

                # Detect Intel C/C++
                if "intel" in output:
                    if "icx" in output or "dpcpp" in output:
                        version = self._extract_intel_version(result.stdout + result.stderr)
                        return "icx", version
                    else:
                        version = self._extract_intel_version(result.stdout + result.stderr)
                        return "icc", version

                # Detect MSVC
                if "microsoft" in output or "msvc" in output:
                    version = self._extract_msvc_version(result.stdout + result.stderr)
                    return "msvc", version

                # Detect NVCC
                if "nvidia" in output or "nvcc" in output:
                    version = self._extract_nvcc_version(result.stdout + result.stderr)
                    return "nvcc", version

                # Detect Rust
                if "rustc" in output:
                    version = self._extract_rust_version(result.stdout + result.stderr)
                    return "rustc", version

                # Detect TinyGo first (before regular Go)
                if "tinygo" in output:
                    version = self._extract_tinygo_version(result.stdout + result.stderr)
                    return "tinygo", version

                # Detect Go
                if "go version" in output or "gccgo" in output:
                    version = self._extract_go_version(result.stdout + result.stderr)
                    return "go" if "go version" in output else "gccgo", version

                # Detect Python
                if "python" in output:
                    version = self._extract_python_version(result.stdout + result.stderr)
                    return "pypy" if "pypy" in output else "python", version

                # Detect Free Pascal
                if "free pascal" in output or "fpc" in output:
                    version = self._extract_fpc_version(result.stdout + result.stderr)
                    return "fpc", version

            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                continue

        return None, None

    def _extract_gcc_version(self, output: str) -> Optional[str]:
        """Extract GCC version from output."""
        match = re.search(r"gcc.*?(\d+\.\d+\.\d+)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(r"g\+\+.*?(\d+\.\d+\.\d+)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_clang_version(self, output: str) -> Optional[str]:
        """Extract Clang version from output."""
        match = re.search(r"clang version (\d+\.\d+\.\d+)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_intel_version(self, output: str) -> Optional[str]:
        """Extract Intel compiler version from output."""
        match = re.search(r"(?:icc|icpc|icx|dpcpp).*?(\d+\.\d+\.\d+)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(r"intel.*?compiler.*?(\d+\.\d+)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_intel_fortran_version(self, output: str) -> Optional[str]:
        """Extract Intel Fortran compiler version from output."""
        # Match "ifx (IFX) 2024.0.0" or "ifort (IFORT) 2021.1.0"
        match = re.search(r"(?:ifx|ifort)\s*\([^)]+\)\s*(\d+\.\d+\.\d+)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        # Fallback patterns
        match = re.search(r"(?:ifx|ifort).*?(\d+\.\d+\.\d+)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_msvc_version(self, output: str) -> Optional[str]:
        """Extract MSVC version from output."""
        match = re.search(r"compiler version (\d+\.\d+\.\d+)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_nvcc_version(self, output: str) -> Optional[str]:
        """Extract NVCC version from output."""
        match = re.search(r"release (\d+\.\d+)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_rust_version(self, output: str) -> Optional[str]:
        """Extract Rust version from output."""
        match = re.search(r"rustc (\d+\.\d+\.\d+)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_go_version(self, output: str) -> Optional[str]:
        """Extract Go version from output."""
        # Match "go version go1.24.2" or similar patterns
        match = re.search(r"go\s*version\s+go(\d+\.\d+(?:\.\d+)?)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        # Fallback to simpler pattern
        match = re.search(r"go(\d+\.\d+(?:\.\d+)?)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_tinygo_version(self, output: str) -> Optional[str]:
        """Extract TinyGo version from output."""
        # Match "tinygo version 0.37.0" or "version: 0.37.0"
        match = re.search(r"(?:tinygo\s+)?version:?\s+(\d+\.\d+(?:\.\d+)?)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_python_version(self, output: str) -> Optional[str]:
        """Extract Python version from output."""
        match = re.search(r"python (\d+\.\d+\.\d+)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(r"pypy.*?(\d+\.\d+\.\d+)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_fpc_version(self, output: str) -> Optional[str]:
        """Extract Free Pascal Compiler version from output."""
        match = re.search(r"Free Pascal Compiler version (\d+\.\d+\.\d+)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(r"fpc.*?(\d+\.\d+\.\d+)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_z88dk_version(self, output: str) -> Optional[str]:
        """Extract z88dk version from output."""
        # Match "v1-9ffe2042-20220728" or similar version patterns
        match = re.search(r"z88dk.*?-\s*v([^-\s]+(?:-[^-\s]+)*)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        # Fallback to simpler pattern
        match = re.search(r"v(\d+[^-\s]*(?:-[^-\s]*)*)", output, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_semver(self, version: Optional[str]) -> Optional[str]:
        """Extract semantic version from version string."""
        if not version:
            return None
        match = re.match(r"(\d+\.\d+(?:\.\d+)?)", version)
        if match:
            return match.group(1)
        return None

    def _detect_target_platform(self, compiler_path: str, compiler_type: Optional[str]) -> Optional[str]:
        """Detect the target platform of the compiler."""
        if not compiler_type:
            return None

        # Try to get target info using -v flag
        try:
            result = subprocess.run([compiler_path, "-v"], capture_output=True, text=True, timeout=5)

            # Look for Target: line in output
            for line in (result.stdout + result.stderr).split("\n"):
                if line.strip().startswith("Target:"):
                    target = line.split(":", 1)[1].strip()
                    return target
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass

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
