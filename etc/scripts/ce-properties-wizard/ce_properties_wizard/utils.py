"""Shared utility functions for CE Properties Wizard."""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Tuple


def find_ce_root_directory(search_targets: List[Tuple[str, Callable]], max_levels: int = 6) -> Optional[Path]:
    """Find CE root directory by looking for specific target paths.

    Args:
        search_targets: List of (relative_path, validation_function) tuples
        max_levels: Maximum directory levels to traverse upward

    Returns:
        Path to CE root directory if found, None otherwise
    """
    current_dir = Path(__file__).resolve().parent

    for _ in range(max_levels):
        for target_path, validator in search_targets:
            target_dir = current_dir / target_path
            if target_dir.exists() and validator(target_dir):
                return current_dir
        current_dir = current_dir.parent

    return None


def find_ce_config_directory() -> Path:
    """Find the etc/config directory containing CE configuration files."""

    def validate_config_dir(path: Path) -> bool:
        return path.is_dir() and any(path.glob("*.defaults.properties"))

    search_targets = [("etc/config", validate_config_dir)]
    ce_root = find_ce_root_directory(search_targets)

    if ce_root:
        return ce_root / "etc" / "config"

    # Fallback: check if we're already in the main CE directory
    if Path("etc/config").exists() and Path("etc/config").is_dir():
        config_dir = Path("etc/config").resolve()
        if any(config_dir.glob("*.defaults.properties")):
            return config_dir

    raise FileNotFoundError("Could not find etc/config directory with CE configuration files")


def find_ce_lib_directory() -> Path:
    """Find the lib directory containing CE TypeScript files."""

    def validate_lib_dir(path: Path) -> bool:
        compilers_dir = path / "compilers"
        return compilers_dir.exists() and compilers_dir.is_dir() and any(compilers_dir.glob("*.ts"))

    search_targets = [("lib", validate_lib_dir)]
    ce_root = find_ce_root_directory(search_targets)

    if ce_root:
        return ce_root / "lib"

    # Fallback: assume we're in the main CE directory
    lib_dir = Path("lib")
    if validate_lib_dir(lib_dir):
        return lib_dir.resolve()

    raise FileNotFoundError("Could not find lib directory with TypeScript files")


def create_backup(file_path: Path) -> Path:
    """Create a backup of the file with .bak extension.

    Args:
        file_path: Path to the file to backup

    Returns:
        Path to the created backup file
    """
    backup_path = file_path.with_suffix(".properties.bak")
    if file_path.exists():
        shutil.copy2(file_path, backup_path)
    return backup_path


class SubprocessRunner:
    """Utility class for running subprocess commands with consistent error handling."""

    @staticmethod
    def run_with_timeout(
        cmd: List[str], timeout: int = 10, capture_output: bool = True, text: bool = True
    ) -> Optional[subprocess.CompletedProcess]:
        """Run a subprocess command with timeout and error handling.

        Args:
            cmd: Command and arguments to execute
            timeout: Timeout in seconds
            capture_output: Whether to capture stdout/stderr
            text: Whether to return text output

        Returns:
            CompletedProcess result if successful, None if failed
        """
        try:
            return subprocess.run(cmd, capture_output=capture_output, text=text, timeout=timeout)
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return None


class VersionExtractor:
    """Utility class for extracting version information from compiler output."""

    # Regex patterns for different compiler types
    PATTERNS = {
        "gcc": [r"gcc.*?(\d+\.\d+\.\d+)", r"g\+\+.*?(\d+\.\d+\.\d+)"],
        "clang": [r"clang version (\d+\.\d+\.\d+)"],
        "intel": [r"(?:icc|icpc|icx|dpcpp).*?(\d+\.\d+\.\d+)", r"intel.*?compiler.*?(\d+\.\d+)"],
        "intel_fortran": [r"(?:ifx|ifort)\s*\([^)]+\)\s*(\d+\.\d+\.\d+)", r"(?:ifx|ifort).*?(\d+\.\d+\.\d+)"],
        "msvc": [r"compiler version (\d+\.\d+\.\d+)"],
        "nvcc": [r"release (\d+\.\d+)"],
        "rust": [r"rustc (\d+\.\d+\.\d+)"],
        "go": [r"go\s*version\s+go(\d+\.\d+(?:\.\d+)?)", r"go(\d+\.\d+(?:\.\d+)?)"],
        "tinygo": [r"(?:tinygo\s+)?version:?\s+(\d+\.\d+(?:\.\d+)?)"],
        "python": [r"python (\d+\.\d+\.\d+)", r"pypy.*?(\d+\.\d+\.\d+)"],
        "fpc": [r"Free Pascal Compiler version (\d+\.\d+\.\d+)", r"fpc.*?(\d+\.\d+\.\d+)"],
        "z88dk": [r"z88dk.*?-\s*v([^-\s]+(?:-[^-\s]+)*)", r"v(\d+[^-\s]*(?:-[^-\s]*)*)"],
    }

    @classmethod
    def extract_version(cls, compiler_type: str, output: str) -> Optional[str]:
        """Extract version string from compiler output.

        Args:
            compiler_type: Type of compiler (gcc, clang, etc.)
            output: Raw output from compiler version command

        Returns:
            Extracted version string if found, None otherwise
        """
        patterns = cls.PATTERNS.get(compiler_type, [])

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    @classmethod
    def extract_semver(cls, version: Optional[str]) -> Optional[str]:
        """Extract semantic version from version string.

        Args:
            version: Version string to parse

        Returns:
            Semantic version (major.minor.patch) if found, None otherwise
        """
        if not version:
            return None
        match = re.match(r"(\d+\.\d+(?:\.\d+)?)", version)
        if match:
            return match.group(1)
        return None


class ArchitectureMapper:
    """Utility class for architecture and instruction set mapping."""

    # Architecture mapping based on lib/instructionsets.ts
    ARCH_MAPPINGS = {
        "aarch64": "aarch64",
        "arm64": "aarch64",
        "arm": "arm32",
        "avr": "avr",
        "bpf": "ebpf",
        "ez80": "ez80",
        "kvx": "kvx",
        "k1": "kvx",
        "loongarch": "loongarch",
        "m68k": "m68k",
        "mips": "mips",
        "mipsel": "mips",
        "mips64": "mips",
        "mips64el": "mips",
        "nanomips": "mips",
        "mrisc32": "mrisc32",
        "msp430": "msp430",
        "powerpc": "powerpc",
        "ppc64": "powerpc",
        "ppc": "powerpc",
        "riscv64": "riscv64",
        "rv64": "riscv64",
        "riscv32": "riscv32",
        "rv32": "riscv32",
        "sh": "sh",
        "sparc": "sparc",
        "sparc64": "sparc",
        "s390x": "s390x",
        "vax": "vax",
        "wasm32": "wasm32",
        "wasm64": "wasm64",
        "xtensa": "xtensa",
        "z180": "z180",
        "z80": "z80",
        "x86_64": "amd64",
        "x86-64": "amd64",
        "amd64": "amd64",
        "i386": "x86",
        "i486": "x86",
        "i586": "x86",
        "i686": "x86",
    }

    @classmethod
    def detect_instruction_set(cls, target: Optional[str], exe_path: str) -> str:
        """Detect instruction set from target platform or executable path.

        Args:
            target: Target platform string (e.g., from compiler -v output)
            exe_path: Path to the compiler executable

        Returns:
            Instruction set name (defaults to "amd64" if not detected)
        """
        if not target:
            target = ""

        target_lower = target.lower()
        exe_lower = exe_path.lower()

        # Check target first
        for arch, instruction_set in cls.ARCH_MAPPINGS.items():
            if arch in target_lower:
                return instruction_set

        # Check executable path as fallback
        for arch, instruction_set in cls.ARCH_MAPPINGS.items():
            if arch in exe_lower:
                return instruction_set

        # Default to amd64 if nothing detected
        return "amd64"
