"""Configuration file management for CE Properties Wizard."""

import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Set

from .compiler_detector import LANGUAGE_CONFIGS
from .models import CompilerInfo
from .surgical_editor import PropertiesFileEditor
from .utils import ArchitectureMapper, create_backup, find_ce_lib_directory


def get_supported_instruction_sets() -> Set[str]:
    """Dynamically extract all supported instruction sets from lib/instructionsets.ts."""
    instruction_sets = set()

    try:
        lib_dir = find_ce_lib_directory()
        instructionsets_file = lib_dir / "instructionsets.ts"
    except FileNotFoundError:
        # Return a minimal fallback set if we can't find the file
        return {"amd64", "aarch64", "arm32", "x86", "sparc", "s390x", "powerpc", "mips", "riscv64", "riscv32"}

    try:
        with open(instructionsets_file, "r", encoding="utf-8") as f:
            content = f.read()

            # Look for instruction set definitions in the supported object
            # Pattern: instructionSetName: {
            pattern = r"(\w+):\s*{"
            matches = re.findall(pattern, content)
            for match in matches:
                if match not in ["target", "path"]:  # Skip property names
                    instruction_sets.add(match)

    except (IOError, UnicodeDecodeError):
        # Return fallback set if file can't be read
        return {"amd64", "aarch64", "arm32", "x86", "sparc", "s390x", "powerpc", "mips", "riscv64", "riscv32"}

    return instruction_sets


def detect_instruction_set_from_target(target: Optional[str], exe_path: str) -> str:
    """Detect instruction set from target platform or executable path."""
    return ArchitectureMapper.detect_instruction_set(target, exe_path)


class ConfigManager:
    """Manages reading and writing of compiler properties files."""

    def __init__(self, config_dir: Path, env: str = "local", debug: bool = False):
        """Initialize with path to etc/config directory and environment."""
        self.config_dir = config_dir
        self.env = env
        self.debug = debug
        if not self.config_dir.exists():
            raise ValueError(f"Config directory not found: {config_dir}")

    def _debug_log(self, message: str):
        """Log debug message if debug mode is enabled."""
        if self.debug:
            print(message)

    def get_properties_path(self, language: str) -> Path:
        """Get path to properties file for a language in the current environment."""
        if language not in LANGUAGE_CONFIGS:
            raise ValueError(f"Unknown language: {language}")

        filename = LANGUAGE_CONFIGS[language].get_properties_file(self.env)
        return self.config_dir / filename

    def get_local_properties_path(self, language: str) -> Path:
        """Get path to local properties file for a language (for backward compatibility)."""
        if language not in LANGUAGE_CONFIGS:
            raise ValueError(f"Unknown language: {language}")

        filename = LANGUAGE_CONFIGS[language].properties_file
        return self.config_dir / filename

    def read_properties_file(self, file_path: Path) -> OrderedDict:
        """Read a properties file and return as ordered dict."""
        properties = OrderedDict()

        if not file_path.exists():
            return properties

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    # Preserve comments and empty lines
                    properties[f"__comment_{line_num}__"] = line
                    continue

                # Parse key=value
                match = re.match(r"^([^=]+)=(.*)$", line)
                if match:
                    key, value = match.groups()
                    properties[key.strip()] = value.strip()
                else:
                    # Preserve malformed lines as comments
                    properties[f"__comment_{line_num}__"] = f"# {line}"

        return properties

    def write_properties_file(self, file_path: Path, properties: OrderedDict):
        """Write properties to file, preserving order and comments."""
        # Create backup if file exists
        if file_path.exists():
            create_backup(file_path)

        with open(file_path, "w", encoding="utf-8") as f:
            previous_key = None
            lines_written = []

            for key, value in properties.items():
                # Add empty line before group definitions (except the first entry)
                if (
                    previous_key is not None
                    and isinstance(key, str)
                    and key.startswith("group.")
                    and not (isinstance(previous_key, str) and previous_key.startswith("group."))
                ):
                    lines_written.append("")

                # Add empty line between different groups
                elif (
                    previous_key is not None
                    and isinstance(key, str)
                    and key.startswith("group.")
                    and isinstance(previous_key, str)
                    and previous_key.startswith("group.")
                ):
                    # Extract group names from the keys (group.{name}.property)
                    current_group_name = key.split(".")[1]
                    previous_group_name = previous_key.split(".")[1]
                    # Add empty line if we're starting a new group
                    if current_group_name != previous_group_name:
                        lines_written.append("")

                # Add empty line before compiler definitions (except the first entry)
                elif (
                    previous_key is not None
                    and isinstance(key, str)
                    and key.startswith("compiler.")
                    and not (isinstance(previous_key, str) and previous_key.startswith("compiler."))
                ):
                    lines_written.append("")

                # Add empty line between different compilers
                elif (
                    previous_key is not None
                    and isinstance(key, str)
                    and key.startswith("compiler.")
                    and isinstance(previous_key, str)
                    and previous_key.startswith("compiler.")
                ):
                    # Extract compiler IDs from the keys
                    current_compiler_id = key.split(".")[1]
                    previous_compiler_id = previous_key.split(".")[1]
                    # Add empty line if we're starting a new compiler
                    if current_compiler_id != previous_compiler_id:
                        lines_written.append("")

                if key.startswith("__comment_"):
                    lines_written.append(value)
                elif key.startswith("__libs_section_"):
                    lines_written.append(value)
                elif key.startswith("__tools_section_"):
                    lines_written.append(value)
                else:
                    lines_written.append(f"{key}={value}")

                previous_key = key

            # Write all lines and ensure at most 1 trailing newline
            for line in lines_written:
                f.write(f"{line}\n")

            # No additional newline needed as each line already has one

    def get_existing_compiler_ids(self, language: str) -> Set[str]:
        """Get all existing compiler IDs for a language."""
        file_path = self.get_properties_path(language)
        if not file_path.exists():
            return set()

        properties = self.read_properties_file(file_path)
        compiler_ids = set()

        # Extract compiler IDs from compiler.*.exe entries
        for key in properties:
            match = re.match(r"^compiler\.([^.]+)\.exe$", key)
            if match:
                compiler_ids.add(match.group(1))

        return compiler_ids

    def get_existing_groups(self, properties: OrderedDict) -> Dict[str, List[str]]:
        """Extract existing groups and their compilers from properties."""
        groups = {}

        for key, value in properties.items():
            # Match group.*.compilers entries
            match = re.match(r"^group\.([^.]+)\.compilers$", key)
            if match:
                group_name = match.group(1)
                # Split compiler list, handling various formats
                compilers = [c.strip() for c in re.split(r"[:;,\s]+", value) if c.strip()]
                groups[group_name] = compilers

        return groups

    def _extract_compiler_version(self, compiler_exe: str) -> Optional[str]:
        """Extract version from compiler executable."""
        import re
        import subprocess

        try:
            # Try common version flags
            version_flags = ["--version", "-version", "-V"]

            for flag in version_flags:
                try:
                    result = subprocess.run([compiler_exe, flag], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 and result.stdout:
                        # Look for version patterns in the output
                        version_patterns = [
                            r"(\d+\.\d+\.\d+)",  # x.y.z
                            r"(\d+\.\d+)",  # x.y
                            r"version\s+(\d+\.\d+\.\d+)",  # version x.y.z
                            r"version\s+(\d+\.\d+)",  # version x.y
                        ]

                        for pattern in version_patterns:
                            match = re.search(pattern, result.stdout, re.IGNORECASE)
                            if match:
                                return match.group(1)

                        # If no pattern matched, try to extract from the first line
                        first_line = result.stdout.split("\n")[0]
                        numbers = re.findall(r"\d+\.\d+(?:\.\d+)?", first_line)
                        if numbers:
                            return numbers[0]

                except (subprocess.SubprocessError, FileNotFoundError):
                    continue

            # If version extraction failed, try to extract from path
            # e.g., /opt/compiler-explorer/gcc-14.1.0/bin/gfortran -> 14.1.0
            path_match = re.search(r"gcc-(\d+\.\d+\.\d+)", compiler_exe)
            if path_match:
                return path_match.group(1)

            path_match = re.search(r"gcc-(\d+\.\d+)", compiler_exe)
            if path_match:
                return path_match.group(1)

        except Exception:
            pass

        return None

    def ensure_compiler_id_unique(self, compiler_id: str, language: str) -> str:
        """Ensure compiler ID is unique, modifying if necessary."""
        existing_ids = self.get_existing_compiler_ids(language)

        if compiler_id not in existing_ids:
            return compiler_id

        # Try adding numbers until we find a unique ID
        for i in range(2, 100):
            new_id = f"{compiler_id}-{i}"
            if new_id not in existing_ids:
                return new_id

        # Fallback to timestamp if somehow we have 98 duplicates
        import time

        return f"{compiler_id}-{int(time.time())}"

    def check_existing_compiler_by_path(self, compiler_exe: str, language: str) -> Optional[str]:
        """Check if a compiler with the same executable path already exists.

        Returns:
            The existing compiler ID if found, None otherwise.
        """
        file_path = self.get_properties_path(language)
        if not file_path.exists():
            return None

        editor = PropertiesFileEditor(file_path)

        # Use Path objects for robust cross-platform path comparison
        from pathlib import Path
        input_path = Path(compiler_exe)

        # Look for any compiler with the same exe path
        for line in editor.lines:
            if ".exe=" in line and line.startswith("compiler."):
                match = re.match(r"^compiler\.([^.]+)\.exe=(.+)$", line)
                if match:
                    compiler_id, existing_exe = match.groups()
                    # Compare using Path objects which handle normalization automatically
                    existing_path = Path(existing_exe)
                    if existing_path == input_path:
                        return compiler_id

        return None

    def suggest_appropriate_group(
        self, compiler: CompilerInfo, existing_compiler_id: Optional[str] = None
    ) -> Optional[str]:
        """Suggest an appropriate group for a compiler based on existing groups.

        Args:
            compiler: The compiler information
            existing_compiler_id: If this is a duplicate, the ID of the existing compiler

        Returns:
            Suggested group name or None if no appropriate group found
        """
        file_path = self.get_properties_path(compiler.language)
        if not file_path.exists():
            return compiler.compiler_type  # Fallback to compiler type

        editor = PropertiesFileEditor(file_path)

        # If this is a duplicate, suggest the group of the existing compiler
        if existing_compiler_id:
            for line in editor.lines:
                if line.startswith("group.") and ".compilers=" in line:
                    # Check if the existing compiler ID is in this group's compiler list
                    if (
                        f":{existing_compiler_id}" in line
                        or f"={existing_compiler_id}" in line
                        or line.endswith(existing_compiler_id)
                    ):
                        # Extract group name from group.{name}.compilers line
                        match = re.match(r"^group\.([^.]+)\.compilers=.*", line)
                        if match:
                            return match.group(1)

        # Get compiler's target architecture
        target_instruction_set = None
        if compiler.target:
            target_instruction_set = detect_instruction_set_from_target(compiler.target, compiler.exe)
        else:
            target_instruction_set = detect_instruction_set_from_target(None, compiler.exe)
            
        # Debug output
        self._debug_log(f"DEBUG: Compiler type: {compiler.compiler_type}")
        self._debug_log(f"DEBUG: Detected instruction set: {target_instruction_set}")
        self._debug_log(f"DEBUG: Compiler path: {compiler.exe}")

        # Find existing groups and their properties
        existing_groups = {}
        for line in editor.lines:
            if line.startswith("group.") and ".compilers=" in line:
                # Extract group name
                match = re.match(r"^group\.([^.]+)\.compilers=", line)
                if match:
                    group_name = match.group(1)
                    existing_groups[group_name] = {
                        "compilers": line.split("=", 1)[1],
                        "compiler_type": None,
                        "compiler_categories": None,
                        "instruction_set": None,
                        "group_name": None,
                    }

        # Get additional properties for each group
        for line in editor.lines:
            for group_name in existing_groups:
                if line.startswith(f"group.{group_name}."):
                    if ".compilerType=" in line:
                        existing_groups[group_name]["compiler_type"] = line.split("=", 1)[1]
                    elif ".compilerCategories=" in line:
                        existing_groups[group_name]["compiler_categories"] = line.split("=", 1)[1]
                    elif ".instructionSet=" in line:
                        existing_groups[group_name]["instruction_set"] = line.split("=", 1)[1]
                    elif ".groupName=" in line:
                        existing_groups[group_name]["group_name"] = line.split("=", 1)[1]

        # Score groups based on compatibility
        group_scores = []

        for group_name, group_info in existing_groups.items():
            score = 0

            # Match instruction set (highest priority - architecture must match)
            if target_instruction_set and group_info["instruction_set"] == target_instruction_set:
                score += 200
                self._debug_log(f"DEBUG: Group {group_name} instruction set match (+200): {group_info['instruction_set']} == {target_instruction_set}")

            # Special architecture matching for MSVC (when instruction sets match)
            if compiler.compiler_type == "win32-vc" and "cl.exe" in compiler.exe.lower():
                if ("hostx64\\x64" in compiler.exe.lower() or "/hostx64/x64" in compiler.exe.lower()) and "x64" in group_name:
                    score += 150
                    self._debug_log(f"DEBUG: Group {group_name} MSVC x64 path match (+150)")
                elif ("hostx86\\x86" in compiler.exe.lower() or "/hostx86/x86" in compiler.exe.lower()) and "x86" in group_name:
                    score += 150
                    self._debug_log(f"DEBUG: Group {group_name} MSVC x86 path match (+150)")
                elif ("hostx64\\arm64" in compiler.exe.lower() or "/hostx64/arm64" in compiler.exe.lower()) and "arm64" in group_name:
                    score += 150
                    self._debug_log(f"DEBUG: Group {group_name} MSVC arm64 path match (+150)")

            # Match compiler type (high priority)
            if group_info["compiler_type"] == compiler.compiler_type:
                score += 100
                self._debug_log(f"DEBUG: Group {group_name} compiler type match (+100): {group_info['compiler_type']} == {compiler.compiler_type}")
            elif group_info["compiler_categories"] == compiler.compiler_type:
                score += 100
                self._debug_log(f"DEBUG: Group {group_name} compiler categories match (+100): {group_info['compiler_categories']} == {compiler.compiler_type}")
            elif compiler.compiler_type and compiler.compiler_type.lower() in group_name.lower():
                score += 80
                self._debug_log(f"DEBUG: Group {group_name} name contains compiler type (+80): {compiler.compiler_type} in {group_name}")
                
            self._debug_log(f"DEBUG: Group {group_name} total score: {score}")

            # Match target architecture in group name (medium priority)
            if compiler.target and compiler.is_cross_compiler:
                target_arch = compiler.target.split("-")[0].lower()
                if group_info["group_name"] and target_arch in group_info["group_name"].lower():
                    score += 70
                elif target_arch in group_name.lower():
                    score += 60

            # Prefer groups with similar naming patterns (low priority)
            if compiler.compiler_type:
                if group_name.lower().startswith(compiler.compiler_type.lower()):
                    score += 30

            # For native compilers, prefer groups without cross-architecture indicators
            if not compiler.is_cross_compiler:
                cross_indicators = ["arm", "aarch64", "mips", "sparc", "powerpc", "riscv", "s390x"]
                if not any(arch in group_name.lower() for arch in cross_indicators):
                    score += 20

            # Prefer larger groups (more compilers = more established)
            if group_info["compilers"]:
                compiler_count = len([c for c in group_info["compilers"].split(":") if c.strip()])
                if compiler_count > 10:
                    score += 15
                elif compiler_count > 5:
                    score += 10
                elif compiler_count > 1:
                    score += 5

            if score > 0:
                group_scores.append((score, group_name))

        # Return the highest scoring group
        if group_scores:
            group_scores.sort(reverse=True)
            return group_scores[0][1]

        # Fallback: create a new group name based on compiler type and architecture
        if compiler.is_cross_compiler and compiler.target:
            arch = compiler.target.split("-")[0]
            return f"{compiler.compiler_type or 'compiler'}{arch}"
        else:
            return compiler.compiler_type or "custom"

    def add_compiler(self, compiler: CompilerInfo):
        """Add a compiler to the configuration using surgical editing."""
        file_path = self.get_properties_path(compiler.language)

        # Ensure unique ID
        compiler.id = self.ensure_compiler_id_unique(compiler.id, compiler.language)

        # Ensure semver is always added - if not detected, try to extract it
        if not compiler.semver:
            compiler.semver = self._extract_compiler_version(compiler.exe)

        # Use surgical editor for minimal changes
        editor = PropertiesFileEditor(file_path)

        # Add group to compilers line if not present
        if compiler.group:
            editor.add_group_to_compilers_line(compiler.group)

            # Create group section if it doesn't exist
            if not editor.group_exists(compiler.group):
                editor.create_group_section(compiler.group, [compiler.id])

                # Add group properties
                self._add_group_properties_surgical(editor, compiler.group, compiler)
            else:
                # Add compiler to existing group
                editor.add_compiler_to_group(compiler.group, compiler.id)

                # Add missing group properties
                self._add_group_properties_surgical(editor, compiler.group, compiler)

        # Create compiler section
        editor.create_compiler_section(compiler)

        # Ensure proper spacing after the new compiler
        editor.ensure_proper_spacing_after_compiler(compiler.id)

        # Ensure libs and tools sections exist
        editor.ensure_libs_tools_sections()

        # Save the file
        editor.save_file()

    def _add_group_properties_surgical(self, editor: PropertiesFileEditor, group_name: str, compiler: CompilerInfo):
        """Add group properties using surgical editing."""
        # Always add isSemVer=true for new groups
        editor.add_group_property(group_name, "isSemVer", "true")

        # Extract architecture and compiler type for naming
        if compiler and compiler.is_cross_compiler and compiler.target:
            # For cross-compilers, extract arch from target
            arch = compiler.target.split("-")[0]
            compiler_type = compiler.compiler_type or "compiler"

            # Detect instruction set
            instruction_set = detect_instruction_set_from_target(compiler.target, compiler.exe)

            # Set group properties for cross-compilers
            editor.add_group_property(group_name, "groupName", f"{compiler_type.title()} {arch}")
            editor.add_group_property(group_name, "baseName", f"{arch} {compiler_type}")
            editor.add_group_property(group_name, "instructionSet", instruction_set)
        else:
            # For native compilers
            compiler_type = compiler.compiler_type if compiler else group_name

            # Detect instruction set from executable path
            instruction_set = detect_instruction_set_from_target(None, compiler.exe if compiler else "")

            # Set group properties for native compilers
            editor.add_group_property(group_name, "groupName", f"{compiler_type.title()}")
            editor.add_group_property(group_name, "baseName", compiler_type)
            editor.add_group_property(group_name, "instructionSet", instruction_set)

        # Add group properties based on known types
        if group_name == "gcc" or (compiler and compiler.compiler_type == "gcc"):
            editor.add_group_property(group_name, "compilerType", "gcc")
            editor.add_group_property(group_name, "compilerCategories", "gcc")
        elif group_name == "clang" or (compiler and compiler.compiler_type == "clang"):
            editor.add_group_property(group_name, "compilerType", "clang")
            editor.add_group_property(group_name, "compilerCategories", "clang")
            editor.add_group_property(group_name, "intelAsm", "-mllvm --x86-asm-syntax=intel")
        elif group_name in ["icc", "icx"] or (compiler and compiler.compiler_type in ["icc", "icx"]):
            editor.add_group_property(group_name, "compilerType", compiler.compiler_type if compiler else group_name)
            editor.add_group_property(group_name, "compilerCategories", "intel")
        elif group_name == "win32-vc" or (compiler and compiler.compiler_type == "win32-vc"):
            # MSVC-specific properties
            editor.add_group_property(group_name, "compilerType", "win32-vc")
            editor.add_group_property(group_name, "compilerCategories", "msvc")
            editor.add_group_property(group_name, "versionFlag", "/?")
            editor.add_group_property(group_name, "versionRe", "^.*Microsoft \\(R\\).*$")
            editor.add_group_property(group_name, "needsMulti", "false")
            editor.add_group_property(group_name, "includeFlag", "/I")
            editor.add_group_property(group_name, "options", "/EHsc /utf-8 /MD")
        elif compiler and compiler.compiler_type:
            # For other known compiler types
            editor.add_group_property(group_name, "compilerType", compiler.compiler_type)

    def _add_to_group(
        self, properties: OrderedDict, group_name: str, compiler_id: str, compiler: Optional[CompilerInfo] = None
    ):
        """Add compiler to a group, creating group if necessary."""
        group_key = f"group.{group_name}.compilers"

        # Get existing groups
        groups = self.get_existing_groups(properties)

        if group_name in groups:
            # Add to existing group if not already there
            if compiler_id not in groups[group_name]:
                groups[group_name].append(compiler_id)
                # Update properties with colon separator
                properties[group_key] = ":".join(groups[group_name])
        else:
            # Create new group
            properties[group_key] = compiler_id

            # Always add isSemVer=true for new groups
            properties[f"group.{group_name}.isSemVer"] = "true"

            # Extract architecture and compiler type for naming
            if compiler and compiler.is_cross_compiler and compiler.target:
                # For cross-compilers, extract arch from target
                arch = compiler.target.split("-")[0]
                compiler_type = compiler.compiler_type or "compiler"

                # Detect instruction set
                instruction_set = detect_instruction_set_from_target(compiler.target, compiler.exe)

                # Set group properties for cross-compilers
                properties[f"group.{group_name}.groupName"] = f"{compiler_type.title()} {arch}"
                properties[f"group.{group_name}.baseName"] = f"{arch} {compiler_type}"
                properties[f"group.{group_name}.instructionSet"] = instruction_set
            else:
                # For native compilers
                compiler_type = compiler.compiler_type if compiler else group_name

                # Detect instruction set from executable path
                instruction_set = detect_instruction_set_from_target(None, compiler.exe if compiler else "")

                # Set group properties for native compilers
                properties[f"group.{group_name}.groupName"] = f"{compiler_type.title()}"
                properties[f"group.{group_name}.baseName"] = compiler_type
                properties[f"group.{group_name}.instructionSet"] = instruction_set

            # Add group properties based on known types
            if group_name == "gcc" or (compiler and compiler.compiler_type == "gcc"):
                properties[f"group.{group_name}.compilerType"] = "gcc"
                properties[f"group.{group_name}.compilerCategories"] = "gcc"
            elif group_name == "clang" or (compiler and compiler.compiler_type == "clang"):
                properties[f"group.{group_name}.compilerType"] = "clang"
                properties[f"group.{group_name}.compilerCategories"] = "clang"
                properties[f"group.{group_name}.intelAsm"] = "-mllvm --x86-asm-syntax=intel"
            elif group_name in ["icc", "icx"] or (compiler and compiler.compiler_type in ["icc", "icx"]):
                properties[f"group.{group_name}.compilerType"] = compiler.compiler_type if compiler else group_name
                properties[f"group.{group_name}.compilerCategories"] = "intel"
            elif compiler and compiler.compiler_type:
                # For other known compiler types
                properties[f"group.{group_name}.compilerType"] = compiler.compiler_type

    def _reorganize_properties(self, properties: OrderedDict):
        """Reorganize properties in the correct order: compilers line, group definitions, compiler definitions."""
        # Find all groups defined in this file
        groups = set()
        for key in properties:
            if isinstance(key, str) and key.startswith("group.") and key.endswith(".compilers"):
                # Extract group name from group.{name}.compilers
                group_name = key.split(".")[1]
                groups.add(group_name)

        # Create new ordered properties
        new_properties = OrderedDict()

        # 1. Add compilers line at the top if we have groups
        if groups:
            compilers_value = ":".join(f"&{group}" for group in sorted(groups))
            new_properties["compilers"] = compilers_value

        # 2. Add other non-group, non-compiler properties (like defaultCompiler, objdumper, etc.)
        # But exclude libs and tools which will be added at the end
        for key, value in properties.items():
            if (
                key != "compilers"
                and key not in ("libs", "tools")
                and not key.startswith("group.")
                and not key.startswith("compiler.")
                and not key.startswith("__comment_")
            ):
                new_properties[key] = value

        # 3. Add all group definitions, ensuring required fields exist
        group_names_processed = set()
        for key, value in properties.items():
            if key.startswith("group."):
                new_properties[key] = value

                # Track group names and ensure required fields exist
                match = re.match(r"^group\.([^.]+)\.compilers$", key)
                if match:
                    group_name = match.group(1)
                    group_names_processed.add(group_name)

                    # Ensure isSemVer exists
                    if f"group.{group_name}.isSemVer" not in properties:
                        new_properties[f"group.{group_name}.isSemVer"] = "true"

                    # Try to determine if this is a cross-compiler group and add missing fields
                    # Look for compiler examples in this group to determine architecture
                    compiler_ids = [c.strip() for c in value.split(":") if c.strip()]
                    sample_compiler_exe = None
                    sample_compiler_type = None
                    sample_target = None

                    # Find a sample compiler from this group
                    for compiler_id in compiler_ids:
                        exe_key = f"compiler.{compiler_id}.exe"
                        type_key = f"compiler.{compiler_id}.compilerType"
                        if exe_key in properties:
                            sample_compiler_exe = properties[exe_key]
                            sample_compiler_type = properties.get(type_key, group_name)
                            # Try to detect if it's a cross-compiler from the exe path
                            if "-" in sample_compiler_exe and any(
                                arch in sample_compiler_exe
                                for arch in ["s390x", "sparc", "aarch64", "arm", "mips", "powerpc"]
                            ):
                                # Extract target from path
                                path_parts = sample_compiler_exe.split("/")
                                for part in path_parts:
                                    if "-" in part and any(
                                        arch in part for arch in ["s390x", "sparc", "aarch64", "arm", "mips", "powerpc"]
                                    ):
                                        sample_target = part
                                        break
                            break

                    # Add missing groupName if not present
                    if f"group.{group_name}.groupName" not in properties and sample_compiler_type:
                        if sample_target:
                            # Cross-compiler
                            arch = sample_target.split("-")[0] if sample_target else "unknown"
                            new_properties[f"group.{group_name}.groupName"] = f"{sample_compiler_type.title()} {arch}"
                        else:
                            # Native compiler
                            new_properties[f"group.{group_name}.groupName"] = f"{sample_compiler_type.title()}"

                    # Add missing baseName if not present
                    if f"group.{group_name}.baseName" not in properties and sample_compiler_type:
                        if sample_target:
                            # Cross-compiler
                            arch = sample_target.split("-")[0] if sample_target else "unknown"
                            new_properties[f"group.{group_name}.baseName"] = f"{arch} {sample_compiler_type}"
                        else:
                            # Native compiler
                            new_properties[f"group.{group_name}.baseName"] = sample_compiler_type

                    # Add missing instructionSet if not present
                    if f"group.{group_name}.instructionSet" not in properties:
                        if sample_compiler_exe:
                            instruction_set = detect_instruction_set_from_target(sample_target, sample_compiler_exe)
                            new_properties[f"group.{group_name}.instructionSet"] = instruction_set

        # 4. Add all compiler definitions, ensuring semver fields exist and removing name if semver exists
        compiler_ids_processed = set()
        compiler_semvers = {}  # Track which compilers have semver

        # First pass: collect all compiler properties and track semvers
        for key, value in properties.items():
            if key.startswith("compiler."):
                match = re.match(r"^compiler\.([^.]+)\.(.+)$", key)
                if match:
                    compiler_id, prop_type = match.groups()
                    if prop_type == "semver":
                        compiler_semvers[compiler_id] = value
                    elif prop_type == "exe":
                        compiler_ids_processed.add(compiler_id)
                        # Check if this compiler has a semver field
                        semver_key = f"compiler.{compiler_id}.semver"
                        if semver_key not in properties:
                            # Try to extract semver from the compiler executable
                            semver = self._extract_compiler_version(value)
                            if semver:
                                compiler_semvers[compiler_id] = semver

        # Second pass: add properties, skipping name if semver exists
        for key, value in properties.items():
            if key.startswith("compiler."):
                match = re.match(r"^compiler\.([^.]+)\.(.+)$", key)
                if match:
                    compiler_id, prop_type = match.groups()

                    # Skip name property if this compiler has semver
                    if prop_type == "name" and compiler_id in compiler_semvers:
                        continue

                    new_properties[key] = value

        # Add any newly extracted semvers
        for compiler_id, semver in compiler_semvers.items():
            semver_key = f"compiler.{compiler_id}.semver"
            if semver_key not in new_properties:
                new_properties[semver_key] = semver

        # 5. Add libs= and tools= at the end (preserve existing values if they exist)
        # Always add the libs section header comments right before libs
        # (any duplicates from old __comment_ entries will be filtered out in step 6)
        new_properties["__libs_section_empty__"] = ""
        new_properties["__libs_section_border1__"] = "#################################"
        new_properties["__libs_section_border2__"] = "#################################"
        new_properties["__libs_section_title__"] = "# Installed libs"

        if "libs" in properties:
            new_properties["libs"] = properties["libs"]
        else:
            new_properties["libs"] = ""

        # Add tools section header comments
        new_properties["__tools_section_empty__"] = ""
        new_properties["__tools_section_border1__"] = "#################################"
        new_properties["__tools_section_border2__"] = "#################################"
        new_properties["__tools_section_title__"] = "# Installed tools"

        if "tools" in properties:
            new_properties["tools"] = properties["tools"]
        else:
            new_properties["tools"] = ""

        # 6. Add comments at the end (but exclude libs/tools section comments and empty lines to avoid duplicates)
        for key, value in properties.items():
            if (
                key.startswith("__comment_")
                and not key.startswith("__comment_libs_header")
                and not key.startswith("__libs_section_")
                and not key.startswith("__tools_section_")
                and "# Installed libs" not in str(value)
                and "# Installed tools" not in str(value)
                and "#################################" not in str(value)
                and str(value).strip() != ""
            ):  # Filter out empty comment lines
                new_properties[key] = value

        # Replace the properties with the new ordered dict
        properties.clear()
        properties.update(new_properties)

    def reorganize_existing_file(self, language: str):
        """Add missing properties to an existing file using surgical editing."""
        file_path = self.get_properties_path(language)
        if not file_path.exists():
            return

        # Use surgical editor for minimal changes
        editor = PropertiesFileEditor(file_path)

        # Add missing semver fields to existing compilers
        self._add_missing_semver_fields_surgical(editor)

        # Add missing group properties to existing groups
        self._add_missing_group_properties_surgical(editor)

        # Ensure libs and tools sections exist
        editor.ensure_libs_tools_sections()

        # Save the file
        editor.save_file()

    def _add_missing_semver_fields_surgical(self, editor: PropertiesFileEditor):
        """Add missing semver fields to existing compilers using surgical editing."""
        # Find all compiler .exe properties
        for i, line in enumerate(editor.lines):
            if ".exe=" in line and line.startswith("compiler."):
                # Extract compiler ID
                match = re.match(r"^compiler\.([^.]+)\.exe=(.+)$", line)
                if match:
                    compiler_id, exe_path = match.groups()

                    # Check if semver already exists
                    has_semver = any(line.startswith(f"compiler.{compiler_id}.semver=") for line in editor.lines)
                    if not has_semver:
                        # Try to extract semver
                        semver = self._extract_compiler_version(exe_path)
                        if semver:
                            editor.add_compiler_property(compiler_id, "semver", semver)

                    # Remove name if semver exists or was just added
                    has_semver_after = any(line.startswith(f"compiler.{compiler_id}.semver=") for line in editor.lines)
                    if has_semver_after:
                        # Remove name property if it exists
                        for j, name_line in enumerate(editor.lines):
                            if name_line.startswith(f"compiler.{compiler_id}.name="):
                                editor.lines.pop(j)
                                break

    def _add_missing_group_properties_surgical(self, editor: PropertiesFileEditor):
        """Add missing group properties to existing groups using surgical editing."""
        # Find all group.*.compilers properties
        for i, line in enumerate(editor.lines):
            if ".compilers=" in line and line.startswith("group."):
                # Extract group name
                match = re.match(r"^group\.([^.]+)\.compilers=(.*)$", line)
                if match:
                    group_name, compilers_list = match.groups()

                    # Add missing properties
                    editor.add_group_property(group_name, "isSemVer", "true")

                    # Try to determine compiler type and architecture from first compiler
                    if compilers_list:
                        first_compiler_id = compilers_list.split(":")[0].strip()
                        if first_compiler_id.startswith("&"):
                            first_compiler_id = first_compiler_id[1:]

                        # Find the first compiler's exe path to determine properties
                        compiler_exe = None
                        compiler_type = None
                        target = None

                        for exe_line in editor.lines:
                            if exe_line.startswith(f"compiler.{first_compiler_id}.exe="):
                                compiler_exe = exe_line.split("=", 1)[1]
                                break

                        for type_line in editor.lines:
                            if type_line.startswith(f"compiler.{first_compiler_id}.compilerType="):
                                compiler_type = type_line.split("=", 1)[1]
                                break

                        # Detect if it's a cross-compiler from the exe path
                        is_cross = False
                        if compiler_exe and "-" in compiler_exe:
                            cross_indicators = ["s390x", "sparc", "aarch64", "arm", "mips", "powerpc"]
                            if any(arch in compiler_exe for arch in cross_indicators):
                                is_cross = True
                                # Extract target from path
                                path_parts = compiler_exe.split("/")
                                for part in path_parts:
                                    if "-" in part and any(arch in part for arch in cross_indicators):
                                        target = part
                                        break

                        # Set group properties
                        if is_cross and target:
                            # Cross-compiler
                            arch = target.split("-")[0] if target else "unknown"
                            comp_type = compiler_type or group_name
                            editor.add_group_property(group_name, "groupName", f"{comp_type.title()} {arch}")
                            editor.add_group_property(group_name, "baseName", f"{arch} {comp_type}")
                            instruction_set = detect_instruction_set_from_target(target, compiler_exe or "")
                            editor.add_group_property(group_name, "instructionSet", instruction_set)
                        else:
                            # Native compiler
                            comp_type = compiler_type or group_name
                            editor.add_group_property(group_name, "groupName", f"{comp_type.title()}")
                            editor.add_group_property(group_name, "baseName", comp_type)
                            instruction_set = detect_instruction_set_from_target(None, compiler_exe or "")
                            editor.add_group_property(group_name, "instructionSet", instruction_set)

                        # Add type-specific properties
                        if comp_type == "gcc" or group_name == "gcc":
                            editor.add_group_property(group_name, "compilerType", "gcc")
                            editor.add_group_property(group_name, "compilerCategories", "gcc")
                        elif comp_type == "clang" or group_name == "clang":
                            editor.add_group_property(group_name, "compilerType", "clang")
                            editor.add_group_property(group_name, "compilerCategories", "clang")
                            editor.add_group_property(group_name, "intelAsm", "-mllvm --x86-asm-syntax=intel")
                        elif comp_type in ["icc", "icx"] or group_name in ["icc", "icx"]:
                            editor.add_group_property(group_name, "compilerType", comp_type or group_name)
                            editor.add_group_property(group_name, "compilerCategories", "intel")
                        elif comp_type:
                            editor.add_group_property(group_name, "compilerType", comp_type)

    def validate_with_discovery(self, language: str, compiler_id: str) -> tuple[bool, str, Optional[str]]:
        """Validate that the compiler is discovered by running npm run dev with discovery-only."""
        import json
        import os
        import subprocess
        import tempfile

        # Check if local properties file exists - skip validation if it does (only for non-local environments)
        if self.env != "local":
            local_file = self.get_local_properties_path(language)
            if local_file.exists():
                return (
                    True,
                    f"Skipping discovery validation for {self.env} environment because {local_file.name} exists",
                    None,
                )

        # Create a temporary file for discovery output
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            discovery_file = f.name

        try:
            # Find the main CE directory (go up from etc/config to the root)
            ce_root = self.config_dir.parent.parent

            # Run npm run dev with discovery-only, including environment if not local
            # On Windows, we might need to use npm.cmd instead of npm
            import platform
            npm_cmd = "npm.cmd" if platform.system() == "Windows" else "npm"
            cmd = [npm_cmd, "run", "dev", "--", "--language", language]
            if self.env != "local":
                cmd.extend(["--env", self.env])
            cmd.extend(["--discovery-only", discovery_file])

            print(f"DEBUG: Running discovery command: {' '.join(cmd)}")
            print(f"DEBUG: Working directory: {ce_root}")

            result = subprocess.run(
                cmd, cwd=ce_root, capture_output=True, text=True, timeout=60  # Discovery should be relatively fast
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()
                # On Windows, if npm fails due to PATH issues, make discovery optional
                if platform.system() == "Windows" and ("Is a directory" in error_msg or "not found" in error_msg):
                    return True, f"Discovery validation skipped on Windows (npm PATH issue): {error_msg}", None
                return False, f"Discovery command failed: {error_msg}", None

            # Read and parse the discovery JSON
            if not os.path.exists(discovery_file):
                return False, f"Discovery file not created: {discovery_file}", None

            with open(discovery_file, "r") as f:
                discovery_data = json.load(f)

            # Check if our compiler ID is in the discovery results
            # The discovery JSON structure might vary, so let's be defensive
            compilers = []
            if isinstance(discovery_data, dict):
                compilers = discovery_data.get("compilers", [])
            elif isinstance(discovery_data, list):
                compilers = discovery_data

            found_compiler = None

            for compiler in compilers:
                # Handle both dict and potentially other formats
                if isinstance(compiler, dict):
                    if compiler.get("id") == compiler_id:
                        found_compiler = compiler
                        break
                elif isinstance(compiler, str):
                    # If it's just a string ID
                    if compiler == compiler_id:
                        found_compiler = {"id": compiler, "name": compiler}
                        break

            if found_compiler:
                compiler_name = found_compiler.get("name", found_compiler.get("id", "unknown"))
                # Extract semver from the discovered compiler
                discovered_semver = found_compiler.get("semver") or found_compiler.get("version")
                return True, f"Compiler '{compiler_id}' successfully discovered as '{compiler_name}'", discovered_semver
            else:
                # List available compiler IDs for debugging
                available_ids = []
                for c in compilers[:10]:  # First 10 for brevity
                    if isinstance(c, dict):
                        available_ids.append(c.get("id", "unknown"))
                    elif isinstance(c, str):
                        available_ids.append(c)
                    else:
                        available_ids.append(str(c))

                # Also show the raw structure for debugging
                data_preview = (
                    str(discovery_data)[:200] + "..." if len(str(discovery_data)) > 200 else str(discovery_data)
                )
                return (
                    False,
                    f"Compiler '{compiler_id}' not found in discovery results. "
                    f"Available IDs (first 10): {available_ids}. "
                    f"Data structure preview: {data_preview}",
                    None,
                )

        except subprocess.TimeoutExpired:
            return False, "Discovery validation timed out (60s)", None
        except json.JSONDecodeError as e:
            return False, f"Discovery JSON parse error: {str(e)}", None
        except Exception as e:
            # On Windows, if subprocess fails due to npm not being found, make discovery optional
            import platform
            if platform.system() == "Windows" and ("The system cannot find the file specified" in str(e) or "WinError 2" in str(e)):
                return True, f"Discovery validation skipped on Windows (npm not found): {str(e)}", None
            return False, f"Discovery validation error: {str(e)}", None
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(discovery_file):
                    os.unlink(discovery_file)
            except OSError:
                pass  # Ignore cleanup errors

    def validate_with_propscheck(self, language: str) -> tuple[bool, str]:
        """Validate properties file with propscheck.py."""
        propscheck_path = self.config_dir.parent / "scripts" / "util" / "propscheck.py"
        if not propscheck_path.exists():
            return True, "Warning: propscheck.py not found, skipping validation"

        file_path = self.get_properties_path(language)
        if not file_path.exists():
            return True, f"No {self.env} properties file to validate"

        import subprocess

        try:
            # propscheck.py takes --config-dir parameter and --check-local for local properties files
            # Use the same Python interpreter that's running this script
            import sys
            cmd = [sys.executable, str(propscheck_path), "--config-dir", str(self.config_dir)]
            if self.env == "local":
                cmd.append("--check-local")
            else:
                # For other environments, we might need to add specific flags or just run without --check-local
                # This depends on how propscheck.py handles non-local environments
                cmd.append("--check-local")  # Keep this for now, may need adjustment

            print(f"DEBUG: Running propscheck command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                return True, "Properties validated successfully"
            else:
                error_output = result.stdout + result.stderr
                # Always return the validation output so we can learn from the issues
                return False, f"Validation issues detected:\n{error_output}"

        except subprocess.TimeoutExpired:
            return False, "Validation timed out"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
