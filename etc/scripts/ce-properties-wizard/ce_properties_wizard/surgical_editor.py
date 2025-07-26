"""Surgical properties file editor that preserves existing structure."""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from .models import CompilerInfo
from .utils import create_backup


class PropertiesFileEditor:
    """Surgical editor that makes minimal changes to properties files."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.lines: List[str] = []
        self.load_file()

    def load_file(self):
        """Load file content, preserving all structure."""
        if self.file_path.exists():
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.lines = [line.rstrip("\n") for line in f.readlines()]
        else:
            self.lines = []

    def save_file(self):
        """Save file with minimal changes."""
        # Create backup if file exists
        if self.file_path.exists():
            create_backup(self.file_path)

        with open(self.file_path, "w", encoding="utf-8") as f:
            for line in self.lines:
                f.write(f"{line}\n")

    def find_compilers_line(self) -> Optional[int]:
        """Find the compilers= line."""
        for i, line in enumerate(self.lines):
            if line.startswith("compilers="):
                return i
        return None

    def find_group_section(self, group_name: str) -> Tuple[Optional[int], Optional[int]]:
        """Find the start and end of a group section.

        Returns:
            (start_line, end_line) where start_line is the first group.{name}. line
            and end_line is the line before the next group or compiler section starts.
        """
        start_line = None
        end_line = None

        # Find start of this group
        group_prefix = f"group.{group_name}."
        for i, line in enumerate(self.lines):
            if line.startswith(group_prefix):
                start_line = i
                break

        if start_line is None:
            return None, None

        # Find end of this group (before next group or compiler section)
        for i in range(start_line + 1, len(self.lines)):
            line = self.lines[i].strip()
            if (
                line.startswith("group.")
                and not line.startswith(group_prefix)
                or line.startswith("compiler.")
                or line.startswith("libs=")
                or line.startswith("tools=")
                or line.startswith("#")
                and ("####" in line or "Installed" in line)
            ):
                end_line = i
                break

        if end_line is None:
            end_line = len(self.lines)

        return start_line, end_line

    def find_compiler_section(self, compiler_id: str) -> Tuple[Optional[int], Optional[int]]:
        """Find the start and end of a compiler section."""
        start_line = None
        end_line = None

        # Find start of this compiler
        compiler_prefix = f"compiler.{compiler_id}."
        for i, line in enumerate(self.lines):
            if line.startswith(compiler_prefix):
                start_line = i
                break

        if start_line is None:
            return None, None

        # Find end of this compiler (before next compiler, group, or libs/tools section)
        for i in range(start_line + 1, len(self.lines)):
            line = self.lines[i].strip()
            if (
                line.startswith("compiler.")
                and not line.startswith(compiler_prefix)
                or line.startswith("group.")
                or line.startswith("libs=")
                or line.startswith("tools=")
                or line.startswith("#")
                and ("####" in line or "Installed" in line)
            ):
                end_line = i
                break

        if end_line is None:
            end_line = len(self.lines)

        return start_line, end_line

    def get_existing_groups_from_compilers_line(self) -> List[str]:
        """Extract group names from the compilers= line."""
        compilers_line_idx = self.find_compilers_line()
        if compilers_line_idx is None:
            return []

        line = self.lines[compilers_line_idx]
        # Extract groups from compilers=&group1:&group2:...
        if "=" in line:
            value = line.split("=", 1)[1]
            groups = []
            for part in value.split(":"):
                part = part.strip()
                if part.startswith("&"):
                    groups.append(part[1:])  # Remove & prefix
            return groups
        return []

    def add_group_to_compilers_line(self, group_name: str):
        """Add a group to the compilers= line if not already present."""
        existing_groups = self.get_existing_groups_from_compilers_line()
        if group_name in existing_groups:
            return  # Already exists
            
        # Check if this group is referenced by any existing parent groups
        # (e.g., vcpp_x64 might be referenced by group.vcpp.compilers=&vcpp_x86:&vcpp_x64:&vcpp_arm64)
        if self._is_group_referenced_elsewhere(group_name):
            return  # Already referenced by another group

        compilers_line_idx = self.find_compilers_line()
        if compilers_line_idx is None:
            # No compilers line exists, create one
            self.lines.insert(0, f"compilers=&{group_name}")
            return

        # Add to existing line
        line = self.lines[compilers_line_idx]
        if line.endswith("="):
            # Empty compilers line, just append without colon
            self.lines[compilers_line_idx] = f"{line}&{group_name}"
        elif line.endswith(":"):
            # Line ends with colon, just append
            self.lines[compilers_line_idx] = f"{line}&{group_name}"
        else:
            # Add with colon separator
            self.lines[compilers_line_idx] = f"{line}:&{group_name}"
            
    def _is_group_referenced_elsewhere(self, group_name: str) -> bool:
        """Check if a group is referenced by any other group's compilers list."""
        for line in self.lines:
            # Look for group.*.compilers= lines that reference this group
            if ".compilers=" in line and not line.startswith(f"group.{group_name}.compilers="):
                # Extract the value part after =
                if "=" in line:
                    value = line.split("=", 1)[1]
                    # Check if this group is referenced (with & prefix)
                    referenced_groups = []
                    for part in value.split(":"):
                        part = part.strip()
                        if part.startswith("&"):
                            referenced_groups.append(part[1:])  # Remove & prefix
                    
                    if group_name in referenced_groups:
                        return True
        return False

    def group_exists(self, group_name: str) -> bool:
        """Check if a group already exists in the file."""
        start_line, _ = self.find_group_section(group_name)
        return start_line is not None

    def compiler_exists(self, compiler_id: str) -> bool:
        """Check if a compiler already exists in the file."""
        start_line, _ = self.find_compiler_section(compiler_id)
        return start_line is not None

    def find_insertion_point_for_group(self, group_name: str) -> int:
        """Find the best place to insert a new group section."""
        # Find the end of both existing groups AND all compilers
        last_group_end = 0
        last_compiler_end = 0

        # Find end of existing groups
        for i, line in enumerate(self.lines):
            if line.startswith("group."):
                # Find end of this group
                group_match = re.match(r"^group\.([^.]+)\.", line)
                if group_match:
                    current_group = group_match.group(1)
                    _, end_line = self.find_group_section(current_group)
                    if end_line is not None:
                        last_group_end = max(last_group_end, end_line)

        # Find end of all compilers
        for i, line in enumerate(self.lines):
            if line.startswith("compiler."):
                # Find end of this compiler
                compiler_match = re.match(r"^compiler\.([^.]+)\.", line)
                if compiler_match:
                    current_compiler = compiler_match.group(1)
                    _, end_line = self.find_compiler_section(current_compiler)
                    if end_line is not None:
                        last_compiler_end = max(last_compiler_end, end_line)

        # Insert after whichever comes last: groups or compilers
        insertion_point = max(last_group_end, last_compiler_end)

        # If neither groups nor compilers found, insert after compilers line
        if insertion_point == 0:
            compilers_line_idx = self.find_compilers_line()
            if compilers_line_idx is not None:
                # Insert after compilers line and any following blank lines
                insertion_point = compilers_line_idx + 1
                while insertion_point < len(self.lines) and self.lines[insertion_point].strip() == "":
                    insertion_point += 1
                return insertion_point
            else:
                return 0

        return insertion_point

    def find_insertion_point_for_compiler(self, compiler_id: str, group_name: Optional[str] = None) -> int:
        """Find the best place to insert a new compiler section."""
        # If we have a group, try to insert at the end of that group's compilers
        if group_name:
            group_start, group_end = self.find_group_section(group_name)
            if group_start is not None:
                # Look for compilers from this group after the group definition
                last_compiler_end = group_end

                # Find compilers that belong to this group
                compilers_in_group = self.get_compilers_in_group(group_name)
                for comp_id in compilers_in_group:
                    if comp_id != compiler_id:  # Don't include ourselves
                        _, comp_end = self.find_compiler_section(comp_id)
                        if comp_end is not None:
                            last_compiler_end = max(last_compiler_end, comp_end)

                return last_compiler_end

        # Fallback: find the end of all compilers, but insert before libs/tools
        last_compiler_end = 0
        libs_tools_start = len(self.lines)  # Default to end of file

        # Find where libs/tools sections start
        for i, line in enumerate(self.lines):
            if (
                line.startswith("libs=")
                or line.startswith("tools=")
                or (line.startswith("#") and ("####" in line or "Installed" in line))
            ):
                libs_tools_start = i
                break

        # Find end of all compilers, but only those before libs/tools
        for i, line in enumerate(self.lines):
            if i >= libs_tools_start:
                break
            if line.startswith("compiler."):
                # Find end of this compiler
                compiler_match = re.match(r"^compiler\.([^.]+)\.", line)
                if compiler_match:
                    current_compiler = compiler_match.group(1)
                    _, end_line = self.find_compiler_section(current_compiler)
                    if end_line is not None and end_line <= libs_tools_start:
                        last_compiler_end = max(last_compiler_end, end_line)

        if last_compiler_end == 0:
            # No compilers found, insert after groups but before libs/tools
            group_insertion = self.find_insertion_point_for_group("dummy")
            return min(group_insertion, libs_tools_start)

        return min(last_compiler_end, libs_tools_start)

    def get_compilers_in_group(self, group_name: str) -> List[str]:
        """Get list of compiler IDs in a group."""
        group_start, group_end = self.find_group_section(group_name)
        if group_start is None:
            return []

        # Look for group.{name}.compilers line
        compilers_key = f"group.{group_name}.compilers"
        for i in range(group_start, group_end):
            line = self.lines[i]
            if line.startswith(compilers_key + "="):
                value = line.split("=", 1)[1]
                # Parse compiler list (could be : separated or & prefixed)
                compilers = []
                for part in value.split(":"):
                    part = part.strip()
                    if part.startswith("&"):
                        part = part[1:]  # Remove & prefix
                    if part:
                        compilers.append(part)
                return compilers

        return []

    def add_compiler_to_group(self, group_name: str, compiler_id: str):
        """Add a compiler to a group's compilers list."""
        group_start, group_end = self.find_group_section(group_name)
        if group_start is None:
            return  # Group doesn't exist

        # Find the group.{name}.compilers line
        compilers_key = f"group.{group_name}.compilers"
        for i in range(group_start, group_end):
            line = self.lines[i]
            if line.startswith(compilers_key + "="):
                # Check if compiler is already in the list
                existing_compilers = self.get_compilers_in_group(group_name)
                if compiler_id in existing_compilers:
                    return  # Already exists

                # Add to the list
                if line.endswith("="):
                    # Empty list
                    self.lines[i] = f"{line}{compiler_id}"
                else:
                    # Add with colon separator
                    self.lines[i] = f"{line}:{compiler_id}"
                return

    def add_group_property(self, group_name: str, property_name: str, value: str):
        """Add a property to a group if it doesn't already exist."""
        group_start, group_end = self.find_group_section(group_name)
        if group_start is None:
            return  # Group doesn't exist

        # Check if property already exists
        prop_key = f"group.{group_name}.{property_name}"
        for i in range(group_start, group_end):
            line = self.lines[i]
            if line.startswith(prop_key + "="):
                return  # Already exists

        # Find a good place to insert (after the compilers line if it exists)
        insertion_point = group_start + 1
        compilers_key = f"group.{group_name}.compilers"
        for i in range(group_start, group_end):
            line = self.lines[i]
            if line.startswith(compilers_key + "="):
                insertion_point = i + 1
                break

        # Insert the new property
        self.lines.insert(insertion_point, f"{prop_key}={value}")

    def get_group_property(self, group_name: str, property_name: str) -> Optional[str]:
        """Get a property value from a group."""
        group_start, group_end = self.find_group_section(group_name)
        if group_start is None:
            return None

        # Check if property exists
        prop_key = f"group.{group_name}.{property_name}"
        for i in range(group_start, group_end):
            line = self.lines[i]
            if line.startswith(prop_key + "="):
                return line.split("=", 1)[1]

        return None

    def add_compiler_property(self, compiler_id: str, property_name: str, value: str):
        """Add a property to a compiler if it doesn't already exist."""
        compiler_start, compiler_end = self.find_compiler_section(compiler_id)
        if compiler_start is None:
            return  # Compiler doesn't exist

        # Check if property already exists
        prop_key = f"compiler.{compiler_id}.{property_name}"
        for i in range(compiler_start, compiler_end):
            line = self.lines[i]
            if line.startswith(prop_key + "="):
                return  # Already exists

        # Insert at the end of the compiler section
        insertion_point = compiler_end

        # Try to insert in a logical order (exe, name, semver, compilerType, options, etc.)
        desired_order = ["exe", "semver", "name", "compilerType", "options"]
        if property_name in desired_order:
            target_index = desired_order.index(property_name)

            # Find where to insert based on order
            for i in range(compiler_start, compiler_end):
                line = self.lines[i]
                if line.startswith(f"compiler.{compiler_id}."):
                    existing_prop = line.split(".", 2)[2].split("=")[0]
                    if existing_prop in desired_order:
                        existing_index = desired_order.index(existing_prop)
                        if existing_index > target_index:
                            insertion_point = i
                            break
                        else:
                            insertion_point = i + 1

        # Insert the new property
        self.lines.insert(insertion_point, f"{prop_key}={value}")

    def create_group_section(self, group_name: str, compilers_list: Optional[List[str]] = None):
        """Create a new group section."""
        if self.group_exists(group_name):
            return  # Already exists

        insertion_point = self.find_insertion_point_for_group(group_name)

        # Ensure proper spacing: blank line after compilers= and before group
        compilers_line_idx = self.find_compilers_line()
        if compilers_line_idx is not None and insertion_point == compilers_line_idx + 1:
            # We're inserting right after compilers= line, add blank line first
            self.lines.insert(insertion_point, "")
            insertion_point += 1
        elif (
            insertion_point > 0 and insertion_point < len(self.lines) and self.lines[insertion_point - 1].strip() != ""
        ):
            # Add empty line before group if previous line is not empty
            self.lines.insert(insertion_point, "")
            insertion_point += 1

        # Create the group.{name}.compilers line
        compilers_value = ":".join(compilers_list) if compilers_list else ""
        self.lines.insert(insertion_point, f"group.{group_name}.compilers={compilers_value}")

    def create_compiler_section(self, compiler: CompilerInfo):
        """Create a new compiler section."""
        if self.compiler_exists(compiler.id):
            return  # Already exists

        insertion_point = self.find_insertion_point_for_compiler(compiler.id, compiler.group)

        # Ensure proper spacing: blank line after group section and before compiler
        if compiler.group:
            group_start, group_end = self.find_group_section(compiler.group)
            if group_end is not None and insertion_point == group_end:
                # We're inserting right after group section, add blank line first
                self.lines.insert(insertion_point, "")
                insertion_point += 1

        # Add empty line before compiler if previous line is not empty
        if insertion_point > 0 and insertion_point < len(self.lines) and self.lines[insertion_point - 1].strip() != "":
            self.lines.insert(insertion_point, "")
            insertion_point += 1

        # Add compiler properties in order
        props_to_add = []
        
        # Normalize exe path for Windows (convert backslashes to forward slashes)
        normalized_exe_path = compiler.exe.replace("\\", "/")
        props_to_add.append(f"compiler.{compiler.id}.exe={normalized_exe_path}")

        # Add semver if available, name if no semver or force_name is True
        if compiler.semver:
            props_to_add.append(f"compiler.{compiler.id}.semver={compiler.semver}")
        if compiler.name and (not compiler.semver or compiler.force_name):
            props_to_add.append(f"compiler.{compiler.id}.name={compiler.name}")

        # Only add compilerType if the group doesn't already have the same one
        if compiler.compiler_type:
            group_compiler_type = None
            if compiler.group:
                group_compiler_type = self.get_group_property(compiler.group, "compilerType")

            # Add compilerType only if group doesn't have it or has a different one
            if not group_compiler_type or group_compiler_type != compiler.compiler_type:
                props_to_add.append(f"compiler.{compiler.id}.compilerType={compiler.compiler_type}")

        if compiler.options:
            props_to_add.append(f"compiler.{compiler.id}.options={compiler.options}")

        # Add Java-related properties for Java-based compilers
        if compiler.java_home:
            props_to_add.append(f"compiler.{compiler.id}.java_home={compiler.java_home}")

        if compiler.runtime:
            props_to_add.append(f"compiler.{compiler.id}.runtime={compiler.runtime}")

        # Add execution wrapper for compilers that need it
        if compiler.execution_wrapper:
            props_to_add.append(f"compiler.{compiler.id}.executionWrapper={compiler.execution_wrapper}")

        # Add MSVC-specific include and library paths
        if compiler.include_path:
            props_to_add.append(f"compiler.{compiler.id}.includePath={compiler.include_path}")
        if compiler.lib_path:
            props_to_add.append(f"compiler.{compiler.id}.libPath={compiler.lib_path}")

        # Insert all properties
        for prop in props_to_add:
            self.lines.insert(insertion_point, prop)
            insertion_point += 1

    def ensure_libs_tools_sections(self):
        """Ensure libs= and tools= sections exist at the end if missing."""
        has_libs = any(line.startswith("libs=") for line in self.lines)
        has_tools = any(line.startswith("tools=") for line in self.lines)

        if has_libs and has_tools:
            # Check if there's proper spacing before libs section
            self._ensure_proper_spacing_before_libs_tools()
            return  # Both exist

        # Find insertion point (end of file, but before any existing libs/tools)
        insertion_point = len(self.lines)
        for i, line in enumerate(self.lines):
            if line.startswith("libs=") or line.startswith("tools="):
                insertion_point = i
                break

        # Add sections if missing
        if not has_libs:
            # Add libs section header
            self.lines.insert(insertion_point, "")
            self.lines.insert(insertion_point + 1, "#################################")
            self.lines.insert(insertion_point + 2, "#################################")
            self.lines.insert(insertion_point + 3, "# Installed libs")
            self.lines.insert(insertion_point + 4, "libs=")
            insertion_point += 5

        if not has_tools:
            # Add tools section header
            self.lines.insert(insertion_point, "")
            self.lines.insert(insertion_point + 1, "#################################")
            self.lines.insert(insertion_point + 2, "#################################")
            self.lines.insert(insertion_point + 3, "# Installed tools")
            self.lines.insert(insertion_point + 4, "tools=")

    def _ensure_proper_spacing_before_libs_tools(self):
        """Ensure there's proper spacing before libs/tools sections."""
        # Find the start of libs/tools sections
        libs_tools_start = None
        for i, line in enumerate(self.lines):
            if (
                line.startswith("libs=")
                or line.startswith("tools=")
                or (line.startswith("#") and ("####" in line or "Installed" in line or "Libraries" in line))
            ):
                libs_tools_start = i
                break

        if libs_tools_start is None:
            return  # No libs/tools sections found

        # Check if there's an empty line before the libs/tools section
        if libs_tools_start > 0 and self.lines[libs_tools_start - 1].strip() != "":
            # No empty line before libs/tools, add one
            self.lines.insert(libs_tools_start, "")

    def ensure_proper_spacing_after_compiler(self, compiler_id: str):
        """Ensure proper spacing after a compiler section before libs/tools."""
        compiler_start, compiler_end = self.find_compiler_section(compiler_id)
        if compiler_start is None:
            return

        # Find if there are libs/tools sections after this compiler
        libs_tools_start = None
        for i in range(compiler_end, len(self.lines)):
            line = self.lines[i]
            if (
                line.startswith("libs=")
                or line.startswith("tools=")
                or (line.startswith("#") and ("####" in line or "Installed" in line or "Libraries" in line))
            ):
                libs_tools_start = i
                break

        if libs_tools_start is None:
            return  # No libs/tools sections after this compiler

        # Check spacing between compiler end and libs/tools start
        empty_lines_count = 0
        for i in range(compiler_end, libs_tools_start):
            if self.lines[i].strip() == "":
                empty_lines_count += 1
            else:
                # Non-empty line found, reset count
                empty_lines_count = 0

        # Ensure exactly one empty line before libs/tools
        if empty_lines_count == 0:
            # No empty lines, add one
            self.lines.insert(libs_tools_start, "")
        elif empty_lines_count > 1:
            # Too many empty lines, remove extras
            lines_to_remove = empty_lines_count - 1
            for _ in range(lines_to_remove):
                for i in range(compiler_end, libs_tools_start):
                    if i < len(self.lines) and self.lines[i].strip() == "":
                        self.lines.pop(i)
                        break
