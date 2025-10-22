"""Data models for the CE Properties Wizard."""

import re
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class CompilerInfo(BaseModel):
    """Model representing compiler information."""

    id: str = Field(..., description="Unique identifier for the compiler")
    name: str = Field(..., description="Display name for the compiler")
    exe: str = Field(..., description="Path to the compiler executable")
    compiler_type: Optional[str] = Field(None, description="Type of compiler (gcc, clang, etc)")
    version: Optional[str] = Field(None, description="Compiler version")
    semver: Optional[str] = Field(None, description="Semantic version")
    group: Optional[str] = Field(None, description="Compiler group to add to")
    options: Optional[str] = Field(None, description="Default compiler options")
    language: str = Field(..., description="Programming language")
    target: Optional[str] = Field(None, description="Target platform (for cross-compilers)")
    is_cross_compiler: bool = Field(False, description="Whether this is a cross-compiler")
    force_name: bool = Field(False, description="Force inclusion of .name property even when semver exists")
    java_home: Optional[str] = Field(None, description="JAVA_HOME path for Java-based compilers")
    runtime: Optional[str] = Field(None, description="Runtime executable path for Java-based compilers")
    execution_wrapper: Optional[str] = Field(None, description="Execution wrapper path for languages like Dart")
    include_path: Optional[str] = Field(None, description="Include paths for MSVC compilers")
    lib_path: Optional[str] = Field(None, description="Library paths for MSVC compilers")
    needs_sdk_prompt: bool = Field(False, description="Whether to prompt user for Windows SDK path")

    @validator("id")
    def validate_id(cls, value):  # noqa: N805
        """Ensure ID is valid for properties files."""
        if not re.match(r"^[a-zA-Z0-9_-]+$", value):
            raise ValueError("ID must contain only alphanumeric characters, hyphens, and underscores")
        return value


class LanguageConfig(BaseModel):
    """Model representing language configuration."""

    name: str = Field(..., description="Language name")
    properties_file: str = Field(..., description="Properties filename (without path, defaults to local)")
    compiler_types: List[str] = Field(default_factory=list, description="Known compiler types for this language")
    extensions: List[str] = Field(default_factory=list, description="File extensions")
    keywords: List[str] = Field(default_factory=list, description="Keywords in compiler path/name")

    def get_properties_file(self, env: str = "local") -> str:
        """Get properties file name for specified environment."""
        if env == "local":
            return self.properties_file
        else:
            # Replace .local. with .{env}.
            return self.properties_file.replace(".local.", f".{env}.")
