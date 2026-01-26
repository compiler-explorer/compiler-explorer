"""Main CLI entry point for CE Properties Wizard."""

import os
import shlex
import sys
from pathlib import Path
from typing import Optional

import click
import inquirer
from colorama import Fore, Style, init

from .compiler_detector import LANGUAGE_CONFIGS, CompilerDetector, get_supported_compiler_types
from .config_manager import ConfigManager
from .models import CompilerInfo
from .utils import find_ce_config_directory

# Initialize colorama for cross-platform color support
init(autoreset=True)


def print_success(message: str):
    """Print success message in green."""
    click.echo(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")


def print_error(message: str):
    """Print error message in red."""
    click.echo(f"{Fore.RED}✗ {message}{Style.RESET_ALL}", err=True)


def print_info(message: str):
    """Print info message in blue."""
    click.echo(f"{Fore.BLUE}ℹ {message}{Style.RESET_ALL}")


def print_warning(message: str):
    """Print warning message in yellow."""
    click.echo(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")


def format_compiler_options(options_input: str) -> str:
    """Format compiler options properly.

    Takes space-separated options and quotes any that contain spaces.

    Args:
        options_input: Raw options string from user input

    Returns:
        Properly formatted options string with quoted options containing spaces
    """
    if not options_input or not options_input.strip():
        return ""

    # Split by spaces but respect quoted strings
    try:
        options = shlex.split(options_input)
    except ValueError:
        # If shlex fails (unmatched quotes), fall back to simple split
        options = options_input.split()

    # Format each option - quote it if it contains spaces
    formatted_options = []
    for opt in options:
        opt = opt.strip()
        if opt:
            if " " in opt and not (opt.startswith('"') and opt.endswith('"')):
                formatted_options.append(f'"{opt}"')
            else:
                formatted_options.append(opt)

    return " ".join(formatted_options)


@click.command()
@click.argument("compiler_path", required=False)
@click.option("--id", "compiler_id", help="Compiler ID (auto-generated if not specified)")
@click.option("--name", "display_name", help="Display name for the compiler")
@click.option("--group", help="Compiler group to add to")
@click.option("--options", help="Default compiler options")
@click.option("--language", help="Programming language (auto-detected if not specified)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts")
@click.option("--non-interactive", is_flag=True, help="Run in non-interactive mode with auto-detected values")
@click.option("--config-dir", type=click.Path(exists=True), help="Path to etc/config directory")
@click.option("--verify-only", is_flag=True, help="Only detect and display compiler information without making changes")
@click.option("--list-types", is_flag=True, help="List all supported compiler types and exit")
@click.option("--reorganize", help="Reorganize an existing properties file for the specified language")
@click.option(
    "--validate-discovery",
    is_flag=True,
    help="Run discovery validation to verify the compiler is detected (default for local environment)",
)
@click.option("--env", default="local", help="Environment to target (local, amazon, etc.)")
@click.option("--debug", is_flag=True, help="Enable debug output including subprocess commands")
@click.option("--sdk-path", help="Windows SDK base path for MSVC compilers (e.g., D:/efs/compilers/windows-kits-10)")
def cli(
    compiler_path: Optional[str],
    compiler_id: Optional[str],
    display_name: Optional[str],
    group: Optional[str],
    options: Optional[str],
    language: Optional[str],
    yes: bool,
    non_interactive: bool,
    config_dir: Optional[str],
    verify_only: bool,
    list_types: bool,
    reorganize: Optional[str],
    validate_discovery: bool,
    env: str,
    debug: bool,
    sdk_path: Optional[str],
):
    """CE Properties Wizard - Add compilers to your Compiler Explorer installation.

    Examples:
        ce-props-wizard                              # Interactive mode (local environment)
        ce-props-wizard /usr/bin/g++-13             # Path-first mode
        ce-props-wizard /usr/bin/g++-13 --yes       # Automated mode
        ce-props-wizard --env amazon /usr/bin/g++   # Target amazon environment
        ce-props-wizard --list-types                 # List all supported compiler types
        ce-props-wizard /usr/bin/g++ --verify-only  # Just detect compiler info

    MSVC Examples:
        ce-props-wizard "D:/efs/compilers/msvc-2022/VC/Tools/MSVC/14.34.31933/bin/Hostx64/x64/cl.exe"
        ce-props-wizard "C:/MSVC/cl.exe" --sdk-path "C:/WindowsKits/10" --non-interactive
    """
    # Handle --list-types flag
    if list_types:
        try:
            supported_types = get_supported_compiler_types()
            click.echo(f"Found {len(supported_types)} supported compiler types:\n")
            for compiler_type in sorted(supported_types):
                click.echo(compiler_type)
            sys.exit(0)
        except Exception as e:
            print_error(f"Error reading compiler types: {e}")
            sys.exit(1)

    # Handle --reorganize flag
    if reorganize:
        try:
            # Find config directory
            if config_dir:
                config_mgr = ConfigManager(Path(config_dir), env, debug=debug)
            else:
                config_mgr = ConfigManager(find_ce_config_directory(), env, debug=debug)

            print_info(f"Reorganizing {reorganize} properties file...")

            # Check if language is valid
            if reorganize not in LANGUAGE_CONFIGS:
                print_error(f"Unknown language: {reorganize}")
                print_info(f"Available languages: {', '.join(LANGUAGE_CONFIGS.keys())}")
                sys.exit(1)

            file_path = config_mgr.get_properties_path(reorganize)
            if not file_path.exists():
                print_error(f"No {env} properties file found for {reorganize}: {file_path}")
                sys.exit(1)

            config_mgr.reorganize_existing_file(reorganize)
            print_success(f"Reorganized {file_path}")

            # Validate properties
            print_info("Validating properties...")
            valid, message = config_mgr.validate_properties(reorganize)
            if valid:
                print_success(message)
            else:
                print_error(message)

            sys.exit(0)
        except Exception as e:
            print_error(f"Error reorganizing file: {e}")
            sys.exit(1)

    # Skip banner in verify-only mode
    if not verify_only:
        click.echo(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        click.echo(f"{Fore.CYAN}Compiler Explorer Properties Wizard{Style.RESET_ALL}")
        click.echo(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

    try:
        # Find config directory only if needed
        if not verify_only:
            if config_dir:
                config_path = Path(config_dir)
            else:
                config_path = find_ce_config_directory()
            print_info(f"Using config directory: {config_path}")
            print_info(f"Targeting environment: {env}")
            config_mgr = ConfigManager(config_path, env, debug=debug)
        else:
            config_mgr = None

        # Initialize detector
        detector = CompilerDetector(debug=debug)

        # Get compiler path if not provided
        if not compiler_path:
            questions = [
                inquirer.Text(
                    "compiler_path",
                    message="Enter the full path to the compiler executable",
                    validate=lambda _, x: os.path.isfile(x),
                )
            ]
            answers = inquirer.prompt(questions)
            if not answers:
                print_error("Cancelled by user")
                sys.exit(1)
            compiler_path = answers["compiler_path"]

        # Validate compiler path
        compiler_path = os.path.abspath(compiler_path)
        if not os.path.isfile(compiler_path):
            print_error(f"Compiler not found: {compiler_path}")
            sys.exit(1)

        if not os.access(compiler_path, os.X_OK):
            print_error(f"File is not executable: {compiler_path}")
            sys.exit(1)

        # Detect compiler information
        print_info("Detecting compiler type and language...")
        try:
            detected_info = detector.detect_from_path(compiler_path)

            if detected_info.compiler_type:
                print_success(f"Detected: {detected_info.name} ({LANGUAGE_CONFIGS[detected_info.language].name})")
            else:
                print_warning("Could not detect compiler type")

        except Exception as e:
            print_error(f"Detection failed: {e}")
            # Create minimal info
            detected_info = CompilerInfo(
                id="custom-compiler",
                name=os.path.basename(compiler_path),
                exe=compiler_path,
                language=language or "c++",
            )

        # Override with command-line options
        if language:
            detected_info.language = language

        # Suggest appropriate group if not already set
        if not detected_info.group:
            if not verify_only and config_mgr is not None:
                # Normal mode - create config manager and suggest group
                suggested_group = config_mgr.suggest_appropriate_group(detected_info)
                if suggested_group:
                    detected_info.group = suggested_group
            else:
                # Verify-only mode - create a temporary config manager just for suggestion
                temp_config_mgr = ConfigManager(find_ce_config_directory(), env, debug=debug)
                suggested_group = temp_config_mgr.suggest_appropriate_group(detected_info)
                if suggested_group:
                    detected_info.group = suggested_group

        # Initialize flag for forcing custom ID/name
        force_custom_id_name = False

        # Check for existing compiler by path early (before prompts)
        if not verify_only and config_mgr is not None:
            existing_compiler_id = config_mgr.check_existing_compiler_by_path(compiler_path, detected_info.language)
            if existing_compiler_id:
                file_path = config_mgr.get_properties_path(detected_info.language)
                print_warning(f"Compiler already exists in {env} environment!")
                print_info(f"Existing compiler ID: {existing_compiler_id}")
                print_info(f"Executable path: {compiler_path}")
                print_info(f"Properties file: {file_path}")

                # If automated mode (-y), exit immediately
                if yes or non_interactive:
                    print_info("No changes were made.")
                    sys.exit(0)

                # In interactive mode, ask if user wants to continue with different ID/name
                if not click.confirm("\nWould you like to add this compiler anyway with a different ID and name?"):
                    print_info("No changes were made.")
                    sys.exit(0)

                print_info("You will need to provide a unique compiler ID and custom name.")
                # Set flag to force custom ID and name prompts
                force_custom_id_name = True
                # Suggest the group from the existing duplicate compiler
                if config_mgr is not None:
                    suggested_group = config_mgr.suggest_appropriate_group(detected_info, existing_compiler_id)
                    if suggested_group and not detected_info.group:
                        detected_info.group = suggested_group

        # If verify-only mode, display info and exit
        if verify_only:
            click.echo("\nDetected compiler information:")
            click.echo(f"  Path: {compiler_path}")
            lang_name = (
                LANGUAGE_CONFIGS[detected_info.language].name
                if detected_info.language in LANGUAGE_CONFIGS
                else detected_info.language
            )
            click.echo(f"  Language: {lang_name}")
            click.echo(f"  Compiler Type: {detected_info.compiler_type or 'unknown'}")
            click.echo(f"  Version: {detected_info.version or 'unknown'}")
            click.echo(f"  Semver: {detected_info.semver or 'unknown'}")
            if detected_info.target:
                click.echo(f"  Target: {detected_info.target}")
                click.echo(f"  Cross-compiler: {'Yes' if detected_info.is_cross_compiler else 'No'}")
            click.echo(f"  Suggested ID: {detected_info.id}")
            click.echo(f"  Suggested Name: {detected_info.name}")
            click.echo(f"  Suggested Group: {detected_info.group or 'none'}")
            sys.exit(0)

        # Handle Windows SDK path for MSVC compilers
        if detected_info.needs_sdk_prompt:
            if sdk_path:
                # Use command-line provided SDK path
                if os.path.isdir(sdk_path.replace("\\", "/")):
                    detected_info = detector.set_windows_sdk_path(detected_info, sdk_path)
                    print_success(f"Windows SDK paths added from: {sdk_path}")
                else:
                    print_error(f"Invalid SDK path: {sdk_path}")
                    sys.exit(1)
            elif not yes and not non_interactive:
                # Interactive prompt for SDK path
                print_info("Windows SDK auto-detection failed. You can optionally specify the Windows SDK path.")
                print_info("Example: Z:/compilers/windows-kits-10 (leave empty to skip)")
                sdk_question = inquirer.Text(
                    "windows_sdk_path",
                    message="Windows SDK base path (optional)",
                    default="",
                    validate=lambda _, x: x == "" or os.path.isdir(x.replace("\\", "/"))
                )
                sdk_answers = inquirer.prompt([sdk_question])
                if sdk_answers and sdk_answers["windows_sdk_path"].strip():
                    # Apply the user-provided SDK path
                    detected_info = detector.set_windows_sdk_path(detected_info, sdk_answers["windows_sdk_path"].strip())
                    print_success(f"Windows SDK paths added from: {sdk_answers['windows_sdk_path']}")

        # Interactive prompts for missing information
        if not yes and not non_interactive:
            questions = []

            # Language selection if needed
            if not language and detected_info.language:
                lang_choices = [(LANGUAGE_CONFIGS[k].name, k) for k in LANGUAGE_CONFIGS.keys()]
                questions.append(
                    inquirer.List(
                        "language", message="Programming language", choices=lang_choices, default=detected_info.language
                    )
                )

            # Compiler ID - force custom if duplicate exists
            if force_custom_id_name:
                questions.append(
                    inquirer.Text(
                        "compiler_id",
                        message="Compiler ID (must be unique)",
                        default=compiler_id or "",
                        validate=lambda _, x: bool(x and x.strip() and x != detected_info.id),
                    )
                )
            else:
                questions.append(
                    inquirer.Text(
                        "compiler_id",
                        message="Compiler ID",
                        default=compiler_id or detected_info.id,
                        validate=lambda _, x: bool(x and x.strip()),
                    )
                )

            # Display name - force custom if duplicate exists
            if force_custom_id_name:
                questions.append(
                    inquirer.Text(
                        "display_name",
                        message="Display name (must be custom)",
                        default=display_name or "",
                        validate=lambda _, x: bool(x and x.strip() and x != detected_info.name),
                    )
                )
            else:
                questions.append(
                    inquirer.Text("display_name", message="Display name", default=display_name or detected_info.name)
                )

            # Compiler type (if not detected)
            if not detected_info.compiler_type:
                # Get all supported compiler types dynamically
                supported_types = sorted(get_supported_compiler_types())
                # Add 'other' as fallback option
                type_choices = supported_types + ["other"]

                questions.append(
                    inquirer.List("compiler_type", message="Compiler type", choices=type_choices, default="other")
                )

            # Group
            questions.append(
                inquirer.Text(
                    "group",
                    message="Add to group",
                    default=group or detected_info.group or detected_info.compiler_type or "",
                )
            )

            # Options
            questions.append(
                inquirer.Text(
                    "options",
                    message="Additional options (space-separated, quote options with spaces)",
                    default=options or "",
                )
            )

            if questions:
                answers = inquirer.prompt(questions)
                if not answers:
                    print_error("Cancelled by user")
                    sys.exit(1)

                # Update detected info
                if "language" in answers:
                    detected_info.language = answers["language"]
                if "compiler_id" in answers:
                    detected_info.id = answers["compiler_id"]
                if "display_name" in answers:
                    detected_info.name = answers["display_name"]
                    # If this is a duplicate override scenario, force the name to be included
                    if force_custom_id_name:
                        detected_info.force_name = True
                if "compiler_type" in answers:
                    compiler_type = answers["compiler_type"]
                    # Validate compiler type against supported types
                    if compiler_type != "other":
                        supported_types = get_supported_compiler_types()
                        if compiler_type not in supported_types:
                            print_warning(f"'{compiler_type}' is not a recognized compiler type in Compiler Explorer")
                    detected_info.compiler_type = compiler_type
                if "group" in answers and answers["group"]:
                    detected_info.group = answers["group"]
                if "options" in answers and answers["options"]:
                    detected_info.options = format_compiler_options(answers["options"])
        else:
            # In automated mode, use command-line values
            if compiler_id:
                detected_info.id = compiler_id
            if display_name:
                detected_info.name = display_name
                # If this is a duplicate override scenario, force the name to be included
                if force_custom_id_name:
                    detected_info.force_name = True
            if group:
                detected_info.group = group
            if options:
                detected_info.options = format_compiler_options(options)

        # Ensure unique ID (config_mgr should not be None at this point)
        assert config_mgr is not None, "config_mgr should not be None in non-verify mode"
        original_id = detected_info.id
        detected_info.id = config_mgr.ensure_compiler_id_unique(detected_info.id, detected_info.language)
        if detected_info.id != original_id:
            print_warning(f"ID already exists, using: {detected_info.id}")

        # Show configuration preview
        print_info("\nConfiguration preview:")
        normalized_exe_path = detected_info.exe.replace("\\", "/")
        click.echo(f"  compiler.{detected_info.id}.exe={normalized_exe_path}")

        # Check if semver will be available (either detected or extracted)
        semver_to_use = detected_info.semver
        if not semver_to_use:
            # Try to extract version like the config manager will do
            try:
                semver_to_use = config_mgr._extract_compiler_version(detected_info.exe)
            except Exception:
                pass

        # Show semver if available
        if semver_to_use:
            click.echo(f"  compiler.{detected_info.id}.semver={semver_to_use}")

        # Show name if semver is not available OR if this is a duplicate override scenario
        if detected_info.name and (not semver_to_use or force_custom_id_name):
            click.echo(f"  compiler.{detected_info.id}.name={detected_info.name}")

        if detected_info.compiler_type:
            click.echo(f"  compiler.{detected_info.id}.compilerType={detected_info.compiler_type}")
        if detected_info.options:
            click.echo(f"  compiler.{detected_info.id}.options={detected_info.options}")
        if detected_info.java_home:
            click.echo(f"  compiler.{detected_info.id}.java_home={detected_info.java_home}")
        if detected_info.runtime:
            click.echo(f"  compiler.{detected_info.id}.runtime={detected_info.runtime}")
        if detected_info.execution_wrapper:
            click.echo(f"  compiler.{detected_info.id}.executionWrapper={detected_info.execution_wrapper}")
        if detected_info.include_path:
            click.echo(f"  compiler.{detected_info.id}.includePath={detected_info.include_path}")
        if detected_info.lib_path:
            click.echo(f"  compiler.{detected_info.id}.libPath={detected_info.lib_path}")
        if detected_info.group:
            click.echo(f"  Will add to group: {detected_info.group}")

        # Confirm
        file_path = config_mgr.get_properties_path(detected_info.language)
        if not yes and not non_interactive:
            if not click.confirm(f"\nUpdate {file_path}?"):
                print_error("Cancelled by user")
                sys.exit(1)

        # Add compiler
        config_mgr.add_compiler(detected_info)
        print_success("Configuration updated successfully!")

        # Validate properties
        print_info("Validating properties...")
        valid, message = config_mgr.validate_properties(detected_info.language)
        if valid:
            print_success(message)
        else:
            print_error(message)
            # Don't exit with error, as the file was written successfully

        # Discovery validation (default for local environment, optional for others)
        should_validate_discovery = validate_discovery or (env == "local")
        if should_validate_discovery:
            print_info("Validating with discovery...")
            valid, message, discovered_semver = config_mgr.validate_with_discovery(
                detected_info.language, detected_info.id
            )
            if valid:
                print_success(message)
                if discovered_semver:
                    print_info(f"Discovered semver: {discovered_semver}")
            else:
                print_error(message)
                print_info(
                    "Note: Discovery validation failed, but the compiler was added to the properties file successfully."
                )

        click.echo(f"\n{Fore.GREEN}Compiler added successfully!{Style.RESET_ALL}")
        click.echo("You may need to restart Compiler Explorer for changes to take effect.")

    except KeyboardInterrupt:
        print_error("\nCancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
