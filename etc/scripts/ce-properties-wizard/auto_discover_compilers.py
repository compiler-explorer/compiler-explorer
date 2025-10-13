#!/usr/bin/env python3
"""
CE Compiler Auto-Discovery Tool

Automatically discovers compilers in PATH directories and adds them using
the CE Properties Wizard.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set


# Compiler patterns for each language
COMPILER_PATTERNS = {
    'c++': ['g++', 'g++-*', 'clang++', 'clang++-*', 'icpc*', 'icx*'],
    'c': ['gcc', 'gcc-[0-9]*', 'clang', 'clang-[0-9]*', 'icc*', 'cc'],
    'cuda': ['nvcc*'],
    'rust': ['rustc*'],
    'go': ['go', 'gccgo*'],
    'python': ['python*', 'python3*', 'pypy*'],
    'java': ['javac*', 'java'],
    'fortran': ['gfortran*', 'ifort*', 'ifx*'],
    'pascal': ['fpc'],
    'kotlin': ['kotlin*', 'kotlinc*'],
    'zig': ['zig'],
    'dart': ['dart'],
    'd': ['dmd*', 'ldc*', 'ldc2*', 'gdc*'],
    'swift': ['swift*', 'swiftc*'],
    'nim': ['nim'],
    'crystal': ['crystal'],
    'v': ['v'],
    'haskell': ['ghc*'],
    'ocaml': ['ocaml*'],
    'scala': ['scala*', 'scalac*'],
    'csharp': ['csc*', 'mcs*', 'dotnet'],
    'fsharp': ['fsharpc*', 'dotnet'],
    'ruby': ['ruby*'],
    'julia': ['julia'],
    'elixir': ['elixir*'],
    'erlang': ['erlc*', 'erl'],
    'assembly': ['nasm*', 'yasm*', 'as'],
    'carbon': ['carbon*'],
    'mojo': ['mojo*'],
    'odin': ['odin*'],
    'ada': ['gnatmake*', 'gprbuild*', 'gnat*'],
    'cobol': ['cobc*', 'gnucobol*', 'gcobol*'],
}

# Default exclude patterns
DEFAULT_EXCLUDES = {
    'wrapper', 'distcc', 'ccache', '-config', 'config-', 
    '-ar', '-nm', '-ranlib', '-strip', 'filt', 'format', 
    'calls', 'flow', 'stat', '-gdb', 'argcomplete', 'build',
    'ldconfig', 'ldconfig.real', '-bpfcc', 'bpfcc', 'scalar',
    'pythongc-bpfcc', 'pythonflow-bpfcc', 'pythoncalls-bpfcc', 'pythonstat-bpfcc'
}


def get_path_dirs() -> List[Path]:
    """Get all directories from PATH environment variable."""
    path = os.environ.get('PATH', '')
    return [Path(p) for p in path.split(':') if p.strip()]


def should_exclude(name: str, excludes: Set[str]) -> bool:
    """Check if a compiler name should be excluded."""
    return any(exclude in name for exclude in excludes)


def find_compilers_in_dir(directory: Path, patterns: List[str], excludes: Set[str]) -> List[Path]:
    """Find compilers matching patterns in a directory."""
    compilers = []
    
    if not directory.exists() or not directory.is_dir():
        return compilers
    
    for pattern in patterns:
        # Simple glob matching
        for compiler in directory.glob(pattern):
            if (compiler.is_file() or compiler.is_symlink()) and \
               os.access(compiler, os.X_OK) and \
               not should_exclude(compiler.name, excludes):
                compilers.append(compiler)
    
    return compilers


def resolve_duplicates(compilers: List[Path]) -> List[Path]:
    """Remove duplicate compilers (same resolved path)."""
    seen = set()
    unique_compilers = []
    
    for compiler in compilers:
        try:
            resolved = compiler.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_compilers.append(compiler)
        except (OSError, RuntimeError):
            # If we can't resolve, keep the original
            unique_compilers.append(compiler)
    
    return unique_compilers


def discover_compilers(languages: List[str], search_dirs: List[Path] = None, 
                      excludes: Set[str] = None) -> Dict[str, List[Path]]:
    """Discover compilers for specified languages."""
    if search_dirs is None:
        search_dirs = get_path_dirs()
    
    if excludes is None:
        excludes = DEFAULT_EXCLUDES
    
    discovered = {}
    
    for language in languages:
        if language not in COMPILER_PATTERNS:
            print(f"Warning: Unknown language '{language}'", file=sys.stderr)
            continue
        
        patterns = COMPILER_PATTERNS[language]
        compilers = []
        
        for directory in search_dirs:
            compilers.extend(find_compilers_in_dir(directory, patterns, excludes))
        
        if compilers:
            # Remove duplicates and sort
            unique_compilers = resolve_duplicates(compilers)
            discovered[language] = sorted(unique_compilers, key=lambda x: x.name)
    
    return discovered


def add_compiler_with_wizard(compiler: Path, language: str, script_dir: Path, 
                           wizard_args: List[str], dry_run: bool) -> bool:
    """Add a compiler using the CE Properties Wizard."""
    if dry_run:
        return True
    
    cmd = [
        str(script_dir / 'run.sh'),
        str(compiler),
        '--yes',
        '--language', language
    ] + wizard_args
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description='CE Compiler Auto-Discovery Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Interactive discovery of all languages
  %(prog)s --dry-run                 # Preview what would be discovered
  %(prog)s --languages c++,rust,go   # Only discover C++, Rust, and Go
  %(prog)s --yes --languages c++,c   # Non-interactive C/C++ discovery
        """)
    
    parser.add_argument('--languages', 
                       help='Comma-separated list of languages to discover (default: all)')
    parser.add_argument('--search-dirs', 
                       help='Colon-separated search directories (default: PATH dirs)')
    parser.add_argument('--exclude', 
                       help='Comma-separated exclude patterns')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be added without making changes')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompts')
    parser.add_argument('--config-dir', 
                       help='Path to etc/config directory')
    parser.add_argument('--env', default='local',
                       help='Environment to target (local, amazon, etc.)')
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    wizard_script = script_dir / 'run.sh'
    
    if not wizard_script.exists():
        print(f"Error: CE Properties Wizard not found at {wizard_script}", file=sys.stderr)
        sys.exit(1)
    
    # Parse languages
    if args.languages:
        languages = [lang.strip() for lang in args.languages.split(',')]
    else:
        languages = list(COMPILER_PATTERNS.keys())
    
    # Parse search directories
    search_dirs = None
    if args.search_dirs:
        search_dirs = [Path(d) for d in args.search_dirs.split(':') if d.strip()]
    
    # Parse excludes
    excludes = DEFAULT_EXCLUDES.copy()
    if args.exclude:
        excludes.update(args.exclude.split(','))
    
    # Discover compilers
    print("CE Compiler Auto-Discovery Tool")
    print("=" * 35)
    print()
    
    if args.dry_run:
        print("DRY RUN MODE - No compilers will actually be added")
        print()
    
    discovered = discover_compilers(languages, search_dirs, excludes)
    
    if not discovered:
        print("No compilers found matching the specified criteria")
        sys.exit(1)
    
    # Show results
    total_count = sum(len(compilers) for compilers in discovered.values())
    print(f"Found {total_count} compilers:")
    print()
    
    for language, compilers in discovered.items():
        print(f"{language.upper()} ({len(compilers)} compilers):")
        for compiler in compilers:
            print(f"  ✓ {compiler}")
        print()
    
    # Confirm before adding
    if not args.dry_run and not args.yes:
        response = input("Add these compilers? [y/N] ")
        if not response.lower().startswith('y'):
            print("Operation cancelled")
            sys.exit(0)
    
    if args.dry_run:
        print("Dry run complete - no changes made")
        sys.exit(0)
    
    # Add compilers
    print("Adding compilers using CE Properties Wizard...")
    print()
    
    wizard_args = []
    if args.config_dir:
        wizard_args.extend(['--config-dir', args.config_dir])
    if args.env != 'local':
        wizard_args.extend(['--env', args.env])
    
    added_count = 0
    failed_count = 0
    
    for language, compilers in discovered.items():
        for compiler in compilers:
            print(f"Adding {compiler} ({language})...", end=' ')
            
            if add_compiler_with_wizard(compiler, language, script_dir, wizard_args, args.dry_run):
                print("✓")
                added_count += 1
            else:
                print("✗")
                failed_count += 1
    
    print()
    print("Summary:")
    print(f"  ✓ Successfully added: {added_count} compilers")
    if failed_count > 0:
        print(f"  ✗ Failed to add: {failed_count} compilers")
    print()
    print("Auto-discovery complete!")


if __name__ == '__main__':
    main()