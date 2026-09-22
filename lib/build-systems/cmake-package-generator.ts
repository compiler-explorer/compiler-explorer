// Copyright (c) 2026, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

import fs from 'node:fs/promises';
import path from 'node:path';

/**
 * Most CE libraries ship no CMake package config: their headers are unpacked source trees and
 * their binaries arrive as conan packages that hold only `lib/`. `find_package()` therefore fails
 * for them however `CMAKE_PREFIX_PATH` is set. This generator synthesises a package config from
 * the properties CE already has, so `find_package(<lib>)` resolves for any selected library.
 *
 * Libraries whose conan package *is* a full cmake install tree (`packagedheaders`) already carry a
 * real config; their extracted directory is offered as a prefix ahead of anything generated, and
 * `skipcmakepackage` turns generation off entirely for a library that needs its own config to win.
 */

/** Subdirectory of the compilation temp dir that generated packages are written under. */
export const GENERATED_PACKAGE_DIRNAME = 'ce-cmake-packages';

/** The parts of a resolved library version this generator needs. */
export type CMakePackageVersionInfo = {
    version?: string;
    path?: string[];
    libpath?: string[];
    staticliblink?: string[];
    liblink?: string[];
    packagedheaders?: boolean;
    skipcmakepackage?: boolean;
};

export type CMakePackageLibrary = {
    id: string;
    version: CMakePackageVersionInfo;
};

/** What a user can actually write for one selected library, for hinting after a find_package failure. */
export type CMakePackageDescription = {
    libId: string;
    /** Names that work as find_package(<name>) */
    packageNames: string[];
    /** Imported targets that work in target_link_libraries() */
    targets: string[];
    /** Raw library names the linker can be given, for explaining "cannot find -lfoo" */
    linkNames: string[];
    /** True when CE generated this package from the library's properties. */
    generated: boolean;
};

export type GeneratedCMakePackage = {
    libId: string;
    packageName: string;
    version: string;
    prefixPath: string;
    configFile: string;
    versionFile: string;
    includeDirs: string[];
    libraryDirs: string[];
    linkNames: string[];
};

/** Escape a value for use inside a CMake double-quoted argument. */
export function escapeCMakeString(value: string): string {
    return value.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\$/g, '\\$');
}

/** CMake wants forward slashes even on Windows; a backslash is an escape inside a quoted argument. */
function toCMakePath(value: string): string {
    return escapeCMakeString(value.replace(/\\/g, '/'));
}

/** A single quoted argument holding a CMake list, for `set(VAR "a;b")`. */
function quotedList(values: string[]): string {
    return `"${values.map(toCMakePath).join(';')}"`;
}

/**
 * Separate quoted arguments, for command keywords like `PATHS`. A single `"a;b"` argument is one
 * path with a semicolon in it as far as find_library is concerned, not two paths.
 */
function quotedArgs(values: string[]): string {
    return values.map(value => `"${toCMakePath(value)}"`).join(' ');
}

function unique(values: string[]): string[] {
    return [...new Set(values.filter(Boolean))];
}

/** Caps so that scanning a large install tree (Qt ships ~29 configs) stays bounded. */
const MAX_CONFIG_FILES_SCANNED = 400;
const MAX_CONFIG_BYTES_SCANNED = 8 * 1024 * 1024;

function packageNameFromConfigFile(filename: string): string | undefined {
    for (const suffix of ['Config.cmake', '-config.cmake']) {
        if (filename.endsWith(suffix)) return filename.slice(0, -suffix.length);
    }
    return undefined;
}

/**
 * Reads the package names and imported targets a real install tree exports. Only called after a
 * build has already failed, so the I/O is affordable in a way it would not be on the happy path.
 */
async function readExportedPackages(prefix: string): Promise<{packageNames: string[]; targets: string[]}> {
    const packageNames: string[] = [];
    const targets: string[] = [];
    let filesRead = 0;
    let bytesRead = 0;

    for (const searchDir of [path.join(prefix, 'lib', 'cmake'), path.join(prefix, 'share', 'cmake')]) {
        let packageDirs;
        try {
            packageDirs = await fs.readdir(searchDir, {withFileTypes: true});
        } catch {
            continue;
        }
        for (const packageDir of packageDirs.filter(entry => entry.isDirectory())) {
            const dir = path.join(searchDir, packageDir.name);
            let entries: string[];
            try {
                entries = await fs.readdir(dir);
            } catch {
                continue;
            }
            for (const entry of entries) {
                const name = packageNameFromConfigFile(entry);
                if (name) packageNames.push(name);
            }
            // Targets live in the *Targets.cmake files next to the config, not in the config itself.
            // Restricting to those keeps a big install tree (Qt's is ~2500 files) affordable.
            for (const entry of entries.filter(e => /Targets.*\.cmake$/.test(e) || e.endsWith('Config.cmake'))) {
                if (filesRead >= MAX_CONFIG_FILES_SCANNED || bytesRead >= MAX_CONFIG_BYTES_SCANNED) break;
                try {
                    const contents = await fs.readFile(path.join(dir, entry), 'utf8');
                    filesRead++;
                    bytesRead += contents.length;
                    for (const match of contents.matchAll(/add_library\(\s*([A-Za-z0-9_]+::[A-Za-z0-9_+.-]+)/g)) {
                        targets.push(match[1]);
                    }
                } catch {
                    // An unreadable file simply contributes nothing.
                }
            }
        }
    }

    // A tree like Qt's also defines detection helpers (WrapSystemZLIB::, DB2::) that nobody should
    // link against. Keeping only targets whose namespace is one of the packages it exports leaves
    // the ones a user can actually ask for.
    const exported = unique(packageNames);
    // A namespace counts as ours if it is an exported package, or the stem of one: Qt exports
    // Qt6Core and defines Qt6::Core, so requiring an exact match would discard every real target.
    const ownTargets = unique(targets).filter(target => {
        const namespace = target.split('::')[0];
        return exported.some(name => name === namespace || name.startsWith(namespace));
    });
    return {packageNames: exported, targets: ownTargets};
}

export class CMakePackageGenerator {
    private readonly buildDirPath: string;
    private readonly packagesExtractedPerLib: boolean;

    /**
     * @param dirPath the compilation temp directory on this host: where packages are written.
     * @param options.buildDirPath the same directory as the *build* sees it. Under nsjail the
     *   compilation directory is bind-mounted at /app, and while exec rewrites command arguments
     *   and environment variables from one to the other, it cannot rewrite the contents of a file
     *   we generate. Paths embedded in the config must therefore already be build-visible.
     *   Defaults to `dirPath` for an unjailed build.
     * @param options.packagesExtractedPerLib mirrors the inverse of buildenvsetup's
     *   `extractAllToRoot`: when true a library's package lands in `<dir>/<libId>/`, else `<dir>/`.
     */
    constructor(
        private readonly dirPath: string,
        options: {buildDirPath?: string; packagesExtractedPerLib?: boolean} = {},
    ) {
        this.buildDirPath = options.buildDirPath ?? dirPath;
        this.packagesExtractedPerLib = options.packagesExtractedPerLib ?? true;
    }

    /** Directory a library's conan package is extracted into, on this host. */
    extractedPackageDir(libId: string): string {
        return this.packagesExtractedPerLib ? path.join(this.dirPath, libId) : this.dirPath;
    }

    /** The same directory as the build sees it; this is what may be embedded in a config file. */
    buildExtractedPackageDir(libId: string): string {
        return this.packagesExtractedPerLib ? path.join(this.buildDirPath, libId) : this.buildDirPath;
    }

    /**
     * Prefix contributed by the library itself: for `packagedheaders` the conan package is a full
     * cmake install tree, so its extraction directory is a genuine install prefix.
     */
    installedPrefixFor(lib: CMakePackageLibrary): string | undefined {
        return lib.version.packagedheaders ? this.extractedPackageDir(lib.id) : undefined;
    }

    packageNameFor(lib: CMakePackageLibrary): string {
        return lib.id;
    }

    includeDirsFor(lib: CMakePackageLibrary): string[] {
        const dirs = [...(lib.version.path ?? [])];
        if (lib.version.packagedheaders) {
            dirs.push(path.join(this.buildExtractedPackageDir(lib.id), 'include'));
        }
        return unique(dirs);
    }

    libraryDirsFor(lib: CMakePackageLibrary): string[] {
        return unique([...(lib.version.libpath ?? []), path.join(this.buildExtractedPackageDir(lib.id), 'lib')]);
    }

    linkNamesFor(lib: CMakePackageLibrary): string[] {
        return unique([...(lib.version.staticliblink ?? []), ...(lib.version.liblink ?? [])]);
    }

    /**
     * Whether generation is switched off for this library: either flag alone is enough.
     *
     * A `packagedheaders` package is a full cmake install tree that already carries its own config
     * - Qt installs `lib/cmake/Qt6/Qt6Config.cmake` and callers write `find_package(Qt6)`, so a
     * generated `qtConfig.cmake` would be noise at best. Treating that as a reason to skip keeps
     * the decision a property lookup rather than a directory scan.
     */
    skipsGeneration(lib: CMakePackageLibrary): boolean {
        return !!(lib.version.skipcmakepackage || lib.version.packagedheaders);
    }

    /** A library with nothing to offer CMake gets no package either. */
    shouldGenerate(lib: CMakePackageLibrary): boolean {
        if (this.skipsGeneration(lib)) return false;
        return this.includeDirsFor(lib).length > 0 || this.linkNamesFor(lib).length > 0;
    }

    renderConfig(lib: CMakePackageLibrary): string {
        const name = this.packageNameFor(lib);
        const version = lib.version.version || '0.0.0';
        const includeDirs = this.includeDirsFor(lib);
        const libraryDirs = this.libraryDirsFor(lib);
        const linkNames = this.linkNamesFor(lib);

        const lines: string[] = [
            `# Generated by Compiler Explorer for library '${lib.id}' version ${version}.`,
            '# Derived from the library properties configured for this compiler; do not edit.',
            '',
            `if(TARGET ${name}::${name})`,
            '  return()',
            'endif()',
            '',
            `set(${name}_VERSION ${quotedList([version])})`,
            `set(${name}_FOUND TRUE)`,
            `set(${name}_INCLUDE_DIRS ${quotedList(includeDirs)})`,
            `set(${name}_LIBRARY_DIRS ${quotedList(libraryDirs)})`,
            `set(${name}_LIBRARIES "")`,
            '',
        ];

        const resolvedVars: string[] = [];
        for (const linkName of linkNames) {
            const variable = `_ce_${name}_${linkName}_LIBRARY`;
            resolvedVars.push(variable);
            const notFound =
                `Compiler Explorer: library '${lib.id}' declares link library '${linkName}' ` +
                `but it was not found in ${escapeCMakeString(libraryDirs.join(', '))}`;
            lines.push(
                `find_library(${variable}`,
                `  NAMES ${linkName}`,
                ...(libraryDirs.length > 0 ? [`  PATHS ${quotedArgs(libraryDirs)}`, '  NO_DEFAULT_PATH'] : []),
                ')',
                `if(NOT ${variable})`,
                // Silently omitting the target turns a missing binary into an inscrutable
                // undefined-reference at link time, long after the cause is visible.
                `  message(FATAL_ERROR "${notFound}")`,
                'endif()',
                `add_library(${name}::${linkName} UNKNOWN IMPORTED)`,
                `set_target_properties(${name}::${linkName} PROPERTIES`,
                `  IMPORTED_LOCATION "\${${variable}}"`,
                ...(includeDirs.length > 0 ? [`  INTERFACE_INCLUDE_DIRECTORIES ${quotedList(includeDirs)}`] : []),
                ')',
                `list(APPEND ${name}_LIBRARIES "\${${variable}}")`,
                '',
            );
        }

        // The aggregate target: what `find_package(foo)` users reach for as `foo::foo`. When the
        // library id is itself one of its link names - benchmark, re2, fmt at some versions - the
        // loop above has already created `foo::foo` as the imported library, and CMake rejects a
        // second add_library() for it. That target is then the aggregate, and linking it to itself
        // would be a cycle, so only the remaining link names are attached.
        if (!linkNames.includes(name)) {
            lines.push(`add_library(${name}::${name} INTERFACE IMPORTED)`);
            if (includeDirs.length > 0) {
                lines.push(
                    `set_target_properties(${name}::${name} PROPERTIES`,
                    `  INTERFACE_INCLUDE_DIRECTORIES ${quotedList(includeDirs)}`,
                    ')',
                );
            }
        }
        for (const linkName of linkNames.filter(linkName => linkName !== name)) {
            lines.push(
                `set_property(TARGET ${name}::${name} APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${name}::${linkName})`,
            );
        }
        lines.push(
            '',
            // Projects that were written against a plain `target_link_libraries(x foo)` still work.
            `if(NOT TARGET ${name})`,
            `  add_library(${name} INTERFACE IMPORTED)`,
            `  set_target_properties(${name} PROPERTIES INTERFACE_LINK_LIBRARIES ${name}::${name})`,
            'endif()',
            '',
        );

        return lines.join('\n');
    }

    renderConfigVersion(lib: CMakePackageLibrary): string {
        const version = lib.version.version || '0.0.0';
        return [
            `set(PACKAGE_VERSION ${quotedList([version])})`,
            '',
            'if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)',
            '  set(PACKAGE_VERSION_COMPATIBLE FALSE)',
            'else()',
            '  set(PACKAGE_VERSION_COMPATIBLE TRUE)',
            '  if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)',
            '    set(PACKAGE_VERSION_EXACT TRUE)',
            '  endif()',
            'endif()',
            '',
        ].join('\n');
    }

    /** Writes a package for one library and returns what was produced. */
    async generateOne(lib: CMakePackageLibrary): Promise<GeneratedCMakePackage | undefined> {
        if (!this.shouldGenerate(lib)) return undefined;

        const name = this.packageNameFor(lib);
        const prefixPath = path.join(this.dirPath, GENERATED_PACKAGE_DIRNAME, lib.id);
        const packageDir = path.join(prefixPath, 'lib', 'cmake', name);
        await fs.mkdir(packageDir, {recursive: true});

        const configFile = path.join(packageDir, `${name}Config.cmake`);
        const versionFile = path.join(packageDir, `${name}ConfigVersion.cmake`);
        await fs.writeFile(configFile, this.renderConfig(lib));
        await fs.writeFile(versionFile, this.renderConfigVersion(lib));

        return {
            libId: lib.id,
            packageName: name,
            version: lib.version.version || '0.0.0',
            prefixPath,
            configFile,
            versionFile,
            includeDirs: this.includeDirsFor(lib),
            libraryDirs: this.libraryDirsFor(lib),
            linkNames: this.linkNamesFor(lib),
        };
    }

    async generate(libraries: CMakePackageLibrary[]): Promise<GeneratedCMakePackage[]> {
        const generated: GeneratedCMakePackage[] = [];
        for (const lib of libraries) {
            const result = await this.generateOne(lib);
            if (result) generated.push(result);
        }
        return generated;
    }

    /**
     * What find_package() can actually resolve for each selected library, for explaining a failure.
     * Reads the filesystem, so this belongs on the failure path only.
     */
    async describePackages(libraries: CMakePackageLibrary[]): Promise<CMakePackageDescription[]> {
        const described: CMakePackageDescription[] = [];
        for (const lib of libraries) {
            if (this.shouldGenerate(lib)) {
                const name = this.packageNameFor(lib);
                described.push({
                    libId: lib.id,
                    packageNames: [name],
                    targets: unique([
                        `${name}::${name}`,
                        ...this.linkNamesFor(lib).map(linkName => `${name}::${linkName}`),
                        name,
                    ]),
                    linkNames: this.linkNamesFor(lib),
                    generated: true,
                });
                continue;
            }
            const prefix = this.installedPrefixFor(lib);
            if (!prefix) continue;
            const exported = await readExportedPackages(prefix);
            if (exported.packageNames.length > 0 || exported.targets.length > 0) {
                described.push({libId: lib.id, ...exported, linkNames: this.linkNamesFor(lib), generated: false});
            }
        }
        return described;
    }

    /**
     * Prefixes to append to CMAKE_PREFIX_PATH, real install trees first so a library's own config
     * always wins over a generated one.
     */
    async prefixPaths(libraries: CMakePackageLibrary[]): Promise<string[]> {
        const installed = libraries.map(lib => this.installedPrefixFor(lib)).filter((p): p is string => !!p);
        const generated = (await this.generate(libraries)).map(pkg => pkg.prefixPath);
        return unique([...installed, ...generated]);
    }
}
