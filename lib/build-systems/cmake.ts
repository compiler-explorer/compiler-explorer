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

import {BuildSystems} from '../../shared/build-systems.js';
import {LIBRARIES_ICON_TOKEN} from '../../shared/common-utils.js';
import {assert} from '../assert.js';
import type {BaseCompiler} from '../base-compiler.js';
import {BaseBuildSystem} from './base.js';
import type {BuildContext, BuildPlan} from './build-system.interfaces.js';
import type {CMakePackageDescription} from './cmake-package-generator.js';

/** Every message has to fit a notification that fades, so these stay deliberately small. */
const MAX_PACKAGES_LISTED = 3;

/** Reduces a package name to letters and digits so `qt` can be matched against `Qt6`. */
function matchKey(name: string): string {
    return name.toLowerCase().replaceAll(/[^a-z0-9]/g, '');
}

/** The package the user most likely meant: an exact-ish match on what they typed. */
function closestPackage(requested: string, available: CMakePackageDescription[]): string | undefined {
    const wanted = matchKey(requested);
    const candidates = available.flatMap(entry => entry.packageNames);
    return (
        candidates.find(name => matchKey(name) === wanted) ??
        // `qt` -> `Qt6`: the shortest name that starts with what they typed is the base package,
        // not one of its many component packages.
        candidates.filter(name => matchKey(name).startsWith(wanted)).sort((a, b) => a.length - b.length)[0]
    );
}

/**
 * Puts the targets a user would plausibly link first. Two signals, both cheap: a target whose
 * namespace+name is itself an exported package is a real component (`Qt6::Core` pairs with the
 * `Qt6Core` package, while `Qt6::Platform` pairs with nothing), and anything marked Private or
 * Internal is plumbing.
 */
function rankTargets(targets: string[], packageNames: string[]): string[] {
    const isComponent = (target: string) => packageNames.includes(target.replace('::', ''));
    const isPlumbing = (target: string) => /Private|Internal|Tools$/.test(target);
    const isBaseModule = (target: string) => /::Core$/.test(target);
    return [...targets].sort((a, b) => {
        const score = (target: string) =>
            (isComponent(target) ? 0 : 4) + (isPlumbing(target) ? 2 : 0) + (isBaseModule(target) ? 0 : 1);
        return score(a) - score(b);
    });
}

/** Same idea for package names: a Private or Internal package is never the one being asked for. */
function rankPackages(packageNames: string[]): string[] {
    return [...packageNames].sort(
        (a, b) => Number(/Private|Internals?$|Tools$/.test(a)) - Number(/Private|Internals?$|Tools$/.test(b)),
    );
}

function summarise(values: string[], limit: number): string {
    if (values.length <= limit) return values.join(', ');
    return `${values.slice(0, limit).join(', ')} (+${values.length - limit} more)`;
}

export class CMakeBuildSystem extends BaseBuildSystem {
    readonly id = 'cmake' as const;
    readonly descriptor = BuildSystems.cmake;

    async getUnsupportedReason(compiler: BaseCompiler): Promise<string | undefined> {
        if (!compiler.compiler.supportsBinary) return 'Compiler does not support compiling to binaries';
        return undefined;
    }

    override getBuildPath(dirPath: string): string {
        return path.join(dirPath, 'build');
    }

    override async prepareBuildDirectory(ctx: BuildContext): Promise<void> {
        await fs.mkdir(ctx.buildPath);
    }

    async getBuildPlan(ctx: BuildContext): Promise<BuildPlan> {
        const compiler = ctx.compiler;

        const execParams = compiler.getDefaultExecOptions();
        execParams.appHome = ctx.dirPath;
        execParams.customCwd = ctx.buildPath;
        compiler.applyOverridesToExecOptions(execParams, ctx.parsedRequest.backendOptions.overrides || []);

        // Note this shares its `env` with execParams, so both steps see the compiler flags it adds.
        const makeExecParams = compiler.createCmakeExecParams(
            execParams,
            ctx.dirPath,
            ctx.libsAndOptions,
            ctx.toolchainPath,
        );

        const cmakePrefixPaths = await compiler.getCMakePrefixPaths(ctx.libsAndOptions.libraries, ctx.dirPath);
        if (cmakePrefixPaths.length > 0) {
            const existingPrefixPath = makeExecParams.env.CMAKE_PREFIX_PATH || process.env.CMAKE_PREFIX_PATH;
            makeExecParams.env.CMAKE_PREFIX_PATH = [existingPrefixPath, ...cmakePrefixPaths]
                .filter(Boolean)
                .join(path.delimiter);
        }

        const toolchainparam = compiler.getCMakeExtToolchainParam(ctx.parsedRequest.backendOptions.overrides || []);

        const partArgs: string[] = [
            toolchainparam,
            ...compiler.getExtraCMakeArgs(ctx.parsedRequest),
            ...ctx.buildSystemArgs,
            '..',
        ].filter(Boolean);
        const useNinja = ctx.env.ceProps('useninja');
        const fullArgs: string[] = useNinja ? ['-GNinja'].concat(partArgs) : partArgs;

        const cmd = ctx.env.ceProps('cmake') as string;
        assert(cmd, 'No cmake command found');

        return {
            getCompilationOptions: () => compiler.getUsedEnvironmentVariableFlags(makeExecParams),
            steps: [
                {
                    name: 'cmake',
                    exe: cmd,
                    args: fullArgs,
                    execParams: makeExecParams,
                    failureMessage: '<CMake configure step failed>',
                    reportsCompilationOptions: true,
                    explainFailure: async output =>
                        CMakeBuildSystem.explainFailure(
                            output,
                            await compiler.describeCMakePackages(ctx.libsAndOptions.libraries, ctx.dirPath),
                            compiler.unselectedLibraryIds(ctx.libsAndOptions.libraries),
                        ),
                },
                {
                    name: 'build',
                    exe: cmd,
                    args: ['--build', '.'],
                    execParams: execParams,
                    failureMessage: '<CMake build step failed>',
                    // The link step fails here, and what the selected libraries provide is the useful answer.
                    explainFailure: async output =>
                        CMakeBuildSystem.explainFailure(
                            output,
                            await compiler.describeCMakePackages(ctx.libsAndOptions.libraries, ctx.dirPath),
                            compiler.unselectedLibraryIds(ctx.libsAndOptions.libraries),
                        ),
                },
            ],
        };
    }

    /**
     * CMake names the package it could not find, but has no idea what Compiler Explorer does have.
     * Saying which package names and targets the selected libraries actually export turns a dead
     * end into the line the user needs to write - the CE library id is rarely the package name
     * (the `qt` library exports `Qt6`).
     */
    static explainFailure(
        output: string,
        available: CMakePackageDescription[],
        unselected: string[] = [],
    ): string | undefined {
        return (
            CMakeBuildSystem.explainMissingHeader(output, unselected) ??
            CMakeBuildSystem.explainMissingPackage(output, available, unselected) ??
            CMakeBuildSystem.explainMissingTarget(output, available) ??
            CMakeBuildSystem.explainLinkFailure(output, available)
        );
    }

    /**
     * A header that does not exist, where its first path component names a library Compiler Explorer has
     * but the user has not selected - `#include <fmt/core.h>` with no fmt. The include path is the only
     * evidence needed, so this also catches builds whose find_package was not REQUIRED and so only warned.
     */
    static explainMissingHeader(output: string, unselected: string[]): string | undefined {
        const missing = /fatal error: ([^:\n]+): No such file or directory/.exec(output);
        if (!missing) return undefined;
        // `fmt/core.h` -> `fmt`, `fmt.h` -> `fmt`: either spelling names the library.
        const stem = missing[1].split('/')[0].replace(/\.[^.]+$/, '');
        const wanted = matchKey(stem);
        const library = unselected.find(id => matchKey(id) === wanted);
        return library
            ? `Select "${library}" in the ${LIBRARIES_ICON_TOKEN} Libraries pane to get ${missing[1]}.`
            : undefined;
    }

    /**
     * `target_link_libraries(x foo::bar)` with a name that does not exist. CMake reports it at the end of the
     * configure step, and the fix is the same shape as a mistyped package: name the closest target we do have.
     */
    static explainMissingTarget(output: string, available: CMakePackageDescription[]): string | undefined {
        // CMake puts the name on its own line ("Target "x" links to:\n\n  Foo::Bar\n\nbut the target
        // was not found"); older versions used a single-line form with the name quoted.
        const missing =
            /links to target "([^"]+)" but the target was not found/.exec(output) ??
            /links to:\s+(\S+)\s+but the target was not found/.exec(output);
        if (!missing) return undefined;
        const requested = missing[1];
        const targets = available.flatMap(entry => entry.targets);
        if (targets.length === 0) return `No target "${requested}" here, and none are available.`;

        const wanted = matchKey(requested);
        const closest =
            targets.find(target => matchKey(target) === wanted) ??
            targets.filter(target => matchKey(target).includes(wanted.split('::').pop() ?? wanted))[0];
        // The target exists in a selected library but CMake has not imported it. Say where it comes from
        // rather than what their CMakeLists should say: we only ever see build output, never the file, and
        // the call may be behind a variable, a branch or an include. Never answer "did you mean X" with the
        // name they already wrote.
        if (closest === requested) {
            const [namespace, component] = requested.split('::');
            return component
                ? `${requested} comes from find_package(${namespace} COMPONENTS ${component}).`
                : `${requested} comes from find_package(${requested}).`;
        }
        if (closest) return `No target "${requested}" - did you mean ${closest}?`;
        // Nothing resembles it, so offer the single most plausible target rather than a list: the names are
        // long (Qt6::GlobalConfigPrivate) and several of them do not fit a notification.
        const best = rankTargets(
            targets,
            available.flatMap(entry => entry.packageNames),
        )[0];
        return `No target "${requested}" - try ${best}.`;
    }

    /**
     * Link-step failures. Two are worth answering: a `-l` the linker cannot resolve, where we know what the
     * selected libraries actually provide, and a missing shared-library dependency of a library we shipped,
     * which the user cannot fix from their CMakeLists at all.
     */
    static explainLinkFailure(output: string, available: CMakePackageDescription[]): string | undefined {
        const missingSo = /warning: (lib\S+\.so[.\d]*), needed by \S+, not found/.exec(output);
        if (missingSo) return `${missingSo[1]} is missing here - please report it.`;

        // Qt's headers emit a reference to qt_version_tag, which only libQt6Core defines. Seeing it
        // undefined means the headers were used without the library ever being linked - typically a
        // find_package(Qt6) with no matching target_link_libraries.
        if (/undefined\s+reference to .?qt_version_tag/.test(output)) {
            const qtTarget = available.flatMap(entry => entry.targets).find(target => /^Qt\d*::Core$/.test(target));
            // Give the line to write, not just the name: there is usually no target_link_libraries at
            // all yet, so naming the target alone leaves the user to guess the call and the target name.
            return `Qt is not linked: add target_link_libraries(output.s PRIVATE ${qtTarget ?? 'Qt6::Core'})`;
        }

        // A mangled C++ symbol carries its namespace, and if that names a library the user selected then
        // the headers were found but the library never linked: fmt::v12::vprint means fmt is on the
        // include path with nothing on the link line. Only fires when the namespace matches a selected
        // library, so an undefined reference to the user's own code stays unexplained.
        const undefinedSymbol = /undefined\s+reference to [`'"]?([A-Za-z_]\w*)::/.exec(output);
        if (undefinedSymbol) {
            const namespace = matchKey(undefinedSymbol[1]);
            const owner = available.find(
                entry =>
                    matchKey(entry.libId) === namespace ||
                    entry.targets.some(target => matchKey(target.split('::')[0]) === namespace),
            );
            const target = owner?.targets.find(
                candidate => candidate.includes('::') && matchKey(candidate.split('::')[0]) === namespace,
            );
            if (owner && target) {
                return `${owner.libId} is not linked: add target_link_libraries(output.s PRIVATE ${target})`;
            }
        }

        // Not \S+: ld writes `cannot find -lfmt: No such file`, and the colon is not part of the name.
        const missingFlag = /cannot find -l([\w.+-]+)/.exec(output);
        if (missingFlag) {
            const best = available.flatMap(entry => entry.targets)[0];
            if (best) return `No -l${missingFlag[1]} - link ${best} instead.`;
            return `No -l${missingFlag[1]} here - select a library first.`;
        }

        return undefined;
    }

    private static explainMissingPackage(
        output: string,
        available: CMakePackageDescription[],
        unselected: string[],
    ): string | undefined {
        // A version mismatch has its own wording, and already tells us the package itself was found.
        const versionMismatch = /Could not find a configuration file for package "([^"]+)" that is compatible/.exec(
            output,
        );
        if (versionMismatch) return `${versionMismatch[1]} is here - check the version you asked for.`;

        const configMode = /Could not find a package configuration file provided by "([^"]+)"/.exec(output);
        const moduleMode = /By not providing "Find([^"]+)\.cmake"/.exec(output);
        const requested = configMode?.[1] ?? moduleMode?.[1];
        if (!requested) return undefined;

        const suggestion = closestPackage(requested, available);
        if (suggestion) {
            if (suggestion === requested) return `${requested} is here - check the version you asked for.`;
            const owner = available.find(entry => entry.packageNames.includes(suggestion));
            const targets = rankTargets(
                (owner?.targets ?? []).filter(target => target.split('::')[0] === suggestion),
                owner?.packageNames ?? [],
            );
            const linkWith = targets.length > 0 ? ` Link ${targets[0]}.` : '';
            return `Did you mean find_package(${suggestion})?${linkWith}`;
        }

        // The commonest mistake of all: the library exists here, it just was not selected.
        const wanted = matchKey(requested);
        const notSelected = unselected.find(id => matchKey(id) === wanted);
        if (notSelected) return `Select "${notSelected}" in the ${LIBRARIES_ICON_TOKEN} Libraries pane.`;

        if (available.length === 0) {
            return `No libraries selected - pick one in the ${LIBRARIES_ICON_TOKEN} Libraries pane.`;
        }

        const names = available.flatMap(entry => rankPackages(entry.packageNames));
        return `No package "${requested}" here - try ${summarise(names, MAX_PACKAGES_LISTED)}.`;
    }

    override getArtifactFilename(ctx: BuildContext): string {
        return ctx.compiler.getExecutableFilename(ctx.buildPath, ctx.compiler.outputFilebase, ctx.key);
    }
}
