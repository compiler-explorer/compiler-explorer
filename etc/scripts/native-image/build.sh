#!/usr/bin/env bash
# Copyright (c) 2026, Compiler Explorer Authors
# All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
set -euo pipefail

graal_home=${1:?Usage: build.sh GRAALVM_HOME OUTPUT_DIRECTORY}
output_dir=${2:?Usage: build.sh GRAALVM_HOME OUTPUT_DIRECTORY}
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$output_dir/classes"
exports=()
for package in util.json code graph debug; do
    exports+=("--add-exports=jdk.graal.compiler/jdk.graal.compiler.$package=ALL-UNNAMED")
done
for package in meta meta.annotation code code.site; do
    exports+=("--add-exports=jdk.internal.vm.ci/jdk.vm.ci.$package=ALL-UNNAMED")
done
"$graal_home/bin/javac" -XDignore.symbol.file \
    --add-modules=jdk.graal.compiler,jdk.internal.vm.ci "${exports[@]}" \
    -cp "$graal_home/lib/svm/builder/*" -d "$output_dir/classes" "$source_dir/ExplorerFeature.java"
"$graal_home/bin/jar" --create --file "$output_dir/explorer-feature.jar" -C "$output_dir/classes" .
"$graal_home/bin/native-image" --version > "$output_dir/toolchain-version.txt"
