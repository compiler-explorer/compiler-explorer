#!/usr/bin/env bash
# Copyright (c) 2026, Compiler Explorer Authors
# All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
set -euo pipefail

graal_home=$(realpath "${1:?Usage: build-layer.sh GRAALVM_HOME OUTPUT_DIRECTORY}")
output_dir=${2:?Usage: build-layer.sh GRAALVM_HOME OUTPUT_DIRECTORY}
mkdir -p "$output_dir"
cd "$output_dir"
"$graal_home/bin/native-image" \
    --parallelism=4 -J-Xmx6g -march=x86-64 -O2 \
    -H:+UnlockExperimentalVMOptions -H:+TrackNodeSourcePosition \
    -H:LayerCreate=base.nil,module=java.base -o libce-base
"$graal_home/bin/native-image" --version > toolchain-version.txt
