FROM node:16-bookworm
MAINTAINER Chinmay Garde <chinmaygarde@gmail.com>

RUN apt update
RUN apt install -y spirv-tools clang-format

RUN mkdir -p /engine_artifacts
WORKDIR /engine_artifacts
RUN wget https://storage.googleapis.com/flutter_infra_release/flutter/e8c834be50bb12a7f521076d5fa8887b923c9cd5/linux-x64/artifacts.zip
RUN unzip artifacts.zip
RUN mv impellerc /usr/local/bin
RUN mv shader_lib /usr/local/include
RUN impellerc --help

COPY . /src
WORKDIR /src
RUN make prebuild

ENTRYPOINT make run-only EXTRA_ARGS="--language impeller --port ${PORT}"
