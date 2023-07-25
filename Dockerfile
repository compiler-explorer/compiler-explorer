FROM node:16-bookworm
MAINTAINER Chinmay Garde <chinmaygarde@gmail.com>

RUN mkdir -p /engine_artifacts
WORKDIR /engine_artifacts
RUN wget https://storage.googleapis.com/flutter_infra_release/flutter/4fded78e5a01f6cf90f0c7f0539b824abb95122d/linux-x64/artifacts.zip
RUN unzip artifacts.zip
RUN mv impellerc /usr/local/bin
RUN impellerc --help

COPY . /src
WORKDIR /src
RUN make prebuild

ENTRYPOINT make run-only EXTRA_ARGS="--language impeller --port ${PORT}"
