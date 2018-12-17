#!/bin/bash

set -ex

OPT=$(pwd)/.travis-compilers
mkdir -p ${OPT}
mkdir -p ${OPT}/tmp

fetch() {
    curl -v ${http_proxy:+--proxy $http_proxy} -L "$*"
}

get_ghc() {
    local VER=$1
    local DIR=ghc-$VER

    pushd ${OPT}/tmp
    fetch https://downloads.haskell.org/~ghc/${VER}/ghc-${VER}-x86_64-deb8-linux.tar.xz | tar Jxf -
    cd ${OPT}/tmp/ghc-${VER}
    ./configure --prefix=${OPT}/ghc
    make install
    rm -rf ${OPT}/ghc/lib/ghc-${VER}/Cabal*
    rm -rf ${OPT}/ghc/share
    popd
    rm -rf ${OPT}/tmp/ghc-${VER}
}

get_gdc() {
    vers=$1
    build=$2
    mkdir ${OPT}/gdc
    pushd ${OPT}/gdc
    fetch ftp://ftp.gdcproject.org/binaries/${vers}/x86_64-linux-gnu/gdc-${vers}+${build}.tar.xz | tar Jxf -
    popd
}

do_rust_install() {
    local DIR=$1
    pushd ${OPT}/tmp
    fetch http://static.rust-lang.org/dist/${DIR}.tar.gz | tar zxf -
    cd ${DIR}
    ./install.sh --prefix=${OPT}/rust --without=rust-docs
    popd
    rm -rf ${OPT}/tmp/${DIR}
}

install_new_rust() {
    local NAME=$1
    
    do_rust_install rust-${NAME}-x86_64-unknown-linux-gnu
}

if [[ ! -d ${OPT}/ghc/bin ]]; then
    get_ghc 8.0.2
fi
if [[ ! -d ${OPT}/gdc/x86_64-pc-linux-gnu/bin ]]; then
    get_gdc 5.2.0 2.066.1
fi
if [[ ! -d ${OPT}/rust/bin ]]; then
    install_new_rust 1.30.0
fi
