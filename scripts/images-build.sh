#!/usr/bin/env bash
# Build the pieces `cargo` cannot, then hand off to cargo with its arguments.
#
#     ./scripts/images-build.sh run --release --features images
#
# llama.cpp and stable-diffusion.cpp both embed ggml, and two static copies in
# one binary collide on every symbol. Both can consume an external one instead,
# so this builds ggml once and points each at it. The catch is GGML_MAX_NAME:
# it sits inside ggml_tensor, sd.cpp static_asserts it is at least 160, and
# upstream defaults to 64 — every object that sees ggml.h must agree, or they
# disagree about how big a tensor is. It is behind #ifndef, so a define is all
# it takes.
set -euo pipefail
cd "$(dirname "$0")/.."

PREFIX="$PWD/target/sd-prefix"
DEF="-DGGML_MAX_NAME=160"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

# cmake wants native paths. Under git-bash on Windows this script sees
# /d/a/offgrid/... , which cmake.exe cannot open; cygpath turns that back into
# a drive letter. Everywhere else it is the identity.
cmpath() {
    if command -v cygpath > /dev/null 2>&1; then
        cygpath -m "$1"
    else
        printf '%s' "$1"
    fi
}

# On macOS the shared ggml keeps Metal, or an images build would take GPU
# acceleration away from chat in the same binary, which is a poor trade for a
# feature you might never open. sd.cpp picks the backend up from the ggml it
# links against, so it needs no flag of its own.
GGML_GPU=()
if [[ "$(uname -s 2>/dev/null)" == "Darwin" ]]; then
    GGML_GPU=(-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON)
fi

# llama.cpp's own ggml is the one to share: sd.cpp has an upstream-ggml mode
# for exactly this, while llama.cpp has no mode for sd.cpp's patched fork.
# Windows runners ship `python`, not always `python3`.
PY="$(command -v python3 || command -v python)"
if [[ -z "$PY" ]]; then
    echo "error: need python to locate llama-cpp-sys-2's sources" >&2
    exit 1
fi

llama_src() {
    cargo metadata --format-version 1 --no-deps > /dev/null 2>&1 || true
    cargo metadata --format-version 1 2>/dev/null | "$PY" -c '
import json, sys, pathlib
meta = json.load(sys.stdin)
for pkg in meta["packages"]:
    if pkg["name"] == "llama-cpp-sys-2":
        print(pathlib.Path(pkg["manifest_path"]).parent / "llama.cpp")
        break
'
}

LLAMA="$(llama_src)"
if [[ -z "$LLAMA" || ! -d "$LLAMA" ]]; then
    echo "error: could not find llama-cpp-sys-2's vendored llama.cpp" >&2
    exit 1
fi

# stable-diffusion.cpp: a checkout you provide, or one fetched next to the
# build. Nothing here is vendored into the repo yet — that is a decision for
# when this stops being a prototype.
SD_SRC="${OFFGRID_SD_SRC:-$PWD/target/stable-diffusion.cpp}"
# Pinned, not master: this is the revision the model list was tested against,
# and sd.cpp moves fast enough that an unpinned build would be a different
# program every week. Bump it deliberately, after running the models.
SD_REF="${OFFGRID_SD_REF:-c92d73c408515c94beef32161bb5960764fde7a0}"
if [[ ! -d "$SD_SRC" ]]; then
    echo "==> fetching stable-diffusion.cpp $SD_REF into $SD_SRC"
    git init -q "$SD_SRC"
    git -C "$SD_SRC" remote add origin https://github.com/leejet/stable-diffusion.cpp
    git -C "$SD_SRC" fetch -q --depth 1 origin "$SD_REF"
    git -C "$SD_SRC" checkout -q FETCH_HEAD
fi

# The marker differs by toolchain, as does everything else about a static
# library's name.
if ! ls "$PREFIX"/lib/libggml-base.a "$PREFIX"/lib/ggml-base.lib > /dev/null 2>&1; then
    echo "==> building ggml (shared by llama.cpp and stable-diffusion.cpp)"
    cmake -S "$(cmpath "$LLAMA")" -B target/ggml-build "${GGML_GPU[@]}" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$(cmpath "$PREFIX")" \
        -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_C_FLAGS="$DEF" -DCMAKE_CXX_FLAGS="$DEF" \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_COMMON=OFF -DLLAMA_CURL=OFF
    cmake --build target/ggml-build --config Release -j"$JOBS" --target install
fi

if ! ls "$PREFIX"/lib/libstable-diffusion.a "$PREFIX"/lib/stable-diffusion.lib > /dev/null 2>&1; then
    echo "==> building stable-diffusion.cpp against that ggml"
    cmake -S "$(cmpath "$SD_SRC")" -B target/sd-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DSD_USE_SYSTEM_GGML=ON -DSD_USE_UPSTREAM_GGML=ON \
        -DSD_GGML_SOURCE_DIR="$(cmpath "$LLAMA/ggml")" -DSD_BUILD_EXAMPLES=OFF \
        -DCMAKE_C_FLAGS="$DEF" -DCMAKE_CXX_FLAGS="$DEF" \
        -DCMAKE_PREFIX_PATH="$(cmpath "$PREFIX")"
    cmake --build target/sd-build --config Release -j"$JOBS"
    # libstable-diffusion.a on unix, stable-diffusion.lib under MSVC, and the
    # latter lands in a per-configuration subdirectory.
    SD_LIB="$(find target/sd-build \( -name "libstable-diffusion.a" -o -name "stable-diffusion.lib" \) | head -1)"
    if [[ -z "$SD_LIB" ]]; then
        echo "error: stable-diffusion.cpp built but produced no static library" >&2
        exit 1
    fi
    cp "$SD_LIB" "$PREFIX/lib/"
    cp "$SD_SRC/include/stable-diffusion.h" "$PREFIX/include/"
fi

echo "==> cargo $*"
# CFLAGS/CXXFLAGS reach llama.cpp's own build through the cc crate, so its
# ggml.h sees the same GGML_MAX_NAME as the library it will link against.
export OFFGRID_SD_PREFIX="$PREFIX"
export CMAKE_PREFIX_PATH="$PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CFLAGS="$DEF ${CFLAGS:-}"
export CXXFLAGS="$DEF ${CXXFLAGS:-}"
exec cargo "$@"
